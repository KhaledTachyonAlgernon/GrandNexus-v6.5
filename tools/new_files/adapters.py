"""
adapters.py — Les trois classes que le CognitiveOrchestrator (15/05) importe
et qui n'ont jamais existé : ReasoningEngine, LearningManager, PerceptionModule.

Elles ne réimplémentent rien : elles traduisent l'API « tâche dict » de
l'orchestrateur vers les modules du cœur (29/05) et la voie texte.

    orchestrateur.perception_module.process(task)  -> PerceptionCore + PremiseExtractor
    orchestrateur.reasoning_engine.reason(task)    -> TextReasoner (-> ReasoningCore)
    orchestrateur.learning_manager.learn(task)     -> LearningCore.submit_learning_task

L'entrée courante est celle que l'orchestrateur range en mémoire de travail
(source "external_input") dans process_input().
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set

from grandnexus.cognition.contracts import CognitiveContext, CognitiveRequest

_log = logging.getLogger("GrandNexus.Adapters")


def latest_input(working_memory: Any, source: str = "external_input") -> Any:
    """Dernier élément rangé par l'orchestrateur en mémoire de travail."""
    if working_memory is None:
        return None
    try:
        items = [i for i in working_memory.get_all_items() if getattr(i, "source", None) == source]
    except Exception:
        return None
    if not items:
        return None
    items.sort(key=lambda i: getattr(i, "creation_time", getattr(i, "timestamp", 0.0)))
    return items[-1].content


class PerceptionModule:
    def __init__(self, perception_core: Any = None, extractor: Any = None, working_memory: Any = None):
        self.perception_core = perception_core
        self.extractor = extractor
        self.working_memory = working_memory
        self._seen_tokens: Set[str] = set()
        self._last_processed: Any = None

    def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        data = task.get("input") if isinstance(task, dict) else None
        if data is None:
            data = latest_input(self.working_memory)
        if data is None:
            return {"success": False, "error": "aucune entrée à percevoir"}
        text = data.get("query") if isinstance(data, dict) and "query" in data else data
        ctx = CognitiveContext()
        perceived = {}
        if self.perception_core is not None:
            perceived = self.perception_core.perceive_input(CognitiveRequest(query=text, context=ctx), ctx, text)
        premises = self.extractor.extract(str(text)).to_dict() if self.extractor is not None else {}
        for k in ("universals", "conditionals"):  # objets internes, non sérialisés en mémoire
            premises.pop(k, None)
        toks = set(str(text).lower().split())
        novelty = 1.0 if not self._seen_tokens else len(toks - self._seen_tokens) / max(1, len(toks))
        self._seen_tokens |= toks
        repeated = text == self._last_processed
        self._last_processed = text
        return {"success": True, "novelty": 0.0 if repeated else round(novelty, 3),
                "processed_data": {"text": text, "input_type": perceived.get("input_type"),
                                   "keywords": (perceived.get("parsed_data") or {}).get("potential_keywords"),
                                   "premises": premises}}


class ReasoningEngine:
    def __init__(self, text_reasoner: Any = None, working_memory: Any = None):
        self.text_reasoner = text_reasoner
        self.working_memory = working_memory
        self._answered: Dict[str, Dict[str, Any]] = {}

    def reason(self, task: Dict[str, Any]) -> Dict[str, Any]:
        data = task.get("input") if isinstance(task, dict) else None
        if data is None:
            data = latest_input(self.working_memory)
        text = data.get("query") if isinstance(data, dict) and "query" in data else data
        if not text or self.text_reasoner is None:
            return {"success": False, "error": "rien à raisonner"}
        key = str(text)
        if key in self._answered:            # l'orchestrateur relance souvent : pas de recalcul
            out = self._answered[key]
        else:
            out = self.text_reasoner.answer(key)
            self._answered[key] = out
        decided = out["answer"] in ("oui", "non", "contradictoire")
        return {"success": decided or bool(out["derived"]),
                "conclusions": [{"question": key, "answer": out["answer"], "verdicts": out["verdicts"]}],
                "confidence": out["confidence"], "task_type": task.get("task_type") if isinstance(task, dict) else None}


class LearningManager:
    def __init__(self, learning_core: Any = None):
        self.learning_core = learning_core
        self.submitted: List[str] = []

    def learn(self, task: Dict[str, Any]) -> Dict[str, Any]:
        if self.learning_core is None:
            return {"success": False, "error": "LearningCore absent"}
        try:
            from grandnexus.cognition.learning.learning_core import LearningTask
            lt = LearningTask(description=task.get("objective", task.get("name", "")),
                              parameters=dict(task), priority=0.5)
            task_id = self.learning_core.submit_learning_task(lt)
            self.submitted.append(task_id)
            return {"success": True, "task_id": task_id, "metrics": {"improvement": 0.0}}
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {e}"}
