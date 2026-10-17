"""
text_reasoner.py — Voie cognitive perception → raisonnement.

    texte ─► PerceptionCore.perceive_input   (type d'entrée, mots-clés)
          ─► PremiseExtractor.extract         (faits, règles, buts)
          ─► ReasoningCore.reason             (stratégies existantes : déduction…)
          ─► verdict par but : oui / non / inconnu / contradictoire
          ─► écritures : graphe partagé (is_a, can), mémoire de travail, bus d'événements

Rien n'est réimplémenté côté raisonnement : la déduction reste celle de
ReasoningCore ; ce module fait la traduction et la confrontation au but.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from grandnexus.cognition.contracts import (
    CognitiveContext, CognitiveMode, CognitiveModuleBase, CognitiveProcessType,
    CognitiveRequest, cognitive_capability,
)
from grandnexus.cognition.perception.premise_extractor import PremiseExtractor


def _negate(p: str) -> str:
    return p[4:] if p.startswith("not ") else "not " + p


class TextReasoner(CognitiveModuleBase):

    def __init__(self, reasoning_core: Any = None, perception_core: Any = None,
                 extractor: Optional[PremiseExtractor] = None, graph_engine: Any = None,
                 working_memory: Any = None, nexus_core: Any = None, **kwargs):
        self.reasoning_core = reasoning_core
        self.perception_core = perception_core
        self.extractor = extractor or PremiseExtractor(nexus_core=nexus_core)
        self.graph_engine = graph_engine
        self.working_memory = working_memory
        self.nexus_core = nexus_core
        self.logger = logging.getLogger("GrandNexus.TextReasoner")
        self._node_ids: Dict[str, str] = {}

    # ------------------------------------------------------------------------
    def answer(self, text: str, max_steps: int = 8, write_back: bool = True) -> Dict[str, Any]:
        t0 = time.time()
        perception = None
        if self.perception_core is not None:
            try:
                ctx = CognitiveContext()
                perception = self.perception_core.perceive_input(CognitiveRequest(query=text, context=ctx), ctx, text)
            except Exception as e:
                perception = {"error": str(e)}
        ex = self.extractor.extract(text)
        derived: List[Dict[str, Any]] = []
        reasoning: Dict[str, Any] = {}
        if self.reasoning_core is not None and ex.facts and ex.rules:
            ctx = CognitiveContext(parameters={
                "premises": list(ex.facts), "rules": list(ex.rules),
                "max_steps": max_steps, "exploration_factor": 0.0,
            })
            reasoning = self.reasoning_core.reason(CognitiveRequest(query=text, context=ctx), ctx, text)
            derived = list(reasoning.get("conclusions", []) or [])
        known = {f: {"statement": f, "confidence": 1.0, "from": "prémisse"} for f in ex.facts}
        for c in derived:
            st = c.get("statement")
            if st and st not in known:
                known[st] = {"statement": st, "confidence": c.get("confidence", 0.95), "from": c.get("derived_from")}

        verdicts = []
        for g in ex.goals:
            pos, neg = g in known, _negate(g) in known
            if pos and neg:
                v, conf = "contradictoire", 0.0
            elif pos:
                v, conf = ("oui" if not g.startswith("not ") else "non"), known[g]["confidence"]
            elif neg:
                v, conf = ("non" if not g.startswith("not ") else "oui"), known[_negate(g)]["confidence"]
            else:
                v, conf = "inconnu", 0.0
            verdicts.append({"goal": g, "answer": v, "confidence": conf,
                             "support": [known[x] for x in (g, _negate(g)) if x in known]})

        out = {
            "answer": verdicts[0]["answer"] if verdicts else ("faits enregistrés" if ex.facts else "inconnu"),
            "verdicts": verdicts,
            "confidence": verdicts[0]["confidence"] if verdicts else 0.0,
            "extraction": {"facts": ex.facts, "rules": [(r["if"], r["then"]) for r in ex.rules],
                           "goals": ex.goals, "unparsed": ex.unparsed, "trace": ex.trace},
            "derived": [c.get("statement") for c in derived],
            "strategies_used": reasoning.get("strategies_used", []),
            "input_type": (perception or {}).get("input_type") if isinstance(perception, dict) else None,
            "processing_time": time.time() - t0,
        }
        if write_back:
            self._write_back(text, ex.facts, out)
        return out

    # ------------------------------------------------------------------------
    def _node(self, name: str) -> Optional[str]:
        if self.graph_engine is None:
            return None
        if name not in self._node_ids:
            self._node_ids[name] = self.graph_engine.add_node(label="Concept",
                                                               properties={"name": name, "source": "text_reasoner"})
        return self._node_ids[name]

    def _write_back(self, text: str, facts: List[str], out: Dict[str, Any]) -> None:
        # graphe partagé : relations is_a / can (positives et négatives)
        statements = list(facts) + [d for d in out["derived"] if d]
        for st in statements:
            neg = st.startswith("not ")
            core = st[4:] if neg else st
            if core.startswith(("is(", "can(")) and core.endswith(")"):
                kind, args = core.split("(", 1)
                parts = args[:-1].split(",")
                if len(parts) != 2:
                    continue
                try:
                    a, b = self._node(parts[0]), self._node(parts[1])
                    if a and b:
                        self.graph_engine.add_edge(a, b, ("not_" if neg else "") + ("is_a" if kind == "is" else "can"),
                                                   properties={"source": "text_reasoner",
                                                               "derived": st not in facts})
                except Exception as e:
                    self.logger.debug(f"écriture graphe: {e}")
        # mémoire de travail : relu par l'orchestrateur (requête « reasoning »)
        if self.working_memory is not None:
            try:
                self.working_memory.store_item(content={"question": text, "answer": out["answer"],
                                                        "verdicts": out["verdicts"], "kind": "reasoning"},
                                               source="reasoning", importance=max(0.3, out["confidence"]))
            except Exception as e:
                self.logger.debug(f"écriture mémoire de travail: {e}")
        if self.nexus_core is not None and hasattr(self.nexus_core, "publish_event"):
            self.nexus_core.publish_event("reasoning.conclusion", {"question": text, "answer": out["answer"],
                                                                    "derived": out["derived"]},
                                          source="text_reasoner")

    # ------------------------------------------------------------------------
    @cognitive_capability(
        capability_type=CognitiveProcessType.REASONING,
        description="Raisonner sur un texte : prémisses extraites puis déduction",
        priority=30,
        compatible_modes={CognitiveMode.DELIBERATIVE, CognitiveMode.REACTIVE},
    )
    def reason_from_text(self, request, context, query, **kwargs) -> Dict[str, Any]:
        return self.answer(str(query), max_steps=int(context.get("max_steps", 8) or 8))
