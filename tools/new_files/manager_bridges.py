"""
manager_bridges.py — Le CognitionManager appelle 12 méthodes que ses modules
n'ont jamais eues (quick_scan, evaluate, adapt, reflect_on, recognize,
process_text, create_plan…). Chaque pont ci-dessous DÉLÈGUE à une méthode qui
existe déjà dans le module ; il n'est installé que si la méthode manque.

Correspondances (appel du manager -> méthode réelle) :
  CognitiveMonitor.quick_scan()                  -> get_active_alerts()
  CognitiveMonitor.perform_monitoring(mods, m)   -> get_current_value() + alertes + dernier rapport
  ConfidenceEvaluator.evaluate(content, ctx)     -> evaluate_perception/reasoning/memory_confidence
  ConfidenceEvaluator.get_global_uncertainty()   -> 1 - moyenne de confidence_history (0.5 si vide)
  AdaptationEngine.adapt(module, params, ctx)    -> set_parameter() ou apply_suggestions()
  ReflectionModule.reflect_on(topic, depth, ctx) -> reflect_on_process()
  PatternRecognizer.recognize(data, ctx)         -> process()
  PerceptionCore.process_text(text, metadata)    -> perceive_input()
  PerceptionCore.shift_attention(focus)          -> mémorise le focus (aucun équivalent existant)
  PlanningModule.create_plan(goal, constraints)  -> create_plan_for_goal()
  PlanningModule.execute_step(goal, step)        -> execute_plan() du plan de ce but
  PlanningModule.revise_plan(goal, why, constr.) -> cancel_plan() puis create_plan_for_goal()
"""

import logging
import statistics
import time
from typing import Any, Dict, List, Optional

_log = logging.getLogger("GrandNexus.ManagerBridges")


def _install(cls, name, fn):
    if not hasattr(cls, name):
        fn.__name__ = name
        fn.__doc__ = (fn.__doc__ or "") + "\n(pont 6.5, voir manager_bridges.py)"
        setattr(cls, name, fn)
        return 1
    return 0


# --- métacognition ----------------------------------------------------------
def _quick_scan(self) -> Dict[str, Any]:
    alerts = self.get_active_alerts() or []
    issues = [getattr(a, "__dict__", a) for a in alerts]
    return {"issues": issues, "alert_count": len(issues), "timestamp": time.time()}


def _perform_monitoring(self, target_modules: List[str] = None, metrics: List[str] = None) -> Dict[str, Any]:
    values = {}
    for m in metrics or []:
        try:
            values[m] = self.get_current_value(m)
        except Exception as e:
            values[m] = f"indisponible: {e}"
    report = None
    try:
        report = self.get_latest_report()
    except Exception:
        pass
    return {"targets": list(target_modules or []), "metrics": values,
            "alerts": _quick_scan(self)["issues"], "latest_report": getattr(report, "__dict__", report)}


def _evaluate(self, content: Any, context: Optional[Dict[str, Any]] = None) -> Any:
    context = context or {}
    if isinstance(content, dict) and ("conclusions" in content or "verdicts" in content):
        return self.evaluate_reasoning_confidence(content, context)
    if isinstance(content, dict) and ("input_type" in content or "parsed_data" in content):
        return self.evaluate_perception_confidence(content, context)
    return self.evaluate_memory_confidence(content, context)


def _global_uncertainty(self) -> float:
    vals = []
    for dq in getattr(self, "confidence_history", {}).values():
        vals.extend(getattr(s, "value", None) for s in dq)
    vals = [v for v in vals if isinstance(v, (int, float))]
    return 0.5 if not vals else round(1.0 - statistics.mean(vals), 4)


def _adapt(self, module: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None,
           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    results = {}
    if parameters:
        for name, value in parameters.items():
            for key in (f"{module}.{name}" if module else None, name):
                if key and self.get_parameter(key) is not None:
                    results[key] = self.set_parameter(key, value, reason=f"CognitionManager: {context or ''}")
                    break
            else:
                results[name] = "paramètre inconnu"
        return {"applied": results}
    return {"suggestions_applied": self.apply_suggestions()}


def _reflect_on(self, topic: Any, depth: int = 1, context: Optional[Dict[str, Any]] = None) -> Any:
    return self.reflect_on_process({"topic": topic, "depth": depth, "context": context or {}})


def _recognize(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
    return self.process(data, context=context or {})


# --- perception -------------------------------------------------------------
def _process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from grandnexus.cognition.contracts import CognitiveContext, CognitiveRequest
    ctx = CognitiveContext(metadata=dict(metadata or {}))
    return self.perceive_input(CognitiveRequest(query=text, context=ctx, metadata=dict(metadata or {})), ctx, text)


def _shift_attention(self, focus_items: Dict[str, float]) -> bool:
    self.attention_focus = dict(focus_items or {})
    return True


# --- planification ----------------------------------------------------------
def _plans_for(self, goal_id: str):
    return [p for p in getattr(self, "plans", {}).values() if getattr(p, "goal_id", None) == goal_id]


def _create_plan(self, goal_id: str, constraints: Optional[Dict[str, Any]] = None) -> Optional[str]:
    return self.create_plan_for_goal(goal_id, context=constraints or {})


def _execute_step(self, goal_id: str, step_id: Optional[str] = None) -> Any:
    plans = _plans_for(self, goal_id)
    if not plans:
        return {"success": False, "error": f"aucun plan pour le but {goal_id}"}
    return self.execute_plan(plans[-1].plan_id)


def _revise_plan(self, goal_id: str, reason: Optional[str] = None,
                 constraints: Optional[Dict[str, Any]] = None) -> Optional[str]:
    for p in _plans_for(self, goal_id):
        try:
            self.cancel_plan(p.plan_id)
        except Exception as e:
            _log.debug(f"cancel_plan({p.plan_id}): {e}")
    return self.create_plan_for_goal(goal_id, context={**(constraints or {}), "revision_reason": reason})


def install() -> int:
    from grandnexus.cognition.metacognition.monitor import CognitiveMonitor
    from grandnexus.cognition.metacognition.confidence import ConfidenceEvaluator
    from grandnexus.cognition.metacognition.adaptation import AdaptationEngine
    from grandnexus.cognition.metacognition.reflection import ReflectionModule
    from grandnexus.cognition.perception.pattern_recognizer import PatternRecognizer
    from grandnexus.cognition.perception.perception_core import PerceptionCore
    from grandnexus.cognition.reasoning.planning import PlanningModule
    n = 0
    n += _install(CognitiveMonitor, "quick_scan", _quick_scan)
    n += _install(CognitiveMonitor, "perform_monitoring", _perform_monitoring)
    n += _install(ConfidenceEvaluator, "evaluate", _evaluate)
    n += _install(ConfidenceEvaluator, "get_global_uncertainty", _global_uncertainty)
    n += _install(AdaptationEngine, "adapt", _adapt)
    n += _install(ReflectionModule, "reflect_on", _reflect_on)
    n += _install(PatternRecognizer, "recognize", _recognize)
    n += _install(PerceptionCore, "process_text", _process_text)
    n += _install(PerceptionCore, "shift_attention", _shift_attention)
    n += _install(PlanningModule, "create_plan", _create_plan)
    n += _install(PlanningModule, "execute_step", _execute_step)
    n += _install(PlanningModule, "revise_plan", _revise_plan)
    return n
