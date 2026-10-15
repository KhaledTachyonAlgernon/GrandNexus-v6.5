"""
event_bus.py — Méthodes de bus d'événements appelées par graph_module_manager et
graph_pattern_interface mais absentes de NexusCore : register_handler,
unregister_handler, publish_event, schedule_task.

Elles sont greffées sur NexusCore par install_event_bus(NexusCore) à la fin de
nexus_core.py. Le send_message/handle_message existant n'est pas modifié :
il reste le canal point-à-point, ceci est le canal publication/abonnement.

Motifs : "graph.add_node" exact, "graph.*" préfixe, "*" tout.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional

_log = logging.getLogger("GrandNexus.EventBus")


def _bus(self) -> Dict[str, Any]:
    b = self.__dict__.get("_gn_bus")
    if b is None:
        b = {"handlers": defaultdict(list), "lock": threading.RLock(),
             "history": deque(maxlen=500), "timers": {}, "stats": defaultdict(int)}
        self.__dict__["_gn_bus"] = b
    return b


def register_handler(self, event_type: str, handler: Callable[[Dict[str, Any]], Any]) -> None:
    b = _bus(self)
    with b["lock"]:
        if handler not in b["handlers"][event_type]:
            b["handlers"][event_type].append(handler)


def unregister_handler(self, event_type: str, handler: Callable) -> bool:
    b = _bus(self)
    with b["lock"]:
        if handler in b["handlers"].get(event_type, []):
            b["handlers"][event_type].remove(handler)
            return True
    return False


def _matching(b, event_type: str) -> List[Callable]:
    out = []
    for pattern, hs in b["handlers"].items():
        if pattern == event_type or pattern == "*" or (
                pattern.endswith(".*") and event_type.startswith(pattern[:-1])):
            out.extend(hs)
    return out


def publish_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None,
                  source: Optional[str] = None) -> int:
    """Diffuse un événement à tous les abonnés. Renvoie le nombre de handlers appelés.
    Le handler reçoit un dict : le payload + 'event_type', 'source', 'timestamp'."""
    b = _bus(self)
    message = dict(payload or {})
    message.setdefault("event_type", event_type)
    message.setdefault("source", source)
    message.setdefault("timestamp", time.time())
    with b["lock"]:
        handlers = list(_matching(b, event_type))
        b["history"].append((event_type, source, message["timestamp"]))
        b["stats"][event_type] += 1
    for h in handlers:
        try:
            h(message)
        except Exception as e:
            _log.error(f"handler {getattr(h, '__qualname__', h)} sur '{event_type}': {e}")
    return len(handlers)


def schedule_task(self, task_id: str, handler: str, data: Optional[Dict[str, Any]] = None,
                  delay: float = 0.0) -> str:
    """Publie l'événement `handler` avec `data` après `delay` secondes.
    Un task_id déjà planifié est remplacé (regroupement des demandes rapprochées)."""
    b = _bus(self)
    with b["lock"]:
        old = b["timers"].pop(task_id, None)
        if old:
            old.cancel()
        t = threading.Timer(delay, lambda: (b["timers"].pop(task_id, None),
                                            self.publish_event(handler, data, source=f"task:{task_id}")))
        t.daemon = True
        b["timers"][task_id] = t
        t.start()
    return task_id


def event_bus_stats(self) -> Dict[str, Any]:
    b = _bus(self)
    with b["lock"]:
        return {"subscriptions": {k: len(v) for k, v in b["handlers"].items() if v},
                "published": dict(b["stats"]), "pending_tasks": list(b["timers"])}


def install_event_bus(cls) -> None:
    for fn in (register_handler, unregister_handler, publish_event, schedule_task, event_bus_stats):
        if not hasattr(cls, fn.__name__):
            setattr(cls, fn.__name__, fn)
