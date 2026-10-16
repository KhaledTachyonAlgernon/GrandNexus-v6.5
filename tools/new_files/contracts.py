"""
contracts.py — Contrat cognitif commun de GrandNexus (classes manquantes recréées).

PerceptionCore et ReasoningCore importaient depuis cognition_manager des classes
qui n'existaient nulle part : CognitiveModuleBase, CognitiveProcessType,
CognitiveMode, CognitiveContext, CognitiveRequest, CognitiveResult, ainsi que le
décorateur CognitionManager.cognitive_capability. Elles sont définies ici, dans
un module feuille sans dépendance interne, ce qui casse aussi l'import circulaire
cognition_manager <-> perception_core / reasoning_core.

Logique inter-module apportée :
  - chaque module déclare ses capacités avec @cognitive_capability ;
  - CapabilityRegistry les indexe par type de processus ;
  - CognitionManagerBase.invoke(...) choisit la capacité adaptée au mode courant,
    l'appelle avec la signature existante (request, context, query) et renvoie un
    CognitiveResult ; un événement est publié sur le bus du NexusCore s'il existe.
"""

import inspect
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("GrandNexus.Contracts")


class CognitiveProcessType(Enum):
    PERCEPTION = auto()
    ATTENTION = auto()
    MEMORY = auto()
    REASONING = auto()
    PLANNING = auto()
    LEARNING = auto()
    METACOGNITION = auto()
    LANGUAGE = auto()
    ACTION = auto()


class CognitiveMode(Enum):
    REACTIVE = auto()       # quick response, low resources
    DELIBERATIVE = auto()   # in-depth reasoning
    CREATIVE = auto()       # exploration, analogies
    REFLECTIVE = auto()     # metacognition, self-analysis
    EXPLORATORY = auto()    # curiosity, information gathering


@dataclass
class CognitiveContext:
    """Contexte transmis à une capacité. Se comporte aussi comme un dict
    (get/keys/[]) sur `parameters`, car certains modules l'utilisent ainsi."""
    parameters: Dict[str, Any] = field(default_factory=dict)
    mode: CognitiveMode = CognitiveMode.DELIBERATIVE
    goal_id: Optional[str] = None
    focus: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    def get(self, key, default=None):
        return self.parameters.get(key, default)

    def keys(self):
        return self.parameters.keys()

    def __getitem__(self, key):
        return self.parameters[key]

    def __contains__(self, key):
        return key in self.parameters


@dataclass
class CognitiveRequest:
    query: Any = None
    context: CognitiveContext = field(default_factory=CognitiveContext)
    process_type: Optional[CognitiveProcessType] = None
    capability: Optional[str] = None          # target capability name (optional)
    source: str = "external"
    priority: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)


@dataclass
class CognitiveResult:
    request_id: str
    success: bool
    output: Any = None
    confidence: float = 0.0
    process_type: Optional[CognitiveProcessType] = None
    module: Optional[str] = None
    capability: Optional[str] = None
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityInfo:
    name: str
    capability_type: CognitiveProcessType
    description: str = ""
    priority: int = 50                          # plus petit = choisi en premier
    compatible_modes: Set[CognitiveMode] = field(default_factory=lambda: set(CognitiveMode))
    module_name: Optional[str] = None
    handler: Optional[Callable] = None
    calls: int = 0
    failures: int = 0
    total_time: float = 0.0


def cognitive_capability(capability_type: CognitiveProcessType,
                         description: str = "",
                         priority: int = 50,
                         compatible_modes: Optional[Set[CognitiveMode]] = None):
    """Marque une méthode comme capacité cognitive. La méthode reste inchangée."""
    def decorator(func):
        func._cognitive_capability = CapabilityInfo(
            name=func.__name__,
            capability_type=capability_type,
            description=description,
            priority=priority,
            compatible_modes=set(compatible_modes) if compatible_modes else set(CognitiveMode),
        )
        return func
    return decorator


class CognitiveModuleBase:
    """Base des modules cognitifs. Tolérante : les sous-classes existantes
    n'appellent pas super().__init__, donc tout est initialisé paresseusement."""

    def initialize(self) -> bool:
        self._gn_status = "initialized"
        return True

    def start(self) -> bool:
        self._gn_status = "running"
        return True

    def stop(self) -> bool:
        self._gn_status = "stopped"
        return True

    def get_status(self) -> Dict[str, Any]:
        return {"module": type(self).__name__,
                "status": getattr(self, "_gn_status", "created"),
                "capabilities": [c.name for c in self.get_capabilities()]}

    def get_capabilities(self) -> List[CapabilityInfo]:
        caps = []
        for name, member in inspect.getmembers(type(self), predicate=callable):
            info = getattr(member, "_cognitive_capability", None)
            if isinstance(info, CapabilityInfo):
                caps.append(info)
        return caps

    def handle_message(self, message: Dict[str, Any]) -> None:
        """Point d'entrée du bus NexusCore.send_message : si le message nomme une
        capacité, elle est exécutée."""
        content = message.get("content") or {}
        cap = content.get("capability") if isinstance(content, dict) else None
        if cap and hasattr(self, cap):
            req = CognitiveRequest(query=content.get("query"), source=message.get("source", "bus"))
            try:
                getattr(self, cap)(req, req.context, req.query)
            except Exception as e:  # pragma: no cover
                logging.getLogger(type(self).__name__).error(f"handle_message({cap}): {e}")


class CapabilityRegistry:
    """Index des capacités de tous les modules, par type de processus."""

    def __init__(self):
        self._lock = threading.RLock()
        self._by_type: Dict[CognitiveProcessType, List[CapabilityInfo]] = {}
        self._by_name: Dict[str, CapabilityInfo] = {}

    def register_module(self, module_name: str, module: Any) -> int:
        n = 0
        with self._lock:
            for attr in dir(type(module)):
                member = getattr(type(module), attr, None)
                proto = getattr(member, "_cognitive_capability", None)
                if not isinstance(proto, CapabilityInfo):
                    continue
                key = f"{module_name}.{proto.name}"
                if key in self._by_name:
                    continue
                info = CapabilityInfo(**{**proto.__dict__, "module_name": module_name,
                                         "handler": getattr(module, attr)})
                self._by_name[key] = info
                self._by_type.setdefault(info.capability_type, []).append(info)
                n += 1
        return n

    def find(self, process_type: Optional[CognitiveProcessType] = None,
             mode: Optional[CognitiveMode] = None,
             name: Optional[str] = None) -> List[CapabilityInfo]:
        with self._lock:
            if name:
                return [c for k, c in self._by_name.items() if k == name or c.name == name]
            pool = self._by_type.get(process_type, []) if process_type else list(self._by_name.values())
            if mode is not None:
                pool = [c for c in pool if mode in c.compatible_modes]
            return sorted(pool, key=lambda c: c.priority)

    def summary(self) -> Dict[str, List[str]]:
        with self._lock:
            return {t.name: [f"{c.module_name}.{c.name} (p={c.priority})" for c in sorted(v, key=lambda c: c.priority)]
                    for t, v in self._by_type.items()}


class CognitionManagerBase:
    """Socle du CognitionManager : décorateur de capacités + routage inter-modules.
    CognitionManager en hérite ; les modules feuilles n'importent que ce contrat."""

    cognitive_capability = staticmethod(cognitive_capability)

    # -- registry (lazy to avoid changing the existing __init__) --------
    @property
    def capabilities(self) -> CapabilityRegistry:
        reg = self.__dict__.get("_gn_capabilities")
        if reg is None:
            reg = CapabilityRegistry()
            self.__dict__["_gn_capabilities"] = reg
        return reg

    def wire_capabilities(self) -> int:
        """Indexe les capacités de tous les modules connus du manager."""
        total = 0
        for attr in ("perception_modules", "reasoning_modules", "learning_modules",
                     "metacognition_modules", "extra_modules"):
            for name, mod in (getattr(self, attr, None) or {}).items():
                total += self.capabilities.register_module(name, mod)
        self.__dict__["_gn_wired"] = True
        return total

    def invoke(self, process_type: Optional[CognitiveProcessType] = None, query: Any = None,
               context: Optional[CognitiveContext] = None, capability: Optional[str] = None,
               fallback: bool = True, **kwargs) -> CognitiveResult:
        """Route une requête vers la meilleure capacité, avec repli sur la suivante
        en cas d'échec si fallback=True."""
        if not self.__dict__.get("_gn_wired"):
            self.wire_capabilities()
        context = context or CognitiveContext()
        request = CognitiveRequest(query=query, context=context, process_type=process_type,
                                   capability=capability)
        candidates = self.capabilities.find(process_type, context.mode, capability)
        if not candidates:
            return CognitiveResult(request.request_id, False, process_type=process_type,
                                   error=f"Aucune capacité pour {process_type} en mode {context.mode}")
        last_error = None
        for cap in candidates:
            t0 = time.time()
            cap.calls += 1
            try:
                out = cap.handler(request, context, query, **kwargs)
                dt = time.time() - t0
                cap.total_time += dt
                conf = out.get("confidence", 0.0) if isinstance(out, dict) else 0.0
                ok = not (isinstance(out, dict) and out.get("error"))
                result = CognitiveResult(request.request_id, ok, out, conf, cap.capability_type,
                                         cap.module_name, cap.name,
                                         out.get("error") if isinstance(out, dict) else None, dt)
                self._gn_publish("cognition.capability_invoked", result)
                if ok or not fallback:
                    return result
                last_error = result.error
            except Exception as e:
                cap.failures += 1
                last_error = f"{cap.module_name}.{cap.name}: {type(e).__name__}: {e}"
                logger.warning(last_error)
                if not fallback:
                    break
        return CognitiveResult(request.request_id, False, process_type=process_type, error=last_error)

    def _gn_publish(self, event_type: str, result: CognitiveResult) -> None:
        core = getattr(self, "nexus_core", None)
        if core is not None and hasattr(core, "publish_event"):
            try:
                core.publish_event(event_type, {"module": result.module, "capability": result.capability,
                                                "success": result.success, "duration": result.duration},
                                   source="cognition_manager")
            except Exception:
                pass


__all__ = ["CognitiveProcessType", "CognitiveMode", "CognitiveContext", "CognitiveRequest",
           "CognitiveResult", "CapabilityInfo", "cognitive_capability", "CognitiveModuleBase",
           "CapabilityRegistry", "CognitionManagerBase"]
