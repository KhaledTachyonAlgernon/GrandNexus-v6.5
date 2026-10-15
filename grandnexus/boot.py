"""
boot.py — Assemblage de GrandNexus : crée et câble les modules dans le bon ordre.

    from grandnexus.boot import boot
    gn = boot()
    gn.cognition.invoke(CognitiveProcessType.PERCEPTION, "2*x + 3 = 7")
    gn.shutdown()

Chaque étape est tracée dans gn.report : un module qui échoue n'arrête pas les
suivants, il est noté (rien n'est masqué, rien n'est jeté).
"""

import logging
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BootStep:
    name: str
    ok: bool
    seconds: float
    error: Optional[str] = None


@dataclass
class GrandNexusSystem:
    core: Any = None
    working: Any = None
    semantic: Any = None
    episodic: Any = None
    graph: Any = None
    graph_manager: Any = None
    cognition: Any = None
    report: List[BootStep] = field(default_factory=list)

    def ok(self) -> bool:
        return all(s.ok for s in self.report)

    def print_report(self) -> None:
        for s in self.report:
            print(f"{'OK  ' if s.ok else 'ÉCHEC'} {s.name:<38} {s.seconds:6.2f}s"
                  + ("" if s.ok else f"\n       └─ {s.error}"))
        n = sum(s.ok for s in self.report)
        print(f"==> {n}/{len(self.report)} étapes de démarrage réussies")

    def modules(self) -> Dict[str, Any]:
        out = {}
        if self.cognition is not None:
            for attr in ("perception_modules", "reasoning_modules", "learning_modules", "metacognition_modules"):
                out.update(getattr(self.cognition, attr, {}) or {})
        return out

    def shutdown(self) -> None:
        for obj, meth in ((self.cognition, "stop"), (self.graph_manager, "stop"), (self.core, "stop")):
            if obj is not None and hasattr(obj, meth):
                try:
                    getattr(obj, meth)()
                except Exception as e:
                    logging.getLogger("GrandNexus.Boot").warning(f"{type(obj).__name__}.{meth}: {e}")


def _step(system: GrandNexusSystem, name: str, fn, verbose: bool = False):
    t0 = time.time()
    try:
        result = fn()
        if result is False:
            raise RuntimeError("a renvoyé False (voir les logs du module)")
        system.report.append(BootStep(name, True, time.time() - t0))
        return result
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        msg = f"{type(e).__name__}: {e} [{tb.filename.split('grandnexus/')[-1]}:{tb.lineno}]"
        system.report.append(BootStep(name, False, time.time() - t0, msg))
        if verbose:
            traceback.print_exc()
        return None


def boot(db_file: Optional[str] = None, start_threads: bool = False,
         log_level: int = logging.WARNING, verbose: bool = False,
         graph_config: Optional[Dict[str, Any]] = None) -> GrandNexusSystem:
    """Démarre GrandNexus.
    start_threads=False : modules initialisés mais boucles de fond non lancées
    (mode laboratoire, déterministe). True : CognitionManager.start()."""
    logging.basicConfig(level=log_level, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    logging.getLogger().setLevel(log_level)
    logging.getLogger("NexusCore").setLevel(log_level)

    from grandnexus.core.nexus_core import NexusCore
    from grandnexus.memory.working import WorkingMemory, WorkingMemoryConfig
    from grandnexus.memory.semantic import SemanticMemory, SemanticMemoryConfig
    from grandnexus.memory.episodic import EpisodicMemory, EpisodicMemoryConfig
    from grandnexus.graph.graph_engine import create_graph_engine
    from grandnexus.cognition.cognition_manager import CognitionManager

    gn = GrandNexusSystem()
    db_file = db_file or tempfile.mktemp(prefix="nexuscore_", suffix=".db")

    gn.core = _step(gn, "NexusCore", lambda: NexusCore(db_file=db_file), verbose)
    gn.graph = _step(gn, "GraphEngine (inmemory)", lambda: create_graph_engine("inmemory"), verbose)
    gn.working = _step(gn, "WorkingMemory", lambda: WorkingMemory(WorkingMemoryConfig()), verbose)
    gn.semantic = _step(gn, "SemanticMemory (+graphe)",
                        lambda: SemanticMemory(SemanticMemoryConfig(), graph_engine=gn.graph), verbose)
    gn.episodic = _step(gn, "EpisodicMemory", lambda: EpisodicMemory(EpisodicMemoryConfig()), verbose)

    def _register_memories():
        for name, mod in (("working_memory", gn.working), ("semantic_memory", gn.semantic),
                          ("episodic_memory", gn.episodic), ("graph_engine", gn.graph)):
            if mod is not None:
                gn.core.register_module(name, mod)
    if gn.core is not None:
        _step(gn, "mémoires enregistrées dans NexusCore", _register_memories, verbose)

    def _graph_manager():
        from grandnexus.graph.graph_module_manager import GraphModuleManager
        cfg = dict(graph_config or {})
        cfg.setdefault("graph_engine_instance", gn.graph)
        return GraphModuleManager(nexus_core=gn.core, config=cfg)
    gn.graph_manager = _step(gn, "GraphModuleManager", _graph_manager, verbose)

    gn.cognition = _step(gn, "CognitionManager()", lambda: CognitionManager(
        nexus_core=gn.core, working_memory=gn.working,
        semantic_memory=gn.semantic, episodic_memory=gn.episodic), verbose)

    if gn.cognition is not None:
        for part in ("perception", "reasoning", "learning", "metacognition"):
            _step(gn, f"modules {part}", getattr(gn.cognition, f"_initialize_{part}_modules"), verbose)
        _step(gn, "voies cognitives (pathways)", gn.cognition._establish_initial_pathways, verbose)
        if gn.core is not None:
            _step(gn, "enregistrement dans NexusCore", gn.cognition._register_with_nexus_core, verbose)
        def _share_graph():
            # un seul graphe pour tout le système : injecté là où le manager n'en a pas passé
            n = 0
            for name, mod in gn.modules().items():
                if hasattr(mod, "graph_engine") and getattr(mod, "graph_engine") is None:
                    mod.graph_engine = gn.graph
                    n += 1
            return n
        if gn.graph is not None:
            _step(gn, "graphe partagé injecté", _share_graph, verbose)
        gn.cognition.initialized = True
        n = _step(gn, "câblage des capacités", gn.cognition.wire_capabilities, verbose)
        if start_threads:
            _step(gn, "CognitionManager.start()", gn.cognition.start, verbose)
    return gn
