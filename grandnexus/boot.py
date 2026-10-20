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
    detail: Optional[str] = None


@dataclass
class GrandNexusSystem:
    core: Any = None
    working: Any = None
    semantic: Any = None
    episodic: Any = None
    graph: Any = None
    graph_manager: Any = None
    cognition: Any = None
    orchestrator: Any = None
    resource_manager: Any = None
    llm_engine: Any = None
    translator: Any = None
    report: List[BootStep] = field(default_factory=list)

    def ok(self) -> bool:
        return all(s.ok for s in self.report)

    def print_report(self) -> None:
        for s in self.report:
            print(f"{'OK  ' if s.ok else 'ÉCHEC'} {s.name:<38} {s.seconds:6.2f}s"
                  + (f"  {s.detail}" if s.ok and s.detail else "")
                  + ("" if s.ok else f"\n       └─ {s.error}"))
        n = sum(s.ok for s in self.report)
        print(f"==> {n}/{len(self.report)} étapes de démarrage réussies | mode : {self.mode}"
              + ("" if self.mode == "vivant" else " (NexusCore assemblé mais non démarré : gn.start() pour le lancer)"))

    def modules(self) -> Dict[str, Any]:
        out = {}
        if self.cognition is not None:
            for attr in ("perception_modules", "reasoning_modules", "learning_modules", "metacognition_modules",
                         "extra_modules"):
                out.update(getattr(self.cognition, attr, {}) or {})
        return out

    def start(self) -> bool:
        """Mode VIVANT : démarre NexusCore, qui démarre à son tour tous les modules
        enregistrés (boucles de fond, messagerie asynchrone, orchestrateur)."""
        if self.core is None:
            return False
        if getattr(self.core, "running", False):
            return True
        ok = self.core.start()
        print(f"NexusCore démarré : {ok} | modules vivants : {len(getattr(self.core, 'modules', {}))}")
        return ok

    @property
    def mode(self) -> str:
        return "vivant" if self.core is not None and getattr(self.core, "running", False) else "laboratoire"

    def shutdown(self) -> None:
        log = logging.getLogger("GrandNexus.Boot")
        if self.core is not None and getattr(self.core, "running", False):
            try:
                self.core.stop()          # arrête les modules que NexusCore a lui-même démarrés
            except Exception as e:
                log.warning(f"NexusCore.stop: {e}")
            # modules démarrés par eux-mêmes (ex. ResourceManager à sa création) : NexusCore les
            # croit « registered » et ne les arrête pas -> on arrête tout ce qui tourne encore
            for name, info in list(getattr(self.core, "modules", {}).items()):
                inst = info.get("instance")
                if getattr(inst, "running", False) is True and hasattr(inst, "stop"):
                    try:
                        inst.stop()
                    except Exception as e:
                        log.warning(f"{name}.stop: {e}")
            return
        # mode laboratoire : NexusCore n'a jamais été démarré, on arrête ce qui l'a été
        for obj, meth in ((self.orchestrator, "stop"), (self.resource_manager, "stop"), (self.cognition, "stop"),
                          (self.graph_manager, "stop"),
                          (getattr(self.core, "error_recovery", None), "shutdown")):  # lancé dès NexusCore()
            if obj is not None and hasattr(obj, meth):
                try:
                    getattr(obj, meth)()
                except Exception as e:
                    log.warning(f"{type(obj).__name__}.{meth}: {e}")


def _step(system: GrandNexusSystem, name: str, fn, verbose: bool = False):
    t0 = time.time()
    try:
        result = fn()
        if result is False:
            raise RuntimeError("a renvoyé False (voir les logs du module)")
        system.report.append(BootStep(name, True, time.time() - t0,
                                      detail=result if isinstance(result, str) else None))
        return result
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        msg = f"{type(e).__name__}: {e} [{tb.filename.split('grandnexus/')[-1]}:{tb.lineno}]"
        system.report.append(BootStep(name, False, time.time() - t0, msg))
        if verbose:
            traceback.print_exc()
        return None


def find_anthropic_key(explicit: Optional[str] = None) -> Optional[str]:
    """API key lookup: explicit argument, then ANTHROPIC_API_KEY env var, then Colab Secrets."""
    import os
    if explicit:
        return explicit
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    try:
        from google.colab import userdata  # type: ignore
        return userdata.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


def boot(db_file: Optional[str] = None, start_threads: bool = False,
         log_level: int = logging.WARNING, verbose: bool = False,
         graph_config: Optional[Dict[str, Any]] = None,
         with_orchestrator: bool = True, orchestrator_threads: bool = False,
         start_core: bool = False,
         llm: str = "auto", llm_model: str = "claude-sonnet-5-5",
         api_key: Optional[str] = None, reasoner_mode: Optional[str] = None) -> GrandNexusSystem:
    """Assemble GrandNexus.
    Par défaut : mode LABORATOIRE — tout est créé et câblé, mais NexusCore n'est pas
    démarré (pas de boucles de fond : résultats déterministes, idéal pour mesurer).
    start_core=True (ou gn.start() ensuite) : mode VIVANT — NexusCore démarre et
    lance tous les modules enregistrés."""
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
    lg = logging.getLogger("NexusCore")
    if lg.handlers:            # NexusCore a son propre handler : évite chaque ligne en double
        lg.propagate = False
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
        def _llm():
            # Language layer backed by Claude, used ONLY as a translator (see llm_translator.py)
            key = find_anthropic_key(api_key)
            if llm == "off" or (llm == "auto" and not key):
                return "désactivé (pas de clé ANTHROPIC_API_KEY)" if not key else "désactivé"
            if not key:
                raise RuntimeError("llm='on' mais aucune clé ANTHROPIC_API_KEY (Secrets Colab ou variable d'environnement)")
            try:
                import anthropic  # noqa: F401
            except ImportError:
                raise RuntimeError("paquet 'anthropic' absent : !pip install -q anthropic")
            from grandnexus.llm_interface.llm_engine import LLMEngine, LLMConfig, LLMProvider
            from grandnexus.llm_interface.llm_translator import LLMTranslator
            engine = LLMEngine(LLMConfig(provider=LLMProvider.ANTHROPIC, model_name=llm_model, api_key=key,
                                         temperature=0.0, max_tokens=1500, synergy_with_memory=False,
                                         synergy_with_graph=False))
            if not engine.initialize():
                raise RuntimeError("LLMEngine.initialize() a échoué (voir logs)")
            gn.llm_engine = engine
            gn.translator = LLMTranslator(llm_engine=engine, model=llm_model, nexus_core=gn.core)
            return llm_model
        _step(gn, "LLM traducteur (Anthropic)", _llm, verbose)

        def _pathways():
            # voie perception -> prémisses -> raisonnement (modules ajoutés, rien remplacé)
            from grandnexus.cognition.perception.premise_extractor import PremiseExtractor
            from grandnexus.cognition.reasoning.text_reasoner import TextReasoner
            mods = gn.modules()
            extractor = PremiseExtractor(nexus_core=gn.core)
            gn.cognition.extra_modules = {
                "premise_extractor": extractor,
                "text_reasoner": TextReasoner(reasoning_core=mods.get("reasoning_core"),
                                              perception_core=mods.get("perception_core"),
                                              extractor=extractor, graph_engine=gn.graph,
                                              working_memory=gn.working, nexus_core=gn.core,
                                              translator=gn.translator, mode=reasoner_mode),
            }
            if gn.translator is not None:
                gn.cognition.extra_modules["llm_translator"] = gn.translator
            for name, mod in gn.cognition.extra_modules.items():
                if gn.core is not None:
                    gn.core.register_module(name, mod)
        _step(gn, "voie perception→raisonnement", _pathways, verbose)
        gn.cognition.initialized = True
        n = _step(gn, "câblage des capacités", gn.cognition.wire_capabilities, verbose)
        if start_threads:
            _step(gn, "CognitionManager.start()", gn.cognition.start, verbose)

    if with_orchestrator and gn.core is not None and gn.cognition is not None:
        def _adapters():
            # noms attendus par CognitiveOrchestrator.start() via nexus_core.get_module(...)
            from grandnexus.cognition.adapters import ReasoningEngine, LearningManager, PerceptionModule
            from grandnexus.computation.resource_manager import ResourceManager
            mods = gn.modules()
            gn.core.register_module("perception_module", PerceptionModule(
                mods.get("perception_core"), mods.get("premise_extractor"), gn.working))
            gn.core.register_module("reasoning_engine", ReasoningEngine(mods.get("text_reasoner"), gn.working))
            gn.core.register_module("learning_manager", LearningManager(mods.get("learning_core")))
            gn.resource_manager = ResourceManager(nexus_core=gn.core)
            gn.core.register_module("resource_manager", gn.resource_manager)
        _step(gn, "adaptateurs orchestrateur (15/05)", _adapters, verbose)

        def _orchestrator():
            from grandnexus.cognition.orchestration.cognitive_orchestrator import CognitiveOrchestrator
            orch = CognitiveOrchestrator(gn.core, {"monitoring_interval": 0.5, "execution_interval": 0.2})
            if not orch.start():
                raise RuntimeError("CognitiveOrchestrator.start() a échoué (voir logs)")
            if not orchestrator_threads:
                orch.stop()   # mode laboratoire : modules câblés, boucles de fond arrêtées
            gn.core.register_module("cognitive_orchestrator", orch)
            gn.orchestrator = orch
            return orch
        _step(gn, "CognitiveOrchestrator", _orchestrator, verbose)
    if start_core:
        _step(gn, "NexusCore.start() (mode vivant)", gn.start, verbose)
    return gn
