"""
boot.py - Assembly of GrandNexus: creates and wires the modules in the right order.

    from grandnexus.boot import boot
    gn = boot()
    gn.cognition.invoke(CognitiveProcessType.PERCEPTION, "2*x + 3 = 7")
    gn.shutdown()

Every step is traced in gn.report: a failing module does not stop the next ones,
it is recorded (nothing is hidden, nothing is thrown away).
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
    nx_translator: Any = None
    report: List[BootStep] = field(default_factory=list)

    def ok(self) -> bool:
        return all(s.ok for s in self.report)

    def print_report(self) -> None:
        for s in self.report:
            print(f"{'OK  ' if s.ok else 'FAIL'} {s.name:<38} {s.seconds:6.2f}s"
                  + (f"  {s.detail}" if s.ok and s.detail else "")
                  + ("" if s.ok else f"\n       └─ {s.error}"))
        n = sum(s.ok for s in self.report)
        print(f"==> {n}/{len(self.report)} boot steps succeeded | mode : {self.mode}"
              + ("" if self.mode == "vivant" else " (NexusCore assembled but not started : gn.start() to start it)"))

    def modules(self) -> Dict[str, Any]:
        out = {}
        if self.cognition is not None:
            for attr in ("perception_modules", "reasoning_modules", "learning_modules", "metacognition_modules",
                         "extra_modules"):
                out.update(getattr(self.cognition, attr, {}) or {})
        return out

    def start(self) -> bool:
        """LIVE mode: starts NexusCore, which in turn starts every registered module
        (background loops, asynchronous messaging, orchestrator)."""
        if self.core is None:
            return False
        if getattr(self.core, "running", False):
            return True
        ok = self.core.start()
        print(f"NexusCore started: {ok} | live modules: {len(getattr(self.core, 'modules', {}))}")
        return ok

    @property
    def mode(self) -> str:
        return "vivant" if self.core is not None and getattr(self.core, "running", False) else "laboratoire"

    def shutdown(self) -> None:
        log = logging.getLogger("GrandNexus.Boot")
        if self.core is not None and getattr(self.core, "running", False):
            try:
                self.core.stop()          # stops the modules NexusCore started itself
            except Exception as e:
                log.warning(f"NexusCore.stop: {e}")
            # modules that started themselves (e.g. ResourceManager at creation): NexusCore thinks
            # they are only 'registered' and does not stop them -> stop everything still running
            for name, info in list(getattr(self.core, "modules", {}).items()):
                inst = info.get("instance")
                if getattr(inst, "running", False) is True and hasattr(inst, "stop"):
                    try:
                        inst.stop()
                    except Exception as e:
                        log.warning(f"{name}.stop: {e}")
            return
        # laboratory mode: NexusCore was never started, stop whatever was
        for obj, meth in ((self.orchestrator, "stop"), (self.resource_manager, "stop"), (self.cognition, "stop"),
                          (self.graph_manager, "stop"),
                          (getattr(self.core, "error_recovery", None), "shutdown")):  # started by NexusCore() itself
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
            raise RuntimeError("returned False (see the module logs)")
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


def find_anthropic_key(explicit: Optional[str] = None, with_reason: bool = False):
    """API key lookup: explicit argument, then ANTHROPIC_API_KEY env var, then Colab Secrets.
    with_reason=True returns (key, reason) so a missing key can be diagnosed precisely."""
    import os
    if explicit:
        return (explicit, "argument api_key") if with_reason else explicit
    if os.environ.get("ANTHROPIC_API_KEY"):
        return (os.environ["ANTHROPIC_API_KEY"], "environment variable") if with_reason else os.environ["ANTHROPIC_API_KEY"]
    reason = "pas dans Colab (ni argument ni environment variable)"
    try:
        from google.colab import userdata  # type: ignore
        key = userdata.get("ANTHROPIC_API_KEY")
        return (key, "Secrets Colab") if with_reason else key
    except ImportError:
        pass
    except Exception as e:  # SecretNotFoundError, NotebookAccessError, TimeoutException...
        hints = {"SecretNotFoundError": "no secret named exactly ANTHROPIC_API_KEY",
                 "NotebookAccessError": "secret present but notebook access is disabled (toggle in the 🔑 panel)",
                 "TimeoutException": "Colab authorisation dialog not confirmed in time"}
        reason = hints.get(type(e).__name__, f"{type(e).__name__}: {e}")
    return (None, reason) if with_reason else None


def boot(db_file: Optional[str] = None, start_threads: bool = False,
         log_level: int = logging.WARNING, verbose: bool = False,
         graph_config: Optional[Dict[str, Any]] = None,
         with_orchestrator: bool = True, orchestrator_threads: bool = False,
         start_core: bool = False,
         llm: str = "auto", llm_model: str = "claude-sonnet-5-5",
         api_key: Optional[str] = None, reasoner_mode: Optional[str] = None) -> GrandNexusSystem:
    """Assemble GrandNexus.
    Default: LABORATORY mode - everything is created and wired, but NexusCore is not
    started (no background loop: deterministic results, ideal for measuring).
    start_core=True (or gn.start() later): LIVE mode - NexusCore starts and launches
    every registered module."""
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
    if lg.handlers:            # NexusCore has its own handler: avoids duplicated lines
        lg.propagate = False
    gn.graph = _step(gn, "GraphEngine (inmemory)", lambda: create_graph_engine("inmemory"), verbose)
    gn.working = _step(gn, "WorkingMemory", lambda: WorkingMemory(WorkingMemoryConfig()), verbose)
    gn.semantic = _step(gn, "SemanticMemory (+graph)",
                        lambda: SemanticMemory(SemanticMemoryConfig(), graph_engine=gn.graph), verbose)
    gn.episodic = _step(gn, "EpisodicMemory", lambda: EpisodicMemory(EpisodicMemoryConfig()), verbose)

    def _register_memories():
        for name, mod in (("working_memory", gn.working), ("semantic_memory", gn.semantic),
                          ("episodic_memory", gn.episodic), ("graph_engine", gn.graph)):
            if mod is not None:
                gn.core.register_module(name, mod)
    if gn.core is not None:
        _step(gn, "memories registered in NexusCore", _register_memories, verbose)

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
        _step(gn, "cognitive pathways", gn.cognition._establish_initial_pathways, verbose)
        if gn.core is not None:
            _step(gn, "registration in NexusCore", gn.cognition._register_with_nexus_core, verbose)
        def _share_graph():
            # one graph for the whole system: injected where the manager did not pass one
            n = 0
            for name, mod in gn.modules().items():
                if hasattr(mod, "graph_engine") and getattr(mod, "graph_engine") is None:
                    mod.graph_engine = gn.graph
                    n += 1
            return n
        if gn.graph is not None:
            _step(gn, "shared graph injected", _share_graph, verbose)
        def _llm():
            # Language layer backed by Claude, used ONLY as a translator (see llm_translator.py)
            if llm == "off":
                return "disabled (llm='off')"
            key, why = find_anthropic_key(api_key, with_reason=True)
            if not key and llm == "auto":
                return f"DISABLED - key not found: {why}"
            if not key:
                raise RuntimeError(f"llm='on' but ANTHROPIC_API_KEY not found: {why}")
            try:
                import anthropic  # noqa: F401
            except ImportError:
                raise RuntimeError("paquet 'anthropic' absent : !pip install -q anthropic")
            from grandnexus.llm_interface.llm_engine import LLMEngine, LLMConfig, LLMProvider
            from grandnexus.llm_interface.llm_translator import LLMTranslator
            engine = LLMEngine(LLMConfig(provider=LLMProvider.ANTHROPIC, model_name=llm_model, api_key=key,
                                         temperature=None, max_tokens=1500, synergy_with_memory=False,
                                         synergy_with_graph=False))
            if not engine.initialize():
                raise RuntimeError("LLMEngine.initialize() failed (see logs)")
            gn.llm_engine = engine
            gn.translator = LLMTranslator(llm_engine=engine, model=llm_model, nexus_core=gn.core)
            # v7.0: the nexus_logic translator (text -> program) is the main pathway; the JSON one stays for comparison
            from grandnexus.llm_interface.translator_v5 import Translator as NxTranslator
            import anthropic
            gn.nx_translator = NxTranslator(client=anthropic.Anthropic(api_key=key), model=llm_model)
            return f"{llm_model} (key: {why}) | main pathway : nexus_logic"
        _step(gn, "LLM translator (Anthropic)", _llm, verbose)

        def _pathways():
            # perception -> premises -> reasoning pathway (modules added, nothing replaced)
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
                                              translator=gn.translator, mode=reasoner_mode,
                                              nx_translator=gn.nx_translator),
            }
            if gn.translator is not None:
                gn.cognition.extra_modules["llm_translator"] = gn.translator
            for name, mod in gn.cognition.extra_modules.items():
                if gn.core is not None:
                    gn.core.register_module(name, mod)
        _step(gn, "perception→reasoning pathway", _pathways, verbose)
        gn.cognition.initialized = True
        n = _step(gn, "capability wiring", gn.cognition.wire_capabilities, verbose)
        if start_threads:
            _step(gn, "CognitionManager.start()", gn.cognition.start, verbose)

    if with_orchestrator and gn.core is not None and gn.cognition is not None:
        def _adapters():
            # noms expecteds par CognitiveOrchestrator.start() via nexus_core.get_module(...)
            from grandnexus.cognition.adapters import ReasoningEngine, LearningManager, PerceptionModule
            from grandnexus.computation.resource_manager import ResourceManager
            mods = gn.modules()
            gn.core.register_module("perception_module", PerceptionModule(
                mods.get("perception_core"), mods.get("premise_extractor"), gn.working))
            gn.core.register_module("reasoning_engine", ReasoningEngine(mods.get("text_reasoner"), gn.working))
            gn.core.register_module("learning_manager", LearningManager(mods.get("learning_core")))
            gn.resource_manager = ResourceManager(nexus_core=gn.core)
            if "resource_manager" not in getattr(gn.core, "modules", {}):   # it may register itself
                gn.core.register_module("resource_manager", gn.resource_manager)
        _step(gn, "orchestrator adapters (15/05)", _adapters, verbose)

        def _orchestrator():
            from grandnexus.cognition.orchestration.cognitive_orchestrator import CognitiveOrchestrator
            orch = CognitiveOrchestrator(gn.core, {"monitoring_interval": 0.5, "execution_interval": 0.2})
            if not orch.start():
                raise RuntimeError("CognitiveOrchestrator.start() failed (see logs)")
            if not orchestrator_threads:
                orch.stop()   # laboratory mode: modules wired, background loops stopped
            gn.core.register_module("cognitive_orchestrator", orch)
            gn.orchestrator = orch
            return orch
        _step(gn, "CognitiveOrchestrator", _orchestrator, verbose)
    if start_core:
        _step(gn, "NexusCore.start() (live mode)", gn.start, verbose)
    return gn
