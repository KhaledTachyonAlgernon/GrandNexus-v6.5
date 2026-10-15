"""
Réparations GrandNexus 6.4 -> 6.5. Chaque réparation est une fonction ;
run_all.sh les applique une par une et commite chacune séparément.
Règle : on répare et on crée, on ne supprime ni ne simplifie.
"""
import sys
from repair_lib import *

def r2_syntax():
    """Erreurs d'indentation qui empêchaient 2 modules de se charger."""
    p = "grandnexus/cognition/perception/filters/attention_filter.py"
    replace_once(p, "\n   def _filter_by_selective_attention(", "\n    def _filter_by_selective_attention(")
    p = "grandnexus/cognition/reasoning/reasoning_core.py"
    lines = read(p).split("\n")
    assert lines[1685].startswith("            if all(\"attributes\" in entity"), lines[1685][:60]
    assert lines[1829].strip().startswith("# Use LLM for more sophisticated analogical")
    reindent(p, 1686, 1829, -4)   # bloc sur-indenté de 4 (un `try:` perdu)

def r3_dataclasses():
    """Champs obligatoires après des champs par défaut : kw_only=True (Python 3.10+),
    toutes les constructions existantes sont déjà par mots-clés."""
    replace_once("grandnexus/computation/resource_manager.py",
                 "@dataclass(order=True)\nclass Task:", "@dataclass(order=True, kw_only=True)\nclass Task:")
    replace_once("grandnexus/computation/resource_manager_ml.py",
                 "@dataclass(order=True)\nclass TaskRequest:", "@dataclass(order=True, kw_only=True)\nclass TaskRequest:")

def r4_graph_factory():
    """create_graph_engine était indentée DANS Neo4jGraphEngine (V2) : remise au niveau module.
    Alias GraphEngine (importé par 3 modules, jamais défini)."""
    p = "grandnexus/graph/graph_engine.py"
    lines = read(p).split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("    def create_graph_engine("))
    for k in range(i, len(lines)):
        if lines[k].startswith("    "): lines[k] = lines[k][4:]
    write(p, "\n".join(lines))
    for p in ("grandnexus/graph/graph_engine.py", "grandnexus/graph/graph_engine_v1.py"):
        append(p, "# --- réparation 6.5 : nom attendu par pattern_recognizer, reasoning_core, llm_engine\nGraphEngine = AbstractGraphEngine")

def r5_aliases():
    """Noms importés par cognition_manager / adaptation / reflection qui existaient sous un autre nom."""
    A = {
        "grandnexus/cognition/reasoning/planning.py": "PlanningEngine = PlanningModule",
        "grandnexus/cognition/learning/reinforcement.py": "ReinforcementLearner = ReinforcementLearning",
        "grandnexus/cognition/metacognition/adaptation.py": "AdaptationManager = AdaptationEngine",
        "grandnexus/cognition/metacognition/reflection.py": "Reflector = ReflectionModule",
        "grandnexus/cognition/metacognition/confidence.py": "ConfidenceEstimator = ConfidenceEvaluator",
        "grandnexus/cognition/metacognition/monitor.py": "MetaMonitor = CognitiveMonitor",
    }
    for p, line in A.items():
        append(p, "# --- réparation 6.5 : alias du nom attendu par les autres modules\n" + line)

def r6_missing_classes():
    """LearningResult (jamais écrite) ; create_pattern_mining (enfermée dans la classe)."""
    append("grandnexus/cognition/learning/learning_core.py", '''
# --- réparation 6.5 : LearningResult (importé par continuous_learner, jamais défini)
@dataclass
class LearningResult:
    """Résultat d'une tâche d'apprentissage."""
    success: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    learned_model: Any = None
    task_id: Optional[str] = None
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)
''')
    # create_pattern_mining existait mais était indentée DANS PatternMiningModule
    p = "grandnexus/graph/pattern_mining.py"
    lines = read(p).split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("    def create_pattern_mining("))
    for k in range(i, len(lines)):
        if lines[k].startswith("    "): lines[k] = lines[k][4:]
    write(p, "\n".join(lines))

def r7_event_bus():
    """NexusCore reçoit register_handler / publish_event / schedule_task (appelés par le graphe)."""
    copy_new("event_bus.py", "grandnexus/core/event_bus.py")
    append("grandnexus/core/nexus_core.py",
           "# --- réparation 6.5 : bus publication/abonnement (voir core/event_bus.py)\n"
           "from grandnexus.core.event_bus import install_event_bus as _install_event_bus\n"
           "_install_event_bus(NexusCore)")

def r8_cognitive_contract():
    """Contrat inter-modules : les classes Cognitive* manquantes viennent de contracts.py ;
    CognitionManager hérite du routage de capacités ; constructeurs alignés sur les appels réels."""
    copy_new("contracts.py", "grandnexus/cognition/contracts.py")
    for p in ("grandnexus/cognition/perception/perception_core.py",
              "grandnexus/cognition/reasoning/reasoning_core.py"):
        s = read(p)
        start = s.index("from grandnexus.cognition.cognition_manager import (")
        end = s.index(")", start) + 1
        write(p, s[:start] + "from grandnexus.cognition.contracts import (\n"
              "    CognitionManagerBase as CognitionManager,  # contrat : casse l'import circulaire\n"
              "    CognitiveModuleBase, CognitiveProcessType, CognitiveMode,\n"
              "    CognitiveContext, CognitiveRequest, CognitiveResult,\n)" + s[end:])
    # PerceptionCore : le manager lui passe nexus_core
    p = "grandnexus/cognition/perception/perception_core.py"
    replace_once(p, "                 llm_engine: Optional[LLMEngine] = None):\n        \"\"\"\n        Initialize the Perception Core.",
                 "                 llm_engine: Optional[LLMEngine] = None,\n                 nexus_core: Any = None,\n                 **kwargs):\n        \"\"\"\n        Initialize the Perception Core.")
    replace_once(p, "        self.cognition_manager = cognition_manager\n        self.working_memory = working_memory\n        self.semantic_memory = semantic_memory\n        self.llm_engine = llm_engine\n",
                 "        self.cognition_manager = cognition_manager\n        self.working_memory = working_memory\n        self.semantic_memory = semantic_memory\n        self.llm_engine = llm_engine\n        self.nexus_core = nexus_core\n")
    # ReasoningCore : le manager lui passe nexus_core et episodic_memory
    p = "grandnexus/cognition/reasoning/reasoning_core.py"
    replace_once(p, "                 llm_engine: Optional[LLMEngine] = None):\n        \"\"\"\n        Initialize the Reasoning Core.",
                 "                 llm_engine: Optional[LLMEngine] = None,\n                 episodic_memory: Any = None,\n                 nexus_core: Any = None,\n                 **kwargs):\n        \"\"\"\n        Initialize the Reasoning Core.")
    replace_once(p, "        self.graph_engine = graph_engine\n        self.llm_engine = llm_engine\n\n        # Logger setup\n        self.logger = logging.getLogger(\"ReasoningCore\")",
                 "        self.graph_engine = graph_engine\n        self.llm_engine = llm_engine\n        self.episodic_memory = episodic_memory\n        self.nexus_core = nexus_core\n\n        # Logger setup\n        self.logger = logging.getLogger(\"ReasoningCore\")")
    # ReinforcementLearning : le manager lui passe learning_core
    p = "grandnexus/cognition/learning/reinforcement.py"
    replace_once(p, "                semantic_memory: Optional[SemanticMemory] = None):\n        \"\"\"\n        Initialize the reinforcement learning module.",
                 "                semantic_memory: Optional[SemanticMemory] = None,\n                learning_core: Any = None):\n        \"\"\"\n        Initialize the reinforcement learning module.")
    s = read(p); i = s.index("class ReinforcementLearning"); j = s.index('"""', s.index("def __init__", i)); j = s.index('"""', j + 3) + 3
    write(p, s[:j] + "\n        self.learning_core = learning_core  # réparation 6.5" + s[j:])
    # PlanningModule : le manager appelle initialize()
    append("grandnexus/cognition/reasoning/planning.py", '''
# --- réparation 6.5 : CognitionManager appelle initialize() avant start()
def _planning_initialize(self) -> bool:
    if hasattr(self, "logger"):
        self.logger.info("PlanningModule initialized")
    return True
if not hasattr(PlanningModule, "initialize"):
    PlanningModule.initialize = _planning_initialize
''')
    # CognitionManager hérite du socle contractuel et ré-exporte le contrat
    p = "grandnexus/cognition/cognition_manager.py"
    replace_once(p, "\nclass CognitionManager:", "\nfrom grandnexus.cognition.contracts import (  # réparation 6.5\n"
                 "    CognitionManagerBase, CognitiveModuleBase, CognitiveProcessType, CognitiveMode,\n"
                 "    CognitiveContext, CognitiveRequest, CognitiveResult, cognitive_capability,\n)\n\n\n"
                 "class CognitionManager(CognitionManagerBase):")

def r9_robustness():
    """Fonction module enfermée dans une classe, champ dataclass qui masque le module ast,
    imports torch sans repli (le module entier tombait si torch/avalanche absent)."""
    p = "grandnexus/cognition/perception/filters/attention_filter.py"
    lines = read(p).split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("    def create_anomaly_filter("))
    lines[i] = lines[i][4:]
    write(p, "\n".join(lines))
    replace_once("grandnexus/cognition/perception/symbolic_parser.py",
                 "    python_ast: Optional[ast.AST] = None",
                 "    python_ast: \"Optional[ast.AST]\" = None  # annotation différée : le champ `ast` masque le module")
    replace_once("grandnexus/cognition/learning/continuous_learner.py",
                 "except ImportError:\n    HAS_AVALANCHE = False\n",
                 "except ImportError:\n    HAS_AVALANCHE = False\n\nif \"Dataset\" not in globals():  # réparation 6.5 : repli sans PyTorch\n    class Dataset:\n        pass\n")
    replace_once("grandnexus/cognition/learning/reinforcement.py",
                 "except ImportError:\n    HAS_TORCH = False\n",
                 "except ImportError:\n    HAS_TORCH = False\n\n    class nn:  # réparation 6.5 : repli sans PyTorch (annotations nn.Module)\n        Module = object\n")

def r10_flat_namespace_imports():
    """Noms utilisés mais jamais importés : dans Colab ils venaient d'une autre cellule.
    Détectés avec pyflakes (undefined name)."""
    for p, line in [
        ("grandnexus/cognition/learning/learning_core.py", "import random"),
        ("grandnexus/cognition/metacognition/monitor.py", "import sys"),
        ("grandnexus/cognition/perception/code_analyzer.py", "import math"),
        ("grandnexus/memory/working.py", "import math"),
        ("grandnexus/cognition/reasoning/reasoning_core.py", "import re"),
        ("grandnexus/core/registry.py", "from collections import deque"),
        ("grandnexus/llm_interface/llm_engine_lite.py", "import uuid"),
        ("grandnexus/core/nexus_core.py", "from grandnexus.core.errorRecoveryManager import ErrorRecoveryManager"),
    ]:
        add_import(p, line)
    replace_once("grandnexus/llm_interface/llm_engine.py",
                 "from grandnexus.graph.graph_engine import GraphEngine\n",
                 "from grandnexus.graph.graph_engine import GraphEngine\nfrom grandnexus.graph.graph_engine import GraphQuery  # réparation 6.5\n")

def r11_logic_bugs():
    """Vrais bugs de logique trouvés par pyflakes.
    - semantic.py : 'remaining -= cluster' coupé en deux lignes -> la boucle de
      clustering ne retirait jamais les concepts traités (boucle infinie possible).
    - nexus_core.stop() : l'arrêt de ErrorRecoveryManager était placé après le return
      (code mort) -> son thread n'était jamais arrêté."""
    replace_once("grandnexus/memory/semantic.py",
                 "                    remainin\n                    g -= cluster\n",
                 "                    remaining -= cluster\n")
    replace_once("grandnexus/core/nexus_core.py",
                 "      return stop_success\n\n      # Arrêter le gestionnaire d'erreurs\n      self.error_recovery.shutdown()\n      return success\n",
                 "      # Arrêter le gestionnaire d'erreurs (réparation 6.5 : était après le return)\n"
                 "      if hasattr(self, \"error_recovery\"):\n          try:\n              self.error_recovery.shutdown()\n"
                 "          except Exception as e:\n              self.logger.error(f\"ErrorRecoveryManager.shutdown: {e}\")\n\n      return stop_success\n")

def r12_graph_registry():
    """GraphModuleManager appelait ModuleRegistry.register(...) comme une méthode de classe,
    alors que ModuleRegistry est une instance liée au core : registre partagé créé sur le core."""
    replace_once("grandnexus/graph/graph_module_manager.py",
                 "        ModuleRegistry.register(\"graph\", self)\n",
                 "        registry = getattr(self.nexus_core, \"module_registry\", None)  # réparation 6.5\n"
                 "        if registry is None:\n"
                 "            registry = ModuleRegistry(self.nexus_core)\n"
                 "            self.nexus_core.module_registry = registry\n"
                 "        registry.register(\"graph\", self)\n")

def r13_shared_graph():
    """GraphModuleManager créait toujours son propre graphe : la mémoire sémantique et le
    gestionnaire de graphe vivaient dans deux graphes séparés. Il accepte maintenant une
    instance existante via config['graph_engine_instance']."""
    replace_once("grandnexus/graph/graph_module_manager.py",
                 "        self.graph_engine = create_graph_engine(graph_engine_type, **graph_engine_config)\n",
                 "        shared = config.get('graph_engine_instance')  # réparation 6.5 : graphe partagé\n"
                 "        self.graph_engine = shared if shared is not None else create_graph_engine(graph_engine_type, **graph_engine_config)\n")

if __name__ == "__main__":
    globals()[sys.argv[1]]()
    print("ok", sys.argv[1])
