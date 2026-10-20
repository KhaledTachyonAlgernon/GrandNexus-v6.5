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

def r14_reason_workspace_contract():
    """reason() rangeait le contexte sous workspace['context_data'] alors que
    _infer_input() et _select_next_strategy() le cherchent sous workspace['context']
    et les entrées (premises, rules...) directement dans le workspace : la déduction
    n'était donc jamais sélectionnable. Le contrat est rétabli des deux côtés."""
    replace_once("grandnexus/cognition/reasoning/reasoning_core.py",
                 "                \"context_data\": context.parameters,\n",
                 "                \"context_data\": context.parameters,\n"
                 "                \"context\": context.parameters,  # réparation 6.5 : clé lue par _infer_input\n"
                 "                **{k: context.parameters[k] for k in (\"premises\", \"rules\", \"examples\", \"observations\",\n"
                 "                                                      \"source_domain\", \"target_domain\")\n"
                 "                   if k in context.parameters},\n")

def r15_orchestrator_imports():
    """Le CognitiveOrchestrator importe ReasoningEngine / LearningManager / PerceptionModule
    depuis les paquets cognition.* : ces classes n'existaient pas. Elles sont fournies
    par cognition/adapters.py et exposées paresseusement par les __init__ (PEP 562)."""
    copy_new("adapters.py", "grandnexus/cognition/adapters.py")
    for pkg, name in (("reasoning", "ReasoningEngine"), ("learning", "LearningManager"),
                      ("perception", "PerceptionModule")):
        write(f"grandnexus/cognition/{pkg}/__init__.py",
              f'"""Paquet cognition.{pkg} (réparation 6.5 : export paresseux de {name})."""\n\n'
              f'def __getattr__(attr):\n'
              f'    if attr == "{name}":\n'
              f'        from grandnexus.cognition.adapters import {name}\n'
              f'        return {name}\n'
              f'    raise AttributeError(attr)\n')
    for p in ("grandnexus/cognition/orchestration/cognitive_orchestrator.py",
              "grandnexus/cognition/orchestration/cognitive_orchestrator_v1.py"):
        dedent_from(p, "def create_cognitive_orchestrator(")

def r16_layer_bugs():
    """Couche 15/05 : défauts de dataclass, texte shell/JS collé dans le code,
    chemin d'import incohérent (grandnexus.dialogue vs grandnexus/io), openai non protégé."""
    p = "grandnexus/homeostasis/homeostasis_manager.py"
    for name, a, b in (("cpu_thr", "0.70", "0.90"), ("mem_thr", "0.75", "0.95")):
        replace_once(p, f"    {name}: Thresholds = Thresholds({a}, {b})\n",
                     f"    {name}: Thresholds = field(default_factory=lambda: Thresholds({a}, {b}))\n")
    for name, a, b, cmt in (("temp_thr", "0.60", "0.80", "        # fraction of Tjmax"),
                            ("batt_thr", "0.30", "0.10", "        # remaining % (inverse)"),
                            ("api_thr", "0.70", "0.90", "         # quota usage"),
                            ("cost_thr", "0.70", "0.90", "        # budget usage")):
        replace_once(p, f"    {name}: Thresholds = Thresholds({a}, {b}){cmt}\n",
                     f"    {name}: Thresholds = field(default_factory=lambda: Thresholds({a}, {b})){cmt}\n")
    s = read(p)
    if "from dataclasses import dataclass, field" not in s:
        add_import(p, "from dataclasses import field")
    p = "grandnexus/io/dialogue_interface.py"
    s = read(p); i = s.index("curl -N -H")
    write(p, s[:i] + "\n".join("# " + l for l in s[i:].split("\n")) + "\n")
    p = "grandnexus/cognition/metacognition/explainability_engine.py"
    replace_once(p, "import openai  # Assurez-vous", "try:\n    import openai\nexcept ImportError:  # réparation 6.5\n    openai = None\n# Assurez-vous")
    replace_once("grandnexus/__init__.py", '    "grandnexus.metacognition": "grandnexus.cognition.metacognition",\n',
                 '    "grandnexus.metacognition": "grandnexus.cognition.metacognition",\n'
                 '    "grandnexus.dialogue": "grandnexus.io",  # couche 15/05 : en-tête io/, import dialogue/\n')

def r17_resource_history():
    """L'orchestrateur appelle resource_manager.get_usage_history() : méthode absente,
    les instantanés existaient déjà (resource_snapshots)."""
    append("grandnexus/computation/resource_manager.py", '''
# --- réparation 6.5 : historique demandé par le CognitiveOrchestrator
def _get_usage_history(self, limit: int = 60):
    return [{"timestamp": s.timestamp, "cpu": s.cpu_usage, "memory": s.memory_usage,
             "disk": s.disk_usage} for s in list(self.resource_snapshots)[-limit:]]
if not hasattr(ResourceManager, "get_usage_history"):
    ResourceManager.get_usage_history = _get_usage_history
''')

def r18_orchestrator_state():
    """La matrice de transition de l'orchestrateur envoie vers CognitiveState.PERCEPTION
    (« new_input ») mais l'énumération ne définissait pas cet état : plantage à start()."""
    for p in ("grandnexus/cognition/orchestration/cognitive_orchestrator.py",
              "grandnexus/cognition/orchestration/cognitive_orchestrator_v1.py"):
        if "CognitiveState.PERCEPTION" in read(p):
            replace_once(p, "    META_REFLECTION = auto()  # Self-analysis and process optimization\n",
                         "    META_REFLECTION = auto()  # Self-analysis and process optimization\n"
                         "    PERCEPTION = auto()       # réparation 6.5 : cible de la transition « new_input »\n")

def r19_working_memory_sort():
    """WorkingMemory triait des tuples (score, item) : à score égal, Python compare
    les items -> TypeError. Tri sur le score seul (ordre stable conservé)."""
    p = "grandnexus/memory/working.py"
    s = read(p)
    assert s.count("            scored_items.sort(reverse=True)\n") == 1 and s.count("        scored_items.sort(reverse=True)\n") == 2
    write(p, s.replace("scored_items.sort(reverse=True)", "scored_items.sort(key=lambda x: x[0], reverse=True)  # réparation 6.5"))

def r20_working_memory_item():
    """WorkingMemoryItem.access_count lu et incrémenté par 4 méthodes, jamais déclaré."""
    replace_once("grandnexus/memory/working.py",
                 "    embedding: Optional[np.ndarray] = None\n\n    def __repr__(self) -> str:\n        return (f\"<WorkingMemoryItem",
                 "    embedding: Optional[np.ndarray] = None\n    access_count: int = 0  # réparation 6.5 : utilisé par retrieve/éviction\n\n"
                 "    def __repr__(self) -> str:\n        return (f\"<WorkingMemoryItem")

def r21_orchestrator_session_output():
    """process_input() renvoyait les résultats de raisonnement de N'IMPORTE QUELLE session :
    la question 2 recevait la réponse de la question 1 (contamination mémoire).
    Les éléments sont maintenant filtrés sur la session courante (horodatage >= début)."""
    p = "grandnexus/cognition/orchestration/cognitive_orchestrator.py"
    replace_once(p, '''                reasoning_results = self.working_memory.retrieve(
                    query_text="reasoning",
                    max_results=5
                )
''', '''                reasoning_results = self.working_memory.retrieve(
                    query_text="reasoning",
                    max_results=50
                )
                # réparation 6.5 : seulement ce qui a été produit pendant cette session
                reasoning_results = [i for i in reasoning_results
                                     if getattr(i, "timestamp", 0) >= start_time
                                     and getattr(i, "source", "") == "reasoning"][:5]
''')
    replace_once(p, '''                    relevant_items = self.working_memory.retrieve(
                        query_text=self.context.focus_target or "current_task",
                        max_results=3
                    )
''', '''                    relevant_items = self.working_memory.retrieve(
                        query_text=self.context.focus_target or "current_task",
                        max_results=50
                    )
                    relevant_items = [i for i in relevant_items
                                      if getattr(i, "timestamp", 0) >= start_time][:3]  # réparation 6.5
''')

def r22_orchestrator_cleanup():
    """Les actions terminées restaient dans active_actions (et y gardaient leurs ressources)
    tant que la boucle de fond n'appelait pas _cleanup_actions(). Sans cette boucle
    (process_input synchrone), plus rien ne s'exécutait après la 1re question.
    Nettoyage fait aussi au début de _process_action_queue()."""
    for p in ("grandnexus/cognition/orchestration/cognitive_orchestrator.py",
              "grandnexus/cognition/orchestration/cognitive_orchestrator_v1.py"):
        replace_once(p, '''        """Process actions from the queue based on priorities and resource constraints."""
        if not self.action_queue:
            return
''', '''        """Process actions from the queue based on priorities and resource constraints."""
        self._cleanup_actions()  # réparation 6.5 : libère les ressources des actions finies
        if not self.action_queue:
            return
''')

def r23_orchestrator_memory_queries():
    """L'orchestrateur passait des dict à SemanticMemory.query_concepts (attend SemanticQuery)
    et EpisodicMemory.retrieve_episodes (attend EpisodicQuery) -> chaque « retrieve » échouait.
    Méta-analyse : moyenne sur un dict vide -> division par zéro."""
    for p in ("grandnexus/cognition/orchestration/cognitive_orchestrator.py",
              "grandnexus/cognition/orchestration/cognitive_orchestrator_v1.py"):
        s = read(p)
        if "semantic_results = self.semantic_memory.query_concepts(semantic_query)" in s:
            replace_once(p, "semantic_results = self.semantic_memory.query_concepts(semantic_query)",
                         "from grandnexus.memory.semantic import SemanticQuery  # réparation 6.5\n"
                         "                semantic_results = self.semantic_memory.query_concepts(SemanticQuery(**semantic_query))")
        if "episodic_results = self.episodic_memory.retrieve_episodes(episodic_query)" in s:
            replace_once(p, "episodic_results = self.episodic_memory.retrieve_episodes(episodic_query)",
                         "from grandnexus.memory.episodic import EpisodicQuery  # réparation 6.5\n"
                         "                _eq = dict(episodic_query)\n"
                         "                _eq[\"context_factors\"] = {\"recency_bias\": _eq.pop(\"recency_bias\", None)}\n"
                         "                episodic_results = self.episodic_memory.retrieve_episodes(EpisodicQuery(**_eq))")
        s = read(p)
        if "if sum(self.success_rate.values()) / len(self.success_rate) < 0.6:" in s:
            replace_once(p, "if sum(self.success_rate.values()) / len(self.success_rate) < 0.6:",
                         "if self.success_rate and sum(self.success_rate.values()) / len(self.success_rate) < 0.6:  # réparation 6.5")

def r24_message_queue_tiebreak():
    """NexusCore (mode asynchrone) : la file de messages stockait (priorité, dict).
    Deux messages de même priorité en attente -> Python compare les dict -> TypeError
    dans send_message(). Le thread émetteur meurt : l'horloge cognitive s'arrêtait
    après 1 tick. Ajout d'un compteur d'ordre d'arrivée (FIFO à priorité égale)."""
    p = "grandnexus/core/nexus_core.py"
    replace_once(p, "            heapq.heappush(self.message_queue, (priority, message))  # Use heapq",
                 "            heapq.heappush(self.message_queue, (priority, next(_MSG_SEQ), message))  # réparation 6.5 : FIFO à égalité")
    replace_once(p, "                priority, message = heapq.heappop(self.message_queue) # Use heappop",
                 "                priority, _seq, message = heapq.heappop(self.message_queue)  # réparation 6.5")
    add_import(p, "import itertools")
    s = read(p)
    lines = s.split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("class NexusCore"))
    lines.insert(i, "_MSG_SEQ = itertools.count()  # réparation 6.5 : départage des messages de même priorité\n")
    write(p, "\n".join(lines))

def r25_manager_bridges():
    """CognitionManager appelait 12 méthodes absentes de ses modules (quick_scan, evaluate,
    adapt, reflect_on, recognize, process_text, create_plan...). Ponts de délégation vers
    les méthodes qui existent, installés seulement si absentes."""
    copy_new("manager_bridges.py", "grandnexus/cognition/manager_bridges.py")
    append("grandnexus/cognition/cognition_manager.py",
           "# --- réparation 6.5 : ponts vers l'API réelle des modules (voir manager_bridges.py)\n"
           "from grandnexus.cognition.manager_bridges import install as _install_bridges\n"
           "_install_bridges()")

def r26_attention_v2_default():
    """AttentionFilter (15/05) : max() sur max_rate vide par défaut -> plantage à la création."""
    replace_once("grandnexus/cognition/perception/filters/attention_filter_v2.py",
                 "window = int(self.config.history_sec * max(self.config.max_rate.values()))",
                 "window = int(self.config.history_sec * max(self.config.max_rate.values(), default=50.0))  # réparation 6.5 : 50 Hz = défaut du defaultdict")

def r27_orchestrator_session_filter_first():
    """Suite de R21 : le filtre « session courante » s'appliquait APRÈS la coupe à 50
    résultats ; après une dizaine de questions, les anciennes réponses (plus importantes)
    occupaient les 50 places et la réponse du moment disparaissait (7/12 au labo).
    Filtre d'abord, coupe ensuite."""
    p = "grandnexus/cognition/orchestration/cognitive_orchestrator.py"
    s = read(p)
    assert s.count("max_results=50\n") == 2
    write(p, s.replace("max_results=50\n", "max_results=10**6  # réparation 6.5 : filtrer la session avant de couper\n"))

def r28_working_memory_v2_name():
    """Décision de Khaled : la WorkingMemory du 15/05 vit À CÔTÉ de celle du cœur.
    Elle s'enregistrait sous « working_memory » (écrasait le cœur). Nouveau nom :
    « working_memory_v2 ». AffectRegulator la visait (decay_half_life n'existe que
    dans la v2) : ses deux références suivent le renommage."""
    replace_once("grandnexus/memory/working_v2.py", 'MODULE_NAME = "working_memory"',
                 'MODULE_NAME = "working_memory_v2"  # réparation 6.5 : cohabite avec la mémoire du cœur')
    p = "grandnexus/cognition/metacognition/affect_regulator.py"
    replace_once(p, '        "executive_planner",\n        "working_memory",\n',
                 '        "executive_planner",\n        "working_memory_v2",  # réparation 6.5 : renommage\n')
    replace_once(p, 'wm = self.nexus.get_module("working_memory")',
                 'wm = self.nexus.get_module("working_memory_v2")  # réparation 6.5 : seule la v2 a decay_half_life')

def r29_layer_flat_imports():
    """Couche 15/05 (pyflakes) : os manquant dans ExplainabilityEngine (plantage à la
    création), time manquant dans le pilote VirtualGripper. Les autres noms indéfinis
    signalés sont dans les démos sous __main__ (fragments d'exemple, laissés tels quels)."""
    add_import("grandnexus/cognition/metacognition/explainability_engine.py", "import os")
    add_import("grandnexus/actuators/drivers/virtual_gripper.py", "import time")

def r30_semantic_v2_dataclass():
    """Révélé sur Colab (sentence_transformers y est installé, le module va plus loin) :
    - Python 3.11+ refuse une instance de dataclass comme valeur par défaut -> default_factory ;
    - _init_annoy testait la CLASSE AnnoyCfg (toujours vraie) au lieu de la config, puis
      appelait AnnoyIndex même si annoy n'est pas installé -> NameError à la création."""
    p = "grandnexus/memory/semantic_v2.py"
    for name, cls in (("pg", "PGVectorCfg"), ("neo4j", "Neo4jCfg"), ("annoy", "AnnoyCfg")):
        replace_once(p, f"    {name}: Optional[{cls}] = {cls}()\n",
                     f"    {name}: Optional[{cls}] = field(default_factory={cls})  # réparation 6.5\n")
    replace_once(p, "    def _init_annoy(self):\n        if not AnnoyCfg:\n            return\n",
                 "    def _init_annoy(self):\n        if not self.cfg.annoy or \"AnnoyIndex\" not in globals():  # réparation 6.5\n"
                 "            return\n")

def r31_interruptible_loops():
    """Arrêt en mode vivant : 17 s et des threads survivants. Les boucles de fond
    dormaient avec time.sleep() alors qu'un Event d'arrêt existait déjà : stop()
    attendait la fin du sommeil (jusqu'à 30 s) puis abandonnait après 5 s de join.
    Les sommeils de boucle deviennent des attentes interruptibles sur cet Event."""
    p = "grandnexus/cognition/metacognition/monitor.py"
    s = read(p)
    for old in ("                time.sleep(self.monitoring_interval)\n",
                "                time.sleep(self.alert_check_interval)\n",
                "                time.sleep(min(30.0, next_report_time - current_time))\n"):
        assert s.count(old) == 1, old
        s = s.replace(old, old.replace("time.sleep(", "self.stop_event.wait("))
    assert s.count("                time.sleep(5.0)  # Sleep longer on error\n") == 3
    s = s.replace("                time.sleep(5.0)  # Sleep longer on error\n",
                  "                self.stop_event.wait(5.0)  # Sleep longer on error (réparation 6.5 : interruptible)\n")
    write(p, s)
    p = "grandnexus/core/errorRecoveryManager.py"
    replace_once(p, "                time.sleep(self.health_check_interval)\n",
                 "                self.stop_monitoring.wait(self.health_check_interval)  # réparation 6.5 : interruptible\n")
    replace_once(p, "                time.sleep(5)  # Court sommeil en cas d'erreur\n",
                 "                self.stop_monitoring.wait(5)  # Court sommeil en cas d'erreur (interruptible)\n")

def r32_deduction_fixpoint():
    """Diagnostic par oracles : la déduction n'enchaînait que 2 pas (modus ponens + un
    seul niveau « transitif ») et ignorait la contraposée. Réparation dans la stratégie
    existante : chaînage jusqu'à stabilité + règles contraposées (si A alors B => si
    non-B alors non-A)."""
    p = "grandnexus/cognition/reasoning/reasoning_core.py"
    replace_once(p, '''        rules = inputs.get("rules", [])

        # Check if we have the necessary inputs
        if not premises or not rules:
            return {
                "outputs": {
                    "message": "Insufficient inputs for deductive reasoning",''', '''        rules = inputs.get("rules", [])

        # réparation 6.5 : contraposée (si A alors B  =>  si non-B alors non-A)
        def _neg(p):
            return p[4:] if p.startswith("not ") else "not " + p
        _seen_rules = {(r.get("if"), r.get("then")) for r in rules if isinstance(r, dict)}
        _contra = []
        for _r in rules:
            if isinstance(_r, dict) and "if" in _r and "then" in _r and not _r.get("contrapositive"):
                _c = (_neg(_r["then"]), _neg(_r["if"]))
                if _c not in _seen_rules:
                    _seen_rules.add(_c)
                    _contra.append({"if": _c[0], "then": _c[1], "confidence": _r.get("confidence", 0.95),
                                    "contrapositive": True})
        rules = list(rules) + _contra

        # Check if we have the necessary inputs
        if not premises or not rules:
            return {
                "outputs": {
                    "message": "Insufficient inputs for deductive reasoning",''')
    replace_once(p, '''        new_conclusions = []
        for conclusion in conclusions:
            statement = conclusion["statement"]
''', '''        new_conclusions = []
        _known = set(premises) | {c["statement"] for c in conclusions}
        _frontier = list(conclusions)
        _i = 0
        while _i < len(_frontier) and len(_known) < 5000:  # réparation 6.5 : jusqu'à stabilité
            conclusion = _frontier[_i]
            _i += 1
            statement = conclusion["statement"]
''')
    replace_once(p, '''                    new_conclusions.append(trans_conclusion)
''', '''                    if rule["then"] in _known:
                        continue
                    _known.add(rule["then"])
                    new_conclusions.append(trans_conclusion)
                    _frontier.append(trans_conclusion)
''')

def r33_llm_engine_anthropic():
    """LLMEngine (chemin Anthropic) : complete() ne transmettait jamais le modèle configuré,
    _call_anthropic retombait sur un modèle retiré (claude-3-sonnet-20240229) et ne savait
    pas passer de consigne système. Modèle et consigne système sont maintenant transmis."""
    p = "grandnexus/llm_interface/llm_engine.py"
    replace_once(p, '''                "request_type": request.request_type,
                "streaming": request.streaming,
                "stream_callback": request.stream_callback,
                "request_id": request.request_id
            })''', '''                "request_type": request.request_type,
                "streaming": request.streaming,
                "stream_callback": request.stream_callback,
                "request_id": request.request_id,
                "model": context.get("model") or self.config.model_name,  # réparation 6.5
                "system": context.get("system"),                          # réparation 6.5
            })''')
    replace_once(p, '''        model = request_params.get("model", "claude-3-sonnet-20240229")''',
                 '''        model = request_params.get("model") or "claude-sonnet-5-5"  # réparation 6.5 : ancien défaut retiré
        if model.startswith(("text-embedding", "gpt-")):
            model = "claude-sonnet-5-5"
        _system = {"system": request_params["system"]} if request_params.get("system") else {}''')
    s = read(p)
    old = '''                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=['''
    assert s.count(old) == 1, s.count(old)
    write(p, s.replace(old, '''                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **_system,
                    messages=['''))

if __name__ == "__main__":
    globals()[sys.argv[1]]()
    print("ok", sys.argv[1])
