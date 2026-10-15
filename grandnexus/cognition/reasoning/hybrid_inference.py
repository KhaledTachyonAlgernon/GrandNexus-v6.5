"""
hybrid_inference.py - Raisonnement Neuro-Symbolique pour GrandNexus

Ce module implémente un moteur d'inférence hybride qui combine:
- Raisonnement symbolique (basé sur logique)
- Raisonnement neuronal (basé sur embeddings/réseaux)
- Frameworks neuro-symboliques (LNN, DeepProbLog ou équivalents)

Caractéristiques principales:
- Intégration bidirectionnelle entre logique formelle et représentations neuronales
- Capacité à manipuler l'incertitude et la confiance dans les inférences
- Flexibilité pour différents types de tâches cognitives
- Coopération avec les modules de mémoire et d'apprentissage
- Plusieurs stratégies d'inférence hybride pour s'adapter au contexte
"""

import logging
import threading
import time
import uuid
import math
from typing import Dict, List, Any, Optional, Callable, Union, Set, Tuple
from enum import Enum, auto
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict, deque
import random

# Imports from GrandNexus core and related modules
from grandnexus.core.nexus_core import NexusCore
from grandnexus.memory.working import WorkingMemory
from grandnexus.memory.semantic import SemanticMemory, Concept, Relation
from grandnexus.memory.episodic import EpisodicMemory
from grandnexus.graph.graph_engine import AbstractGraphEngine, GraphNode, GraphEdge
from grandnexus.graph.graph_engine import create_graph_engine, AbstractGraphEngine
from grandnexus.reasoning.reasoning_core import ReasoningCore
from grandnexus.reasoning.symbolic_engine import SymbolicEngine
from grandnexus.reasoning.neural_reasoner import NeuralReasoner
from grandnexus.llm_interface.llm_engine import LLMEngine

# Types d'inférence hybride
class InferenceType(Enum):
    """Types d'inférence hybride supportés par le moteur."""
    DEDUCTION = auto()         # Raisonnement déductif (A → B, A ⊢ B)
    INDUCTION = auto()         # Raisonnement inductif (généralisation)
    ABDUCTION = auto()         # Raisonnement abductif (hypothèse explicative)
    ANALOGY = auto()           # Raisonnement par analogie
    PROBABILISTIC = auto()     # Raisonnement probabiliste
    FUZZY = auto()             # Logique floue
    SPATIAL = auto()           # Raisonnement spatial
    TEMPORAL = auto()          # Raisonnement temporel
    CAUSAL = auto()            # Raisonnement causal
    LLM_GUIDED = auto()        # Inférence guidée par un LLM

# Mode d'opération du moteur hybride
class HybridMode(Enum):
    """Modes de fonctionnement du moteur d'inférence hybride."""
    SYMBOLIC_FIRST = auto()    # Commence par le raisonnement symbolique
    NEURAL_FIRST = auto()      # Commence par le raisonnement neuronal
    PARALLEL = auto()          # Les deux approches en parallèle, puis fusion
    ITERATIVE = auto()         # Alterne entre approches en plusieurs passes
    ADAPTIVE = auto()          # Choisit dynamiquement la meilleure approche
    LLM_ORCHESTRATED = auto()  # LLM dirige le flux de raisonnement

@dataclass
class SymbolicConstraint:
    """Représente une contrainte logique formelle."""
    constraint_id: str
    expression: str            # Représentation textuelle (ex: "A → B")
    variables: Set[str] = field(default_factory=set)
    weight: float = 1.0        # Poids/importance (0-1)
    confidence: float = 1.0    # Confiance (0-1)
    source: str = "user"       # Source de la contrainte

@dataclass
class InferenceQuery:
    """Requête d'inférence à traiter par le moteur hybride."""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    target_variables: Set[str] = field(default_factory=set)
    constraints: List[SymbolicConstraint] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    known_facts: Dict[str, Any] = field(default_factory=dict)
    inference_types: List[InferenceType] = field(default_factory=list)
    hybrid_mode: HybridMode = HybridMode.ADAPTIVE
    time_limit: Optional[float] = None
    max_inference_depth: int = 5
    use_llm: bool = True       # Utiliser ou non l'assistance d'un LLM

@dataclass
class InferenceStep:
    """Une étape unique dans un processus d'inférence."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""             # Type d'étape (symbolique, neurale, etc.)
    description: str = ""
    input_facts: Dict[str, Any] = field(default_factory=dict)
    output_facts: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    reasoning: str = ""        # Explication textuelle du raisonnement
    timestamp: float = field(default_factory=time.time)

@dataclass
class InferenceResult:
    """Résultat complet d'un processus d'inférence."""
    query_id: str
    success: bool = False
    answers: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    steps: List[InferenceStep] = field(default_factory=list)
    explanation: str = ""
    execution_time: float = 0.0
    symbolic_contribution: float = 0.0  # 0-1 proportion de contribution
    neural_contribution: float = 0.0    # 0-1 proportion de contribution
    llm_contribution: float = 0.0       # 0-1 proportion de contribution

class HybridInferenceEngine:
    """
    Moteur d'inférence neuro-symbolique pour GrandNexus.

    Ce moteur intègre des approches symboliques et neuronales pour le raisonnement,
    permettant une inférence plus robuste, explicable et flexible que les approches
    purement symboliques ou purement neuronales.
    """

    def __init__(self,
                 reasoning_core: Optional[ReasoningCore] = None,
                 symbolic_engine: Optional[SymbolicEngine] = None,
                 neural_reasoner: Optional[NeuralReasoner] = None,
                 semantic_memory: Optional[SemanticMemory] = None,
                 working_memory: Optional[WorkingMemory] = None,
                 episodic_memory: Optional[EpisodicMemory] = None,
                 graph_engine: Optional[AbstractGraphEngine] = None,
                 llm_engine: Optional[LLMEngine] = None,
                 nexus_core: Optional[NexusCore] = None):
        """
        Initialise le moteur d'inférence hybride.

        Args:
            reasoning_core: Référence au module de raisonnement principal
            symbolic_engine: Moteur de raisonnement symbolique
            neural_reasoner: Moteur de raisonnement neuronal
            semantic_memory: Mémoire sémantique du système
            working_memory: Mémoire de travail du système
            episodic_memory: Mémoire épisodique du système
            graph_engine: Moteur de graphe pour navigation de connaissances
            llm_engine: Interface avec un modèle de langage
            nexus_core: Référence au noyau central de GrandNexus
        """
        self.logger = logging.getLogger("GrandNexus.Reasoning.HybridInference")

        # Références aux autres modules
        self.reasoning_core = reasoning_core
        self.symbolic_engine = symbolic_engine
        self.neural_reasoner = neural_reasoner
        self.semantic_memory = semantic_memory
        self.working_memory = working_memory
        self.episodic_memory = episodic_memory
        self.graph_engine = graph_engine
        self.llm_engine = llm_engine
        self.nexus_core = nexus_core

        # État interne
        self.instance_id = str(uuid.uuid4())[:8]
        self.lock = threading.RLock()
        self.running = False
        self.initialized = False

        # Registre des stratégies d'inférence
        self.inference_strategies = {}

        # Cache de résultats récents
        self.results_cache = {}
        self.cache_expiry = 3600  # 1 heure

        # Métriques de performance
        self.metrics = defaultdict(lambda: deque(maxlen=100))

        # Règles symboliques et représentations neuronales
        self.symbolic_rules = []
        self.neural_models = {}

        # Compteurs d'utilisation pour ajustement adaptatif
        self.strategy_usage_counts = defaultdict(int)
        self.strategy_success_rates = defaultdict(lambda: (0, 0))  # (success, total)

        # Paramètres adaptatifs
        self.params = {
            "symbolic_weight": 0.5,      # Poids accordé à l'inférence symbolique
            "neural_weight": 0.5,        # Poids accordé à l'inférence neurale
            "confidence_threshold": 0.7, # Seuil minimal de confiance
            "max_branching_factor": 5,   # Facteur de branchement maximum en recherche
            "use_approximate": False,    # Permettre des inférences approximatives
            "combine_method": "weighted" # Comment combiner inférences multiples
        }

        self.logger.info(f"HybridInferenceEngine initialisé avec ID={self.instance_id}")

    def initialize(self) -> bool:
        """Initialise les composants du moteur d'inférence hybride."""
        if self.initialized:
            return True

        try:
            with self.lock:
                # Enregistrer les stratégies d'inférence par défaut
                self._register_default_strategies()

                # Configurer les interfaces avec les autres modules
                if self.nexus_core:
                    self._register_with_nexus_core()

                # Initialiser les modèles neuro-symboliques
                self._initialize_neuro_symbolic_models()

                self.initialized = True
                self.logger.info("HybridInferenceEngine initialization complete")
                return True
        except Exception as e:
            self.logger.error(f"Échec de l'initialisation: {str(e)}")
            return False

    def start(self) -> bool:
        """Démarre le moteur d'inférence hybride."""
        if not self.initialized:
            if not self.initialize():
                return False

        with self.lock:
            if self.running:
                return True

            self.running = True
            self.logger.info("HybridInferenceEngine démarré")
            return True

    def stop(self) -> bool:
        """Arrête le moteur d'inférence hybride."""
        with self.lock:
            if not self.running:
                return True

            self.running = False
            self.logger.info("HybridInferenceEngine arrêté")
            return True

    def process_query(self, query: InferenceQuery) -> InferenceResult:
        """
        Traite une requête d'inférence en utilisant le mode hybride spécifié.

        Args:
            query: La requête d'inférence à traiter

        Returns:
            InferenceResult: Le résultat de l'inférence avec explications
        """
        if not self.running:
            self.logger.warning("Tentative d'utilisation du moteur non démarré")
            if not self.start():
                raise RuntimeError("Impossible de démarrer le moteur d'inférence")

        self.logger.info(f"Traitement de la requête d'inférence: {query.question}")

        # Vérifier le cache pour des requêtes identiques récentes
        cache_key = self._compute_cache_key(query)
        if cache_key in self.results_cache:
            result, timestamp = self.results_cache[cache_key]
            if time.time() - timestamp < self.cache_expiry:
                self.logger.info(f"Résultat trouvé en cache pour {query.query_id}")
                # Créer une copie pour éviter la modification du cache
                return self._copy_result(result)

        # Chronométrer l'exécution
        start_time = time.time()

        # Sélectionner la stratégie d'inférence basée sur le mode hybride
        if query.hybrid_mode == HybridMode.ADAPTIVE:
            hybrid_mode = self._select_adaptive_mode(query)
        else:
            hybrid_mode = query.hybrid_mode

        # Exécuter l'inférence selon le mode hybride choisi
        result = self._execute_hybrid_inference(query, hybrid_mode)

        # Compléter le résultat
        result.execution_time = time.time() - start_time

        # Mettre en cache le résultat
        self.results_cache[cache_key] = (result, time.time())

        # Enregistrer des métriques pour l'apprentissage adaptatif
        self._update_metrics(query, result)

        return result

    def add_symbolic_rule(self, rule: SymbolicConstraint) -> bool:
        """
        Ajoute une règle symbolique au moteur d'inférence.

        Args:
            rule: La règle/contrainte à ajouter

        Returns:
            bool: True si l'ajout a réussi, False sinon
        """
        with self.lock:
            # Vérifier que la règle est valide
            if not rule.expression:
                self.logger.warning("Tentative d'ajout d'une règle vide")
                return False

            # Vérifier les doublons
            for existing_rule in self.symbolic_rules:
                if existing_rule.constraint_id == rule.constraint_id:
                    self.logger.warning(f"Règle {rule.constraint_id} déjà existante, mise à jour")
                    # Remplacer l'ancienne règle
                    self.symbolic_rules.remove(existing_rule)
                    break

            # Ajouter la règle
            self.symbolic_rules.append(rule)
            self.logger.info(f"Règle symbolique ajoutée: {rule.constraint_id} - {rule.expression}")

            # Si le moteur symbolique est disponible, lui passer aussi la règle
            if self.symbolic_engine:
                try:
                    # Interface supposée - peut nécessiter une adaptation
                    self.symbolic_engine.add_rule(rule)
                except Exception as e:
                    self.logger.warning(f"Échec de l'ajout au moteur symbolique: {str(e)}")

            return True

    def adjust_parameters(self, new_params: Dict[str, Any]) -> bool:
        """
        Ajuste les paramètres du moteur d'inférence hybride.

        Args:
            new_params: Dictionnaire de paramètres à modifier

        Returns:
            bool: True si les paramètres ont été modifiés, False sinon
        """
        with self.lock:
            for key, value in new_params.items():
                if key in self.params:
                    old_value = self.params[key]
                    self.params[key] = value
                    self.logger.info(f"Paramètre {key} modifié: {old_value} -> {value}")
                else:
                    self.logger.warning(f"Paramètre inconnu: {key}")

            return True

    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques de performance du moteur d'inférence.

        Returns:
            Dict: Métriques de performance
        """
        with self.lock:
            metrics_snapshot = {}
            for key, values in self.metrics.items():
                if values:
                    metrics_snapshot[key] = {
                        'current': values[-1],
                        'average': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    }
            return metrics_snapshot

    def explain_inference(self, result: InferenceResult) -> str:
        """
        Génère une explication en langage naturel du processus d'inférence.

        Args:
            result: Résultat d'inférence à expliquer

        Returns:
            str: Explication détaillée du raisonnement
        """
        if not result.steps:
            return "Aucune étape d'inférence disponible pour explication."

        # Si le LLM est disponible, l'utiliser pour générer une explication cohérente
        if self.llm_engine and result.llm_contribution > 0.1:
            try:
                # Préparer le prompt avec les étapes
                steps_text = "\n".join([
                    f"Étape {i+1}: {step.type} - {step.description}\n"
                    f"Raisonnement: {step.reasoning}\n"
                    f"Confiance: {step.confidence:.2f}"
                    for i, step in enumerate(result.steps)
                ])

                prompt = (
                    f"Explication du raisonnement:\n\n"
                    f"Question: {result.query_id}\n"
                    f"Étapes d'inférence:\n{steps_text}\n\n"
                    f"Réponses: {result.answers}\n\n"
                    f"Veuillez expliquer ce raisonnement de manière claire et cohérente:"
                )

                # Appeler le LLM pour obtenir une explication
                response = self.llm_engine.complete({
                    "prompt": prompt,
                    "max_tokens": 300,
                    "temperature": 0.7
                })

                if response and hasattr(response, 'text') and response.text:
                    return response.text
            except Exception as e:
                self.logger.warning(f"Échec de l'explication via LLM: {str(e)}")

        # Méthode alternative: construire une explication structurée
        explanation = [f"Explication du processus d'inférence pour: {result.query_id}\n"]

        for i, step in enumerate(result.steps):
            explanation.append(f"Étape {i+1}: {step.type}")
            explanation.append(f"  {step.description}")
            if step.reasoning:
                explanation.append(f"  Raisonnement: {step.reasoning}")
            if step.confidence < 1.0:
                explanation.append(f"  Confiance: {step.confidence:.2f}")
            explanation.append("")

        explanation.append(f"Conclusion:")
        for var, value in result.answers.items():
            explanation.append(f"  {var} = {value}")
        explanation.append(f"Confiance globale: {result.confidence:.2f}")

        contribution_info = []
        if result.symbolic_contribution > 0.01:
            contribution_info.append(f"{result.symbolic_contribution:.0%} symbolique")
        if result.neural_contribution > 0.01:
            contribution_info.append(f"{result.neural_contribution:.0%} neuronal")
        if result.llm_contribution > 0.01:
            contribution_info.append(f"{result.llm_contribution:.0%} LLM")

        if contribution_info:
            explanation.append(f"Contributions: {', '.join(contribution_info)}")

        return "\n".join(explanation)

    def register_inference_strategy(self, name: str,
                                   strategy_fn: Callable[[InferenceQuery], InferenceResult],
                                   applicable_modes: List[HybridMode],
                                   applicable_types: List[InferenceType]) -> bool:
        """
        Enregistre une stratégie d'inférence personnalisée.

        Args:
            name: Nom unique de la stratégie
            strategy_fn: Fonction implémentant la stratégie
            applicable_modes: Modes hybrides compatibles
            applicable_types: Types d'inférence compatibles

        Returns:
            bool: True si l'enregistrement a réussi, False sinon
        """
        with self.lock:
            if name in self.inference_strategies:
                self.logger.warning(f"Stratégie {name} déjà enregistrée, remplacement")

            self.inference_strategies[name] = {
                'function': strategy_fn,
                'modes': applicable_modes,
                'types': applicable_types,
                'last_used': 0,
                'usage_count': 0,
                'success_rate': 0.0
            }

            self.logger.info(f"Stratégie d'inférence enregistrée: {name}")
            return True

    def unregister_inference_strategy(self, name: str) -> bool:
        """
        Supprime une stratégie d'inférence du registre.

        Args:
            name: Nom de la stratégie à supprimer

        Returns:
            bool: True si la suppression a réussi, False sinon
        """
        with self.lock:
            if name not in self.inference_strategies:
                self.logger.warning(f"Stratégie {name} non trouvée")
                return False

            del self.inference_strategies[name]
            self.logger.info(f"Stratégie d'inférence supprimée: {name}")
            return True

    # -----------------------------------------------------------------------
    # Méthodes internes
    # -----------------------------------------------------------------------

    def _register_default_strategies(self) -> None:
        """Enregistre les stratégies d'inférence par défaut."""
        # Stratégie symbolique-première
        self.register_inference_strategy(
            name="symbolic_first",
            strategy_fn=self._symbolic_first_strategy,
            applicable_modes=[HybridMode.SYMBOLIC_FIRST, HybridMode.ADAPTIVE],
            applicable_types=[InferenceType.DEDUCTION, InferenceType.ABDUCTION,
                             InferenceType.PROBABILISTIC, InferenceType.FUZZY]
        )

        # Stratégie neurale-première
        self.register_inference_strategy(
            name="neural_first",
            strategy_fn=self._neural_first_strategy,
            applicable_modes=[HybridMode.NEURAL_FIRST, HybridMode.ADAPTIVE],
            applicable_types=[InferenceType.INDUCTION, InferenceType.ANALOGY,
                             InferenceType.SPATIAL, InferenceType.CAUSAL]
        )

        # Stratégie parallèle
        self.register_inference_strategy(
            name="parallel",
            strategy_fn=self._parallel_strategy,
            applicable_modes=[HybridMode.PARALLEL, HybridMode.ADAPTIVE],
            applicable_types=[t for t in InferenceType]  # Tous les types
        )

        # Stratégie itérative
        self.register_inference_strategy(
            name="iterative",
            strategy_fn=self._iterative_strategy,
            applicable_modes=[HybridMode.ITERATIVE, HybridMode.ADAPTIVE],
            applicable_types=[InferenceType.DEDUCTION, InferenceType.INDUCTION,
                             InferenceType.ABDUCTION, InferenceType.PROBABILISTIC]
        )

        # Stratégie LLM-orchestrée (si LLM disponible)
        if self.llm_engine:
            self.register_inference_strategy(
                name="llm_orchestrated",
                strategy_fn=self._llm_orchestrated_strategy,
                applicable_modes=[HybridMode.LLM_ORCHESTRATED, HybridMode.ADAPTIVE],
                applicable_types=[InferenceType.LLM_GUIDED, InferenceType.ANALOGY,
                                 InferenceType.CAUSAL, InferenceType.TEMPORAL]
            )

    def _register_with_nexus_core(self) -> None:
        """Enregistre ce module auprès du NexusCore."""
        if not self.nexus_core:
            self.logger.warning("Pas de NexusCore disponible pour l'enregistrement")
            return

        try:
            # Préparer les dépendances
            dependencies = []
            if self.reasoning_core:
                dependencies.append("reasoning_core")
            if self.symbolic_engine:
                dependencies.append("symbolic_engine")
            if self.neural_reasoner:
                dependencies.append("neural_reasoner")
            if self.semantic_memory:
                dependencies.append("semantic_memory")

            # Enregistrement auprès du NexusCore
            self.nexus_core.register_module(
                name="hybrid_inference",
                module=self,
                dependencies=dependencies
            )

            self.logger.info("Enregistrement réussi auprès du NexusCore")
        except Exception as e:
            self.logger.error(f"Échec de l'enregistrement auprès du NexusCore: {str(e)}")

    def _initialize_neuro_symbolic_models(self) -> None:
        """Initialise les modèles neuro-symboliques spécifiques."""
        try:
            # Charger les modèles LNN (Logical Neural Networks) si disponible
            # Note: Ceci est un placeholder - dans une implémentation réelle,
            # nous initialiserions les frameworks spécifiques
            self.neural_models["lnn"] = {
                "type": "lnn",
                "initialized": True,
                "description": "Logical Neural Network pour inférence symbolique avec activation neurale"
            }

            # DeepProbLog
            self.neural_models["deepproblog"] = {
                "type": "deepproblog",
                "initialized": True,
                "description": "DeepProbLog pour programmation logique probabiliste neuronale"
            }

            self.logger.info("Modèles neuro-symboliques initialisés")
        except Exception as e:
            self.logger.warning(f"Échec de l'initialisation des modèles neuro-symboliques: {str(e)}")

    def _execute_hybrid_inference(self, query: InferenceQuery,
                                 hybrid_mode: HybridMode) -> InferenceResult:
        """
        Exécute l'inférence selon le mode hybride spécifié.

        Args:
            query: La requête d'inférence
            hybrid_mode: Le mode hybride à utiliser

        Returns:
            InferenceResult: Résultat de l'inférence
        """
        # Initialiser le résultat
        result = InferenceResult(query_id=query.query_id)
        result.steps = []

        # Sélectionner la stratégie d'inférence appropriée
        strategy_name = self._select_strategy(query, hybrid_mode)

        if not strategy_name or strategy_name not in self.inference_strategies:
            self.logger.warning(f"Aucune stratégie trouvée pour le mode {hybrid_mode}")
            return result

        # Exécuter la stratégie
        strategy = self.inference_strategies[strategy_name]
        try:
            # Chronométrer l'exécution
            start_time = time.time()

            # Appeler la fonction de stratégie
            result = strategy['function'](query)

            # Mettre à jour les statistiques d'utilisation
            strategy['last_used'] = time.time()
            strategy['usage_count'] += 1

            # Mettre à jour le taux de succès
            success = bool(result.success)
            old_success, old_total = self.strategy_success_rates[strategy_name]
            self.strategy_success_rates[strategy_name] = (old_success + (1 if success else 0), old_total + 1)

            self.logger.info(f"Inférence exécutée via stratégie '{strategy_name}', succès={success}")

            # Compléter le résultat si nécessaire
            if not result.explanation:
                result.explanation = self.explain_inference(result)

            return result

        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution de la stratégie {strategy_name}: {str(e)}")
            result.explanation = f"Erreur d'inférence: {str(e)}"
            return result

    def _select_strategy(self, query: InferenceQuery,
                        hybrid_mode: HybridMode) -> Optional[str]:
        """
        Sélectionne la stratégie d'inférence la plus appropriée.

        Args:
            query: La requête d'inférence
            hybrid_mode: Le mode d'inférence hybride

        Returns:
            str: Nom de la stratégie sélectionnée, ou None si aucune trouvée
        """
        # Trouver les stratégies compatibles avec le mode et les types d'inférence
        compatible_strategies = []

        for name, strategy in self.inference_strategies.items():
            # Vérifier la compatibilité du mode
            if hybrid_mode not in strategy['modes']:
                continue

            # Vérifier la compatibilité des types d'inférence
            types_compatible = False
            for inference_type in query.inference_types:
                if inference_type in strategy['types']:
                    types_compatible = True
                    break

            if not types_compatible and query.inference_types:
                continue

            compatible_strategies.append(name)

        if not compatible_strategies:
            return None

        # Si une seule stratégie est compatible, la retourner
        if len(compatible_strategies) == 1:
            return compatible_strategies[0]

        # Si mode adaptatif, utiliser les statistiques d'utilisation et de succès
        if hybrid_mode == HybridMode.ADAPTIVE:
            # Calculer un score pour chaque stratégie basé sur:
            # - Taux de succès historique
            # - Pertinence pour les types d'inférence demandés
            # - Avec un peu d'exploration aléatoire

            scores = {}
            for name in compatible_strategies:
                # Calculer le taux de succès
                success, total = self.strategy_success_rates[name]
                success_rate = success / total if total > 0 else 0.5

                # Score de base = taux de succès
                scores[name] = success_rate

                # Bonus pour l'expérience (mais éviter de trop favoriser les stratégies déjà utilisées)
                usage_bonus = min(0.2, self.inference_strategies[name]['usage_count'] / 100)
                scores[name] += usage_bonus

                # Petit facteur aléatoire pour l'exploration (max 0.1)
                scores[name] += random.random() * 0.1

            # Sélectionner la stratégie avec le meilleur score
            return max(scores, key=scores.get) if scores else None

        # Sinon, utiliser une stratégie compatible de manière déterministe
        else:
            # Par défaut, prendre la première stratégie compatible
            return compatible_strategies[0]

    def _select_adaptive_mode(self, query: InferenceQuery) -> HybridMode:
        """
        Sélectionne dynamiquement le meilleur mode hybride pour la requête.

        Args:
            query: La requête d'inférence

        Returns:
            HybridMode: Le mode hybride sélectionné
        """
        # Pour la décision adaptative, nous considérons:
        # 1. La nature de la requête (types d'inférence)
        # 2. Les données disponibles (faits connus, contraintes)
        # 3. L'historique des succès/échecs précédents

        # Règles heuristiques pour la sélection du mode:

        # Si nous avons beaucoup de contraintes symboliques, privilégier symbolique
        if len(query.constraints) > 5:
            return HybridMode.SYMBOLIC_FIRST

        # Si nous avons peu de contraintes mais beaucoup de contexte, privilégier neuronal
        if len(query.constraints) < 2 and len(query.context) > 5:
            return HybridMode.NEURAL_FIRST

        # Si la requête est une LLM_GUIDED, utiliser ce mode
        if InferenceType.LLM_GUIDED in query.inference_types and self.llm_engine:
            return HybridMode.LLM_ORCHESTRATED

        # Si la requête inclut des raisonnements spatiaux ou analogiques, privilégier neuronal
        if (InferenceType.SPATIAL in query.inference_types or
            InferenceType.ANALOGY in query.inference_types):
            return HybridMode.NEURAL_FIRST

        # Si la requête est déductive ou abductive, privilégier symbolique
        if (InferenceType.DEDUCTION in query.inference_types or
            InferenceType.ABDUCTION in query.inference_types):
            return HybridMode.SYMBOLIC_FIRST

        # Si timing est critique (petit time_limit), utiliser parallèle pour optimiser
        if query.time_limit and query.time_limit < 1.0:
            return HybridMode.PARALLEL

        # Pour les questions complexes et multi-étapes, préférer itératif
        if query.max_inference_depth > 3:
            return HybridMode.ITERATIVE

        # Par défaut, utiliser parallèle qui est généralement équilibré
        return HybridMode.PARALLEL

    def _compute_cache_key(self, query: InferenceQuery) -> str:
        """
        Génère une clé de cache pour une requête d'inférence.

        Args:
            query: La requête d'inférence

        Returns:
            str: Clé de cache unique
        """
        # Éléments clés à inclure dans la signature de cache
        elements = [
            query.question,
            sorted(query.target_variables),
            tuple((c.constraint_id, c.expression) for c in query.constraints),
            sorted((k, str(v)) for k, v in query.known_facts.items())
        ]

        # Créer une signature en chaîne
        signature = str(elements)

        # Hacher pour avoir une clé compacte
        import hashlib
        return hashlib.md5(signature.encode('utf-8')).hexdigest()

    def _copy_result(self, result: InferenceResult) -> InferenceResult:
        """
        Crée une copie d'un résultat d'inférence.

        Args:
            result: Le résultat à copier

        Returns:
            InferenceResult: Copie du résultat
        """
        # Cette implémentation est simplifiée, une vraie copie profonde serait plus robuste
        import copy
        return copy.deepcopy(result)

    def _update_metrics(self, query: InferenceQuery, result: InferenceResult) -> None:
        """
        Met à jour les métriques de performance.

        Args:
            query: La requête d'inférence
            result: Le résultat d'inférence
        """
        self.metrics["success_rate"].append(1.0 if result.success else 0.0)
        self.metrics["confidence"].append(result.confidence)
        self.metrics["execution_time"].append(result.execution_time)
        self.metrics["steps_count"].append(len(result.steps))

        # Suivre les contribuions
        self.metrics["symbolic_contribution"].append(result.symbolic_contribution)
        self.metrics["neural_contribution"].append(result.neural_contribution)
        self.metrics["llm_contribution"].append(result.llm_contribution)

    # -----------------------------------------------------------------------
    # Implémentations des stratégies d'inférence
    # -----------------------------------------------------------------------

    def _symbolic_first_strategy(self, query: InferenceQuery) -> InferenceResult:
        """
        Stratégie symbolique-première: commence par un raisonnement
        symbolique puis affine avec neuronal si nécessaire.

        Args:
            query: La requête d'inférence

        Returns:
            InferenceResult: Résultat d'inférence
        """
        # Initialiser le résultat
        result = InferenceResult(query_id=query.query_id)
        result.steps = []

        # Étape 1: Appliquer l'inférence symbolique
        symbolic_result = self._apply_symbolic_reasoning(query)
        result.steps.extend(symbolic_result.steps)

        # Si le résultat symbolique est suffisant (haute confiance), le retourner
        if symbolic_result.success and symbolic_result.confidence >= self.params["confidence_threshold"]:
            result.success = True
            result.answers = symbolic_result.answers
            result.confidence = symbolic_result.confidence
            result.symbolic_contribution = 1.0
            result.neural_contribution = 0.0
            return result

        # Étape 2: Affiner avec inférence neurale
        neural_input = query
        # Injecter les résultats symboliques comme faits connus pour l'inférence neurale
        for var, value in symbolic_result.answers.items():
            neural_input.known_facts[var] = value

        neural_result = self._apply_neural_reasoning(neural_input)
        result.steps.extend(neural_result.steps)

        # Combiner les résultats
        combined_answers = {}
        combined_answers.update(symbolic_result.answers)

        # Pour les variables qui n'ont pas été résolues symboliquement ou avec faible confiance
        for var, value in neural_result.answers.items():
            if var not in combined_answers or symbolic_result.confidence < 0.7:
                combined_answers[var] = value

        result.answers = combined_answers

        # Calculer la confiance combinée - plus élevée pour les réponses symboliques
        symbolic_weight = self.params["symbolic_weight"]
        neural_weight = self.params["neural_weight"]

        # Calcul de la contribution proportionnelle
        if symbolic_result.success and neural_result.success:
            # Si les deux ont réussi, pondérer leurs contributions
            result.symbolic_contribution = 0.7  # Dominant car stratégie symbolique-première
            result.neural_contribution = 0.3
            result.confidence = (symbolic_result.confidence * symbolic_weight +
                               neural_result.confidence * neural_weight) / (symbolic_weight + neural_weight)
        elif symbolic_result.success:
            result.symbolic_contribution = 1.0
            result.confidence = symbolic_result.confidence
        elif neural_result.success:
            result.neural_contribution = 1.0
            result.confidence = neural_result.confidence
        else:
            # Aucun n'a réussi
            result.confidence = 0.0

        result.success = bool(result.answers)

        return result

    def _neural_first_strategy(self, query: InferenceQuery) -> InferenceResult:
        """
        Stratégie neurale-première: commence par un raisonnement
        neuronal puis vérifie/corrige avec symbolique.

        Args:
            query: La requête d'inférence

        Returns:
            InferenceResult: Résultat d'inférence
        """
        # Initialiser le résultat
        result = InferenceResult(query_id=query.query_id)
        result.steps = []

        # Étape 1: Appliquer l'inférence neurale
        neural_result = self._apply_neural_reasoning(query)
        result.steps.extend(neural_result.steps)

        # Si l'inférence neurale a échoué complètement, pas la peine de continuer
        if not neural_result.success:
            # Essayer quand même l'inférence symbolique comme fallback
            symbolic_result = self._apply_symbolic_reasoning(query)
            result.steps.extend(symbolic_result.steps)
            result.success = symbolic_result.success
            result.answers = symbolic_result.answers
            result.confidence = symbolic_result.confidence
            result.symbolic_contribution = 1.0
            return result

        # Étape 2: Vérifier et contraindre avec inférence symbolique
        symbolic_input = query
        # Injecter les résultats neuraux comme hypothèses pour l'inférence symbolique
        for var, value in neural_result.answers.items():
            if var not in symbolic_input.known_facts:
                symbolic_input.known_facts[var] = value

        symbolic_result = self._apply_symbolic_reasoning(symbolic_input)
        result.steps.extend(symbolic_result.steps)

        # Combiner les résultats - privilégier symbolique pour les contraintes dures
        combined_answers = {}

        # Commencer par les réponses neurales
        combined_answers.update(neural_result.answers)

        # Remplacer par les réponses symboliques qui sont plus fiables
        for var, value in symbolic_result.answers.items():
            # Si confiance élevée, toujours préférer la réponse symbolique
            if var in symbolic_result.answers and symbolic_result.confidence > 0.8:
                combined_answers[var] = value

        result.answers = combined_answers

        # Calculer la confiance combinée et les contributions
        if symbolic_result.success and neural_result.success:
            # Si les deux ont réussi, pondérer leurs contributions
            result.neural_contribution = 0.7  # Dominant car stratégie neurale-première
            result.symbolic_contribution = 0.3

            symbolic_weight = self.params["symbolic_weight"] * 0.5  # Réduit car stratégie neurale-première
            neural_weight = self.params["neural_weight"] * 1.5  # Augmenté car stratégie neurale-première

            result.confidence = (symbolic_result.confidence * symbolic_weight +
                               neural_result.confidence * neural_weight) / (symbolic_weight + neural_weight)
        elif neural_result.success:
            result.neural_contribution = 1.0
            result.confidence = neural_result.confidence
        elif symbolic_result.success:
            result.symbolic_contribution = 1.0
            result.confidence = symbolic_result.confidence
        else:
            # Aucun n'a réussi
            result.confidence = 0.0

        result.success = bool(result.answers)

        return result

    def _parallel_strategy(self, query: InferenceQuery) -> InferenceResult:
        """
        Stratégie parallèle: exécute les inférences symbolique et neurale
        en parallèle, puis fusionne les résultats.

        Args:
            query: La requête d'inférence

        Returns:
            InferenceResult: Résultat d'inférence
        """
        # Initialiser le résultat
        result = InferenceResult(query_id=query.query_id)
        result.steps = []

        # Exécuter les deux types d'inférence en parallèle
        # Note: Dans une implémentation réelle, ceci utiliserait des threads ou asyncio
        # Pour simplifier, nous les exécutons séquentiellement

        symbolic_result = self._apply_symbolic_reasoning(query)
        neural_result = self._apply_neural_reasoning(query)

        # Ajouter les étapes des deux raisonnements
        result.steps.extend(symbolic_result.steps)
        result.steps.extend(neural_result.steps)

        # Fusionner les réponses
        combined_answers = {}

        # Méthode de fusion basée sur la confiance
        for var in set(list(symbolic_result.answers.keys()) + list(neural_result.answers.keys())):
            symbolic_value = symbolic_result.answers.get(var)
            neural_value = neural_result.answers.get(var)

            symbolic_confidence = symbolic_result.confidence if var in symbolic_result.answers else 0
            neural_confidence = neural_result.confidence if var in neural_result.answers else 0

            # Si une seule source a une réponse, l'utiliser
            if symbolic_value is not None and neural_value is None:
                combined_answers[var] = symbolic_value
            elif neural_value is not None and symbolic_value is None:
                combined_answers[var] = neural_value
            # Si les deux ont une réponse, prendre celle avec la plus haute confiance
            elif symbolic_value is not None and neural_value is not None:
                if symbolic_confidence >= neural_confidence:
                    combined_answers[var] = symbolic_value
                else:
                    combined_answers[var] = neural_value

        result.answers = combined_answers

        # Calculer la confiance et les contributions
        if symbolic_result.success and neural_result.success:
            # Si les deux ont réussi, leur contribution est à peu près égale
            result.symbolic_contribution = 0.5
            result.neural_contribution = 0.5

            # Confiance pondérée basée sur les poids configurés
            symbolic_weight = self.params["symbolic_weight"]
            neural_weight = self.params["neural_weight"]

            result.confidence = (symbolic_result.confidence * symbolic_weight +
                               neural_result.confidence * neural_weight) / (symbolic_weight + neural_weight)
        elif symbolic_result.success:
            result.symbolic_contribution = 1.0
            result.confidence = symbolic_result.confidence
        elif neural_result.success:
            result.neural_contribution = 1.0
            result.confidence = neural_result.confidence
        else:
            result.confidence = 0.0

        result.success = bool(result.answers)

        return result

    def _iterative_strategy(self, query: InferenceQuery) -> InferenceResult:
        """
        Stratégie itérative: alterne entre raisonnements symbolique et
        neuronal en plusieurs passes, chacun utilisant les résultats précédents.

        Args:
            query: La requête d'inférence

        Returns:
            InferenceResult: Résultat d'inférence
        """
        # Initialiser le résultat
        result = InferenceResult(query_id=query.query_id)
        result.steps = []

        # Copier la requête pour la modifier à chaque itération
        current_query = query  # Dans une implémentation réelle, faire une copie profonde

        # Variables pour suivre les contributions
        symbolic_contrib = 0.0
        neural_contrib = 0.0

        # Exécuter plusieurs passes, alternant symbolique et neuronal
        max_iterations = query.max_inference_depth or 3

        for i in range(max_iterations):
            # Iteration paire: symbolique, impaire: neuronale
            if i % 2 == 0:
                iter_result = self._apply_symbolic_reasoning(current_query)
                symbolic_weight = 1.0 / (i + 1)  # Poids diminue à chaque itération
                symbolic_contrib += symbolic_weight
            else:
                iter_result = self._apply_neural_reasoning(current_query)
                neural_weight = 1.0 / (i + 1)  # Poids diminue à chaque itération
                neural_contrib += neural_weight

            # Ajouter les étapes
            result.steps.extend(iter_result.steps)

            # Mettre à jour les faits connus pour la prochaine itération
            for var, value in iter_result.answers.items():
                if var not in current_query.known_facts:
                    current_query.known_facts[var] = value

            # Si nous avons toutes les variables cibles, ou si la confiance est très haute, arrêter
            all_targets_found = all(var in current_query.known_facts for var in query.target_variables)
            if all_targets_found and iter_result.confidence > 0.9:
                break

            # Si aucun progrès dans cette itération, arrêter
            if not iter_result.answers:
                break

        # Construire le résultat final
        result.answers = current_query.known_facts.copy()

        # Filtrer pour ne garder que les variables cibles si spécifiées
        if query.target_variables:
            result.answers = {var: val for var, val in result.answers.items()
                            if var in query.target_variables}

        # Calculer les contributions normalisées
        total_contrib = symbolic_contrib + neural_contrib
        if total_contrib > 0:
            result.symbolic_contribution = symbolic_contrib / total_contrib
            result.neural_contribution = neural_contrib / total_contrib

        # Estimer la confiance globale - plus complexe dans une stratégie itérative
        # Ici nous utilisons une heuristique simple: moyenne des confiances des étapes
        confidences = [step.confidence for step in result.steps if hasattr(step, 'confidence')]
        result.confidence = sum(confidences) / len(confidences) if confidences else 0.0

        result.success = bool(result.answers)

        return result

    def _llm_orchestrated_strategy(self, query: InferenceQuery) -> InferenceResult:
        """
        Stratégie LLM-orchestrée: utilise un LLM pour guider et intégrer
        les inférences symboliques et neuronales.

        Args:
            query: La requête d'inférence

        Returns:
            InferenceResult: Résultat d'inférence
        """
        # Cette stratégie n'est disponible que si un LLM est configuré
        if not self.llm_engine:
            self.logger.warning("Stratégie LLM-orchestrée invoquée sans LLM disponible")
            return InferenceResult(query_id=query.query_id, success=False)

        # Initialiser le résultat
        result = InferenceResult(query_id=query.query_id)
        result.steps = []

        # Construire un prompt pour le LLM qui explique la requête et demande un plan
        constraints_text = "\n".join([f"- {c.expression}" for c in query.constraints])
        facts_text = "\n".join([f"- {var}: {value}" for var, value in query.known_facts.items()])

        plan_prompt = (
            f"Question/Requête: {query.question}\n\n"
            f"Variables cibles: {', '.join(query.target_variables)}\n\n"
            f"Contraintes connues:\n{constraints_text}\n\n"
            f"Faits connus:\n{facts_text}\n\n"
            f"Veuillez élaborer un plan d'inférence structuré pour résoudre cette question. "
            f"Précisez quand il faut utiliser le raisonnement symbolique (logique, contraintes) "
            f"et quand il faut utiliser le raisonnement neuronal (similarité, analogie)."
        )

        # Demander un plan au LLM
        try:
            plan_response = self.llm_engine.complete({
                "prompt": plan_prompt,
                "max_tokens": 500,
                "temperature": 0.7
            })

            plan_text = plan_response.text if hasattr(plan_response, 'text') else str(plan_response)

            # Ajouter l'étape de planification
            result.steps.append(InferenceStep(
                step_id=str(uuid.uuid4()),
                type="llm_planning",
                description="Planification du processus d'inférence par LLM",
                reasoning=plan_text,
                confidence=0.9
            ))

            # Analyser le plan pour extraire les étapes d'inférence
            # Cette implémentation est simplifiée - une version réelle utiliserait
            # une analyse plus sophistiquée du plan généré par le LLM

            # Supposons que le plan liste clairement "Étape 1", "Étape 2", etc.
            plan_steps = []
            current_step = None

            for line in plan_text.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # Détecter les en-têtes d'étapes
                if line.lower().startswith(("étape", "step", "phase")):
                    if current_step:
                        plan_steps.append(current_step)
                    current_step = {"description": line, "type": "unknown", "content": []}

                    # Essayer de déterminer le type d'étape
                    lower_line = line.lower()
                    if any(kw in lower_line for kw in ["symbolique", "logique", "déduction", "contrainte"]):
                        current_step["type"] = "symbolic"
                    elif any(kw in lower_line for kw in ["neural", "neuronal", "analogie", "similarité"]):
                        current_step["type"] = "neural"
                elif current_step:
                    current_step["content"].append(line)

            # Ajouter la dernière étape
            if current_step:
                plan_steps.append(current_step)

            # Exécuter les étapes du plan
            current_facts = query.known_facts.copy()

            for i, plan_step in enumerate(plan_steps):
                step_type = plan_step["type"]
                step_desc = plan_step["description"]

                # Créer une sous-requête pour cette étape
                sub_query = query  # Dans un vrai code, faire une copie profonde
                sub_query.known_facts = current_facts

                # Exécuter l'étape selon son type
                if step_type == "symbolic":
                    step_result = self._apply_symbolic_reasoning(sub_query)
                elif step_type == "neural":
                    step_result = self._apply_neural_reasoning(sub_query)
                else:
                    # Type inconnu, par défaut essayer les deux et prendre le meilleur
                    symbolic_result = self._apply_symbolic_reasoning(sub_query)
                    neural_result = self._apply_neural_reasoning(sub_query)

                    # Choisir le résultat avec le plus de réponses ou la plus haute confiance
                    if len(symbolic_result.answers) > len(neural_result.answers):
                        step_result = symbolic_result
                    elif len(neural_result.answers) > len(symbolic_result.answers):
                        step_result = neural_result
                    else:
                        step_result = symbolic_result if symbolic_result.confidence >= neural_result.confidence else neural_result

                # Mettre à jour les faits connus
                current_facts.update(step_result.answers)

                # Ajouter les étapes de cette sous-inférence
                result.steps.extend(step_result.steps)

            # Intégration finale des résultats par le LLM
            integration_prompt = (
                f"Question initiale: {query.question}\n\n"
                f"Variables cibles: {', '.join(query.target_variables)}\n\n"
                f"Résultats obtenus:\n"
            )

            for var, value in current_facts.items():
                integration_prompt += f"- {var}: {value}\n"

            integration_prompt += (
                f"\nVeuillez fournir une réponse finale intégrée à la question initiale, "
                f"en indiquant votre niveau de confiance."
            )

            integration_response = self.llm_engine.complete({
                "prompt": integration_prompt,
                "max_tokens": 300,
                "temperature": 0.7
            })

            integration_text = integration_response.text if hasattr(integration_response, 'text') else str(integration_response)

            # Ajouter l'étape d'intégration
            result.steps.append(InferenceStep(
                step_id=str(uuid.uuid4()),
                type="llm_integration",
                description="Intégration finale des résultats par LLM",
                reasoning=integration_text,
                confidence=0.85
            ))

            # Définir les résultats finaux
            result.answers = current_facts
            if query.target_variables:
                result.answers = {var: val for var, val in result.answers.items()
                               if var in query.target_variables}

            # Évaluer les contributions
            step_types = [step.type for step in result.steps if hasattr(step, 'type')]
            symbolic_count = sum(1 for t in step_types if 'symbolic' in t)
            neural_count = sum(1 for t in step_types if 'neural' in t)
            llm_count = sum(1 for t in step_types if 'llm' in t)

            total_steps = symbolic_count + neural_count + llm_count
            if total_steps > 0:
                result.symbolic_contribution = symbolic_count / total_steps
                result.neural_contribution = neural_count / total_steps
                result.llm_contribution = llm_count / total_steps
            else:
                result.llm_contribution = 1.0

            # Estimer la confiance globale
            result.confidence = 0.8  # Valeur par défaut pour la stratégie LLM-orchestrée

            # Essayer d'extraire une confiance du texte d'intégration
            confidence_keywords = {
                "très confiant": 0.9,
                "confiant": 0.8,
                "assez confiant": 0.7,
                "moyennement confiant": 0.6,
                "peu confiant": 0.5,
                "incertain": 0.4,
                "très incertain": 0.3
            }

            for keyword, conf_value in confidence_keywords.items():
                if keyword in integration_text.lower():
                    result.confidence = conf_value
                    break

            result.success = bool(result.answers)
            result.explanation = integration_text

            return result

        except Exception as e:
            self.logger.error(f"Erreur dans la stratégie LLM-orchestrée: {str(e)}")
            result.steps.append(InferenceStep(
                step_id=str(uuid.uuid4()),
                type="error",
                description=f"Erreur lors de l'exécution de la stratégie LLM-orchestrée: {str(e)}",
                confidence=0.0
            ))
            result.success = False
            return result

    def _apply_symbolic_reasoning(self, query: InferenceQuery) -> InferenceResult:
        """
        Applique le raisonnement symbolique à une requête.

        Args:
            query: La requête d'inférence

        Returns:
            InferenceResult: Résultat de l'inférence symbolique
        """
        # Cette méthode est un placeholder - dans une implémentation réelle,
        # elle appellerait un moteur symbolique sophistiqué (par ex. via symbolic_engine)

        # Initialiser le résultat
        result = InferenceResult(query_id=query.query_id)
        result.steps = []

        # Si un moteur symbolique est disponible, l'utiliser
        if self.symbolic_engine:
            try:
                # Interface supposée - adapter selon l'API réelle du moteur
                engine_result = self.symbolic_engine.solve(
                    constraints=[c.expression for c in query.constraints],
                    known_facts=query.known_facts,
                    target_variables=query.target_variables
                )

                # Transformer le résultat du moteur symbolique
                result.answers = engine_result.get("solutions", {})
                result.confidence = engine_result.get("confidence", 0.8)
                result.success = engine_result.get("success", False)

                # Ajouter les étapes de raisonnement
                for i, step in enumerate(engine_result.get("steps", [])):
                    result.steps.append(InferenceStep(
                        step_id=str(uuid.uuid4()),
                        type="symbolic",
                        description=f"Étape symbolique {i+1}",
                        reasoning=step.get("explanation", ""),
                        input_facts=step.get("input", {}),
                        output_facts=step.get("output", {}),
                        confidence=step.get("confidence", 0.8)
                    ))

                return result

            except Exception as e:
                self.logger.error(f"Erreur du moteur symbolique: {str(e)}")
                # Continuer avec l'implémentation de secours

        # Implémentation simplifiée de secours (si pas de moteur ou erreur)
        # Ceci n'est qu'une simulation très basique de raisonnement symbolique

        # Simuler une étape d'analyse des contraintes
        result.steps.append(InferenceStep(
            step_id=str(uuid.uuid4()),
            type="symbolic_analysis",
            description="Analyse des contraintes symboliques",
            input_facts=query.known_facts,
            confidence=0.7,
            reasoning="Analyse des contraintes disponibles pour déterminer les relations logiques."
        ))

        # Simuler une résolution de contraintes extrêmement simple
        # Dans une vraie implémentation, ce serait un solveur de contraintes
        answers = query.known_facts.copy()

        # Raisonnement symbolique simple sur les contraintes
        for constraint in query.constraints:
            # Exemple: traitement très basique pour des règles d'implication simples A → B
            if "→" in constraint.expression or "->" in constraint.expression:
                parts = constraint.expression.replace("→", "->").split("->")
                if len(parts) == 2:
                    antecedent = parts[0].strip()
                    consequent = parts[1].strip()

                    # Si l'antécédent est connu et vrai, inférer le conséquent
                    if antecedent in answers and answers[antecedent]:
                        answers[consequent] = True

                        result.steps.append(InferenceStep(
                            step_id=str(uuid.uuid4()),
                            type="symbolic_deduction",
                            description=f"Application de la règle d'implication: {constraint.expression}",
                            input_facts={antecedent: answers[antecedent]},
                            output_facts={consequent: True},
                            confidence=constraint.confidence,
                            reasoning=f"Puisque {antecedent} est vrai, on peut déduire que {consequent} est vrai."
                        ))

        # Déterminer le succès et filtrer les réponses pour les variables cibles
        target_answers = {}
        for var in query.target_variables:
            if var in answers:
                target_answers[var] = answers[var]

        result.answers = target_answers if query.target_variables else answers
        result.success = bool(result.answers)
        result.confidence = 0.7 if result.success else 0.0

        return result

    def _apply_neural_reasoning(self, query: InferenceQuery) -> InferenceResult:
        """
        Applique le raisonnement neuronal à une requête.

        Args:
            query: La requête d'inférence

        Returns:
            InferenceResult: Résultat de l'inférence neuronale
        """
        # Cette méthode est un placeholder - dans une implémentation réelle,
        # elle appellerait un modèle neuronal (par ex. via neural_reasoner)

        # Initialiser le résultat
        result = InferenceResult(query_id=query.query_id)
        result.steps = []

        # Si un raisonneur neuronal est disponible, l'utiliser
        if self.neural_reasoner:
            try:
                # Interface supposée - adapter selon l'API réelle du raisonneur
                reasoner_result = self.neural_reasoner.infer(
                    question=query.question,
                    context=query.context,
                    known_facts=query.known_facts
                )

                # Transformer le résultat du raisonneur neuronal
                result.answers = reasoner_result.get("predictions", {})
                result.confidence = reasoner_result.get("confidence", 0.7)
                result.success = reasoner_result.get("success", False)

                # Ajouter les étapes de raisonnement
                for i, step in enumerate(reasoner_result.get("steps", [])):
                    result.steps.append(InferenceStep(
                        step_id=str(uuid.uuid4()),
                        type="neural",
                        description=f"Étape neuronale {i+1}",
                        reasoning=step.get("explanation", ""),
                        input_facts=step.get("input", {}),
                        output_facts=step.get("output", {}),
                        confidence=step.get("confidence", 0.7)
                    ))

                return result

            except Exception as e:
                self.logger.error(f"Erreur du raisonneur neuronal: {str(e)}")
                # Continuer avec l'implémentation de secours

        # Implémentation simplifiée de secours (si pas de raisonneur ou erreur)
        # Ceci n'est qu'une simulation très basique de raisonnement neuronal

        # Simuler une étape d'encodage
        result.steps.append(InferenceStep(
            step_id=str(uuid.uuid4()),
            type="neural_encoding",
            description="Encodage neuronal de la requête et du contexte",
            confidence=0.8,
            reasoning="Transformation de la requête et du contexte en représentations vectorielles."
        ))

        # Simuler une recherche de similitude
        has_semantic_memory = self.semantic_memory is not None

        if has_semantic_memory and query.question:
            # Simuler une recherche dans la mémoire sémantique
            result.steps.append(InferenceStep(
                step_id=str(uuid.uuid4()),
                type="neural_retrieval",
                description="Recherche de concepts similaires en mémoire sémantique",
                confidence=0.6,
                reasoning="Recherche de concepts et relations pertinents à la requête."
            ))

            # Simuler des réponses basées sur les variables cibles
            answers = {}
            for var in query.target_variables:
                # Simuler une prédiction avec une valeur aléatoire pour la démonstration
                # Dans une vraie implémentation, cela serait une inférence basée sur embeddings
                answers[var] = f"valeur_prédite_pour_{var}"

                result.steps.append(InferenceStep(
                    step_id=str(uuid.uuid4()),
                    type="neural_prediction",
                    description=f"Prédiction neuronale pour la variable {var}",
                    output_facts={var: answers[var]},
                    confidence=0.65,
                    reasoning=f"Inférence de la valeur la plus probable pour {var} basée sur similitudes."
                ))

            result.answers = answers
            result.success = bool(result.answers)
            result.confidence = 0.65 if result.success else 0.0
        else:
            # Pas de mémoire sémantique ou pas de question
            result.steps.append(InferenceStep(
                step_id=str(uuid.uuid4()),
                type="neural_fallback",
                description="Raisonnement neuronal non applicable",
                confidence=0.0,
                reasoning="Impossible d'appliquer un raisonnement neuronal dans ce contexte."
            ))

            result.success = False
            result.confidence = 0.0

        return result

    def _apply_neuro_symbolic_reasoning(self, query: InferenceQuery) -> InferenceResult:
        """
        Applique un raisonnement neuro-symbolique unifié à une requête.
        Utilise des frameworks comme LNN ou DeepProbLog pour combiner
        les approches symboliques et neuronales en un seul modèle.

        Args:
            query: La requête d'inférence

        Returns:
            InferenceResult: Résultat de l'inférence neuro-symbolique
        """
        # Cette méthode est un placeholder - dans une implémentation réelle,
        # elle utiliserait un framework neuro-symbolique complet

        # Initialiser le résultat
        result = InferenceResult(query_id=query.query_id)
        result.steps = []

        # Vérifier si nous avons un modèle LNN initialisé
        if "lnn" in self.neural_models and self.neural_models["lnn"]["initialized"]:
            # Simuler une étape LNN
            result.steps.append(InferenceStep(
                step_id=str(uuid.uuid4()),
                type="neuro_symbolic_lnn",
                description="Raisonnement avec Logical Neural Network",
                confidence=0.75,
                reasoning="Application de LNN pour intégrer règles logiques et activation neuronale."
            ))

            # Simuler la propagation de l'activation dans LNN
            result.steps.append(InferenceStep(
                step_id=str(uuid.uuid4()),
                type="neuro_symbolic_activation",
                description="Propagation de l'activation dans le réseau logique",
                confidence=0.8,
                reasoning="Initialisation des neurones logiques et propagation des valeurs jusqu'à convergence."
            ))

            # Simuler extraction de résultats depuis LNN
            answers = {}
            for var in query.target_variables:
                # Simuler une prédiction
                answers[var] = f"ns_valeur_pour_{var}"

                result.steps.append(InferenceStep(
                    step_id=str(uuid.uuid4()),
                    type="neuro_symbolic_extraction",
                    description=f"Extraction de valeur pour {var} depuis LNN",
                    output_facts={var: answers[var]},
                    confidence=0.85,
                    reasoning=f"La valeur pour {var} a été extraite avec une confiance élevée via LNN."
                ))

            result.answers = answers
            result.success = bool(result.answers)
            result.confidence = 0.85 if result.success else 0.0

            # Dans un vrai système, les contributions seraient plus nuancées
            result.symbolic_contribution = 0.5
            result.neural_contribution = 0.5

            return result

        # Si pas de framework neuro-symbolique disponible, retomber sur l'approche hybride
        self.logger.info("Aucun framework neuro-symbolique disponible, recours à l'approche hybride")

        # Utiliser la stratégie parallèle comme fallback
        return self._parallel_strategy(query)