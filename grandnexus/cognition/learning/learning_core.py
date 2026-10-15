"""
learning_core.py - Core Learning Module for GrandNexus

This module serves as the central orchestrator for all learning-related
capabilities in the GrandNexus cognitive architecture. It manages various
learning paradigms through a flexible, emergence-oriented approach rather
than strict prescriptive rules.

The design philosophy emphasizes:
- Adaptivity: Learning strategies adapt based on context and performance
- Emergence: Complex learning behaviors emerge from simple interaction rules
- Integration: Seamless connection with other cognitive modules
- Flexibility: Support for multiple learning paradigms without rigidity
"""

import logging
import threading
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from enum import Enum, auto
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict, deque

# Import from GrandNexus core modules (assuming these exist)
from grandnexus.core.nexus_core import NexusCore
from grandnexus.memory.working import WorkingMemory
from grandnexus.memory.semantic import SemanticMemory
from grandnexus.memory.episodic import EpisodicMemory

# Learning paradigm types
class LearningParadigm(Enum):
    SUPERVISED = auto()
    UNSUPERVISED = auto()
    REINFORCEMENT = auto()
    META = auto()
    CONTINUOUS = auto()
    CURRICULUM = auto()
    INTRINSIC = auto()
    TRANSFER = auto()
    EMERGENT = auto()  # Special type for emergent learning patterns

# Learning event types
class LearningEventType(Enum):
    OBSERVATION = auto()    # New observation received
    FEEDBACK = auto()       # Feedback on a prediction/action
    REFLECTION = auto()     # Internal learning assessment
    ADAPTATION = auto()     # Learning parameter adjustment
    DISCOVERY = auto()      # New pattern or knowledge discovered
    INTEGRATION = auto()    # Knowledge integration across domains
    FORGETTING = auto()     # Strategic forgetting to avoid overfitting
    CURIOSITY = auto()      # Intrinsic interest in unexplored areas

@dataclass
class LearningEvent:
    """Represents a discrete learning event in the system"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: LearningEventType = LearningEventType.OBSERVATION
    timestamp: float = field(default_factory=time.time)
    paradigm: LearningParadigm = LearningParadigm.SUPERVISED
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_module: Optional[str] = None
    confidence: float = 1.0
    importance: float = 0.5

@dataclass
class LearningTask:
    """Represents a learning task to be processed"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    paradigm: LearningParadigm = LearningParadigm.SUPERVISED
    data: Any = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None
    priority: float = 0.5
    deadline: Optional[float] = None
    status: str = "pending"
    result: Any = None

@dataclass
class LearningStrategy:
    """Defines a learning strategy with its parameters and activation conditions"""
    strategy_id: str
    name: str
    paradigm: LearningParadigm
    parameters: Dict[str, Any] = field(default_factory=dict)
    activation_threshold: float = 0.5
    current_activation: float = 0.0
    last_used: float = 0.0
    success_rate: float = 0.5
    affinity_map: Dict[str, float] = field(default_factory=dict)  # Affinity to contexts/domains

    def is_applicable(self, task: LearningTask, context: Dict[str, Any]) -> float:
        """
        Determine applicability score of this strategy to the given task and context.
        Higher scores indicate better fit.

        This is a simplistic implementation - in a real system, this would be more sophisticated,
        potentially using meta-learning or contextual bandits to determine strategy selection.
        """
        # Base score is the activation level
        score = self.current_activation

        # Paradigm matching bonus
        if task.paradigm == self.paradigm:
            score += 0.3

        # Context affinity - match task elements with the strategy's affinities
        context_keys = set(context.keys())
        for key, affinity in self.affinity_map.items():
            if key in context_keys:
                score += affinity * context[key]

        # Success rate bonus
        score += self.success_rate * 0.2

        # Recency penalty (gradual decay of relevance if not used recently)
        time_since_used = time.time() - self.last_used if self.last_used > 0 else 0
        recency_penalty = min(0.5, time_since_used / 3600.0)  # Max 0.5 penalty after 1 hour
        score -= recency_penalty

        return max(0.0, min(1.0, score))  # Bound between 0 and 1

class AdaptiveLearningParameter:
    """
    Represents a learning parameter that can self-adjust based on feedback
    and contextual cues. This enables emergent adaptation without requiring
    manual parameter tuning.
    """
    def __init__(self, name: str, initial_value: float,
                 min_value: float, max_value: float,
                 adaptation_rate: float = 0.1):
        self.name = name
        self.value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.adaptation_rate = adaptation_rate
        self.history = deque(maxlen=100)  # Track recent values for analysis
        self.context_influence = {}  # How different contexts influence this parameter

    def adapt(self, feedback: float, context: Dict[str, float]) -> None:
        """
        Adapt parameter based on feedback and context.
        feedback: -1.0 to 1.0 (negative means decrease, positive means increase)
        context: dict mapping context factors to their strength (0.0 to 1.0)
        """
        # Base adaptation from direct feedback
        adaptation = feedback * self.adaptation_rate

        # Context-specific adaptations
        for ctx_name, ctx_strength in context.items():
            if ctx_name in self.context_influence:
                ctx_factor = self.context_influence[ctx_name]
                adaptation += ctx_factor * ctx_strength * self.adaptation_rate * 0.5

        # Apply adaptation with bounds
        new_value = self.value + adaptation
        self.value = max(self.min_value, min(self.max_value, new_value))
        self.history.append((time.time(), self.value, feedback, context))

    def learn_context_influence(self) -> None:
        """
        Analyze history to learn how different contexts influence this parameter.
        This is a simple implementation - more sophisticated approaches could be used.
        """
        if len(self.history) < 30:  # Need sufficient data
            return

        # Extract all unique context factors
        context_factors = set()
        for _, _, _, context in self.history:
            context_factors.update(context.keys())

        # Analyze correlation between each context factor and parameter changes
        for factor in context_factors:
            correlations = []
            for i in range(1, len(self.history)):
                prev_time, prev_value, _, prev_ctx = self.history[i-1]
                curr_time, curr_value, _, curr_ctx = self.history[i]

                if factor in curr_ctx:
                    factor_presence = curr_ctx[factor]
                    value_change = curr_value - prev_value
                    correlations.append((factor_presence, value_change))

            if correlations:
                # Calculate simple correlation coefficient
                factor_values = [c[0] for c in correlations]
                changes = [c[1] for c in correlations]

                if len(set(factor_values)) > 1 and len(set(changes)) > 1:
                    covariance = np.cov(factor_values, changes)[0, 1]
                    std_factor = np.std(factor_values)
                    std_changes = np.std(changes)

                    if std_factor > 0 and std_changes > 0:
                        correlation = covariance / (std_factor * std_changes)
                        self.context_influence[factor] = correlation

class LearningCore:
    """
    Central orchestrator for all learning functions in GrandNexus.

    This module coordinates between different learning paradigms and strategies,
    manages learning tasks, and facilitates the emergence of novel learning
    behaviors through interaction patterns.
    """

    def __init__(self, nexus_core: Optional[NexusCore] = None,
                 working_memory: Optional[WorkingMemory] = None,
                 semantic_memory: Optional[SemanticMemory] = None,
                 episodic_memory: Optional[EpisodicMemory] = None):
        """
        Initialize the learning core with references to other system modules.

        Args:
            nexus_core: Reference to the NexusCore for system-wide coordination
            working_memory: Working memory module for temporary storage
            semantic_memory: Semantic memory for structured knowledge
            episodic_memory: Episodic memory for experience-based learning
        """
        self.logger = logging.getLogger("GrandNexus.Learning.Core")
        self.nexus_core = nexus_core
        self.working_memory = working_memory
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory

        # Internal state
        self.instance_id = str(uuid.uuid4())[:8]
        self.running = False
        self.initialized = False
        self.lock = threading.RLock()

        # Learning strategies registry
        self.strategies: Dict[str, LearningStrategy] = {}

        # Event tracking
        self.event_history = deque(maxlen=1000)
        self.event_patterns = {}  # Discovered patterns in learning events

        # Task management
        self.tasks = deque()
        self.active_tasks = set()
        self.completed_tasks = deque(maxlen=100)

        # Adaptive parameters
        self.parameters = {}

        # Module connections and signal pathways
        self.signal_pathways = defaultdict(float)  # (source, target) -> strength

        # Emergence tracking
        self.emergent_patterns = []
        self.activation_landscape = defaultdict(float)  # Concept activation levels

        # Performance metrics
        self.metrics = defaultdict(lambda: deque(maxlen=100))

        self.logger.info(f"LearningCore initialized with ID={self.instance_id}")

    def initialize(self) -> bool:
        """Set up initial learning strategies and parameters"""
        if self.initialized:
            return True

        with self.lock:
            try:
                # Register basic learning strategies
                self._register_default_strategies()

                # Set up adaptive parameters
                self._initialize_adaptive_parameters()

                # Connect to other modules
                if self.nexus_core:
                    self._register_with_nexus_core()

                self.initialized = True
                self.logger.info("LearningCore initialization complete")
                return True
            except Exception as e:
                self.logger.error(f"LearningCore initialization failed: {str(e)}")
                return False

    def start(self) -> bool:
        """Start the learning core operations"""
        if not self.initialized:
            if not self.initialize():
                return False

        with self.lock:
            if self.running:
                return True

            try:
                # Start background processes
                self._start_background_processes()

                self.running = True
                self.logger.info("LearningCore started successfully")
                return True
            except Exception as e:
                self.logger.error(f"Failed to start LearningCore: {str(e)}")
                return False

    def stop(self) -> bool:
        """Stop the learning core operations"""
        with self.lock:
            if not self.running:
                return True

            try:
                # Stop background processes
                self._stop_background_processes()

                self.running = False
                self.logger.info("LearningCore stopped successfully")
                return True
            except Exception as e:
                self.logger.error(f"Error stopping LearningCore: {str(e)}")
                return False

    def register_strategy(self, strategy: LearningStrategy) -> bool:
        """
        Register a new learning strategy.

        Args:
            strategy: The LearningStrategy to register

        Returns:
            bool: True if registration succeeded, False otherwise
        """
        with self.lock:
            if strategy.strategy_id in self.strategies:
                self.logger.warning(f"Strategy {strategy.strategy_id} already exists, updating")

            self.strategies[strategy.strategy_id] = strategy
            self.logger.info(f"Registered learning strategy: {strategy.name} ({strategy.strategy_id})")
            return True

    def unregister_strategy(self, strategy_id: str) -> bool:
        """Remove a learning strategy from the registry"""
        with self.lock:
            if strategy_id not in self.strategies:
                self.logger.warning(f"Strategy {strategy_id} not found")
                return False

            del self.strategies[strategy_id]
            self.logger.info(f"Unregistered learning strategy: {strategy_id}")
            return True

    def submit_learning_task(self, task: LearningTask) -> str:
        """
        Submit a learning task for processing.

        Args:
            task: The LearningTask to process

        Returns:
            str: The task ID for tracking
        """
        with self.lock:
            self.tasks.append(task)
            self.logger.info(f"Learning task submitted: {task.task_id} - {task.description}")
            return task.task_id

    def process_learning_event(self, event: LearningEvent) -> None:
        """
        Process a learning event, updating internal state and triggering
        appropriate responses based on the event type.

        Args:
            event: The LearningEvent to process
        """
        with self.lock:
            # Record the event
            self.event_history.append(event)

            # Update activation landscape based on event
            self._update_activation_landscape(event)

            # Process different event types
            if event.event_type == LearningEventType.OBSERVATION:
                self._process_observation(event)
            elif event.event_type == LearningEventType.FEEDBACK:
                self._process_feedback(event)
            elif event.event_type == LearningEventType.REFLECTION:
                self._process_reflection(event)
            elif event.event_type == LearningEventType.DISCOVERY:
                self._process_discovery(event)
            elif event.event_type == LearningEventType.CURIOSITY:
                self._process_curiosity(event)

            # Check for emergent patterns in recent events
            self._detect_event_patterns()

    def get_task_status(self, task_id: str) -> Optional[LearningTask]:
        """Get the current status of a learning task"""
        with self.lock:
            # Check active tasks
            for task in self.active_tasks:
                if task.task_id == task_id:
                    return task

            # Check pending tasks
            for task in self.tasks:
                if task.task_id == task_id:
                    return task

            # Check completed tasks
            for task in self.completed_tasks:
                if task.task_id == task_id:
                    return task

            return None

    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get current learning performance metrics"""
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

    def adapt_to_context(self, context: Dict[str, Any]) -> None:
        """
        Adapt learning strategies and parameters based on the current context.

        Args:
            context: Dictionary of contextual factors
        """
        with self.lock:
            # Adjust strategy activations based on context
            self._adapt_strategies_to_context(context)

            # Adjust parameters based on context
            self._adapt_parameters_to_context(context)

            # Log adaptation
            self.logger.info(f"Adapted learning system to context: {context}")

    def get_strategy_activations(self) -> Dict[str, float]:
        """Get current activation levels of all learning strategies"""
        with self.lock:
            return {s_id: strategy.current_activation
                    for s_id, strategy in self.strategies.items()}

    def get_activation_landscape(self) -> Dict[str, float]:
        """Get the current concept activation landscape"""
        with self.lock:
            return dict(self.activation_landscape)

    def get_emergent_patterns(self) -> List[Dict[str, Any]]:
        """Get discovered emergent patterns in learning behavior"""
        with self.lock:
            return self.emergent_patterns.copy()

    def reinforce_pathway(self, source: str, target: str,
                           strength: float = 0.1) -> None:
        """
        Strengthen a signal pathway between source and target concepts.
        This is a core mechanism for emergent learning - strengthening
        pathways that are frequently activated together.

        Args:
            source: Source concept/node identifier
            target: Target concept/node identifier
            strength: Amount to strengthen the pathway (0.0 to 1.0)
        """
        with self.lock:
            key = (source, target)
            current = self.signal_pathways[key]
            # Sigmoid-like update to keep within bounds
            self.signal_pathways[key] = current + strength * (1.0 - current)

            # If connection becomes strong, log it as potentially emergent
            if current < 0.7 and self.signal_pathways[key] >= 0.7:
                self.logger.info(f"Emergent strong connection: {source} -> {target}")

                # Check if this connects previously separate subsystems
                self._check_for_systemic_emergence(source, target)

    def propagate_activation(self, initial_activations: Dict[str, float],
                             steps: int = 3) -> Dict[str, float]:
        """
        Propagate activation through the network of concepts and signal pathways.
        This is a key mechanism for associative learning and emergence.

        Args:
            initial_activations: Dictionary mapping concepts to initial activation
            steps: Number of propagation steps to run

        Returns:
            Dictionary of resulting concept activations
        """
        with self.lock:
            # Start with initial activations
            current = defaultdict(float)
            for concept, activation in initial_activations.items():
                current[concept] = activation

            # Propagate for specified number of steps
            for _ in range(steps):
                next_activations = defaultdict(float)

                # First, copy decay of current activations
                decay_factor = 0.8  # Signals decay by 20% per step
                for concept, activation in current.items():
                    next_activations[concept] += activation * decay_factor

                # Then propagate signals along pathways
                for (source, target), strength in self.signal_pathways.items():
                    if source in current and current[source] > 0.1:  # Threshold
                        propagated = current[source] * strength
                        next_activations[target] += propagated

                # Normalize activations (prevent runaway feedback loops)
                for concept in next_activations:
                    next_activations[concept] = min(1.0, next_activations[concept])

                current = next_activations

            return dict(current)

    # -----------------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------------

    def _register_default_strategies(self) -> None:
        """Register the default set of learning strategies"""
        # Supervised learning strategy
        self.register_strategy(LearningStrategy(
            strategy_id="supervised_basic",
            name="Basic Supervised Learning",
            paradigm=LearningParadigm.SUPERVISED,
            parameters={
                "learning_rate": 0.01,
                "batch_size": 32,
                "epochs": 10
            },
            activation_threshold=0.3,
            current_activation=0.5
        ))

        # Reinforcement learning strategy
        self.register_strategy(LearningStrategy(
            strategy_id="reinforcement_basic",
            name="Basic Reinforcement Learning",
            paradigm=LearningParadigm.REINFORCEMENT,
            parameters={
                "discount_factor": 0.99,
                "exploration_rate": 0.1,
                "learning_rate": 0.001
            },
            activation_threshold=0.3,
            current_activation=0.4
        ))

        # Continuous learning strategy
        self.register_strategy(LearningStrategy(
            strategy_id="continuous_ewc",
            name="Elastic Weight Consolidation",
            paradigm=LearningParadigm.CONTINUOUS,
            parameters={
                "ewc_lambda": 5000,
                "fisher_sample_size": 200
            },
            activation_threshold=0.3,
            current_activation=0.2,
            affinity_map={"memory_constraint": 0.8}
        ))

        # Meta-learning strategy
        self.register_strategy(LearningStrategy(
            strategy_id="meta_maml",
            name="Model-Agnostic Meta-Learning",
            paradigm=LearningParadigm.META,
            parameters={
                "inner_lr": 0.01,
                "outer_lr": 0.001,
                "inner_steps": 5
            },
            activation_threshold=0.4,
            current_activation=0.1,
            affinity_map={"new_task": 0.9, "few_shot": 0.8}
        ))

        # Intrinsic motivation strategy
        self.register_strategy(LearningStrategy(
            strategy_id="intrinsic_rnd",
            name="Random Network Distillation",
            paradigm=LearningParadigm.INTRINSIC,
            parameters={
                "reward_scale": 0.1,
                "predictor_lr": 0.001
            },
            activation_threshold=0.2,
            current_activation=0.3,
            affinity_map={"exploration": 0.9, "sparse_reward": 0.7}
        ))

    def _initialize_adaptive_parameters(self) -> None:
        """Initialize adaptive parameters for learning system"""
        # Learning rate parameter
        self.parameters["learning_rate"] = AdaptiveLearningParameter(
            name="learning_rate",
            initial_value=0.01,
            min_value=0.0001,
            max_value=0.1,
            adaptation_rate=0.05
        )

        # Exploration rate parameter
        self.parameters["exploration_rate"] = AdaptiveLearningParameter(
            name="exploration_rate",
            initial_value=0.2,
            min_value=0.01,
            max_value=0.5,
            adaptation_rate=0.1
        )

        # Memory retention parameter
        self.parameters["memory_retention"] = AdaptiveLearningParameter(
            name="memory_retention",
            initial_value=0.8,
            min_value=0.5,
            max_value=0.99,
            adaptation_rate=0.02
        )

        # Curiosity factor parameter
        self.parameters["curiosity_factor"] = AdaptiveLearningParameter(
            name="curiosity_factor",
            initial_value=0.3,
            min_value=0.1,
            max_value=0.9,
            adaptation_rate=0.1
        )

    def _register_with_nexus_core(self) -> None:
        """Register with NexusCore for system-wide coordination"""
        if not self.nexus_core:
            self.logger.warning("No NexusCore available for registration")
            return

        try:
            # Register as a module with NexusCore
            dependencies = []
            if self.working_memory:
                dependencies.append("working_memory")
            if self.semantic_memory:
                dependencies.append("semantic_memory")
            if self.episodic_memory:
                dependencies.append("episodic_memory")

            self.nexus_core.register_module(
                name="learning_core",
                module=self,
                dependencies=dependencies
            )

            # Set up message handlers
            # (implementation depends on NexusCore's messaging system)

            self.logger.info("Successfully registered with NexusCore")
        except Exception as e:
            self.logger.error(f"Failed to register with NexusCore: {str(e)}")

    def _start_background_processes(self) -> None:
        """Start background threads for ongoing learning processes"""
        # Implementation would depend on specific background processes needed
        pass

    def _stop_background_processes(self) -> None:
        """Stop background learning processes"""
        # Implementation would depend on specific background processes
        pass

    def _process_observation(self, event: LearningEvent) -> None:
        """Process an observation learning event"""
        # Implementation would depend on how observations are handled
        pass

    def _process_feedback(self, event: LearningEvent) -> None:
        """Process feedback on a learning outcome"""
        # Extract feedback information
        task_id = event.metadata.get("task_id")
        if not task_id:
            self.logger.warning("Feedback event missing task_id")
            return

        success = event.metadata.get("success", False)
        score = event.metadata.get("score", 0.0)

        # Update metrics
        self.metrics["feedback_scores"].append(score)

        # Update strategy success rates
        if "strategy_id" in event.metadata:
            strategy_id = event.metadata["strategy_id"]
            if strategy_id in self.strategies:
                strategy = self.strategies[strategy_id]
                # Weighted update of success rate
                current = strategy.success_rate
                strategy.success_rate = 0.9 * current + 0.1 * (1.0 if success else 0.0)
                strategy.last_used = time.time()

                # If strategy performed very well or very poorly, adjust activation
                if score > 0.8:
                    strategy.current_activation = min(1.0, strategy.current_activation + 0.05)
                elif score < 0.2:
                    strategy.current_activation = max(0.1, strategy.current_activation - 0.05)

    def _process_reflection(self, event: LearningEvent) -> None:
        """Process a reflection on internal learning processes"""
        # Implementation would depend on how reflections are used
        pass

    def _process_discovery(self, event: LearningEvent) -> None:
        """Process a knowledge discovery event"""
        if "concept" in event.metadata:
            concept = event.metadata["concept"]
            # Boost activation for the discovered concept
            self.activation_landscape[concept] += 0.5

            # If related concepts are provided, create or strengthen pathways
            if "related_concepts" in event.metadata:
                for related in event.metadata["related_concepts"]:
                    self.reinforce_pathway(concept, related, 0.3)
                    self.reinforce_pathway(related, concept, 0.1)  # Asymmetric reinforcement

    def _process_curiosity(self, event: LearningEvent) -> None:
        """Process a curiosity-driven learning event"""
        # Implementation would depend on curiosity mechanisms
        pass

    def _update_activation_landscape(self, event: LearningEvent) -> None:
        """Update the activation landscape based on learning events"""
        # Extract concepts from event
        concepts = []

        # Check event content for concepts
        if isinstance(event.content, str):
            # Simple example - in real implementation, would use NLP or concept extraction
            words = event.content.split()
            concepts.extend(words)

        # Check metadata for explicit concepts
        if "concepts" in event.metadata:
            concepts.extend(event.metadata["concepts"])

        # Update activation for each concept
        for concept in concepts:
            # Increase activation (with saturation)
            current = self.activation_landscape[concept]
            # Sigmoid-like update to prevent runaway activation
            self.activation_landscape[concept] = current + event.importance * (1.0 - current)

        # Decay activations for all concepts
        for concept in list(self.activation_landscape.keys()):
            self.activation_landscape[concept] *= 0.95  # 5% decay per event

            # Remove concepts with negligible activation
            if self.activation_landscape[concept] < 0.01:
                del self.activation_landscape[concept]

    def _detect_event_patterns(self) -> None:
        """
        Detect patterns in recent learning events that might indicate
        emergent learning behaviors or opportunities.
        """
        # We need enough events to detect patterns
        if len(self.event_history) < 10:
            return

        # Look for repeated sequences of event types
        recent_types = [e.event_type for e in self.event_history][-10:]

        # Check for alternating patterns
        alternating = True
        for i in range(2, len(recent_types), 2):
            if recent_types[i] != recent_types[i-2]:
                alternating = False
                break

        if alternating and len(recent_types) >= 6:
            pattern_key = f"{recent_types[0]}-{recent_types[1]}"
            if pattern_key not in self.event_patterns:
                self.event_patterns[pattern_key] = {
                    'count': 1,
                    'first_seen': time.time()
                }
                self.logger.info(f"Detected new event pattern: Alternating {pattern_key}")
            else:
                self.event_patterns[pattern_key]['count'] += 1

        # Check for frequency of specific event types
        type_counts = {}
        for e_type in recent_types:
            type_counts[e_type] = type_counts.get(e_type, 0) + 1

        # Look for dominance of specific event types
        for e_type, count in type_counts.items():
            if count >= 7:  # 70% or more
                pattern_key = f"dominant-{e_type}"
                if pattern_key not in self.event_patterns:
                    self.event_patterns[pattern_key] = {
                        'count': 1,
                        'first_seen': time.time()
                    }
                    self.logger.info(f"Detected new event pattern: Dominant {e_type}")
                else:
                    self.event_patterns[pattern_key]['count'] += 1

    def _adapt_strategies_to_context(self, context: Dict[str, Any]) -> None:
        """
        Adapt strategy activation levels based on the current context.
        This allows different learning approaches to emerge as dominant
        depending on the situation.

        Args:
            context: Dictionary of contextual factors
        """
        # Convert context to normalized form for strategy evaluation
        normalized_context = {}
        for key, value in context.items():
            if isinstance(value, bool):
                normalized_context[key] = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                # Scale to 0-1 range (assuming most values are in 0-100 range)
                normalized_context[key] = min(1.0, max(0.0, value / 100.0))
            else:
                # For strings or other types, just indicate presence
                normalized_context[key] = 1.0

        # Calculate applicability scores for each strategy
        scores = {}
        for strategy_id, strategy in self.strategies.items():
            scores[strategy_id] = strategy.is_applicable(
                LearningTask(paradigm=strategy.paradigm),  # Dummy task matching the strategy
                normalized_context
            )

        # Adjust activations based on scores
        # Using a softmax-like normalization to create competition
        if scores:
            max_score = max(scores.values())
            exp_scores = {k: np.exp(v - max_score) for k, v in scores.items()}
            sum_exp = sum(exp_scores.values())

            for strategy_id, exp_score in exp_scores.items():
                # Calculate target activation (softmax-normalized)
                target = exp_score / sum_exp

                # Gradually move current activation toward target
                current = self.strategies[strategy_id].current_activation
                # Move 10% of the way toward target
                new_activation = current + 0.1 * (target - current)

                # Update strategy activation
                self.strategies[strategy_id].current_activation = new_activation

                # Log significant changes
                if abs(new_activation - current) > 0.1:
                    self.logger.info(f"Strategy {strategy_id} activation changed: {current:.2f} -> {new_activation:.2f}")

    def _adapt_parameters_to_context(self, context: Dict[str, Any]) -> None:
        """
        Adapt learning parameters based on the current context.

        Args:
            context: Dictionary of contextual factors
        """
        # Normalize context for parameter adaptation
        normalized_context = {}
        for key, value in context.items():
            if isinstance(value, bool):
                normalized_context[key] = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                normalized_context[key] = min(1.0, max(0.0, value / 100.0))
            else:
                normalized_context[key] = 1.0

        # Process specific contextual factors that should influence parameters

        # If high time pressure, decrease exploration
        if "time_pressure" in normalized_context and normalized_context["time_pressure"] > 0.7:
            self.parameters["exploration_rate"].adapt(-0.5, normalized_context)

        # If new environment, increase exploration
        if "new_environment" in normalized_context and normalized_context["new_environment"] > 0.5:
            self.parameters["exploration_rate"].adapt(0.7, normalized_context)
            self.parameters["curiosity_factor"].adapt(0.5, normalized_context)

        # If performance is recently poor, adjust learning rate
        if "recent_performance" in normalized_context:
            perf = normalized_context["recent_performance"]
            if perf < 0.3:  # Poor performance
                # Try a bigger learning rate change
                self.parameters["learning_rate"].adapt(0.4, normalized_context)
            elif perf > 0.8:  # Great performance
                # Slightly reduce learning rate to stabilize
                self.parameters["learning_rate"].adapt(-0.1, normalized_context)

        # If this is critical task, increase memory retention
        if "task_importance" in normalized_context and normalized_context["task_importance"] > 0.8:
            self.parameters["memory_retention"].adapt(0.3, normalized_context)

        # Periodically analyze parameter history to learn context influences
        if random.random() < 0.05:  # 5% chance per adaptation
            for param in self.parameters.values():
                param.learn_context_influence()

    def _check_for_systemic_emergence(self, source: str, target: str) -> None:
        """
        Check if a new strong connection between source and target
        represents an emergent bridge between previously separate subsystems.

        Args:
            source: Source concept/node
            target: Target concept/node
        """
        # Build a graph of strong connections
        strong_connections = defaultdict(set)
        for (s, t), strength in self.signal_pathways.items():
            if strength >= 0.5:  # Consider connections with at least 0.5 strength
                strong_connections[s].add(t)

        # Check if source and target were previously in separate connected components
        # by performing a flood fill from source before this new connection

        # Temporarily remove the direct connection
        if target in strong_connections[source]:
            strong_connections[source].remove(target)

        # Perform a breadth-first search from source
        visited = set([source])
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for neighbor in strong_connections[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # If target wasn't reachable before, this is a bridge between subsystems
        if target not in visited:
            # This is an emergent bridge!
            self.logger.info(f"Emergent bridge detected between subsystems via {source} -> {target}")

            # Record this as an emergent pattern
            self.emergent_patterns.append({
                'type': 'bridge',
                'source': source,
                'target': target,
                'time': time.time(),
                'source_subsystem': list(visited),
                'target_subsystem': [n for n in strong_connections if n not in visited]
            })

    def _select_learning_strategy(self, task: LearningTask, context: Dict[str, Any]) -> Optional[str]:
        """
        Select the most appropriate learning strategy for a given task and context.

        Args:
            task: The learning task to process
            context: Current contextual factors

        Returns:
            strategy_id of selected strategy, or None if no suitable strategy found
        """
        candidates = []

        # Evaluate each strategy's applicability to the task
        for strategy_id, strategy in self.strategies.items():
            if strategy.current_activation >= strategy.activation_threshold:
                score = strategy.is_applicable(task, context)
                if score > 0:
                    candidates.append((strategy_id, score))

        if not candidates:
            return None

        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Introduce some exploration (don't always pick the highest)
        if random.random() < self.parameters["exploration_rate"].value:
            # Pick randomly from top half of candidates
            top_half = candidates[:max(1, len(candidates)//2)]
            return random.choice(top_half)[0]
        else:
            # Pick the highest scoring strategy
            return candidates[0][0]

    def _execute_learning_task(self, task: LearningTask) -> None:
        """
        Execute a learning task using the appropriate strategy.

        Args:
            task: The learning task to execute
        """
        # Prepare execution context
        context = {}

        # Extract task-relevant contextual factors
        if self.working_memory:
            # Get current working memory state
            try:
                working_mem_items = self.working_memory.get_all_items()
                context["working_memory_size"] = len(working_mem_items)

                # Extract relevant items for the task
                relevant_items = []
                if task.description:
                    # Find items related to the task description
                    for item in working_mem_items:
                        # Simplified relevance check - would be more sophisticated in real system
                        if any(word in item.content for word in task.description.split()):
                            relevant_items.append(item)

                context["relevant_items"] = len(relevant_items)
            except Exception as e:
                self.logger.warning(f"Error accessing working memory: {str(e)}")

        # Determine task complexity
        if isinstance(task.data, dict) and "complexity" in task.data:
            context["task_complexity"] = task.data["complexity"]
        else:
            # Estimate complexity based on data size
            try:
                data_size = len(task.data) if hasattr(task.data, "__len__") else 1
                context["task_complexity"] = min(1.0, data_size / 1000)  # Normalize to 0-1
            except (TypeError, AttributeError):
                context["task_complexity"] = 0.5  # Default medium complexity

        # Select strategy
        strategy_id = self._select_learning_strategy(task, context)

        if not strategy_id:
            self.logger.warning(f"No suitable strategy found for task {task.task_id}")
            task.status = "failed"
            if task.callback:
                task.callback(task)
            return

        strategy = self.strategies[strategy_id]
        self.logger.info(f"Selected strategy '{strategy.name}' for task {task.task_id}")

        # Mark task as in progress
        task.status = "running"

        # Execute the appropriate learning algorithm based on strategy
        try:
            if strategy.paradigm == LearningParadigm.SUPERVISED:
                result = self._execute_supervised_learning(task, strategy, context)
            elif strategy.paradigm == LearningParadigm.REINFORCEMENT:
                result = self._execute_reinforcement_learning(task, strategy, context)
            elif strategy.paradigm == LearningParadigm.CONTINUOUS:
                result = self._execute_continuous_learning(task, strategy, context)
            elif strategy.paradigm == LearningParadigm.META:
                result = self._execute_meta_learning(task, strategy, context)
            elif strategy.paradigm == LearningParadigm.INTRINSIC:
                result = self._execute_intrinsic_learning(task, strategy, context)
            else:
                raise ValueError(f"Unsupported learning paradigm: {strategy.paradigm}")

            # Update task with result
            task.result = result
            task.status = "completed"

            # Record success metrics
            self.metrics["task_success_rate"].append(1.0)

            # Update strategy usage statistics
            strategy.last_used = time.time()

            # Log completion
            self.logger.info(f"Task {task.task_id} completed successfully")

        except Exception as e:
            self.logger.error(f"Error executing task {task.task_id}: {str(e)}")
            task.status = "failed"
            task.result = {"error": str(e)}

            # Record failure metrics
            self.metrics["task_success_rate"].append(0.0)

        # Move task to completed
        if task in self.active_tasks:
            self.active_tasks.remove(task)
        self.completed_tasks.append(task)

        # Execute callback if provided
        if task.callback:
            try:
                task.callback(task)
            except Exception as e:
                self.logger.error(f"Error in task callback: {str(e)}")

    def _execute_supervised_learning(self, task: LearningTask,
                                     strategy: LearningStrategy,
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a supervised learning task.

        Args:
            task: The learning task with data
            strategy: The selected learning strategy
            context: Contextual factors

        Returns:
            Dictionary with learning results
        """
        # This is a placeholder implementation
        # In a real system, this would be much more sophisticated

        self.logger.info(f"Executing supervised learning for task {task.task_id}")

        # Extract parameters
        learning_rate = task.parameters.get("learning_rate",
                                           strategy.parameters.get("learning_rate",
                                                                 self.parameters["learning_rate"].value))

        batch_size = task.parameters.get("batch_size",
                                        strategy.parameters.get("batch_size", 32))

        epochs = task.parameters.get("epochs",
                                    strategy.parameters.get("epochs", 10))

        # Extract data
        if not task.data or not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary for supervised learning")

        X_train = task.data.get("X_train")
        y_train = task.data.get("y_train")
        X_val = task.data.get("X_val")
        y_val = task.data.get("y_val")

        if X_train is None or y_train is None:
            raise ValueError("Missing training data (X_train, y_train)")

        # In a real implementation, this would create and train a model
        # Here we'll just simulate the training process

        # Simulated training
        model = {"type": "supervised_model"}
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Simulate training epoch
            train_loss = 1.0 / (epoch + 1)  # Simulated loss that decreases over time
            train_losses.append(train_loss)

            # Simulate validation if validation data provided
            if X_val is not None and y_val is not None:
                val_loss = train_loss * 1.2  # Slightly higher than training loss
                val_losses.append(val_loss)

        # Prepare results
        result = {
            "model": model,
            "train_losses": train_losses,
            "epochs_completed": epochs
        }

        if val_losses:
            result["val_losses"] = val_losses
            result["best_val_loss"] = min(val_losses)

        return result

    def _execute_reinforcement_learning(self, task: LearningTask,
                                       strategy: LearningStrategy,
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a reinforcement learning task.

        Args:
            task: The learning task with environment data
            strategy: The selected learning strategy
            context: Contextual factors

        Returns:
            Dictionary with learning results
        """
        # Placeholder implementation
        self.logger.info(f"Executing reinforcement learning for task {task.task_id}")

        # Extract parameters
        learning_rate = task.parameters.get("learning_rate",
                                          strategy.parameters.get("learning_rate",
                                                                self.parameters["learning_rate"].value))

        discount_factor = task.parameters.get("discount_factor",
                                            strategy.parameters.get("discount_factor", 0.99))

        exploration_rate = task.parameters.get("exploration_rate",
                                             strategy.parameters.get("exploration_rate",
                                                                   self.parameters["exploration_rate"].value))

        # Extract environment data
        if not task.data or not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary for reinforcement learning")

        env = task.data.get("environment")
        episodes = task.data.get("episodes", 100)

        if env is None:
            raise ValueError("Missing environment specification")

        # In a real implementation, this would interact with the environment
        # Here we'll just simulate the learning process

        # Simulated training
        policy = {"type": "reinforcement_policy"}
        rewards = []

        for episode in range(episodes):
            # Simulate an episode
            episode_reward = (1 - np.exp(-episode/20)) * 100  # Reward improves over time
            rewards.append(episode_reward)

        # Prepare results
        result = {
            "policy": policy,
            "episode_rewards": rewards,
            "episodes_completed": episodes,
            "final_exploration_rate": exploration_rate * (0.1 ** (episodes / 100))
        }

        return result

    def _execute_continuous_learning(self, task: LearningTask,
                                    strategy: LearningStrategy,
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a continuous learning task.

        Args:
            task: The learning task with stream data
            strategy: The selected learning strategy
            context: Contextual factors

        Returns:
            Dictionary with learning results
        """
        # Placeholder implementation
        self.logger.info(f"Executing continuous learning for task {task.task_id}")

        # Extract parameters based on strategy type
        if strategy.strategy_id == "continuous_ewc":
            ewc_lambda = task.parameters.get("ewc_lambda",
                                           strategy.parameters.get("ewc_lambda", 5000))

            fisher_sample_size = task.parameters.get("fisher_sample_size",
                                                    strategy.parameters.get("fisher_sample_size", 200))

        # Extract data
        if not task.data or not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary for continuous learning")

        previous_model = task.data.get("previous_model")
        new_data = task.data.get("new_data")
        task_id = task.data.get("task_id")

        if previous_model is None:
            raise ValueError("Missing previous model")
        if new_data is None:
            raise ValueError("Missing new data")

        # In a real implementation, this would update the model with EWC or other CL method
        # Here we'll just simulate the process

        # Simulate model update
        updated_model = {"type": "updated_model", "base": previous_model, "task_id": task_id}
        performance = {}

        # Simulate performance on current and previous tasks
        performance["current_task"] = 0.85  # Good performance on current task
        performance["previous_tasks"] = 0.75  # Slight forgetting on previous tasks

        # Prepare results
        result = {
            "model": updated_model,
            "performance": performance,
            "forgetting_measure": 1 - performance["previous_tasks"]
        }

        return result

    def _execute_meta_learning(self, task: LearningTask,
                              strategy: LearningStrategy,
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a meta-learning task.

        Args:
            task: The learning task with meta-tasks
            strategy: The selected learning strategy
            context: Contextual factors

        Returns:
            Dictionary with learning results
        """
        # Placeholder implementation
        self.logger.info(f"Executing meta-learning for task {task.task_id}")

        # Extract parameters based on strategy
        if strategy.strategy_id == "meta_maml":
            inner_lr = task.parameters.get("inner_lr",
                                         strategy.parameters.get("inner_lr", 0.01))

            outer_lr = task.parameters.get("outer_lr",
                                         strategy.parameters.get("outer_lr", 0.001))

            inner_steps = task.parameters.get("inner_steps",
                                            strategy.parameters.get("inner_steps", 5))

        # Extract data
        if not task.data or not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary for meta-learning")

        meta_train_tasks = task.data.get("meta_train_tasks")
        meta_val_tasks = task.data.get("meta_val_tasks")
        meta_test_tasks = task.data.get("meta_test_tasks")

        if meta_train_tasks is None:
            raise ValueError("Missing meta-training tasks")

        # In a real implementation, this would perform MAML training
        # Here we'll just simulate the process

        # Simulate meta-training
        meta_model = {"type": "meta_model"}
        meta_train_losses = []
        meta_val_losses = []

        for epoch in range(10):  # Simulate 10 meta-epochs
            # Simulate meta-training epoch
            train_loss = 1.0 / (epoch + 1)  # Simulated loss that decreases over time
            meta_train_losses.append(train_loss)

            # Simulate meta-validation if available
            if meta_val_tasks:
                val_loss = train_loss * 1.1  # Slightly higher than training loss
                meta_val_losses.append(val_loss)

        # Simulate meta-testing if available
        meta_test_results = None
        if meta_test_tasks:
            meta_test_results = []
            for i, test_task in enumerate(meta_test_tasks):
                # Simulate adaptation to test task
                task_result = {
                    "task_id": f"test_task_{i}",
                    "pre_adaptation_loss": 0.8,
                    "post_adaptation_loss": 0.3,
                    "adaptation_steps": inner_steps
                }
                meta_test_results.append(task_result)

        # Prepare results
        result = {
            "meta_model": meta_model,
            "meta_train_losses": meta_train_losses
        }

        if meta_val_losses:
            result["meta_val_losses"] = meta_val_losses

        if meta_test_results:
            result["meta_test_results"] = meta_test_results
            result["average_adaptation_improvement"] = 0.5  # Simulated improvement after adaptation

        return result

    def _execute_intrinsic_learning(self, task: LearningTask,
                                   strategy: LearningStrategy,
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an intrinsic motivation-based learning task.

        Args:
            task: The learning task with environment data
            strategy: The selected learning strategy
            context: Contextual factors

        Returns:
            Dictionary with learning results
        """
        # Placeholder implementation
        self.logger.info(f"Executing intrinsic motivation learning for task {task.task_id}")

        # Extract parameters based on strategy
        if strategy.strategy_id == "intrinsic_rnd":
            reward_scale = task.parameters.get("reward_scale",
                                             strategy.parameters.get("reward_scale", 0.1))

            predictor_lr = task.parameters.get("predictor_lr",
                                             strategy.parameters.get("predictor_lr", 0.001))

        # Extract data
        if not task.data or not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary for intrinsic learning")

        env = task.data.get("environment")
        episodes = task.data.get("episodes", 100)

        if env is None:
            raise ValueError("Missing environment specification")

        # In a real implementation, this would set up RND networks and train
        # Here we'll just simulate the process

        # Simulate training
        policy = {"type": "intrinsic_policy"}
        extrinsic_rewards = []
        intrinsic_rewards = []
        prediction_errors = []

        for episode in range(episodes):
            # Simulate an episode
            extrinsic_reward = (1 - np.exp(-episode/30)) * 100  # External reward improves over time

            # Intrinsic reward starts high and decreases as environment becomes familiar
            intrinsic_reward = 100 * np.exp(-episode/20)

            # Prediction error follows the same pattern as intrinsic reward
            prediction_error = 1.0 * np.exp(-episode/20)

            extrinsic_rewards.append(extrinsic_reward)
            intrinsic_rewards.append(intrinsic_reward)
            prediction_errors.append(prediction_error)

        # Prepare results
        result = {
            "policy": policy,
            "extrinsic_rewards": extrinsic_rewards,
            "intrinsic_rewards": intrinsic_rewards,
            "prediction_errors": prediction_errors,
            "episodes_completed": episodes,
            "exploration_coverage": 0.85  # Simulated coverage of state space
        }

        return result