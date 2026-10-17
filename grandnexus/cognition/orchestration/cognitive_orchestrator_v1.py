# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 54751-56536
"""Module experimentale à proposer à Claude après avoir mis en squelette tout le code de GrandNexus"""

"""
╔══════════════════════════════════════════════════════════════════════════╗
║  Module Name   : cognitive_orchestrator.py                              ║
║  Purpose       : Metacognitive Orchestration Layer for GrandNexus       ║
║  Description   : Coordinates cognitive processes (reasoning, learning,   ║
║                  perception) by regulating their activation, prioritizing║
║                  focus, and optimizing resource allocation based on      ║
║                  contextual needs and internal state.                    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import logging
import threading
import time
import uuid
import math
import random
import heapq
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque

from grandnexus.core.nexus_core import NexusCore
from grandnexus.core.errorRecoveryManager import ErrorRecoveryManager
from grandnexus.memory.working import WorkingMemory
from grandnexus.memory.episodic import EpisodicMemory
from grandnexus.memory.semantic import SemanticMemory
from grandnexus.cognition.reasoning import ReasoningEngine
from grandnexus.cognition.learning import LearningManager
from grandnexus.cognition.perception import PerceptionModule
from grandnexus.graph.graph_engine import AbstractGraphEngine
from grandnexus.computation.resource_manager import ResourceManager


class CognitiveState(Enum):
    """States representing different cognitive operational modes."""
    EXPLORATION = auto()      # Focus on curiosity and knowledge acquisition
    EXPLOITATION = auto()     # Focus on applying known solutions efficiently
    DELIBERATION = auto()     # Deep reasoning and causal analysis
    INTUITION = auto()        # Rapid pattern-matching and heuristic responses
    INTEGRATION = auto()      # Consolidation and synthesis of knowledge
    ADAPTATION = auto()       # Adjusting to new information or environments
    META_REFLECTION = auto()  # Self-analysis and process optimization
    PERCEPTION = auto()       # réparation 6.5 : cible de la transition « new_input »


@dataclass
class CognitiveContext:
    """Current cognitive operational context."""
    state: CognitiveState = CognitiveState.EXPLOITATION
    task_complexity: float = 0.5  # 0.0 to 1.0
    time_pressure: float = 0.3    # 0.0 to 1.0
    confidence: float = 0.7       # 0.0 to 1.0
    creativity_need: float = 0.4  # 0.0 to 1.0
    novelty: float = 0.5         # How novel is current input (0.0 to 1.0)
    focus_target: Optional[str] = None  # What cognitive aspect to prioritize
    state_duration: int = 0  # How many cycles in current state
    timestamp: float = field(default_factory=time.time)


@dataclass
class CognitiveAction:
    """A planned cognitive action with priority and target modules."""
    action_id: str
    action_type: str  # "reason", "learn", "perceive", "memorize", "retrieve", etc.
    priority: float
    target_modules: List[str]
    parameters: Dict[str, Any]
    estimated_resources: Dict[str, float]  # cpu, memory, time
    dependencies: Set[str] = field(default_factory=set)  # IDs of prerequisite actions
    timestamp: float = field(default_factory=time.time)
    state: str = "pending"  # pending, executing, completed, failed

    def __lt__(self, other):
        # For priority queue comparison
        return self.priority > other.priority  # Higher priority first


@dataclass
class CognitiveEpisode:
    goal: str
    trigger_context: CognitiveContext
    actions: List[CognitiveAction]
    outcome: str
    timestamp: float = field(default_factory=time.time)


class CognitiveOrchestrator:
    """
    Orchestrates the interplay between cognitive modules (reasoning, learning, perception)
    based on contextual needs, internal state, and resource constraints.

    Key responsibilities:
    1. Determine cognitive state based on task and context
    2. Regulate activation of modules and their interactions
    3. Prioritize cognitive operations based on context
    4. Monitor cognitive performance and adjust allocation
    5. Trigger metacognitive reflection when needed
    6. Facilitate synergistic operation of modules
    """

    def __init__(self, nexus_core: NexusCore, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cognitive orchestrator.

        Args:
            nexus_core: Reference to the NexusCore for module access
            config: Configuration dictionary
        """
        self.logger = logging.getLogger("CognitiveOrchestrator")
        self.nexus_core = nexus_core
        self.config = config or {}

        # Internal state
        self.context = CognitiveContext()
        self.meta_trace_log: List[Dict[str, Any]] = []
        self.action_queue: List[CognitiveAction] = []  # Priority queue of actions
        self.active_actions: Dict[str, CognitiveAction] = {}  # Currently executing actions
        self.completed_actions: Dict[str, CognitiveAction] = {}  # Recently completed actions
        self.state_transition_matrix = self._initialize_state_transitions()
        self.module_performance_history: Dict[str, List[Tuple[float, float]]] = defaultdict(list)  # (timestamp, score)
        self.resource_usage_history: List[Dict[str, float]] = []

        # Cognitive state metrics
        self.cognitive_coherence: float = 0.7  # How well different cognitive processes are aligned
        self.adaptation_rate: float = 0.5     # How quickly system adapts to new information
        self.metacognitive_depth: float = 0.3  # Depth of self-analysis

        # Performance tracking
        self.success_rate: Dict[str, float] = defaultdict(lambda: 0.8)  # Start with assumption of success
        self.response_times: Dict[str, List[float]] = defaultdict(list)

        # Module references (to be populated on start)
        self.reasoning_engine = None
        self.learning_manager = None
        self.perception_module = None
        self.working_memory = None
        self.episodic_memory = None
        self.semantic_memory = None
        self.graph_engine = None
        self.resource_manager = None

        # Background threads
        self._running = False
        self._monitoring_thread = None
        self._execution_thread = None

        self.logger.info("CognitiveOrchestrator initialized")

    def start(self) -> bool:
        """Start the orchestrator and connect to required modules."""
        try:
            # Get references to required modules
            self.reasoning_engine = self.nexus_core.get_module('reasoning_engine')
            self.learning_manager = self.nexus_core.get_module('learning_manager')
            self.perception_module = self.nexus_core.get_module('perception_module')
            self.working_memory = self.nexus_core.get_module('working_memory')
            self.episodic_memory = self.nexus_core.get_module('episodic_memory')
            self.semantic_memory = self.nexus_core.get_module('semantic_memory')
            self.graph_engine = self.nexus_core.get_module('graph_engine')
            self.resource_manager = self.nexus_core.get_module('resource_manager')

            # Check if all required modules are available
            if not all([
                self.reasoning_engine, self.learning_manager, self.perception_module,
                self.working_memory, self.episodic_memory, self.semantic_memory,
                self.graph_engine, self.resource_manager
            ]):
                missing_modules = []
                if not self.reasoning_engine: missing_modules.append('reasoning_engine')
                if not self.learning_manager: missing_modules.append('learning_manager')
                if not self.perception_module: missing_modules.append('perception_module')
                if not self.working_memory: missing_modules.append('working_memory')
                if not self.episodic_memory: missing_modules.append('episodic_memory')
                if not self.semantic_memory: missing_modules.append('semantic_memory')
                if not self.graph_engine: missing_modules.append('graph_engine')
                if not self.resource_manager: missing_modules.append('resource_manager')

                self.logger.error(f"Missing required modules: {', '.join(missing_modules)}")
                return False

            # Start background threads
            self._running = True
            self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
            self._monitoring_thread.start()
            self._execution_thread.start()

            self.logger.info("CognitiveOrchestrator started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error starting CognitiveOrchestrator: {e}")
            return False

    def stop(self) -> bool:
        """Stop the orchestrator gracefully."""
        try:
            self._running = False
            # Wait for threads to finish with timeout
            if self._monitoring_thread:
                self._monitoring_thread.join(timeout=5.0)
            if self._execution_thread:
                self._execution_thread.join(timeout=5.0)

            self.logger.info("CognitiveOrchestrator stopped")
            return True
        except Exception as e:
            self.logger.error(f"Error stopping CognitiveOrchestrator: {e}")
            return False

    def _initialize_state_transitions(self) -> Dict[CognitiveState, Dict[str, List[Tuple[CognitiveState, float]]]]:
        """
        Initialize the transition matrix between cognitive states.
        For each current state, defines possible transitions based on trigger conditions.
        Returns a nested dictionary mapping current states to trigger conditions
        and then to target states with transition probabilities.
        """
        transitions = {}

        # From EXPLORATION state
        transitions[CognitiveState.EXPLORATION] = {
            "high_time_pressure": [(CognitiveState.INTUITION, 0.7), (CognitiveState.EXPLOITATION, 0.3)],
            "solution_found": [(CognitiveState.EXPLOITATION, 0.8), (CognitiveState.INTEGRATION, 0.2)],
            "high_complexity": [(CognitiveState.DELIBERATION, 0.6), (CognitiveState.META_REFLECTION, 0.4)],
            "low_novelty": [(CognitiveState.EXPLOITATION, 0.9), (CognitiveState.INTEGRATION, 0.1)],
            "default": [(CognitiveState.EXPLORATION, 0.7), (CognitiveState.DELIBERATION, 0.3)]
        }

        # From EXPLOITATION state
        transitions[CognitiveState.EXPLOITATION] = {
            "failure_detected": [(CognitiveState.ADAPTATION, 0.6), (CognitiveState.EXPLORATION, 0.4)],
            "high_novelty": [(CognitiveState.EXPLORATION, 0.8), (CognitiveState.ADAPTATION, 0.2)],
            "high_complexity": [(CognitiveState.DELIBERATION, 0.7), (CognitiveState.INTEGRATION, 0.3)],
            "low_confidence": [(CognitiveState.DELIBERATION, 0.5), (CognitiveState.META_REFLECTION, 0.5)],
            "default": [(CognitiveState.EXPLOITATION, 0.8), (CognitiveState.INTUITION, 0.2)]
        }

        # From DELIBERATION state
        transitions[CognitiveState.DELIBERATION] = {
            "high_time_pressure": [(CognitiveState.INTUITION, 0.8), (CognitiveState.EXPLOITATION, 0.2)],
            "solution_refined": [(CognitiveState.INTEGRATION, 0.7), (CognitiveState.EXPLOITATION, 0.3)],
            "no_progress": [(CognitiveState.EXPLORATION, 0.5), (CognitiveState.META_REFLECTION, 0.5)],
            "default": [(CognitiveState.DELIBERATION, 0.7), (CognitiveState.INTEGRATION, 0.3)]
        }

        # From INTUITION state
        transitions[CognitiveState.INTUITION] = {
            "more_time_available": [(CognitiveState.DELIBERATION, 0.6), (CognitiveState.EXPLOITATION, 0.4)],
            "high_uncertainty": [(CognitiveState.DELIBERATION, 0.7), (CognitiveState.EXPLORATION, 0.3)],
            "solution_needed": [(CognitiveState.EXPLOITATION, 0.9), (CognitiveState.INTEGRATION, 0.1)],
            "default": [(CognitiveState.INTUITION, 0.6), (CognitiveState.EXPLOITATION, 0.4)]
        }

        # From INTEGRATION state
        transitions[CognitiveState.INTEGRATION] = {
            "integration_complete": [(CognitiveState.EXPLOITATION, 0.7), (CognitiveState.META_REFLECTION, 0.3)],
            "inconsistency_detected": [(CognitiveState.DELIBERATION, 0.6), (CognitiveState.EXPLORATION, 0.4)],
            "new_input": [(CognitiveState.PERCEPTION, 0.8), (CognitiveState.ADAPTATION, 0.2)],
            "default": [(CognitiveState.INTEGRATION, 0.7), (CognitiveState.EXPLOITATION, 0.3)]
        }

        # From ADAPTATION state
        transitions[CognitiveState.ADAPTATION] = {
            "adaptation_complete": [(CognitiveState.EXPLOITATION, 0.6), (CognitiveState.INTEGRATION, 0.4)],
            "continuing_change": [(CognitiveState.EXPLORATION, 0.7), (CognitiveState.ADAPTATION, 0.3)],
            "adaptation_failed": [(CognitiveState.META_REFLECTION, 0.8), (CognitiveState.EXPLORATION, 0.2)],
            "default": [(CognitiveState.ADAPTATION, 0.6), (CognitiveState.EXPLOITATION, 0.4)]
        }

        # From META_REFLECTION state
        transitions[CognitiveState.META_REFLECTION] = {
            "insight_gained": [(CognitiveState.ADAPTATION, 0.7), (CognitiveState.INTEGRATION, 0.3)],
            "system_optimized": [(CognitiveState.EXPLOITATION, 0.8), (CognitiveState.EXPLORATION, 0.2)],
            "default": [(CognitiveState.META_REFLECTION, 0.3), (CognitiveState.DELIBERATION, 0.7)]
        }

        return transitions

    def _load_state_transitions_from_yaml(self, path="config/state_transitions.yaml"):
        try:
            with open(Path(path), "r") as f:
                transitions = yaml.safe_load(f)
                parsed = {}
                for state, triggers in transitions.items():
                    enum_state = CognitiveState[state.upper()]
                    parsed[enum_state] = {
                        trigger: [(CognitiveState[target.upper()], weight) for target, weight in outcomes]
                        for trigger, outcomes in triggers.items()
                    }
                return parsed
        except Exception as e:
            self.logger.error(f"Failed to load state transitions from YAML: {e}")
            return {}

    def _monitoring_loop(self):
        """Background thread for monitoring system state and triggering state transitions."""
        monitoring_interval = self.config.get("monitoring_interval", 1.0)  # seconds

        while self._running:
            try:
                start_time = time.time()

                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.resource_usage_history.append(system_metrics)
                if len(self.resource_usage_history) > 100:
                    self.resource_usage_history = self.resource_usage_history[-100:]

                # Evaluate cognitive context
                self._update_cognitive_context(system_metrics)

                # Check for state transition triggers
                self._evaluate_state_transitions()

                # Perform metacognitive reflection periodically
                if self.context.state_duration % 10 == 0 and self.context.state_duration > 0:
                    self._perform_metacognitive_reflection()

                # Sleep for the remainder of the monitoring interval
                elapsed = time.time() - start_time
                sleep_time = max(0.0, monitoring_interval - elapsed)
                time.sleep(sleep_time)

                # Increment state duration
                self.context.state_duration += 1

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(monitoring_interval)  # Sleep even on error to prevent CPU spinning

    def _execution_loop(self):
        """Background thread for executing cognitive actions."""
        execution_interval = self.config.get("execution_interval", 0.2)  # seconds

        while self._running:
            try:
                start_time = time.time()

                # Process action queue
                self._process_action_queue()

                # Generate new actions based on current context
                if len(self.action_queue) < 5:  # If queue is getting empty
                    self._generate_cognitive_actions()

                # Cleanup completed actions
                self._cleanup_actions()

                # 🔁 Nouvelle trace méta-cognitive ajoutée ici
                self.meta_trace_log.append({
                    "cycle": self.context.state_duration,
                    "state": self.context.state.name,
                    "focus": self.context.focus_target,
                    "confidence": self.context.confidence,
                    "actions_active": list(self.active_actions.keys()),
                    "actions_queued": [a.action_type for a in self.action_queue[:5]],
                })

                # Sleep for the remainder of the execution interval
                elapsed = time.time() - start_time
                sleep_time = max(0.0, execution_interval - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}")
                time.sleep(execution_interval)

    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics for monitoring."""
        metrics = {}

        # Try to get resource usage from resource manager
        if self.resource_manager:
            try:
                resource_info = self.resource_manager.get_usage_history(1)
                if resource_info and len(resource_info) > 0:
                    metrics["cpu_usage"] = resource_info[0].cpu_percent / 100.0
                    metrics["memory_usage"] = resource_info[0].mem_percent / 100.0
            except Exception as e:
                self.logger.warning(f"Could not get resource metrics: {e}")
                metrics["cpu_usage"] = 0.5  # Default values if unavailable
                metrics["memory_usage"] = 0.5

        # Calculate module-specific metrics
        metrics["reasoning_active"] = 1.0 if self.reasoning_engine and len(self.active_actions.get("reason", [])) > 0 else 0.0
        metrics["learning_active"] = 1.0 if self.learning_manager and len(self.active_actions.get("learn", [])) > 0 else 0.0
        metrics["perception_active"] = 1.0 if self.perception_module and len(self.active_actions.get("perceive", [])) > 0 else 0.0

        # Calculate cognitive metrics
        metrics["cognitive_coherence"] = self.cognitive_coherence
        metrics["adaptation_rate"] = self.adaptation_rate
        metrics["metacognitive_depth"] = self.metacognitive_depth

        # Add context metrics
        metrics["task_complexity"] = self.context.task_complexity
        metrics["time_pressure"] = self.context.time_pressure
        metrics["confidence"] = self.context.confidence
        metrics["novelty"] = self.context.novelty

        # Add success rates
        metrics["reasoning_success_rate"] = self.success_rate.get("reason", 0.8)
        metrics["learning_success_rate"] = self.success_rate.get("learn", 0.8)
        metrics["perception_success_rate"] = self.success_rate.get("perceive", 0.8)

        return metrics

    def _update_cognitive_context(self, metrics: Dict[str, float]):
        """Update cognitive context based on current metrics."""
        # Apply small random drift to simulate environmental changes
        self.context.task_complexity = min(1.0, max(0.0, self.context.task_complexity + (0.1 * (2 * random.random() - 1))))
        self.context.time_pressure = min(1.0, max(0.0, self.context.time_pressure + (0.05 * (2 * random.random() - 1))))

        # Update confidence based on recent success rates
        avg_success = (
            metrics["reasoning_success_rate"] +
            metrics["learning_success_rate"] +
            metrics["perception_success_rate"]
        ) / 3.0

        # Blend current confidence with success rate (exponential moving average)
        alpha = 0.2  # Blending factor
        self.context.confidence = alpha * avg_success + (1 - alpha) * self.context.confidence

        # Update novelty based on recent inputs (would come from perception in real system)
        # For simulation, slowly decrease novelty unless refreshed
        self.context.novelty = max(0.1, self.context.novelty * 0.95)

        # Use working memory size as a proxy for cognitive load
        if self.working_memory:
            try:
                memory_stats = self.working_memory.get_stats()
                memory_load = memory_stats.get("usage_percent", 50) / 100.0
                self.context.task_complexity = max(self.context.task_complexity, memory_load)
            except Exception as e:
                self.logger.debug(f"Could not get working memory stats: {e}")

    def _evaluate_state_transitions(self):
        """Evaluate whether to transition to a different cognitive state."""
        current_state = self.context.state
        transitions = self.state_transition_matrix.get(current_state, {})

        # Determine which trigger conditions apply
        triggers = []

        if self.context.time_pressure > 0.7:
            triggers.append("high_time_pressure")

        if self.context.time_pressure < 0.3 and "more_time_available" in transitions:
            triggers.append("more_time_available")

        if self.context.novelty > 0.7:
            triggers.append("high_novelty")

        if self.context.novelty < 0.3 and "low_novelty" in transitions:
            triggers.append("low_novelty")

        if self.context.task_complexity > 0.7:
            triggers.append("high_complexity")

        if self.context.confidence < 0.3:
            triggers.append("low_confidence")

        if len(self.completed_actions) > 0 and any(a.action_type == "reason" for a in self.completed_actions.values()):
            if self.context.confidence > 0.7:
                triggers.append("solution_found")

            if current_state == CognitiveState.DELIBERATION and self.context.confidence > 0.6:
                triggers.append("solution_refined")

        if len(self.active_actions) == 0 and current_state == CognitiveState.ADAPTATION:
            triggers.append("adaptation_complete")

        if len(self.active_actions) == 0 and current_state == CognitiveState.INTEGRATION:
            triggers.append("integration_complete")

        if self.context.state_duration > 20 and self.context.confidence < 0.4:
            triggers.append("no_progress")

        # If no specific triggers, use default
        if not triggers:
            triggers = ["default"]

        # For each applicable trigger, consider possible transitions
        possible_transitions = []
        for trigger in triggers:
            if trigger in transitions:
                possible_transitions.extend(transitions[trigger])

        # If no transitions found, stay in current state
        if not possible_transitions:
            return

        # Calculate weighted probability for transitions
        total_weight = sum(weight for _, weight in possible_transitions)
        transition_point = random.random() * total_weight

        cumulative_weight = 0
        for next_state, weight in possible_transitions:
            cumulative_weight += weight
            if cumulative_weight >= transition_point:
                # Transition to this state
                if next_state != current_state:
                    self.logger.info(f"Transitioning from {current_state.name} to {next_state.name} (triggers: {triggers})")
                    self.context.state = next_state
                    self.context.state_duration = 0
                break

    def _perform_metacognitive_reflection(self):
        """
        Perform metacognitive reflection to optimize system operation.
        This analyzes performance data and adjusts internal parameters.
        """
        self.logger.debug("Performing metacognitive reflection")

        # Calculate success rates for different cognitive operations
        for action_type in ["reason", "learn", "perceive", "memorize", "retrieve"]:
            actions = [a for a in self.completed_actions.values() if a.action_type == action_type]
            if actions:
                success_count = sum(1 for a in actions if a.state == "completed")
                self.success_rate[action_type] = success_count / len(actions)

        # Analyze resource usage patterns
        if self.resource_usage_history:
            recent_cpu = [m.get("cpu_usage", 0.5) for m in self.resource_usage_history[-10:]]
            avg_cpu = sum(recent_cpu) / len(recent_cpu)

            # If CPU usage is consistently high, reduce parallelism
            if avg_cpu > 0.8:
                self.logger.info("High CPU detected, reducing cognitive parallelism")
                # Implementation: Will reduce number of concurrent actions in queue processing

            # If CPU usage is consistently low, increase parallelism
            if avg_cpu < 0.3:
                self.logger.info("Low CPU detected, increasing cognitive parallelism")
                # Implementation: Will increase number of concurrent actions in queue processing

        # Adjust adaptation rate based on context
        novelty_trend = [m.get("novelty", 0.5) for m in self.resource_usage_history[-5:]]
        if len(novelty_trend) >= 5:
            if novelty_trend[0] < novelty_trend[-1]:  # Increasing novelty
                self.adaptation_rate = min(1.0, self.adaptation_rate + 0.05)
            else:
                self.adaptation_rate = max(0.1, self.adaptation_rate - 0.02)

        # Increase metacognitive depth periodically
        self.metacognitive_depth = min(1.0, self.metacognitive_depth + 0.01)

        # If in META_REFLECTION state, do deeper analysis
        if self.context.state == CognitiveState.META_REFLECTION:
            # Analyze balance between cognitive processes
            reasoning_time = sum(self.response_times.get("reason", [0.0]))
            learning_time = sum(self.response_times.get("learn", [0.0]))
            perception_time = sum(self.response_times.get("perceive", [0.0]))

            total_time = reasoning_time + learning_time + perception_time
            if total_time > 0:
                reasoning_ratio = reasoning_time / total_time
                learning_ratio = learning_time / total_time
                perception_ratio = perception_time / total_time

                self.logger.info(f"Cognitive balance - Reasoning: {reasoning_ratio:.2f}, "
                               f"Learning: {learning_ratio:.2f}, Perception: {perception_ratio:.2f}")

                # Balance appears optimal when no single process dominates and none are neglected
                balance_score = 1.0 - (max(reasoning_ratio, learning_ratio, perception_ratio) -
                                     min(reasoning_ratio, learning_ratio, perception_ratio))

                self.cognitive_coherence = 0.3 * balance_score + 0.7 * self.cognitive_coherence

    def _generate_cognitive_actions(self):
        """Generate new cognitive actions based on current state and context."""
        # The actions generated depend on the current cognitive state

        # In EXPLORATION state, focus on perception and learning
        if self.context.state == CognitiveState.EXPLORATION:
            # Generate a perception action to explore environment
            self._add_perception_action(priority=0.8)

            # Generate a learning action to assimilate new information
            self._add_learning_action(priority=0.7)

            # Generate a low-priority reasoning action to generate hypotheses
            self._add_reasoning_action(priority=0.4, reasoning_type="generative")

        # In EXPLOITATION state, focus on applying knowledge
        elif self.context.state == CognitiveState.EXPLOITATION:
            # Generate high-priority reasoning action to solve problems
            self._add_reasoning_action(priority=0.9, reasoning_type="deductive")

            # Generate memory retrieval actions to support reasoning
            self._add_retrieval_action(priority=0.8)

            # Low-priority learning to reinforce success patterns
            self._add_learning_action(priority=0.3, learning_type="reinforcement")

        # In DELIBERATION state, focus on deep reasoning
        elif self.context.state == CognitiveState.DELIBERATION:
            # Generate high-priority reasoning actions
            self._add_reasoning_action(priority=0.9, reasoning_type="causal")

            # Generate memory retrieval and integration actions
            self._add_retrieval_action(priority=0.8)
            self._add_integration_action(priority=0.7)

            # Low-priority perception to stay aware of environment
            self._add_perception_action(priority=0.3)

        # In INTUITION state, focus on rapid pattern recognition
        elif self.context.state == CognitiveState.INTUITION:
            # Generate rapid pattern-matching reasoning
            self._add_reasoning_action(priority=0.9, reasoning_type="analogical")

            # High-priority perception to stay responsive
            self._add_perception_action(priority=0.8)

            # Medium-priority retrieval for pattern matching
            self._add_retrieval_action(priority=0.6)

        # In INTEGRATION state, focus on consolidating knowledge
        elif self.context.state == CognitiveState.INTEGRATION:
            # Generate high-priority integration actions
            self._add_integration_action(priority=0.9)

            # Medium-priority reasoning to resolve inconsistencies
            self._add_reasoning_action(priority=0.6, reasoning_type="abductive")

            # Medium-priority learning to consolidate
            self._add_learning_action(priority=0.5, learning_type="consolidation")

        # In ADAPTATION state, focus on learning and updating
        elif self.context.state == CognitiveState.ADAPTATION:
            # Generate high-priority learning actions
            self._add_learning_action(priority=0.9, learning_type="adaptive")

            # Medium-priority reasoning to evaluate changes
            self._add_reasoning_action(priority=0.6, reasoning_type="inductive")

            # Medium-priority perception to monitor environment
            self._add_perception_action(priority=0.5)

         # In META_REFLECTION state, focus on self-analysis
        elif self.context.state == CognitiveState.META_REFLECTION:
            # Generate high-priority self-analysis actions
            self._add_meta_analysis_action(priority=0.9)

            # Medium-priority reasoning to optimize processes
            self._add_reasoning_action(priority=0.6, reasoning_type="evaluative")

            # Add performance analysis action
            self._add_performance_analysis_action(priority=0.7)

    def _add_perception_action(self, priority=0.5):
        """Add a perception action to the queue."""
        action_id = str(uuid.uuid4())

        # Determine perception focus based on context
        if self.context.novelty > 0.7:
            perception_mode = "exploratory"  # Focus on detecting novelty
        elif self.context.time_pressure > 0.7:
            perception_mode = "selective"    # Focus on most relevant aspects
        else:
            perception_mode = "balanced"     # Normal operation

        # Create perception task
        action = CognitiveAction(
            action_id=action_id,
            action_type="perceive",
            priority=priority,
            target_modules=["perception_module"],
            parameters={
                "mode": perception_mode,
                "focus": self.context.focus_target,
                "depth": min(1.0, max(0.3, 1.0 - self.context.time_pressure))
            },
            estimated_resources={
                "cpu": 0.3,
                "memory": 0.2,
                "time": 0.5 if perception_mode == "exploratory" else 0.2
            }
        )

        # Add to queue
        heapq.heappush(self.action_queue, action)
        self.logger.debug(f"Added perception action: {action_id} (mode: {perception_mode})")

    def _add_reasoning_action(self, priority=0.5, reasoning_type="deductive"):
        """Add a reasoning action to the queue."""
        action_id = str(uuid.uuid4())

        # Set parameters based on reasoning type
        if reasoning_type == "deductive":
            confidence_threshold = 0.8
            time_limit = 0.5
        elif reasoning_type == "inductive":
            confidence_threshold = 0.6
            time_limit = 0.8
        elif reasoning_type == "abductive":
            confidence_threshold = 0.5
            time_limit = 0.7
        elif reasoning_type == "analogical":
            confidence_threshold = 0.4
            time_limit = 0.3
        elif reasoning_type == "causal":
            confidence_threshold = 0.7
            time_limit = 1.0
        elif reasoning_type == "generative":
            confidence_threshold = 0.3
            time_limit = 0.6
        elif reasoning_type == "evaluative":
            confidence_threshold = 0.7
            time_limit = 0.5
        else:
            confidence_threshold = 0.6
            time_limit = 0.5

        # Adjust time limit based on time pressure
        time_limit = time_limit * (1.0 - 0.5 * self.context.time_pressure)

        # Create reasoning task
        action = CognitiveAction(
            action_id=action_id,
            action_type="reason",
            priority=priority,
            target_modules=["reasoning_engine"],
            parameters={
                "reasoning_type": reasoning_type,
                "confidence_threshold": confidence_threshold,
                "time_limit": time_limit,
                "depth": min(1.0, max(0.3, self.context.task_complexity))
            },
            estimated_resources={
                "cpu": 0.5,
                "memory": 0.4,
                "time": time_limit
            }
        )

        # Add to queue
        heapq.heappush(self.action_queue, action)
        self.logger.debug(f"Added reasoning action: {action_id} (type: {reasoning_type})")

    def _add_learning_action(self, priority=0.5, learning_type="standard"):
        """Add a learning action to the queue."""
        action_id = str(uuid.uuid4())

        # Set parameters based on learning type
        if learning_type == "standard":
            batch_size = 32
            epochs = 1
        elif learning_type == "consolidation":
            batch_size = 64
            epochs = 3
        elif learning_type == "reinforcement":
            batch_size = 16
            epochs = 1
        elif learning_type == "adaptive":
            batch_size = 24
            epochs = 2
        else:
            batch_size = 32
            epochs = 1

        # Create learning task
        action = CognitiveAction(
            action_id=action_id,
            action_type="learn",
            priority=priority,
            target_modules=["learning_manager"],
            parameters={
                "learning_type": learning_type,
                "batch_size": batch_size,
                "epochs": epochs,
                "adaptation_rate": self.adaptation_rate
            },
            estimated_resources={
                "cpu": 0.6,
                "memory": 0.5,
                "time": 0.3 * epochs
            }
        )

        # Add to queue
        heapq.heappush(self.action_queue, action)
        self.logger.debug(f"Added learning action: {action_id} (type: {learning_type})")

    def _add_retrieval_action(self, priority=0.5):
        """Add a memory retrieval action to the queue."""
        action_id = str(uuid.uuid4())

        # Determine memory sources based on context
        memory_sources = []
        if self.context.state in [CognitiveState.INTUITION, CognitiveState.EXPLOITATION]:
            memory_sources.append("working_memory")

        if self.context.task_complexity > 0.5 or self.context.state in [CognitiveState.DELIBERATION, CognitiveState.INTEGRATION]:
            memory_sources.append("semantic_memory")

        if self.context.novelty < 0.5 or self.context.state in [CognitiveState.DELIBERATION, CognitiveState.META_REFLECTION]:
            memory_sources.append("episodic_memory")

        # Create retrieval task
        action = CognitiveAction(
            action_id=action_id,
            action_type="retrieve",
            priority=priority,
            target_modules=memory_sources,
            parameters={
                "query": self.context.focus_target or "current_task",
                "max_results": 5,
                "recency_weight": 0.7 if "episodic_memory" in memory_sources else 0.3,
                "relevance_threshold": 0.6 - (0.2 * self.context.time_pressure)
            },
            estimated_resources={
                "cpu": 0.2,
                "memory": 0.3,
                "time": 0.2 + (0.1 * len(memory_sources))
            }
        )

        # Add to queue
        heapq.heappush(self.action_queue, action)
        self.logger.debug(f"Added retrieval action: {action_id} (sources: {memory_sources})")

    def _add_integration_action(self, priority=0.5):
        """Add a knowledge integration action to the queue."""
        action_id = str(uuid.uuid4())

        # Create integration task
        action = CognitiveAction(
            action_id=action_id,
            action_type="integrate",
            priority=priority,
            target_modules=["semantic_memory", "graph_engine"],
            parameters={
                "consistency_threshold": 0.7,
                "integration_depth": min(1.0, max(0.4, self.context.task_complexity)),
                "update_ontology": self.context.state == CognitiveState.INTEGRATION,
                "consolidate_relations": True
            },
            estimated_resources={
                "cpu": 0.4,
                "memory": 0.6,
                "time": 0.5
            }
        )

        # Add to queue
        heapq.heappush(self.action_queue, action)
        self.logger.debug(f"Added integration action: {action_id}")

    def _add_meta_analysis_action(self, priority=0.5):
        """Add a metacognitive analysis action to the queue."""
        action_id = str(uuid.uuid4())

        # Create meta-analysis task
        action = CognitiveAction(
            action_id=action_id,
            action_type="meta_analyze",
            priority=priority,
            target_modules=["reasoning_engine", "learning_manager"],
            parameters={
                "target": "cognitive_process",
                "depth": self.metacognitive_depth,
                "analyze_success_factors": True,
                "analyze_failure_patterns": True,
                "generate_improvements": True
            },
            estimated_resources={
                "cpu": 0.3,
                "memory": 0.4,
                "time": 0.6
            }
        )

        # Add to queue
        heapq.heappush(self.action_queue, action)
        self.logger.debug(f"Added meta-analysis action: {action_id}")

    def _add_performance_analysis_action(self, priority=0.5):
        """Add a performance analysis action to the queue."""
        action_id = str(uuid.uuid4())

        # Create performance analysis task
        action = CognitiveAction(
            action_id=action_id,
            action_type="analyze_performance",
            priority=priority,
            target_modules=["resource_manager"],
            parameters={
                "target_metrics": ["response_time", "success_rate", "resource_efficiency"],
                "history_length": 50,
                "generate_optimization_recommendations": True
            },
            estimated_resources={
                "cpu": 0.2,
                "memory": 0.3,
                "time": 0.4
            }
        )

        # Add to queue
        heapq.heappush(self.action_queue, action)
        self.logger.debug(f"Added performance analysis action: {action_id}")

    def _process_action_queue(self):
        """Process actions from the queue based on priorities and resource constraints."""
        self._cleanup_actions()  # réparation 6.5 : libère les ressources des actions finies
        if not self.action_queue:
            return

        # Check for available resources
        available_resources = {
            "cpu": 1.0,
            "memory": 1.0,
            "time": 1.0
        }

        # Subtract resources already in use by active actions
        for action in self.active_actions.values():
            for resource, amount in action.estimated_resources.items():
                available_resources[resource] = max(0.0, available_resources[resource] - amount)

        # Process as many actions as resources allow, in priority order
        processed_actions = []
        temp_queue = []

        while self.action_queue:
            # Get highest priority action
            action = heapq.heappop(self.action_queue)

            # Check if it can be executed with available resources
            can_execute = True
            for resource, amount in action.estimated_resources.items():
                if amount > available_resources.get(resource, 0):
                    can_execute = False
                    break

            if can_execute:
                # Reserve resources
                for resource, amount in action.estimated_resources.items():
                    available_resources[resource] = max(0.0, available_resources[resource] - amount)

                # Execute action
                self._execute_action(action)
                processed_actions.append(action)

                # Limit number of actions processed per cycle to prevent overload
                if len(processed_actions) >= 3:
                    break
            else:
                # Put back in temporary queue if can't execute
                temp_queue.append(action)

        # Put unprocessed actions back in the queue
        for action in temp_queue:
            heapq.heappush(self.action_queue, action)

    def _execute_action(self, action: CognitiveAction):
        """Execute a cognitive action by sending to appropriate module."""
        start_time = time.time()
        action.state = "executing"
        self.active_actions[action.action_id] = action

        try:
            # Dispatch to appropriate handler based on action type
            if action.action_type == "perceive":
                self._execute_perception_action(action)
            elif action.action_type == "reason":
                self._execute_reasoning_action(action)
            elif action.action_type == "learn":
                self._execute_learning_action(action)
            elif action.action_type == "retrieve":
                self._execute_retrieval_action(action)
            elif action.action_type == "integrate":
                self._execute_integration_action(action)
            elif action.action_type == "meta_analyze":
                self._execute_meta_analysis_action(action)
            elif action.action_type == "analyze_performance":
                self._execute_performance_analysis_action(action)
            else:
                self.logger.warning(f"Unknown action type: {action.action_type}")
                action.state = "failed"

            # Update action state
            if action.state == "executing":
                action.state = "completed"

            # Record response time
            elapsed = time.time() - start_time
            if action.action_type in self.response_times:
                self.response_times[action.action_type].append(elapsed)
                if len(self.response_times[action.action_type]) > 50:
                    self.response_times[action.action_type] = self.response_times[action.action_type][-50:]
            else:
                self.response_times[action.action_type] = [elapsed]

            self.logger.debug(f"Executed action {action.action_id} ({action.action_type}) in {elapsed:.3f}s")

        except Exception as e:
            self.logger.error(f"Error executing action {action.action_id}: {e}")
            action.state = "failed"

    def _execute_perception_action(self, action: CognitiveAction):
        """Execute a perception action using the perception module."""
        if not self.perception_module:
            action.state = "failed"
            return

        try:
            # Create perception task
            task = {
                "mode": action.parameters.get("mode", "balanced"),
                "focus": action.parameters.get("focus"),
                "depth": action.parameters.get("depth", 0.5)
            }

            # Call perception module
            result = self.perception_module.process(task)

            # Store result in working memory if successful
            if result and result.get("success", False):
                if self.working_memory:
                    self.working_memory.store_item(
                        content=result.get("processed_data"),
                        source="perception",
                        importance=max(0.5, self.context.novelty)
                    )

                # Update novelty based on perception result
                if "novelty" in result:
                    self.context.novelty = result["novelty"]
            else:
                action.state = "failed"

        except Exception as e:
            self.logger.error(f"Error in perception: {e}")
            action.state = "failed"

    def _execute_reasoning_action(self, action: CognitiveAction):
        """Execute a reasoning action using the reasoning engine."""
        if not self.reasoning_engine:
            action.state = "failed"
            return

        try:
            # Prepare reasoning task
            task = {
                "task_type": action.parameters.get("reasoning_type", "deductive"),
                "confidence_threshold": action.parameters.get("confidence_threshold", 0.6),
                "time_limit": action.parameters.get("time_limit", 0.5),
                "depth": action.parameters.get("depth", 0.5),
                "context": {
                    "state": self.context.state.name,
                    "confidence": self.context.confidence,
                    "focus": self.context.focus_target
                }
            }

            # Call reasoning engine
            result = self.reasoning_engine.reason(task)

            # Process result
            if result and result.get("success", False):
                # Store conclusions in working memory
                if self.working_memory and "conclusions" in result:
                    for conclusion in result["conclusions"]:
                        self.working_memory.store_item(
                            content=conclusion,
                            source="reasoning",
                            importance=result.get("confidence", 0.5)
                        )

                # Update context confidence based on reasoning result
                if "confidence" in result:
                    self.context.confidence = 0.3 * result["confidence"] + 0.7 * self.context.confidence
            else:
                action.state = "failed"

        except Exception as e:
            self.logger.error(f"Error in reasoning: {e}")
            action.state = "failed"

    def _execute_learning_action(self, action: CognitiveAction):
        """Execute a learning action using the learning manager."""
        if not self.learning_manager:
            action.state = "failed"
            return

        try:
            # Prepare learning task
            task = {
                "name": f"learning_{action.action_id[-8:]}",
                "task_type": action.parameters.get("learning_type", "standard"),
                "epochs": action.parameters.get("epochs", 1),
                "batch_size": action.parameters.get("batch_size", 32),
                "adaptation_rate": action.parameters.get("adaptation_rate", self.adaptation_rate),
                "objective": f"optimize_for_{self.context.state.name.lower()}"
            }

            # Call learning manager
            result = self.learning_manager.learn(task)

            # Process result
            if result and result.get("success", False):
                # Update adaptation rate based on learning success
                if "metrics" in result:
                    improvement = result["metrics"].get("improvement", 0)
                    self.adaptation_rate = min(1.0, max(0.1, self.adaptation_rate + 0.1 * improvement))
            else:
                action.state = "failed"

        except Exception as e:
            self.logger.error(f"Error in learning: {e}")
            action.state = "failed"

    def _execute_retrieval_action(self, action: CognitiveAction):
        """Execute a memory retrieval action."""
        try:
            query = action.parameters.get("query", "current_task")
            max_results = action.parameters.get("max_results", 5)
            recency_weight = action.parameters.get("recency_weight", 0.5)
            relevance_threshold = action.parameters.get("relevance_threshold", 0.6)

            results = []

            # Query working memory if specified
            if "working_memory" in action.target_modules and self.working_memory:
                working_results = self.working_memory.retrieve(
                    query_text=query,
                    max_results=max_results,
                    min_importance=relevance_threshold
                )
                results.extend([(item, "working") for item in working_results])

            # Query semantic memory if specified
            if "semantic_memory" in action.target_modules and self.semantic_memory:
                semantic_query = {
                    "text": query,
                    "importance_threshold": relevance_threshold,
                    "max_results": max_results
                }
                from grandnexus.memory.semantic import SemanticQuery  # réparation 6.5
                semantic_results = self.semantic_memory.query_concepts(SemanticQuery(**semantic_query))
                results.extend([(item, "semantic") for item in semantic_results])

            # Query episodic memory if specified
            if "episodic_memory" in action.target_modules and self.episodic_memory:
                episodic_query = {
                    "text": query,
                    "importance_threshold": relevance_threshold,
                    "max_results": max_results,
                    "recency_bias": recency_weight
                }
                from grandnexus.memory.episodic import EpisodicQuery  # réparation 6.5
                _eq = dict(episodic_query)
                _eq["context_factors"] = {"recency_bias": _eq.pop("recency_bias", None)}
                episodic_results = self.episodic_memory.retrieve_episodes(EpisodicQuery(**_eq))
                results.extend([(item, "episodic") for item in episodic_results])

            # Store results in working memory
            if results and self.working_memory:
                self.working_memory.store_item(
                    content={
                        "query": query,
                        "results": results,
                        "source_modules": action.target_modules
                    },
                    source="memory_retrieval",
                    importance=0.7
                )

            # If no results found, mark as partial success
            if not results:
                self.logger.debug(f"No results found for retrieval: {query}")
                action.state = "completed"  # Still mark as completed, just with empty results

        except Exception as e:
            self.logger.error(f"Error in retrieval: {e}")
            action.state = "failed"

    def _execute_integration_action(self, action: CognitiveAction):
        """Execute a knowledge integration action."""
        try:
            # Integration requires both semantic memory and graph engine
            if not (self.semantic_memory and self.graph_engine):
                action.state = "failed"
                return

            consistency_threshold = action.parameters.get("consistency_threshold", 0.7)
            integration_depth = action.parameters.get("integration_depth", 0.5)
            update_ontology = action.parameters.get("update_ontology", False)
            consolidate_relations = action.parameters.get("consolidate_relations", True)

            # Extract current working memory items for integration
            working_items = []
            if self.working_memory:
                working_items = self.working_memory.get_all_items()

            # If no items to integrate, skip
            if not working_items:
                self.logger.debug("No items to integrate")
                return

            # Step 1: Extract concepts and relations from working memory
            concepts = []
            relations = []

            for item in working_items:
                # Process based on content type and source
                content = item.content
                if isinstance(content, dict) and "type" in content:
                    if content["type"] == "concept":
                        concepts.append(content)
                    elif content["type"] == "relation":
                        relations.append(content)

            # Step 2: Check consistency with existing knowledge
            inconsistencies = []

            # Check concept consistency
            for concept in concepts:
                # Query semantic memory for the concept
                existing = self.semantic_memory.get_concept(concept.get("name", ""))
                if existing:
                    # Check for property inconsistencies
                    for prop, value in concept.get("properties", {}).items():
                        if prop in existing.properties and existing.properties[prop] != value:
                            inconsistencies.append({
                                "type": "concept_property",
                                "concept": concept.get("name", ""),
                                "property": prop,
                                "new_value": value,
                                "existing_value": existing.properties[prop]
                            })

            # Check relation consistency
            for relation in relations:
                # Skip if missing required fields
                if not all(k in relation for k in ["source", "relation_type", "target"]):
                    continue

                # Check if the inverse relation exists and is contradictory
                query_result = self.semantic_memory.query_relations(
                    source_concept_id=relation["target"],
                    relation_type=relation["relation_type"],
                    target_concept_id=relation["source"]
                )

                if query_result and not relation.get("bidirectional", False):
                    inconsistencies.append({
                        "type": "contradictory_relation",
                        "relation": relation,
                        "existing_relation": query_result[0]
                    })

            # Step 3: Resolve inconsistencies or report them
            if inconsistencies and len(inconsistencies) > 5:
                # Too many inconsistencies, requires manual resolution
                self.logger.warning(f"Too many inconsistencies ({len(inconsistencies)}) during integration")
                action.state = "failed"
                return

            # If few inconsistencies, resolve them based on confidence
            for inconsistency in inconsistencies:
                if inconsistency["type"] == "concept_property":
                    # Use the value with higher confidence or more recent
                    # For now, just log the inconsistency
                    self.logger.debug(f"Resolved property inconsistency for {inconsistency['concept']}.{inconsistency['property']}")

                elif inconsistency["type"] == "contradictory_relation":
                    # For simplicity, just log the inconsistency
                    self.logger.debug(f"Detected contradictory relation: {inconsistency['relation']} vs {inconsistency['existing_relation']}")

            # Step 4: Integrate consistent knowledge
            # Add concepts
            for concept in concepts:
                if "name" not in concept:
                    continue

                existing = self.semantic_memory.get_concept(concept.get("name", ""))
                if existing:
                    # Update existing concept
                    self.semantic_memory.update_concept({
                        "concept_id": existing.concept_id,
                        "properties": concept.get("properties", {}),
                        "importance": max(existing.importance, concept.get("importance", 0.5))
                    }, partial_update=True)
                else:
                    # Create new concept
                    self.semantic_memory.add_concept({
                        "name": concept["name"],
                        "description": concept.get("description", ""),
                        "properties": concept.get("properties", {}),
                        "importance": concept.get("importance", 0.5)
                    })

            # Add relations
            for relation in relations:
                if not all(k in relation for k in ["source", "relation_type", "target"]):
                    continue

                # Check if source and target concepts exist
                source = self.semantic_memory.get_concept(relation["source"])
                target = self.semantic_memory.get_concept(relation["target"])

                if source and target:
                    self.semantic_memory.add_relation({
                        "source_concept_id": source.concept_id,
                        "relation_type": relation["relation_type"],
                        "target_concept_id": target.concept_id,
                        "weight": relation.get("weight", 1.0),
                        "bidirectional": relation.get("bidirectional", False)
                    })

            # Step 5: Update ontology if requested
            if update_ontology:
                # This would require more complex logic to systematically update the ontology
                # For now, just log that it would happen
                self.logger.info("Ontology update would occur here")

            # Step 6: Consolidate redundant relations if requested
            if consolidate_relations:
                # This would merge similar relations or clean up unnecessary ones
                # For now, just log that it would happen
                self.logger.info("Relation consolidation would occur here")

            # Successfully integrated knowledge
            self.cognitive_coherence = min(1.0, self.cognitive_coherence + 0.05)

        except Exception as e:
            self.logger.error(f"Error in integration: {e}")
            action.state = "failed"

    def _execute_meta_analysis_action(self, action: CognitiveAction):
        """Execute a metacognitive analysis action."""
        try:
            target = action.parameters.get("target", "cognitive_process")
            depth = action.parameters.get("depth", 0.5)
            analyze_success = action.parameters.get("analyze_success_factors", True)
            analyze_failure = action.parameters.get("analyze_failure_patterns", True)
            generate_improvements = action.parameters.get("generate_improvements", True)

            # Collect metrics for analysis
            metrics = {
                "cognitive_coherence": self.cognitive_coherence,
                "adaptation_rate": self.adaptation_rate,
                "metacognitive_depth": self.metacognitive_depth,
                "success_rates": dict(self.success_rate),
                "state_durations": {state.name: 0 for state in CognitiveState},
                "response_times": {k: sum(v)/len(v) if v else 0 for k, v in self.response_times.items()}
            }

            # Analyze success factors if requested
            insights = []
            if analyze_success:
                # Find operation types with high success rates
                successful_ops = [op for op, rate in self.success_rate.items() if rate > 0.8]
                if successful_ops:
                    insights.append(f"High success in operations: {', '.join(successful_ops)}")

                # Find fast response times
                fast_ops = [op for op, times in self.response_times.items()
                           if times and sum(times)/len(times) < 0.3]
                if fast_ops:
                    insights.append(f"Fast response in operations: {', '.join(fast_ops)}")

            # Analyze failure patterns if requested
            if analyze_failure:
                # Find operation types with low success rates
                failing_ops = [op for op, rate in self.success_rate.items() if rate < 0.5]
                if failing_ops:
                    insights.append(f"Low success in operations: {', '.join(failing_ops)}")

                # Find slow response times
                slow_ops = [op for op, times in self.response_times.items()
                           if times and sum(times)/len(times) > 0.8]
                if slow_ops:
                    insights.append(f"Slow response in operations: {', '.join(slow_ops)}")

            # Generate improvement recommendations if requested
            recommendations = []
            if generate_improvements:
                # Recommend balanced resource allocation
                if metrics.get("resource_imbalance", 0) > 0.3:
                    recommendations.append("Rebalance resource allocation across cognitive processes")

                # Recommend more exploration if stuck in exploitation
                if self.context.state == CognitiveState.EXPLOITATION and self.context.state_duration > 10:
                    recommendations.append("Increase exploration to discover new approaches")

                # Recommend more learning if success rates are low
                if self.success_rate and sum(self.success_rate.values()) / len(self.success_rate) < 0.6:  # réparation 6.5
                    recommendations.append("Increase learning rate to improve performance")

            # Store analysis results in working memory
            if self.working_memory:
                self.working_memory.store_item(
                    content={
                        "type": "meta_analysis",
                        "timestamp": time.time(),
                        "metrics": metrics,
                        "insights": insights,
                        "recommendations": recommendations
                    },
                    source="meta_cognition",
                    importance=0.8
                )

            # Update metacognitive depth
            self.metacognitive_depth = min(1.0, self.metacognitive_depth + 0.05)

        except Exception as e:
            self.logger.error(f"Error in meta-analysis: {e}")
            action.state = "failed"

    def _execute_performance_analysis_action(self, action: CognitiveAction):
        """Execute a performance analysis action."""
        try:
            target_metrics = action.parameters.get("target_metrics", ["response_time"])
            history_length = action.parameters.get("history_length", 50)
            generate_recommendations = action.parameters.get("generate_optimization_recommendations", True)

            # Collect resource usage history
            resource_history = self.resource_usage_history[-history_length:] if self.resource_usage_history else []

            # Analyze performance trends
            trends = {}

            # Analyze CPU usage trend
            if resource_history and "cpu_usage" in resource_history[0]:
                cpu_values = [r.get("cpu_usage", 0) for r in resource_history]
                cpu_avg = sum(cpu_values) / len(cpu_values)
                cpu_trend = "stable"

                if len(cpu_values) > 5:
                    # Check if trend is increasing or decreasing
                    first_half = cpu_values[:len(cpu_values)//2]
                    second_half = cpu_values[len(cpu_values)//2:]
                    first_avg = sum(first_half) / len(first_half)
                    second_avg = sum(second_half) / len(second_half)

                    if second_avg > first_avg * 1.1:
                        cpu_trend = "increasing"
                    elif second_avg < first_avg * 0.9:
                        cpu_trend = "decreasing"

                trends["cpu_usage"] = {
                    "average": cpu_avg,
                    "trend": cpu_trend,
                    "current": cpu_values[-1] if cpu_values else 0
                }

            # Analyze memory usage trend
            if resource_history and "memory_usage" in resource_history[0]:
                memory_values = [r.get("memory_usage", 0) for r in resource_history]
                memory_avg = sum(memory_values) / len(memory_values)
                memory_trend = "stable"

                if len(memory_values) > 5:
                    # Check if trend is increasing or decreasing
                    first_half = memory_values[:len(memory_values)//2]
                    second_half = memory_values[len(memory_values)//2:]
                    first_avg = sum(first_half) / len(first_half)
                    second_avg = sum(second_half) / len(second_half)

                    if second_avg > first_avg * 1.1:
                        memory_trend = "increasing"
                    elif second_avg < first_avg * 0.9:
                        memory_trend = "decreasing"

                trends["memory_usage"] = {
                    "average": memory_avg,
                    "trend": memory_trend,
                    "current": memory_values[-1] if memory_values else 0
                }

            # Analyze response time trends
            for action_type, times in self.response_times.items():
                if times:
                    avg_time = sum(times) / len(times)
                    time_trend = "stable"

                    if len(times) > 5:
                        # Check if trend is increasing or decreasing
                        first_half = times[:len(times)//2]
                        second_half = times[len(times)//2:]
                        first_avg = sum(first_half) / len(first_half)
                        second_avg = sum(second_half) / len(second_half)

                        if second_avg > first_avg * 1.1:
                            time_trend = "increasing"
                        elif second_avg < first_avg * 0.9:
                            time_trend = "decreasing"

                    trends[f"{action_type}_response_time"] = {
                        "average": avg_time,
                        "trend": time_trend,
                        "current": times[-1] if times else 0
                    }

            # Generate optimization recommendations
            recommendations = []

            if generate_recommendations:
                # CPU usage recommendations
                if "cpu_usage" in trends:
                    cpu_info = trends["cpu_usage"]
                    if cpu_info["trend"] == "increasing" and cpu_info["average"] > 0.7:
                        recommendations.append("Reduce parallelism to manage increasing CPU load")
                    elif cpu_info["trend"] == "stable" and cpu_info["average"] < 0.3:
                        recommendations.append("Increase parallelism to better utilize CPU resources")

                # Memory usage recommendations
                if "memory_usage" in trends:
                    memory_info = trends["memory_usage"]
                    if memory_info["trend"] == "increasing" and memory_info["average"] > 0.7:
                        recommendations.append("Schedule more frequent memory consolidation to reduce pressure")
                    elif memory_info["current"] > 0.9:
                        recommendations.append("Perform immediate memory cleanup to prevent out-of-memory errors")

                # Response time recommendations
                slow_operations = []
                for metric, info in trends.items():
                    if "_response_time" in metric and info["average"] > 0.8:
                        operation = metric.replace("_response_time", "")
                        slow_operations.append(operation)

                if slow_operations:
                    recommendations.append(f"Optimize performance of slow operations: {', '.join(slow_operations)}")

            # Store analysis results in working memory
            if self.working_memory:
                self.working_memory.store_item(
                    content={
                        "type": "performance_analysis",
                        "timestamp": time.time(),
                        "trends": trends,
                        "recommendations": recommendations
                    },
                    source="meta_cognition",
                    importance=0.7
                )

        except Exception as e:
            self.logger.error(f"Error in performance analysis: {e}")
            action.state = "failed"

    def _cleanup_actions(self):
        """Clean up completed and failed actions."""
        # Move completed actions to history
        completed_ids = [action_id for action_id, action in self.active_actions.items()
                        if action.state in ["completed", "failed"]]

        for action_id in completed_ids:
            action = self.active_actions.pop(action_id)
            self.completed_actions[action_id] = action

        # Limit size of completed actions history
        if len(self.completed_actions) > 100:
            # Keep only the most recent 100 actions
            sorted_actions = sorted(self.completed_actions.items(),
                                   key=lambda x: x[1].timestamp,
                                   reverse=True)
            self.completed_actions = dict(sorted_actions[:100])

    def process_input(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an input and coordinate cognitive modules to generate a response.
        This is the main entry point for external systems to interact with the orchestrator.

        Args:
            input_data: The input to process (could be text, structured data, etc.)
            context: Optional context information to guide processing

        Returns:
            Dict containing the processed result and metadata
        """
        # Initialize result structure
        result = {
            "success": False,
            "output": None,
            "metadata": {
                "processing_time": 0,
                "cognitive_state": self.context.state.name,
                "confidence": self.context.confidence
            }
        }

        start_time = time.time()

        try:
            # Update context based on input
            if context:
                self._update_context_from_external(context)

            # Store input in working memory
            if self.working_memory:
                self.working_memory.store_item(
                    content=input_data,
                    source="external_input",
                    importance=0.8
                )

            # Set focus target based on input
            if isinstance(input_data, str):
                self.context.focus_target = input_data[:50]  # Use first 50 chars as focus
            elif isinstance(input_data, dict) and "query" in input_data:
                self.context.focus_target = input_data["query"]

            # Determine input novelty (this would be done by perception in a real system)
            # For demonstration, we use a random value weighted toward moderate novelty
            self.context.novelty = min(1.0, max(0.1, random.gauss(0.5, 0.15)))

            # Generate initial perception action
            self._add_perception_action(priority=0.9)

            # Create a cognitive processing session
            session_id = str(uuid.uuid4())
            session_complete = False
            iteration_count = 0
            max_iterations = 10  # Limit iterations to prevent endless processing

            # Process input using cognitive cycle
            while not session_complete and iteration_count < max_iterations:
                # Process action queue
                self._process_action_queue()

                # Check if processing is complete
                if not self.action_queue and not self.active_actions:
                    session_complete = True

                # Add appropriate actions based on current state
                if not session_complete and len(self.action_queue) < 3:
                    if self.context.state == CognitiveState.EXPLORATION:
                        self._add_perception_action(priority=0.8)
                        self._add_learning_action(priority=0.6)
                    elif self.context.state == CognitiveState.EXPLOITATION:
                        self._add_reasoning_action(priority=0.9, reasoning_type="deductive")
                        self._add_retrieval_action(priority=0.7)
                    elif self.context.state == CognitiveState.DELIBERATION:
                        self._add_reasoning_action(priority=0.9, reasoning_type="causal")
                        self._add_integration_action(priority=0.7)
                    elif self.context.state == CognitiveState.INTUITION:
                        self._add_reasoning_action(priority=0.9, reasoning_type="analogical")

                    # Always consider meta-reflection periodically
                    if iteration_count % 3 == 0:
                        self._add_meta_analysis_action(priority=0.5)

                # Small wait to allow background threads to process
                time.sleep(0.1)
                iteration_count += 1

            # Retrieve results from working memory
            if self.working_memory:
                # Look for reasoning outputs
                reasoning_results = self.working_memory.retrieve(
                    query_text="reasoning",
                    max_results=5
                )

                if reasoning_results:
                    # Combine reasoning results
                    output = []
                    for item in reasoning_results:
                        if item.content:
                            output.append(item.content)

                    result["output"] = output
                    result["success"] = True
                else:
                    # No explicit reasoning results, return whatever is most relevant
                    relevant_items = self.working_memory.retrieve(
                        query_text=self.context.focus_target or "current_task",
                        max_results=3
                    )

                    if relevant_items:
                        result["output"] = [item.content for item in relevant_items]
                        result["success"] = True

            # Update result metadata
            result["metadata"]["processing_time"] = time.time() - start_time
            result["metadata"]["cognitive_state"] = self.context.state.name
            result["metadata"]["confidence"] = self.context.confidence
            result["metadata"]["iterations"] = iteration_count
            self.record_episode(
            goal=self.context.focus_target or "unspecified_task",
            outcome="success" if result["success"] else "failure")

            return result

        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            result["metadata"]["error"] = str(e)
            result["metadata"]["processing_time"] = time.time() - start_time
            self.record_episode(
            goal=self.context.focus_target or "unspecified_task",
            outcome="failure")

            return result

    def record_episode(self, goal: str, outcome: str):
        """Enregistre un épisode cognitif structuré dans une mémoire interne ou externe."""
        try:
            episode = CognitiveEpisode(
                goal=goal,
                trigger_context=self.context,
                actions=list(self.completed_actions.values()),
                outcome=outcome
            )
            # Ajout en mémoire de travail pour trace rapide
            if self.working_memory:
                self.working_memory.store_item(
                    content={
                        "type": "cognitive_episode",
                        "goal": episode.goal,
                        "state": episode.trigger_context.state.name,
                        "confidence": episode.trigger_context.confidence,
                        "actions": [a.action_type for a in episode.actions],
                        "outcome": episode.outcome,
                        "timestamp": episode.timestamp
                    },
                    source="orchestrator_episode",
                    importance=0.85
                )
            self.logger.info(f"Episode enregistré: {goal} → {outcome}")
        except Exception as e:
            self.logger.error(f"Erreur enregistrement épisode: {e}")


    def _update_context_from_external(self, context: Dict[str, Any]):
        """Update cognitive context based on external context info."""
        if "task_complexity" in context:
            self.context.task_complexity = min(1.0, max(0.0, context["task_complexity"]))

        if "time_pressure" in context:
            self.context.time_pressure = min(1.0, max(0.0, context["time_pressure"]))

        if "focus_target" in context:
            self.context.focus_target = context["focus_target"]

        if "cognitive_state" in context:
            try:
                self.context.state = CognitiveState[context["cognitive_state"]]
                self.context.state_duration = 0
            except (KeyError, ValueError):
                self.logger.warning(f"Invalid cognitive state: {context['cognitive_state']}")

    def get_cognitive_status(self) -> Dict[str, Any]:
        """
        Get the current status of the cognitive orchestration system.
        Returns a rich status object for monitoring and debugging.
        """
        status = {
            "cognitive_state": self.context.state.name,
            "context": {
                "task_complexity": self.context.task_complexity,
                "time_pressure": self.context.time_pressure,
                "confidence": self.context.confidence,
                "novelty": self.context.novelty,
                "focus_target": self.context.focus_target,
                "state_duration": self.context.state_duration
            },
            "action_queue_size": len(self.action_queue),
            "active_actions": len(self.active_actions),
            "completed_actions": len(self.completed_actions),
            "metrics": {
                "cognitive_coherence": self.cognitive_coherence,
                "adaptation_rate": self.adaptation_rate,
                "metacognitive_depth": self.metacognitive_depth
            },
            "success_rates": dict(self.success_rate),
            "resource_usage": self.resource_usage_history[-1] if self.resource_usage_history else {}
        }

        return status

           # Module registration function

def create_cognitive_orchestrator(nexus_core, config=None):
    """Factory function to create a CognitiveOrchestrator instance."""
    return CognitiveOrchestrator(nexus_core, config)
