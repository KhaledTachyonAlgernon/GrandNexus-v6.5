"""
cognition_manager.py - Central Cognitive Orchestrator for GrandNexus

This module serves as the central coordinator for all cognitive processes in the GrandNexus
architecture. It orchestrates the interaction between perception, reasoning, learning, and
metacognition modules, creating an emergent cognitive system that is greater than the sum
of its parts.

Key responsibilities:
- Managing cognitive workflows and information flow between modules
- Coordinating perceptual input processing through various filters and analyzers
- Orchestrating reasoning processes using different reasoning strategies
- Facilitating learning from experiences and feedback
- Enabling metacognitive monitoring and adaptation of cognitive processes
- Creating a cognitive workspace where modules can contribute and interact

Design principles:
- Organic interaction rather than rigid control - modules influence each other
- Emergence from simple interaction rules rather than top-down prescription
- Flexible cognitive pathways that can evolve based on experience
- Metacognitive awareness and self-modification capabilities
- Balance between specialization (modularity) and integration (holism)
"""

import logging
import threading
import time
import uuid
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import random
import json
import numpy as np

# Imports from GrandNexus modules
from grandnexus.core.nexus_core import NexusCore
from grandnexus.memory.working import WorkingMemory
from grandnexus.memory.semantic import SemanticMemory
from grandnexus.memory.episodic import EpisodicMemory

# Cognitive modules imports
# Perception
from grandnexus.cognition.perception.perception_core import PerceptionCore
from grandnexus.cognition.perception.symbolic_parser import SymbolicParser
from grandnexus.cognition.perception.code_analyzer import CodeAnalyzer
from grandnexus.cognition.perception.pattern_recognizer import PatternRecognizer

# Reasoning
from grandnexus.cognition.reasoning.reasoning_core import ReasoningCore
from grandnexus.cognition.reasoning.symbolic_engine import SymbolicEngine
from grandnexus.cognition.reasoning.neural_reasoner import NeuralReasoner
from grandnexus.cognition.reasoning.hybrid_inference import HybridInferenceEngine
from grandnexus.cognition.reasoning.planning import PlanningEngine
from grandnexus.cognition.reasoning.uncertainty import UncertaintyManager

# Learning
from grandnexus.cognition.learning.learning_core import LearningCore
from grandnexus.cognition.learning.continuous_learner import ContinuousLearner
from grandnexus.cognition.learning.meta_learner import MetaLearner
from grandnexus.cognition.learning.reinforcement import ReinforcementLearner
from grandnexus.cognition.learning.intrinsic_motivation import IntrinsicMotivation
from grandnexus.cognition.learning.curriculum import CurriculumManager

# Metacognition
from grandnexus.cognition.metacognition.monitor import CognitiveMonitor
from grandnexus.cognition.metacognition.confidence import ConfidenceEvaluator
from grandnexus.cognition.metacognition.adaptation import AdaptationManager
from grandnexus.cognition.metacognition.reflection import Reflector

# Define cognitive event types
class CognitiveEventType(Enum):
    """Types of cognitive events that can occur in the system"""
    PERCEPTION = auto()      # New perception input
    REASONING = auto()       # Reasoning process event
    LEARNING = auto()        # Learning event
    METACOGNITION = auto()   # Metacognitive reflection or adaptation
    GOAL = auto()            # Goal-related event
    ATTENTION = auto()       # Attentional shift
    ACTION = auto()          # Action taken by the system
    ERROR = auto()           # Error condition
    INTEGRATION = auto()     # Cross-module integration event

@dataclass
class CognitiveEvent:
    """Represents a cognitive event in the system"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: CognitiveEventType = CognitiveEventType.PERCEPTION
    timestamp: float = field(default_factory=time.time)
    source_module: str = ""
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5    # Higher values indicate higher priority
    processed: bool = False  # Whether this event has been processed

@dataclass
class CognitiveTask:
    """Represents a cognitive task to be executed"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""      # Type of task (perception, reasoning, etc.)
    description: str = ""
    source: str = ""         # Module that created this task
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # IDs of tasks this depends on
    priority: float = 0.5
    deadline: Optional[float] = None
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None
    created_at: float = field(default_factory=time.time)

@dataclass
class CognitiveState:
    """Represents the current cognitive state of the system"""
    attention_focus: Dict[str, float] = field(default_factory=dict)  # Concept -> activation
    active_goals: List[Dict[str, Any]] = field(default_factory=list)
    active_contexts: Dict[str, float] = field(default_factory=dict)  # Context -> strength
    recent_events: List[CognitiveEvent] = field(default_factory=list)
    working_load: float = 0.0  # Cognitive load measure (0.0-1.0)
    uncertainty: float = 0.0   # Global uncertainty measure (0.0-1.0)
    arousal: float = 0.5       # Cognitive arousal/activation level (0.0-1.0)
    mode: str = "normal"       # Cognitive mode: normal, focused, exploratory, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

from grandnexus.cognition.contracts import (  # réparation 6.5
    CognitionManagerBase, CognitiveModuleBase, CognitiveProcessType, CognitiveMode,
    CognitiveContext, CognitiveRequest, CognitiveResult, cognitive_capability,
)


class CognitionManager(CognitionManagerBase):
    """
    Central orchestrator for GrandNexus's cognitive functions, coordinating
    between perception, reasoning, learning, and metacognitive processes.

    This manager facilitates the emergence of a flexible, adaptive cognitive
    system by establishing pathways for information flow and feedback loops
    between specialized modules.
    """

    def __init__(self, nexus_core: Optional[NexusCore] = None,
                 working_memory: Optional[WorkingMemory] = None,
                 semantic_memory: Optional[SemanticMemory] = None,
                 episodic_memory: Optional[EpisodicMemory] = None):
        """
        Initialize the cognition manager.

        Args:
            nexus_core: Reference to the NexusCore for system-wide coordination
            working_memory: Reference to working memory for temporary storage
            semantic_memory: Reference to semantic memory for structured knowledge
            episodic_memory: Reference to episodic memory for experience-based storage
        """
        self.logger = logging.getLogger("GrandNexus.Cognition.Manager")
        self.nexus_core = nexus_core
        self.working_memory = working_memory
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory

        # Internal state
        self.instance_id = str(uuid.uuid4())[:8]
        self.running = False
        self.initialized = False
        self.lock = threading.RLock()

        # Cognitive state
        self.cognitive_state = CognitiveState()

        # Event management
        self.event_queue = deque()
        self.event_history = deque(maxlen=1000)

        # Task management
        self.task_queue = []  # Heap queue for priority-based tasks
        self.active_tasks = {}  # task_id -> task
        self.completed_tasks = deque(maxlen=100)

        # Module references
        self.perception_modules = {}
        self.reasoning_modules = {}
        self.learning_modules = {}
        self.metacognition_modules = {}

        # Cognitive pathways (dynamic connections between modules)
        self.pathways = defaultdict(float)  # (source, target, event_type) -> strength

        # Metrics tracking
        self.metrics = defaultdict(lambda: deque(maxlen=100))

        self.logger.info(f"CognitionManager initialized with ID={self.instance_id}")

    def initialize(self) -> bool:
        """Initialize all cognitive modules and establish base pathways"""
        if self.initialized:
            return True

        with self.lock:
            try:
                # Initialize all cognitive modules
                self._initialize_perception_modules()
                self._initialize_reasoning_modules()
                self._initialize_learning_modules()
                self._initialize_metacognition_modules()

                # Establish initial cognitive pathways
                self._establish_initial_pathways()

                # Connect to NexusCore if available
                if self.nexus_core:
                    self._register_with_nexus_core()

                self.initialized = True
                self.logger.info("CognitionManager initialization complete")
                return True
            except Exception as e:
                self.logger.error(f"CognitionManager initialization failed: {str(e)}")
                return False

    def start(self) -> bool:
        """Start the cognition manager and begin cognitive processing"""
        if not self.initialized:
            if not self.initialize():
                return False

        with self.lock:
            if self.running:
                return True

            try:
                # Start all cognitive modules
                self._start_perception_modules()
                self._start_reasoning_modules()
                self._start_learning_modules()
                self._start_metacognition_modules()

                # Start cognitive threads
                self._start_cognitive_threads()

                self.running = True
                self.logger.info("CognitionManager started successfully")
                return True
            except Exception as e:
                self.logger.error(f"Failed to start CognitionManager: {str(e)}")
                return False

    def stop(self) -> bool:
        """Stop the cognition manager and all cognitive processes"""
        with self.lock:
            if not self.running:
                return True

            try:
                # Stop all cognitive threads
                self._stop_cognitive_threads()

                # Stop all cognitive modules
                self._stop_metacognition_modules()
                self._stop_learning_modules()
                self._stop_reasoning_modules()
                self._stop_perception_modules()

                self.running = False
                self.logger.info("CognitionManager stopped successfully")
                return True
            except Exception as e:
                self.logger.error(f"Error stopping CognitionManager: {str(e)}")
                return False

    def process_input(self, input_data: Any, input_type: str = "text",
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Process a new input through the cognitive pipeline.

        Args:
            input_data: The input data to process
            input_type: Type of input (text, image, code, etc.)
            metadata: Additional information about the input

        Returns:
            task_id: ID for tracking the processing task
        """
        if not self.initialized:
            self.initialize()

        if metadata is None:
            metadata = {}

        # Create a cognitive task for this input
        task = CognitiveTask(
            task_type="perception",
            description=f"Process {input_type} input",
            source="external",
            parameters={
                "input_data": input_data,
                "input_type": input_type
            },
            priority=metadata.get("priority", 0.5)
        )

        # Create a corresponding cognitive event
        event = CognitiveEvent(
            event_type=CognitiveEventType.PERCEPTION,
            source_module="external",
            content={
                "input_data": input_data,
                "input_type": input_type,
                "task_id": task.task_id
            },
            metadata=metadata,
            priority=metadata.get("priority", 0.5)
        )

        with self.lock:
            # Add the task to the queue
            self._add_task(task)

            # Add the event to the queue
            self.event_queue.append(event)

        self.logger.info(f"Input processing task created: {task.task_id}")
        return task.task_id

    def add_goal(self, description: str, parameters: Optional[Dict[str, Any]] = None,
                priority: float = 0.5, deadline: Optional[float] = None) -> str:
        """
        Add a new goal for the cognitive system to pursue.

        Args:
            description: Description of the goal
            parameters: Additional parameters for the goal
            priority: Goal priority (0.0-1.0)
            deadline: Optional deadline timestamp

        Returns:
            goal_id: ID for tracking the goal
        """
        if not self.initialized:
            self.initialize()

        if parameters is None:
            parameters = {}

        # Create a goal object
        goal_id = str(uuid.uuid4())
        goal = {
            "goal_id": goal_id,
            "description": description,
            "parameters": parameters,
            "priority": priority,
            "deadline": deadline,
            "created_at": time.time(),
            "status": "active"
        }

        # Create a corresponding cognitive event
        event = CognitiveEvent(
            event_type=CognitiveEventType.GOAL,
            source_module="external",
            content={
                "goal": goal,
                "action": "add"
            },
            priority=priority
        )

        with self.lock:
            # Add the goal to active goals
            self.cognitive_state.active_goals.append(goal)

            # Add the event to the queue
            self.event_queue.append(event)

        self.logger.info(f"Goal added: {goal_id} - {description}")
        return goal_id

    def update_goal(self, goal_id: str, status: Optional[str] = None,
                  priority: Optional[float] = None,
                  parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing goal.

        Args:
            goal_id: ID of the goal to update
            status: New status for the goal (active, completed, abandoned)
            priority: New priority for the goal
            parameters: New or updated parameters

        Returns:
            success: Whether the update was successful
        """
        with self.lock:
            # Find the goal
            for i, goal in enumerate(self.cognitive_state.active_goals):
                if goal["goal_id"] == goal_id:
                    # Update fields as specified
                    updates = {}

                    if status is not None:
                        goal["status"] = status
                        updates["status"] = status

                        # If goal is no longer active, we might remove it
                        if status in ["completed", "abandoned"]:
                            self.cognitive_state.active_goals.pop(i)

                    if priority is not None:
                        goal["priority"] = priority
                        updates["priority"] = priority

                    if parameters is not None:
                        goal["parameters"].update(parameters)
                        updates["parameters"] = parameters

                    # Create event for goal update
                    event = CognitiveEvent(
                        event_type=CognitiveEventType.GOAL,
                        source_module="external",
                        content={
                            "goal_id": goal_id,
                            "action": "update",
                            "updates": updates
                        },
                        priority=goal.get("priority", 0.5)
                    )

                    # Add the event to the queue
                    self.event_queue.append(event)

                    self.logger.info(f"Goal updated: {goal_id}")
                    return True

            self.logger.warning(f"Goal not found: {goal_id}")
            return False

    def get_goal_status(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a goal.

        Args:
            goal_id: ID of the goal

        Returns:
            goal_info: Dictionary with goal information
        """
        with self.lock:
            for goal in self.cognitive_state.active_goals:
                if goal["goal_id"] == goal_id:
                    return goal.copy()

            # Check completed tasks for this goal
            for task in self.completed_tasks:
                if task.task_type == "goal" and task.parameters.get("goal_id") == goal_id:
                    return {
                        "goal_id": goal_id,
                        "status": "completed" if task.status == "completed" else "failed",
                        "result": task.result
                    }

            return None

    def get_cognitive_state(self) -> Dict[str, Any]:
        """
        Get a snapshot of the current cognitive state.

        Returns:
            state_dict: Dictionary representation of the cognitive state
        """
        with self.lock:
            # Convert to dictionary and make a deep copy
            state_dict = {
                "attention_focus": dict(self.cognitive_state.attention_focus),
                "active_goals": [goal.copy() for goal in self.cognitive_state.active_goals],
                "active_contexts": dict(self.cognitive_state.active_contexts),
                "recent_events": [
                    {
                        "event_id": e.event_id,
                        "event_type": e.event_type.name,
                        "timestamp": e.timestamp,
                        "source_module": e.source_module,
                        "priority": e.priority,
                        "processed": e.processed
                    }
                    for e in self.cognitive_state.recent_events
                ],
                "working_load": self.cognitive_state.working_load,
                "uncertainty": self.cognitive_state.uncertainty,
                "arousal": self.cognitive_state.arousal,
                "mode": self.cognitive_state.mode,
                "metadata": dict(self.cognitive_state.metadata)
            }

            return state_dict

    def shift_attention(self, focus_items: Dict[str, float]) -> None:
        """
        Shift the system's attention to specific concepts or items.

        Args:
            focus_items: Dictionary mapping concepts to activation levels
        """
        with self.lock:
            # Update attention focus
            for concept, activation in focus_items.items():
                self.cognitive_state.attention_focus[concept] = activation

            # Create a corresponding cognitive event
            event = CognitiveEvent(
                event_type=CognitiveEventType.ATTENTION,
                source_module="external",
                content={
                    "focus_items": focus_items
                },
                priority=0.7  # Attention shifts are important
            )

            # Add the event to the queue
            self.event_queue.append(event)

        self.logger.info(f"Attention shifted to {list(focus_items.keys())}")

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a task.

        Args:
            task_id: ID of the task

        Returns:
            task_info: Dictionary with task information
        """
        with self.lock:
            # Check active tasks
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                return {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "status": task.status,
                    "progress": self._estimate_task_progress(task),
                    "created_at": task.created_at
                }

            # Check task queue
            for task in self.task_queue:
                if task.task_id == task_id:
                    return {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "status": "pending",
                        "progress": 0.0,
                        "created_at": task.created_at
                    }

            # Check completed tasks
            for task in self.completed_tasks:
                if task.task_id == task_id:
                    return {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "status": task.status,
                        "result": task.result,
                        "created_at": task.created_at
                    }

            return None

    def register_module(self, module_type: str, module_name: str,
                       module_instance: Any) -> bool:
        """
        Register a new cognitive module with the manager.

        Args:
            module_type: Type of module (perception, reasoning, learning, metacognition)
            module_name: Name for the module
            module_instance: The module instance to register

        Returns:
            success: Whether registration was successful
        """
        with self.lock:
            if module_type == "perception":
                self.perception_modules[module_name] = module_instance
            elif module_type == "reasoning":
                self.reasoning_modules[module_name] = module_instance
            elif module_type == "learning":
                self.learning_modules[module_name] = module_instance
            elif module_type == "metacognition":
                self.metacognition_modules[module_name] = module_instance
            else:
                self.logger.error(f"Unknown module type: {module_type}")
                return False

            self.logger.info(f"Registered {module_type} module: {module_name}")
            return True

    def add_cognitive_pathway(self, source: str, target: str,
                            event_type: CognitiveEventType,
                            strength: float = 0.5) -> None:
        """
        Add or strengthen a cognitive pathway between modules.

        Args:
            source: Source module name
            target: Target module name
            event_type: Type of event that flows through this pathway
            strength: Connection strength (0.0-1.0)
        """
        with self.lock:
            key = (source, target, event_type)
            self.pathways[key] = strength

            self.logger.info(f"Added pathway: {source} -> {target} ({event_type.name}, {strength:.2f})")

    def submit_cognitive_event(self, event: CognitiveEvent) -> None:
        """
        Submit a cognitive event for processing.

        Args:
            event: The cognitive event to process
        """
        with self.lock:
            self.event_queue.append(event)

            self.logger.debug(f"Event submitted: {event.event_type.name} from {event.source_module}")

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics for cognitive processes.

        Returns:
            metrics_dict: Dictionary of metrics
        """
        with self.lock:
            metrics_dict = {}

            for key, values in self.metrics.items():
                if values:
                    metrics_dict[key] = {
                        "current": values[-1],
                        "average": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values)
                    }

            return metrics_dict

    # -----------------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------------

    def _initialize_perception_modules(self) -> None:
        """Initialize perception modules"""
        try:
            # Initialize PerceptionCore
            perception_core = PerceptionCore(
                nexus_core=self.nexus_core,
                working_memory=self.working_memory,
                semantic_memory=self.semantic_memory
            )
            perception_core.initialize()
            self.perception_modules["perception_core"] = perception_core

            # Initialize SymbolicParser
            symbolic_parser = SymbolicParser(
                nexus_core=self.nexus_core,
                semantic_memory=self.semantic_memory,
                working_memory=self.working_memory
            )
            symbolic_parser.initialize()
            self.perception_modules["symbolic_parser"] = symbolic_parser

            # Initialize CodeAnalyzer
            code_analyzer = CodeAnalyzer(
                nexus_core=self.nexus_core,
                semantic_memory=self.semantic_memory
            )
            code_analyzer.initialize()
            self.perception_modules["code_analyzer"] = code_analyzer

            # Initialize PatternRecognizer
            pattern_recognizer = PatternRecognizer(
                nexus_core=self.nexus_core,
                semantic_memory=self.semantic_memory,
                episodic_memory=self.episodic_memory
            )
            pattern_recognizer.initialize()
            self.perception_modules["pattern_recognizer"] = pattern_recognizer

            self.logger.info("Perception modules initialized")
        except Exception as e:
            self.logger.error(f"Error initializing perception modules: {str(e)}")
            raise

    def _initialize_reasoning_modules(self) -> None:
        """Initialize reasoning modules"""
        try:
            # Initialize ReasoningCore
            reasoning_core = ReasoningCore(
                nexus_core=self.nexus_core,
                working_memory=self.working_memory,
                semantic_memory=self.semantic_memory,
                episodic_memory=self.episodic_memory
            )
            reasoning_core.initialize()
            self.reasoning_modules["reasoning_core"] = reasoning_core

            # Initialize SymbolicEngine
            symbolic_engine = SymbolicEngine(
                nexus_core=self.nexus_core,
                semantic_memory=self.semantic_memory
            )
            symbolic_engine.initialize()
            self.reasoning_modules["symbolic_engine"] = symbolic_engine

            # Initialize NeuralReasoner
            neural_reasoner = NeuralReasoner(
                nexus_core=self.nexus_core,
                semantic_memory=self.semantic_memory
            )
            neural_reasoner.initialize()
            self.reasoning_modules["neural_reasoner"] = neural_reasoner

            # Initialize HybridInferenceEngine
            hybrid_inference = HybridInferenceEngine(
                nexus_core=self.nexus_core,
                symbolic_engine=symbolic_engine,
                neural_reasoner=neural_reasoner
            )
            hybrid_inference.initialize()
            self.reasoning_modules["hybrid_inference"] = hybrid_inference

            # Initialize PlanningEngine
            planning_engine = PlanningEngine(
                nexus_core=self.nexus_core,
                working_memory=self.working_memory,
                semantic_memory=self.semantic_memory
            )
            planning_engine.initialize()
            self.reasoning_modules["planning_engine"] = planning_engine

            # Initialize UncertaintyManager
            uncertainty_manager = UncertaintyManager(
                nexus_core=self.nexus_core
            )
            uncertainty_manager.initialize()
            self.reasoning_modules["uncertainty_manager"] = uncertainty_manager

            self.logger.info("Reasoning modules initialized")
        except Exception as e:
            self.logger.error(f"Error initializing reasoning modules: {str(e)}")
            raise

    def _initialize_learning_modules(self) -> None:
        """Initialize learning modules"""
        try:
            # Initialize LearningCore
            learning_core = LearningCore(
                nexus_core=self.nexus_core,
                working_memory=self.working_memory,
                semantic_memory=self.semantic_memory,
                episodic_memory=self.episodic_memory
            )
            learning_core.initialize()
            self.learning_modules["learning_core"] = learning_core

            # Initialize ContinuousLearner
            continuous_learner = ContinuousLearner(
                nexus_core=self.nexus_core,
                learning_core=learning_core
            )
            continuous_learner.initialize()
            self.learning_modules["continuous_learner"] = continuous_learner

            # Initialize MetaLearner
            meta_learner = MetaLearner(
                nexus_core=self.nexus_core,
                learning_core=learning_core
            )
            meta_learner.initialize()
            self.learning_modules["meta_learner"] = meta_learner

            # Initialize ReinforcementLearner
            reinforcement_learner = ReinforcementLearner(
                nexus_core=self.nexus_core,
                learning_core=learning_core
            )
            reinforcement_learner.initialize()
            self.learning_modules["reinforcement_learner"] = reinforcement_learner

            # Initialize IntrinsicMotivation
            intrinsic_motivation = IntrinsicMotivation(
                nexus_core=self.nexus_core,
                learning_core=learning_core
            )
            intrinsic_motivation.initialize()
            self.learning_modules["intrinsic_motivation"] = intrinsic_motivation

            # Initialize CurriculumManager
            curriculum_manager = CurriculumManager(
                nexus_core=self.nexus_core,
                learning_core=learning_core
            )
            curriculum_manager.initialize()
            self.learning_modules["curriculum_manager"] = curriculum_manager

            self.logger.info("Learning modules initialized")
        except Exception as e:
            self.logger.error(f"Error initializing learning modules: {str(e)}")
            raise

    def _initialize_metacognition_modules(self) -> None:
        """Initialize metacognition modules"""
        try:
            # Initialize CognitiveMonitor
            cognitive_monitor = CognitiveMonitor(
                nexus_core=self.nexus_core
            )
            cognitive_monitor.initialize()
            self.metacognition_modules["cognitive_monitor"] = cognitive_monitor

            # Initialize ConfidenceEvaluator
            confidence_evaluator = ConfidenceEvaluator(
                nexus_core=self.nexus_core
            )
            confidence_evaluator.initialize()
            self.metacognition_modules["confidence_evaluator"] = confidence_evaluator

            # Initialize AdaptationManager
            adaptation_manager = AdaptationManager(
                nexus_core=self.nexus_core
            )
            adaptation_manager.initialize()
            self.metacognition_modules["adaptation_manager"] = adaptation_manager

            # Initialize Reflector
            reflector = Reflector(
                nexus_core=self.nexus_core,
                episodic_memory=self.episodic_memory
            )
            reflector.initialize()
            self.metacognition_modules["reflector"] = reflector

            self.logger.info("Metacognition modules initialized")
        except Exception as e:
            self.logger.error(f"Error initializing metacognition modules: {str(e)}")
            raise

    def _establish_initial_pathways(self) -> None:
        """Establish initial cognitive pathways between modules"""
        # Perception to Reasoning pathways
        self.add_cognitive_pathway("perception_core", "reasoning_core",
                                  CognitiveEventType.PERCEPTION, 0.8)
        self.add_cognitive_pathway("symbolic_parser", "symbolic_engine",
                                  CognitiveEventType.PERCEPTION, 0.9)
        self.add_cognitive_pathway("code_analyzer", "reasoning_core",
                                  CognitiveEventType.PERCEPTION, 0.7)
        self.add_cognitive_pathway("pattern_recognizer", "neural_reasoner",
                                  CognitiveEventType.PERCEPTION, 0.8)

        # Reasoning to Learning pathways
        self.add_cognitive_pathway("reasoning_core", "learning_core",
                                  CognitiveEventType.REASONING, 0.7)
        self.add_cognitive_pathway("symbolic_engine", "learning_core",
                                  CognitiveEventType.REASONING, 0.6)
        self.add_cognitive_pathway("neural_reasoner", "continuous_learner",
                                  CognitiveEventType.REASONING, 0.7)
        self.add_cognitive_pathway("hybrid_inference", "meta_learner",
                                  CognitiveEventType.REASONING, 0.8)

        # Learning to Metacognition pathways
        self.add_cognitive_pathway("learning_core", "cognitive_monitor",
                                  CognitiveEventType.LEARNING, 0.7)
        self.add_cognitive_pathway("continuous_learner", "adaptation_manager",
                                  CognitiveEventType.LEARNING, 0.6)
        self.add_cognitive_pathway("meta_learner", "reflector",
                                  CognitiveEventType.LEARNING, 0.8)

        # Metacognition feedback loops
        self.add_cognitive_pathway("cognitive_monitor", "reasoning_core",
                                  CognitiveEventType.METACOGNITION, 0.7)
        self.add_cognitive_pathway("confidence_evaluator", "uncertainty_manager",
                                  CognitiveEventType.METACOGNITION, 0.9)
        self.add_cognitive_pathway("adaptation_manager", "learning_core",
                                  CognitiveEventType.METACOGNITION, 0.8)
        self.add_cognitive_pathway("reflector", "intrinsic_motivation",
                                  CognitiveEventType.METACOGNITION, 0.7)

        # Goal-related pathways
        self.add_cognitive_pathway("reasoning_core", "planning_engine",
                                  CognitiveEventType.GOAL, 0.9)
        self.add_cognitive_pathway("planning_engine", "reinforcement_learner",
                                  CognitiveEventType.GOAL, 0.7)

        # Attention-related pathways
        self.add_cognitive_pathway("cognitive_monitor", "perception_core",
                                  CognitiveEventType.ATTENTION, 0.8)
        self.add_cognitive_pathway("intrinsic_motivation", "perception_core",
                                  CognitiveEventType.ATTENTION, 0.7)

        self.logger.info("Initial cognitive pathways established")

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
                name="cognition_manager",
                module=self,
                dependencies=dependencies
            )

            # Set up message handlers - specific implementation would depend on NexusCore

            self.logger.info("Successfully registered with NexusCore")
        except Exception as e:
            self.logger.error(f"Failed to register with NexusCore: {str(e)}")

    def _start_perception_modules(self) -> None:
        """Start all perception modules"""
        for name, module in self.perception_modules.items():
            try:
                if hasattr(module, "start") and callable(module.start):
                    module.start()
                    self.logger.info(f"Started perception module: {name}")
            except Exception as e:
                self.logger.error(f"Error starting perception module {name}: {str(e)}")

    def _start_reasoning_modules(self) -> None:
        """Start all reasoning modules"""
        for name, module in self.reasoning_modules.items():
            try:
                if hasattr(module, "start") and callable(module.start):
                    module.start()
                    self.logger.info(f"Started reasoning module: {name}")
            except Exception as e:
                self.logger.error(f"Error starting reasoning module {name}: {str(e)}")

    def _start_learning_modules(self) -> None:
        """Start all learning modules"""
        for name, module in self.learning_modules.items():
            try:
                if hasattr(module, "start") and callable(module.start):
                    module.start()
                    self.logger.info(f"Started learning module: {name}")
            except Exception as e:
                self.logger.error(f"Error starting learning module {name}: {str(e)}")

    def _start_metacognition_modules(self) -> None:
        """Start all metacognition modules"""
        for name, module in self.metacognition_modules.items():
            try:
                if hasattr(module, "start") and callable(module.start):
                    module.start()
                    self.logger.info(f"Started metacognition module: {name}")
            except Exception as e:
                self.logger.error(f"Error starting metacognition module {name}: {str(e)}")

    def _stop_perception_modules(self) -> None:
        """Stop all perception modules"""
        for name, module in self.perception_modules.items():
            try:
                if hasattr(module, "stop") and callable(module.stop):
                    module.stop()
                    self.logger.info(f"Stopped perception module: {name}")
            except Exception as e:
                self.logger.error(f"Error stopping perception module {name}: {str(e)}")

    def _stop_reasoning_modules(self) -> None:
        """Stop all reasoning modules"""
        for name, module in self.reasoning_modules.items():
            try:
                if hasattr(module, "stop") and callable(module.stop):
                    module.stop()
                    self.logger.info(f"Stopped reasoning module: {name}")
            except Exception as e:
                self.logger.error(f"Error stopping reasoning module {name}: {str(e)}")

    def _stop_learning_modules(self) -> None:
        """Stop all learning modules"""
        for name, module in self.learning_modules.items():
            try:
                if hasattr(module, "stop") and callable(module.stop):
                    module.stop()
                    self.logger.info(f"Stopped learning module: {name}")
            except Exception as e:
                self.logger.error(f"Error stopping learning module {name}: {str(e)}")

    def _stop_metacognition_modules(self) -> None:
        """Stop all metacognition modules"""
        for name, module in self.metacognition_modules.items():
            try:
                if hasattr(module, "stop") and callable(module.stop):
                    module.stop()
                    self.logger.info(f"Stopped metacognition module: {name}")
            except Exception as e:
                self.logger.error(f"Error stopping metacognition module {name}: {str(e)}")

    def _start_cognitive_threads(self) -> None:
        """Start cognitive processing threads"""
        # Create and start threads for different cognitive processes
        self.stop_threads = False

        # Event processing thread
        self.event_thread = threading.Thread(
            target=self._event_processing_loop,
            name="event_processor",
            daemon=True
        )
        self.event_thread.start()

        # Task execution thread
        self.task_thread = threading.Thread(
            target=self._task_execution_loop,
            name="task_executor",
            daemon=True
        )
        self.task_thread.start()

        # State update thread
        self.state_thread = threading.Thread(
            target=self._state_update_loop,
            name="state_updater",
            daemon=True
        )
        self.state_thread.start()

        # Metacognitive monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._metacognitive_monitoring_loop,
            name="meta_monitor",
            daemon=True
        )
        self.monitor_thread.start()

        self.logger.info("Cognitive threads started")

    def _stop_cognitive_threads(self) -> None:
        """Stop cognitive processing threads"""
        self.stop_threads = True

        # Wait for threads to exit gracefully
        if hasattr(self, 'event_thread') and self.event_thread.is_alive():
            self.event_thread.join(timeout=2.0)

        if hasattr(self, 'task_thread') and self.task_thread.is_alive():
            self.task_thread.join(timeout=2.0)

        if hasattr(self, 'state_thread') and self.state_thread.is_alive():
            self.state_thread.join(timeout=2.0)

        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)

        self.logger.info("Cognitive threads stopped")

    def _event_processing_loop(self) -> None:
        """Main event processing loop"""
        self.logger.info("Event processing loop started")

        while not self.stop_threads:
            try:
                # Process events in queue
                events_processed = self._process_events_batch()

                # If no events processed, sleep briefly
                if not events_processed:
                    time.sleep(0.01)

            except Exception as e:
                self.logger.error(f"Error in event processing loop: {str(e)}")
                time.sleep(0.1)  # Sleep longer on error

        self.logger.info("Event processing loop stopped")

    def _process_events_batch(self, max_events: int = 10) -> int:
        """
        Process a batch of events from the queue.

        Args:
            max_events: Maximum number of events to process in this batch

        Returns:
            processed_count: Number of events processed
        """
        processed_count = 0

        with self.lock:
            # Process up to max_events
            for _ in range(min(max_events, len(self.event_queue))):
                if not self.event_queue:
                    break

                # Get the next event
                event = self.event_queue.popleft()

                # Add to event history
                self.event_history.append(event)

                # Add to recent events in cognitive state
                self.cognitive_state.recent_events.append(event)
                if len(self.cognitive_state.recent_events) > 20:
                    self.cognitive_state.recent_events.pop(0)

                # Release lock during actual processing to avoid blocking
                self.lock.release()
                try:
                    # Process the event
                    self._process_single_event(event)
                    processed_count += 1
                finally:
                    # Re-acquire lock
                    self.lock.acquire()

        return processed_count

    def _process_single_event(self, event: CognitiveEvent) -> None:
        """
        Process a single cognitive event.

        Args:
            event: The event to process
        """
        try:
            # Update metrics
            self.metrics["events_processed"].append(1.0)

            # Mark as processed
            event.processed = True

            # Find relevant pathways for this event
            pathways = []
            for (source, target, event_type), strength in self.pathways.items():
                if source == event.source_module and event_type == event.event_type:
                    pathways.append((target, strength))

            # Route event to target modules based on pathways
            for target, strength in pathways:
                # Check if strength exceeds threshold based on event priority
                threshold = 0.3 - (event.priority * 0.2)  # Lower threshold for higher priority

                if strength >= threshold:
                    self._route_event_to_module(event, target)

            # Special handling based on event type
            if event.event_type == CognitiveEventType.PERCEPTION:
                self._handle_perception_event(event)
            elif event.event_type == CognitiveEventType.REASONING:
                self._handle_reasoning_event(event)
            elif event.event_type == CognitiveEventType.LEARNING:
                self._handle_learning_event(event)
            elif event.event_type == CognitiveEventType.METACOGNITION:
                self._handle_metacognition_event(event)
            elif event.event_type == CognitiveEventType.GOAL:
                self._handle_goal_event(event)
            elif event.event_type == CognitiveEventType.ATTENTION:
                self._handle_attention_event(event)

        except Exception as e:
            self.logger.error(f"Error processing event {event.event_id}: {str(e)}")

            # Generate an error event
            error_event = CognitiveEvent(
                event_type=CognitiveEventType.ERROR,
                source_module="cognition_manager",
                content={
                    "error": str(e),
                    "original_event": {
                        "event_id": event.event_id,
                        "event_type": event.event_type.name,
                        "source_module": event.source_module
                    }
                },
                priority=0.8  # Errors are high priority
            )

            # Add error event to queue
            with self.lock:
                self.event_queue.append(error_event)

    def _route_event_to_module(self, event: CognitiveEvent, target_module: str) -> None:
        """
        Route an event to a specific module.

        Args:
            event: The event to route
            target_module: The target module name
        """
        # Find the target module in the appropriate category
        module = None

        if target_module in self.perception_modules:
            module = self.perception_modules[target_module]
        elif target_module in self.reasoning_modules:
            module = self.reasoning_modules[target_module]
        elif target_module in self.learning_modules:
            module = self.learning_modules[target_module]
        elif target_module in self.metacognition_modules:
            module = self.metacognition_modules[target_module]

        if module is None:
            self.logger.warning(f"Target module not found: {target_module}")
            return

        # Check if module has event handling capability
        if hasattr(module, "handle_event") and callable(module.handle_event):
            try:
                module.handle_event(event)
                self.logger.debug(f"Routed event {event.event_id} to {target_module}")
            except Exception as e:
                self.logger.error(f"Error routing event to {target_module}: {str(e)}")
        else:
            self.logger.warning(f"Module {target_module} cannot handle events")

    def _handle_perception_event(self, event: CognitiveEvent) -> None:
        """
        Handle a perception event.

        Args:
            event: The perception event to handle
        """
        # Check if this is a new input to process
        if event.source_module == "external" and "input_data" in event.content:
            # Extract information from event
            input_data = event.content["input_data"]
            input_type = event.content.get("input_type", "text")
            task_id = event.content.get("task_id")

            # Create a task to process this input if not already created
            if task_id is None:
                task = CognitiveTask(
                    task_type="perception",
                    description=f"Process {input_type} input",
                    source="external",
                    parameters={
                        "input_data": input_data,
                        "input_type": input_type
                    },
                    priority=event.priority
                )

                with self.lock:
                    self._add_task(task)
                    task_id = task.task_id

            # Route to appropriate perception module based on input type
            if input_type == "text":
                if "perception_core" in self.perception_modules:
                    self.perception_modules["perception_core"].process_text(
                        input_data,
                        metadata={"task_id": task_id, "priority": event.priority}
                    )
            elif input_type == "symbolic":
                if "symbolic_parser" in self.perception_modules:
                    self.perception_modules["symbolic_parser"].parse(
                        input_data,
                        context={"task_id": task_id}
                    )
            elif input_type == "code":
                if "code_analyzer" in self.perception_modules:
                    self.perception_modules["code_analyzer"].analyze(
                        input_data,
                        {"task_id": task_id}
                    )
            elif input_type == "pattern":
                if "pattern_recognizer" in self.perception_modules:
                    self.perception_modules["pattern_recognizer"].recognize(
                        input_data,
                        {"task_id": task_id}
                    )
            else:
                self.logger.warning(f"Unsupported input type: {input_type}")

    def _handle_reasoning_event(self, event: CognitiveEvent) -> None:
        """
        Handle a reasoning event.

        Args:
            event: The reasoning event to handle
        """
        # Implementation would depend on specific reasoning events
        pass

    def _handle_learning_event(self, event: CognitiveEvent) -> None:
        """
        Handle a learning event.

        Args:
            event: The learning event to handle
        """
        # Implementation would depend on specific learning events
        pass

    def _handle_metacognition_event(self, event: CognitiveEvent) -> None:
        """
        Handle a metacognition event.

        Args:
            event: The metacognition event to handle
        """
        # Implementation would depend on specific metacognition events
        pass

    def _handle_goal_event(self, event: CognitiveEvent) -> None:
        """
        Handle a goal-related event.

        Args:
            event: The goal event to handle
        """
        if "action" in event.content:
            action = event.content["action"]

            if action == "add":
                goal = event.content.get("goal")
                if goal and "goal_id" in goal:
                    # Update cognitive state with new goal
                    with self.lock:
                        # Check if goal already exists
                        for existing in self.cognitive_state.active_goals:
                            if existing["goal_id"] == goal["goal_id"]:
                                return

                        # Add new goal
                        self.cognitive_state.active_goals.append(goal)

                    # Route to planning engine
                    if "planning_engine" in self.reasoning_modules:
                        self.reasoning_modules["planning_engine"].add_goal(goal)

            elif action == "update":
                goal_id = event.content.get("goal_id")
                updates = event.content.get("updates", {})

                if goal_id:
                    # Update goal in cognitive state
                    with self.lock:
                        for goal in self.cognitive_state.active_goals:
                            if goal["goal_id"] == goal_id:
                                # Apply updates
                                for key, value in updates.items():
                                    if key == "status" and value in ["completed", "abandoned"]:
                                        # If goal is completed or abandoned, might remove it
                                        self.cognitive_state.active_goals.remove(goal)
                                        break
                                    elif isinstance(value, dict) and isinstance(goal.get(key, {}), dict):
                                        # Merge dictionaries for nested updates
                                        goal.setdefault(key, {}).update(value)
                                    else:
                                        goal[key] = value
                                break

                    # Route to planning engine
                    if "planning_engine" in self.reasoning_modules:
                        self.reasoning_modules["planning_engine"].update_goal(goal_id, updates)

    def _handle_attention_event(self, event: CognitiveEvent) -> None:
        """
        Handle an attention-related event.

        Args:
            event: The attention event to handle
        """
        if "focus_items" in event.content:
            focus_items = event.content["focus_items"]

            # Update attention focus in cognitive state
            with self.lock:
                # Apply decay to existing focus items
                for concept in list(self.cognitive_state.attention_focus.keys()):
                    self.cognitive_state.attention_focus[concept] *= 0.8

                    # Remove concepts with negligible activation
                    if self.cognitive_state.attention_focus[concept] < 0.1:
                        del self.cognitive_state.attention_focus[concept]

                # Add or update new focus items
                for concept, activation in focus_items.items():
                    if concept in self.cognitive_state.attention_focus:
                        # Combine activations (max)
                        self.cognitive_state.attention_focus[concept] = max(
                            self.cognitive_state.attention_focus[concept],
                            activation
                        )
                    else:
                        # Add new focus item
                        self.cognitive_state.attention_focus[concept] = activation

            # Notify perception modules of attention shift
            if "perception_core" in self.perception_modules:
                self.perception_modules["perception_core"].shift_attention(focus_items)

    def _task_execution_loop(self) -> None:
        """Main task execution loop"""
        self.logger.info("Task execution loop started")

        while not self.stop_threads:
            try:
                # Get the highest priority task
                task = self._get_next_task()

                if task:
                    # Execute the task
                    self._execute_task(task)
                else:
                    # No tasks available, sleep briefly
                    time.sleep(0.01)

            except Exception as e:
                self.logger.error(f"Error in task execution loop: {str(e)}")
                time.sleep(0.1)  # Sleep longer on error

        self.logger.info("Task execution loop stopped")

    def _get_next_task(self) -> Optional[CognitiveTask]:
        """
        Get the next task to execute based on priority and dependencies.

        Returns:
            task: The next task to execute, or None if no tasks are ready
        """
        with self.lock:
            # Check if there are tasks in the queue
            if not self.task_queue:
                return None

            # Find the highest priority task that has all dependencies satisfied
            eligible_tasks = []
            for task in self.task_queue:
                # Check if all dependencies are satisfied
                dependencies_satisfied = True

                for dep_id in task.dependencies:
                    # Check if dependency is complete
                    dep_complete = False

                    # Check in completed tasks
                    for completed_task in self.completed_tasks:
                        if completed_task.task_id == dep_id and completed_task.status == "completed":
                            dep_complete = True
                            break

                    if not dep_complete:
                        dependencies_satisfied = False
                        break

                if dependencies_satisfied:
                    eligible_tasks.append(task)

            if not eligible_tasks:
                return None

            # Sort by priority (higher value = higher priority)
            eligible_tasks.sort(key=lambda t: t.priority, reverse=True)

            # Get the highest priority task
            selected_task = eligible_tasks[0]

            # Remove from queue
            self.task_queue.remove(selected_task)

            # Mark as active
            selected_task.status = "running"
            self.active_tasks[selected_task.task_id] = selected_task

            return selected_task

    def _add_task(self, task: CognitiveTask) -> None:
        """
        Add a task to the queue.

        Args:
            task: The task to add
        """
        self.task_queue.append(task)
        self.logger.debug(f"Task added to queue: {task.task_id}")

    def _execute_task(self, task: CognitiveTask) -> None:
        """
        Execute a cognitive task.

        Args:
            task: The task to execute
        """
        self.logger.info(f"Executing task: {task.task_id} - {task.description}")

        try:
            # Update metrics
            self.metrics["tasks_executed"].append(1.0)

            # Execute based on task type
            if task.task_type == "perception":
                self._execute_perception_task(task)
            elif task.task_type == "reasoning":
                self._execute_reasoning_task(task)
            elif task.task_type == "learning":
                self._execute_learning_task(task)
            elif task.task_type == "metacognition":
                self._execute_metacognition_task(task)
            elif task.task_type == "goal":
                self._execute_goal_task(task)
            else:
                self.logger.warning(f"Unknown task type: {task.task_type}")

            # Mark task as completed
            with self.lock:
                task.status = "completed"
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                self.completed_tasks.append(task)

            self.logger.info(f"Task completed: {task.task_id}")

            # Generate task completion event
            completion_event = CognitiveEvent(
                event_type=CognitiveEventType.INTEGRATION,
                source_module="cognition_manager",
                content={
                    "completed_task": {
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "result": task.result
                    }
                }
            )

            # Add event to queue
            with self.lock:
                self.event_queue.append(completion_event)

        except Exception as e:
            self.logger.error(f"Error executing task {task.task_id}: {str(e)}")

            # Mark task as failed
            with self.lock:
                task.status = "failed"
                task.result = {"error": str(e)}
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                self.completed_tasks.append(task)

            # Generate error event
            error_event = CognitiveEvent(
                event_type=CognitiveEventType.ERROR,
                source_module="cognition_manager",
                content={
                    "error": str(e),
                    "failed_task": {
                        "task_id": task.task_id,
                        "task_type": task.task_type
                    }
                },
                priority=0.7
            )

            # Add error event to queue
            with self.lock:
                self.event_queue.append(error_event)

    def _execute_perception_task(self, task: CognitiveTask) -> None:
        """
        Execute a perception task.

        Args:
            task: The perception task to execute
        """
        input_data = task.parameters.get("input_data")
        input_type = task.parameters.get("input_type", "text")

        if input_data is None:
            raise ValueError("No input data provided")

        # Default result is empty
        task.result = {}

        # Process based on input type
        if input_type == "text":
            if "perception_core" in self.perception_modules:
                result = self.perception_modules["perception_core"].process_text(
                    input_data,
                    metadata={"task_id": task.task_id}
                )
                task.result["perception"] = result
        elif input_type == "symbolic":
            if "symbolic_parser" in self.perception_modules:
                result = self.perception_modules["symbolic_parser"].parse(
                    input_data,
                    context={"task_id": task.task_id}
                )
                task.result["parsed"] = result
        elif input_type == "code":
            if "code_analyzer" in self.perception_modules:
                result = self.perception_modules["code_analyzer"].analyze(
                    input_data,
                    {"task_id": task.task_id}
                )
                task.result["code_analysis"] = result
        elif input_type == "pattern":
            if "pattern_recognizer" in self.perception_modules:
                result = self.perception_modules["pattern_recognizer"].recognize(
                    input_data,
                    {"task_id": task.task_id}
                )
                task.result["patterns"] = result
        else:
            raise ValueError(f"Unsupported input type: {input_type}")

    def _execute_reasoning_task(self, task: CognitiveTask) -> None:
        """
        Execute a reasoning task.

        Args:
            task: The reasoning task to execute
        """
        # Implementation would depend on reasoning task types
        pass

    def _execute_learning_task(self, task: CognitiveTask) -> None:
        """
        Execute a learning task.

        Args:
            task: The learning task to execute
        """
        # Implementation would depend on learning task types
        pass

    def _execute_metacognition_task(self, task: CognitiveTask) -> None:
        """
        Execute a metacognition task.

        Args:
            task: The metacognition task to execute
        """
        task_subtype = task.parameters.get("subtype")

        if task_subtype == "monitor":
            # Execute monitoring task
            if "cognitive_monitor" in self.metacognition_modules:
                result = self.metacognition_modules["cognitive_monitor"].perform_monitoring(
                    task.parameters.get("target_modules", []),
                    task.parameters.get("metrics", [])
                )
                task.result = {"monitoring_results": result}

        elif task_subtype == "evaluate_confidence":
            # Execute confidence evaluation task
            if "confidence_evaluator" in self.metacognition_modules:
                result = self.metacognition_modules["confidence_evaluator"].evaluate(
                    task.parameters.get("content"),
                    task.parameters.get("context", {})
                )
                task.result = {"confidence": result}

        elif task_subtype == "adapt":
            # Execute adaptation task
            if "adaptation_manager" in self.metacognition_modules:
                result = self.metacognition_modules["adaptation_manager"].adapt(
                    task.parameters.get("module"),
                    task.parameters.get("parameters", {}),
                    task.parameters.get("context", {})
                )
                task.result = {"adaptation_result": result}

        elif task_subtype == "reflect":
            # Execute reflection task
            if "reflector" in self.metacognition_modules:
                result = self.metacognition_modules["reflector"].reflect_on(
                    task.parameters.get("topic"),
                    task.parameters.get("depth", 1),
                    task.parameters.get("context", {})
                )
                task.result = {"reflection": result}

        else:
            raise ValueError(f"Unknown metacognition task subtype: {task_subtype}")

    def _execute_goal_task(self, task: CognitiveTask) -> None:
        """
        Execute a goal-related task.

        Args:
            task: The goal task to execute
        """
        action = task.parameters.get("action")
        goal_id = task.parameters.get("goal_id")

        if action == "plan":
            # Create plan for a goal
            if "planning_engine" in self.reasoning_modules:
                plan = self.reasoning_modules["planning_engine"].create_plan(
                    goal_id,
                    task.parameters.get("constraints", {})
                )
                task.result = {"plan": plan}

        elif action == "execute_step":
            # Execute a step in a plan
            if "planning_engine" in self.reasoning_modules:
                step_result = self.reasoning_modules["planning_engine"].execute_step(
                    goal_id,
                    task.parameters.get("step_id")
                )
                task.result = {"step_result": step_result}

        elif action == "evaluate":
            # Evaluate progress on a goal
            goal_progress = self._evaluate_goal_progress(goal_id)
            task.result = {"progress": goal_progress}

        elif action == "revise":
            # Revise a goal or plan
            if "planning_engine" in self.reasoning_modules:
                revised_plan = self.reasoning_modules["planning_engine"].revise_plan(
                    goal_id,
                    task.parameters.get("reason"),
                    task.parameters.get("constraints", {})
                )
                task.result = {"revised_plan": revised_plan}

        else:
            raise ValueError(f"Unknown goal task action: {action}")

    def _evaluate_goal_progress(self, goal_id: str) -> Dict[str, Any]:
        """
        Evaluate progress on a goal.

        Args:
            goal_id: ID of the goal to evaluate

        Returns:
            progress_info: Dictionary with progress information
        """
        # Find the goal
        goal = None
        with self.lock:
            for g in self.cognitive_state.active_goals:
                if g["goal_id"] == goal_id:
                    goal = g
                    break

        if goal is None:
            return {"error": f"Goal not found: {goal_id}"}

        # Check if goal has a plan
        if "planning_engine" in self.reasoning_modules:
            plan = self.reasoning_modules["planning_engine"].get_plan(goal_id)

            if plan:
                # Calculate progress based on completed steps
                completed_steps = sum(1 for step in plan.get("steps", [])
                                    if step.get("status") == "completed")
                total_steps = len(plan.get("steps", []))

                if total_steps > 0:
                    progress_pct = (completed_steps / total_steps) * 100
                else:
                    progress_pct = 0.0

                return {
                    "goal_id": goal_id,
                    "has_plan": True,
                    "completed_steps": completed_steps,
                    "total_steps": total_steps,
                    "progress_pct": progress_pct,
                    "status": goal.get("status", "active")
                }

        # No plan available
        return {
            "goal_id": goal_id,
            "has_plan": False,
            "status": goal.get("status", "active")
        }

    def _state_update_loop(self) -> None:
        """Loop to periodically update the cognitive state"""
        self.logger.info("State update loop started")

        while not self.stop_threads:
            try:
                # Update the cognitive state
                self._update_cognitive_state()

                # Sleep for a short time
                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error in state update loop: {str(e)}")
                time.sleep(0.5)  # Sleep longer on error

        self.logger.info("State update loop stopped")

    def _update_cognitive_state(self) -> None:
        """Update the cognitive state based on current system state"""
        with self.lock:
            # Update working load based on queue lengths
            queue_size = len(self.event_queue) + len(self.task_queue) + len(self.active_tasks)
            max_queue_size = 200  # Arbitrary maximum for normalization

            self.cognitive_state.working_load = min(1.0, queue_size / max_queue_size)

            # Update arousal based on recent activity and importance
            recent_activity = len(self.cognitive_state.recent_events)
            avg_importance = (sum(goal.get("priority", 0.5) for goal in self.cognitive_state.active_goals) /
                            max(1, len(self.cognitive_state.active_goals)))

            self.cognitive_state.arousal = 0.3 * (recent_activity / 20) + 0.7 * avg_importance

            # Decay old context activations
            for context in list(self.cognitive_state.active_contexts.keys()):
                self.cognitive_state.active_contexts[context] *= 0.95

                # Remove contexts with negligible activation
                if self.cognitive_state.active_contexts[context] < 0.05:
                    del self.cognitive_state.active_contexts[context]

            # Update uncertainty if confidence evaluator is available
            if "confidence_evaluator" in self.metacognition_modules:
                try:
                    # Get current uncertainty estimate
                    uncertainty = self.metacognition_modules["confidence_evaluator"].get_global_uncertainty()
                    self.cognitive_state.uncertainty = uncertainty
                except Exception as e:
                    self.logger.warning(f"Error getting uncertainty: {str(e)}")

    def _metacognitive_monitoring_loop(self) -> None:
        """Loop to perform periodic metacognitive monitoring"""
        self.logger.info("Metacognitive monitoring loop started")

        monitor_interval = 1.0  # seconds
        last_monitor_time = time.time()

        while not self.stop_threads:
            try:
                current_time = time.time()

                # Check if it's time to monitor
                if current_time - last_monitor_time >= monitor_interval:
                    self._perform_metacognitive_monitoring()
                    last_monitor_time = current_time

                # Sleep for a short time
                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error in metacognitive monitoring loop: {str(e)}")
                time.sleep(1.0)  # Sleep longer on error

        self.logger.info("Metacognitive monitoring loop stopped")

    def _perform_metacognitive_monitoring(self) -> None:
        """Perform metacognitive monitoring of the system"""
        try:
            # Skip if cognitive monitor is not available
            if "cognitive_monitor" not in self.metacognition_modules:
                return

            # Monitor active modules
            monitor_results = self.metacognition_modules["cognitive_monitor"].quick_scan()

            # Check for issues
            issues = monitor_results.get("issues", [])

            for issue in issues:
                severity = issue.get("severity", "info")

                if severity in ["warning", "error"]:
                    # Create metacognition event for this issue
                    event = CognitiveEvent(
                        event_type=CognitiveEventType.METACOGNITION,
                        source_module="cognitive_monitor",
                        content={"issue": issue},
                        priority=0.7 if severity == "error" else 0.5
                    )

                    with self.lock:
                        self.event_queue.append(event)

                    # Log the issue
                    self.logger.warning(f"Metacognitive issue detected: {issue}")

            # Update cognitive state with monitoring results
            with self.lock:
                self.cognitive_state.metadata["last_monitoring"] = {
                    "timestamp": time.time(),
                    "summary": monitor_results.get("summary", {})
                }

        except Exception as e:
            self.logger.error(f"Error in metacognitive monitoring: {str(e)}")

    def _estimate_task_progress(self, task: CognitiveTask) -> float:
        """
        Estimate the progress of a task.

        Args:
            task: The task to estimate progress for

        Returns:
            progress: Estimated progress as a fraction (0.0-1.0)
        """
        # For now, just use a simple heuristic based on task type and time elapsed
        if task.status == "completed":
            return 1.0
        elif task.status == "failed":
            return 0.0
        elif task.status == "pending":
            return 0.0

        # For running tasks, estimate based on elapsed time
        time_elapsed = time.time() - task.created_at

        # Estimate expected duration based on task type
        if task.task_type == "perception":
            expected_duration = 2.0  # seconds
        elif task.task_type == "reasoning":
            expected_duration = 5.0
        elif task.task_type == "learning":
            expected_duration = 10.0
        elif task.task_type == "metacognition":
            expected_duration = 3.0
        elif task.task_type == "goal":
            expected_duration = 7.0
        else:
            expected_duration = 5.0

        # Calculate progress fraction
        progress = min(0.95, time_elapsed / expected_duration)

        return progress