"""
reflection.py - Self-Analysis and Cognitive Process Reflection Module for GrandNexus

This module implements meta-cognitive self-reflection capabilities, allowing
the system to analyze its own reasoning processes, decisions, and cognitive
operations. It supports both automatic periodic reflection and on-demand
reflection triggered by specific events or confidence thresholds.

Key capabilities:
- Process trace analysis of reasoning and decision chains
- Detection of cognitive biases and reasoning errors
- Explanatory generation for system decisions and actions
- Identification of knowledge gaps and uncertainty sources
- Abstraction of lessons learned from episodic experiences
- Strategic adjustment of cognitive processes based on reflection outcomes

The module works closely with monitor.py (for process monitoring),
confidence.py (for uncertainty assessment), and adaption.py (for parameter
adjustment) to form the complete metacognitive subsystem of GrandNexus.
"""

import logging
import time
import threading
import uuid
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from collections import deque, defaultdict
import re
import json

# Assuming these modules exist in the GrandNexus framework
from grandnexus.core.nexus_core import NexusCore
from grandnexus.memory.episodic import EpisodicMemory
from grandnexus.memory.semantic import SemanticMemory
from grandnexus.memory.working import WorkingMemory
from grandnexus.metacognition.monitor import MetaMonitor
from grandnexus.metacognition.confidence import ConfidenceEstimator
from grandnexus.metacognition.adaptation import AdaptationEngine

# Reflection types
class ReflectionType(Enum):
    """Types of reflective analysis"""
    PROCESS_TRACE = auto()      # Analysis of cognitive process traces
    DECISION_ANALYSIS = auto()   # Analysis of decision-making
    ERROR_ANALYSIS = auto()      # Analysis of errors or failures
    KNOWLEDGE_GAP = auto()       # Identification of knowledge gaps
    BIAS_DETECTION = auto()      # Detection of cognitive biases
    STRATEGY_EVALUATION = auto() # Evaluation of cognitive strategies
    PERFORMANCE_ANALYSIS = auto() # Analysis of system performance
    MODEL_EVALUATION = auto()    # Evaluation of internal models
    EXPLANATION_GENERATION = auto() # Generation of explanations
    LESSON_EXTRACTION = auto()   # Extraction of lessons learned

# Targets of reflection (what is being reflected upon)
class ReflectionTarget(Enum):
    """Targets of reflection"""
    REASONING_TRACE = auto()     # A specific reasoning process
    DECISION = auto()            # A specific decision
    ERROR = auto()               # A specific error
    KNOWLEDGE = auto()           # Knowledge state
    BIAS = auto()                # Potential cognitive bias
    STRATEGY = auto()            # Cognitive strategy
    PERFORMANCE = auto()         # System performance
    MODEL = auto()               # Internal model
    EXPLANATION = auto()         # Explanation generation
    LESSON = auto()              # Lesson learned

@dataclass
class ReflectionTrigger:
    """Represents an event or condition that triggers reflection"""
    trigger_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    target_type: ReflectionTarget = ReflectionTarget.REASONING_TRACE
    reflection_type: ReflectionType = ReflectionType.PROCESS_TRACE
    condition: Optional[Callable] = None  # Function that evaluates if reflection is needed
    priority: float = 0.5  # 0.0 to 1.0
    enabled: bool = True
    cooldown: float = 60.0  # Minimum seconds between triggers of this type
    last_triggered: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReflectionTask:
    """A reflection task to be processed"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target_type: ReflectionTarget = ReflectionTarget.REASONING_TRACE
    reflection_type: ReflectionType = ReflectionType.PROCESS_TRACE
    target_data: Any = None  # The data being reflected upon
    context: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    trigger_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReflectionResult:
    """The result of a reflection task"""
    task_id: str
    success: bool = False
    insights: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    completion_time: float = field(default_factory=time.time)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class ReflectionStrategy:
    """Abstract base class for reflection strategies"""
    def __init__(self, name: str, reflection_type: ReflectionType):
        self.name = name
        self.reflection_type = reflection_type
        self.logger = logging.getLogger(f"GrandNexus.Metacognition.Reflection.Strategy.{name}")

    def can_handle(self, task: ReflectionTask) -> bool:
        """Check if this strategy can handle the given task"""
        return task.reflection_type == self.reflection_type

    def reflect(self, task: ReflectionTask) -> ReflectionResult:
        """
        Perform reflection on the given task.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

class ReflectionModule:
    """
    Core module for system self-reflection and meta-cognitive analysis.

    This module enables GrandNexus to analyze its own cognitive processes,
    detect errors or inefficiencies, generate explanations, and learn from
    its experiences to improve its operations over time.
    """

    def __init__(self, nexus_core: Optional[NexusCore] = None,
                 meta_monitor: Optional[MetaMonitor] = None,
                 confidence_estimator: Optional[ConfidenceEstimator] = None,
                 adaptation_engine: Optional[AdaptationEngine] = None,
                 episodic_memory: Optional[EpisodicMemory] = None,
                 semantic_memory: Optional[SemanticMemory] = None,
                 working_memory: Optional[WorkingMemory] = None):
        """
        Initialize the reflection module.

        Args:
            nexus_core: Reference to the NexusCore for system-wide coordination
            meta_monitor: Reference to the metacognitive monitoring module
            confidence_estimator: Reference to the confidence estimation module
            adaptation_engine: Reference to the parameter adaptation engine
            episodic_memory: Reference to episodic memory for reflection
            semantic_memory: Reference to semantic memory for reflection
            working_memory: Reference to working memory for reflection context
        """
        self.logger = logging.getLogger("GrandNexus.Metacognition.Reflection")
        self.nexus_core = nexus_core
        self.meta_monitor = meta_monitor
        self.confidence_estimator = confidence_estimator
        self.adaptation_engine = adaptation_engine
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.working_memory = working_memory

        # Core state
        self.instance_id = str(uuid.uuid4())[:8]
        self.running = False
        self.initialized = False
        self.lock = threading.RLock()

        # Task management
        self.current_tasks = set()
        self.pending_tasks = []  # Priority queue (heapq)
        self.completed_tasks = deque(maxlen=100)

        # Reflection strategies
        self.strategies: Dict[ReflectionType, List[ReflectionStrategy]] = {
            rtype: [] for rtype in ReflectionType
        }

        # Reflection triggers
        self.triggers: Dict[str, ReflectionTrigger] = {}

        # Reflection history
        self.reflection_history = deque(maxlen=1000)

        # Thread pool for parallel reflection
        self.thread_pool = []
        self.max_threads = 4

        # LLM integration - if the system has LLM capabilities
        self.llm_available = False
        self.llm_module = None

        # Learning from reflection
        self.lessons_learned = []

        # Performance metrics
        self.metrics = defaultdict(lambda: deque(maxlen=100))

        self.logger.info(f"ReflectionModule initialized with ID={self.instance_id}")

    def initialize(self) -> bool:
        """Initialize the reflection module"""
        if self.initialized:
            return True

        with self.lock:
            try:
                # Register with NexusCore if available
                if self.nexus_core:
                    self._register_with_nexus_core()

                # Register default reflection strategies
                self._register_default_strategies()

                # Register default reflection triggers
                self._register_default_triggers()

                # Check for LLM availability
                self._check_llm_availability()

                self.initialized = True
                self.logger.info("ReflectionModule initialization complete")
                return True
            except Exception as e:
                self.logger.error(f"ReflectionModule initialization failed: {str(e)}")
                return False

    def start(self) -> bool:
        """Start the reflection module"""
        if not self.initialized:
            if not self.initialize():
                return False

        with self.lock:
            if self.running:
                return True

            try:
                # Start the reflection background thread
                self.running = True
                self.reflection_thread = threading.Thread(
                    target=self._reflection_loop,
                    daemon=True
                )
                self.reflection_thread.start()

                self.logger.info("ReflectionModule started successfully")
                return True
            except Exception as e:
                self.logger.error(f"Failed to start ReflectionModule: {str(e)}")
                return False

    def stop(self) -> bool:
        """Stop the reflection module"""
        with self.lock:
            if not self.running:
                return True

            try:
                # Signal the reflection thread to stop
                self.running = False

                # Wait for the thread to terminate (with timeout)
                if hasattr(self, 'reflection_thread'):
                    self.reflection_thread.join(timeout=5.0)

                self.logger.info("ReflectionModule stopped successfully")
                return True
            except Exception as e:
                self.logger.error(f"Error stopping ReflectionModule: {str(e)}")
                return False

    def register_strategy(self, strategy: ReflectionStrategy) -> bool:
        """
        Register a reflection strategy.

        Args:
            strategy: The strategy to register

        Returns:
            bool: True if registration succeeded, False otherwise
        """
        with self.lock:
            rtype = strategy.reflection_type
            self.strategies[rtype].append(strategy)
            self.logger.info(f"Registered reflection strategy: {strategy.name} for {rtype.name}")
            return True

    def register_trigger(self, trigger: ReflectionTrigger) -> bool:
        """
        Register a reflection trigger.

        Args:
            trigger: The trigger to register

        Returns:
            bool: True if registration succeeded, False otherwise
        """
        with self.lock:
            if trigger.trigger_id in self.triggers:
                self.logger.warning(f"Trigger {trigger.trigger_id} already exists, updating")

            self.triggers[trigger.trigger_id] = trigger
            self.logger.info(f"Registered reflection trigger: {trigger.name}")
            return True

    def submit_reflection_task(self, task: ReflectionTask) -> str:
        """
        Submit a reflection task for processing.

        Args:
            task: The ReflectionTask to process

        Returns:
            str: The task ID for tracking
        """
        with self.lock:
            # Use heapq to maintain a priority queue
            import heapq

            # Create a tuple of (priority, creation time, task) for sorting
            # We negate priority because heapq is a min-heap (lower values come first)
            # but we want higher priority tasks to come first
            entry = (-task.priority, task.created_at, task)
            heapq.heappush(self.pending_tasks, entry)

            self.logger.info(f"Reflection task submitted: {task.task_id} - {task.reflection_type.name}")
            return task.task_id

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a reflection task"""
        with self.lock:
            # Check current tasks
            for task in self.current_tasks:
                if task.task_id == task_id:
                    return {
                        "task_id": task_id,
                        "status": "in_progress",
                        "type": task.reflection_type.name,
                        "created_at": task.created_at
                    }

            # Check pending tasks
            for _, _, task in self.pending_tasks:
                if task.task_id == task_id:
                    return {
                        "task_id": task_id,
                        "status": "pending",
                        "type": task.reflection_type.name,
                        "created_at": task.created_at
                    }

            # Check completed tasks
            for result in self.completed_tasks:
                if result.task_id == task_id:
                    return {
                        "task_id": task_id,
                        "status": "completed",
                        "success": result.success,
                        "completion_time": result.completion_time,
                        "processing_time": result.processing_time
                    }

            return None

    def get_reflection_result(self, task_id: str) -> Optional[ReflectionResult]:
        """Get the result of a completed reflection task"""
        with self.lock:
            for result in self.completed_tasks:
                if result.task_id == task_id:
                    return result

            return None

    def get_reflection_insights(self, limit: int = 10,
                               reflection_type: Optional[ReflectionType] = None) -> List[Dict[str, Any]]:
        """
        Get recent insights from reflection activities.

        Args:
            limit: Maximum number of insights to return
            reflection_type: Optional filter by reflection type

        Returns:
            List of insight dictionaries
        """
        with self.lock:
            # Collect all insights from recent reflection results
            all_insights = []

            for result in self.completed_tasks:
                if not result.success or not result.insights:
                    continue

                # Skip if filtering by type and doesn't match
                if reflection_type is not None:
                    task_found = False
                    for task in self.reflection_history:
                        if task.task_id == result.task_id and task.reflection_type == reflection_type:
                            task_found = True
                            break

                    if not task_found:
                        continue

                # Add insights with metadata
                for insight in result.insights:
                    all_insights.append({
                        "task_id": result.task_id,
                        "timestamp": result.completion_time,
                        "confidence": result.confidence,
                        **insight
                    })

            # Sort by timestamp (newest first) and limit
            all_insights.sort(key=lambda x: x["timestamp"], reverse=True)
            return all_insights[:limit]

    def process_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Process an event that might trigger reflection.

        Args:
            event_type: The type of event
            event_data: Event data dictionary
        """
        with self.lock:
            # Check all triggers to see if any apply
            current_time = time.time()

            for trigger_id, trigger in self.triggers.items():
                # Skip if trigger is disabled
                if not trigger.enabled:
                    continue

                # Skip if in cooldown
                if current_time - trigger.last_triggered < trigger.cooldown:
                    continue

                # Check if condition applies
                should_trigger = False

                if trigger.condition is not None:
                    try:
                        # Call the condition function with event data
                        should_trigger = trigger.condition(event_type, event_data)
                    except Exception as e:
                        self.logger.error(f"Error evaluating trigger condition: {str(e)}")
                        continue
                else:
                    # Simple matching based on event type
                    if event_type == trigger.name:
                        should_trigger = True

                if should_trigger:
                    # Create a reflection task from the trigger
                    task = ReflectionTask(
                        target_type=trigger.target_type,
                        reflection_type=trigger.reflection_type,
                        target_data=event_data,
                        context={"event_type": event_type},
                        priority=trigger.priority,
                        trigger_id=trigger_id,
                        metadata={"triggered_by_event": True}
                    )

                    # Submit the task
                    self.submit_reflection_task(task)

                    # Update trigger timestamp
                    trigger.last_triggered = current_time

                    self.logger.info(f"Event {event_type} triggered reflection: {trigger.name}")

    def reflect_on_process(self, process_data: Dict[str, Any],
                          priority: float = 0.5) -> str:
        """
        Initiate reflection on a cognitive process.

        Args:
            process_data: Data describing the cognitive process
            priority: Priority level for the reflection task

        Returns:
            Task ID for the new reflection task
        """
        task = ReflectionTask(
            target_type=ReflectionTarget.REASONING_TRACE,
            reflection_type=ReflectionType.PROCESS_TRACE,
            target_data=process_data,
            priority=priority,
            metadata={"source": "explicit_request"}
        )

        return self.submit_reflection_task(task)

    def reflect_on_decision(self, decision_data: Dict[str, Any],
                           priority: float = 0.5) -> str:
        """
        Initiate reflection on a decision.

        Args:
            decision_data: Data describing the decision
            priority: Priority level for the reflection task

        Returns:
            Task ID for the new reflection task
        """
        task = ReflectionTask(
            target_type=ReflectionTarget.DECISION,
            reflection_type=ReflectionType.DECISION_ANALYSIS,
            target_data=decision_data,
            priority=priority,
            metadata={"source": "explicit_request"}
        )

        return self.submit_reflection_task(task)

    def reflect_on_error(self, error_data: Dict[str, Any],
                        priority: float = 0.7) -> str:
        """
        Initiate reflection on an error.

        Args:
            error_data: Data describing the error
            priority: Priority level for the reflection task

        Returns:
            Task ID for the new reflection task
        """
        task = ReflectionTask(
            target_type=ReflectionTarget.ERROR,
            reflection_type=ReflectionType.ERROR_ANALYSIS,
            target_data=error_data,
            priority=priority,
            metadata={"source": "explicit_request"}
        )

        return self.submit_reflection_task(task)

    def generate_explanation(self, explanandum: Any,
                            context: Optional[Dict[str, Any]] = None,
                            priority: float = 0.6) -> str:
        """
        Generate an explanation for a system output or process.

        Args:
            explanandum: The thing to be explained
            context: Additional context for explanation
            priority: Priority level for the explanation task

        Returns:
            Task ID for the new reflection task
        """
        if context is None:
            context = {}

        task = ReflectionTask(
            target_type=ReflectionTarget.EXPLANATION,
            reflection_type=ReflectionType.EXPLANATION_GENERATION,
            target_data=explanandum,
            context=context,
            priority=priority,
            metadata={"source": "explicit_request"}
        )

        return self.submit_reflection_task(task)

    def extract_lessons(self, experience_data: Dict[str, Any],
                       priority: float = 0.4) -> str:
        """
        Extract lessons learned from an experience.

        Args:
            experience_data: Data describing the experience
            priority: Priority level for the extraction task

        Returns:
            Task ID for the new reflection task
        """
        task = ReflectionTask(
            target_type=ReflectionTarget.LESSON,
            reflection_type=ReflectionType.LESSON_EXTRACTION,
            target_data=experience_data,
            priority=priority,
            metadata={"source": "explicit_request"}
        )

        return self.submit_reflection_task(task)

    def get_lessons_learned(self, category: Optional[str] = None,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get lessons learned from past reflections.

        Args:
            category: Optional category filter
            limit: Maximum number of lessons to return

        Returns:
            List of lesson dictionaries
        """
        with self.lock:
            if category:
                filtered_lessons = [l for l in self.lessons_learned if l.get("category") == category]
            else:
                filtered_lessons = self.lessons_learned

            # Sort by importance or recency
            sorted_lessons = sorted(filtered_lessons,
                                    key=lambda x: x.get("importance", 0.5),
                                    reverse=True)

            return sorted_lessons[:limit]

    def evaluate_strategy(self, strategy_data: Dict[str, Any],
                         priority: float = 0.5) -> str:
        """
        Evaluate the effectiveness of a cognitive strategy.

        Args:
            strategy_data: Data describing the strategy
            priority: Priority level for the evaluation task

        Returns:
            Task ID for the new reflection task
        """
        task = ReflectionTask(
            target_type=ReflectionTarget.STRATEGY,
            reflection_type=ReflectionType.STRATEGY_EVALUATION,
            target_data=strategy_data,
            priority=priority,
            metadata={"source": "explicit_request"}
        )

        return self.submit_reflection_task(task)

    # -----------------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------------

    def _register_with_nexus_core(self) -> None:
        """Register with NexusCore for system-wide coordination"""
        if not self.nexus_core:
            self.logger.warning("No NexusCore available for registration")
            return

        try:
            # Register as a module with NexusCore
            dependencies = []

            if self.meta_monitor:
                dependencies.append("meta_monitor")
            if self.confidence_estimator:
                dependencies.append("confidence_estimator")
            if self.adaptation_engine:
                dependencies.append("adaptation_engine")
            if self.episodic_memory:
                dependencies.append("episodic_memory")
            if self.semantic_memory:
                dependencies.append("semantic_memory")
            if self.working_memory:
                dependencies.append("working_memory")

            self.nexus_core.register_module(
                name="reflection_module",
                module=self,
                dependencies=dependencies
            )

            self.logger.info("Successfully registered with NexusCore")
        except Exception as e:
            self.logger.error(f"Failed to register with NexusCore: {str(e)}")

    def _register_default_strategies(self) -> None:
        """Register default reflection strategies"""
        # Process trace analysis
        self.register_strategy(ProcessTraceReflectionStrategy())

        # Decision analysis
        self.register_strategy(DecisionAnalysisStrategy())

        # Error analysis
        self.register_strategy(ErrorAnalysisStrategy())

        # Knowledge gap detection
        self.register_strategy(KnowledgeGapStrategy())

        # Explanation generation
        self.register_strategy(ExplanationGenerationStrategy())

        # Lesson extraction
        self.register_strategy(LessonExtractionStrategy())

        # Add more default strategies as needed

    def _register_default_triggers(self) -> None:
        """Register default reflection triggers"""
        # Low confidence decision trigger
        self.register_trigger(ReflectionTrigger(
            name="low_confidence_decision",
            description="Trigger reflection when a decision has low confidence",
            target_type=ReflectionTarget.DECISION,
            reflection_type=ReflectionType.DECISION_ANALYSIS,
            condition=lambda event_type, data: (
                event_type == "decision_made" and
                "confidence" in data and
                data["confidence"] < 0.5
            ),
            priority=0.7,
            cooldown=30.0
        ))

        # Error trigger
        self.register_trigger(ReflectionTrigger(
            name="error_occurred",
            description="Trigger reflection when an error occurs",
            target_type=ReflectionTarget.ERROR,
            reflection_type=ReflectionType.ERROR_ANALYSIS,
            condition=lambda event_type, data: event_type == "error_occurred",
            priority=0.8,
            cooldown=60.0
        ))

        # Performance anomaly trigger
        self.register_trigger(ReflectionTrigger(
            name="performance_anomaly",
            description="Trigger reflection when performance deviates significantly",
            target_type=ReflectionTarget.PERFORMANCE,
            reflection_type=ReflectionType.PERFORMANCE_ANALYSIS,
            condition=lambda event_type, data: (
                event_type == "performance_metrics" and
                "anomaly_detected" in data and
                data["anomaly_detected"]
            ),
            priority=0.6,
            cooldown=300.0
        ))

        # Add more default triggers as needed

    def _check_llm_availability(self) -> None:
        """Check for the availability of LLM module for reflection"""
        try:
            # Try to get the LLM module from NexusCore
            if self.nexus_core:
                llm_module = self.nexus_core.get_module("llm_interface")
                if llm_module:
                    self.llm_module = llm_module
                    self.llm_available = True
                    self.logger.info("LLM interface available for reflection")
        except Exception as e:
            self.logger.warning(f"Failed to check LLM availability: {str(e)}")

    def _reflection_loop(self) -> None:
        """Background thread for processing reflection tasks"""
        self.logger.info("Reflection loop started")

        while self.running:
            try:
                # Process pending tasks if any
                self._process_next_task()

                # Sleep a bit to avoid busy-waiting
                # We use small sleeps and check self.running frequently to ensure quick shutdown
                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error in reflection loop: {str(e)}")
                time.sleep(1.0)  # Sleep a bit longer on error

    def _process_next_task(self) -> None:
        """Process the next task from the queue"""
        with self.lock:
            # Check if there are any pending tasks
            if not self.pending_tasks:
                return

            # Check if we have capacity to process more tasks
            if len(self.current_tasks) >= self.max_threads:
                return

            import heapq

            # Get the highest priority task
            priority, created_at, task = heapq.heappop(self.pending_tasks)

            # Add to current tasks
            self.current_tasks.add(task)

            # Add to reflection history
            self.reflection_history.append(task)

        # Start a new thread to process the task
        thread = threading.Thread(
            target=self._execute_reflection_task,
            args=(task,),
            daemon=True
        )
        thread.start()

        # Add to thread pool
        with self.lock:
            self.thread_pool.append(thread)

    def _execute_reflection_task(self, task: ReflectionTask) -> None:
        """
        Execute a reflection task in a separate thread.

        Args:
            task: The task to execute
        """
        start_time = time.time()
        result = None

        try:
            # Find strategies that can handle this task
            strategies = [s for s in self.strategies[task.reflection_type] if s.can_handle(task)]

            if not strategies:
                self.logger.warning(f"No strategy available for task: {task.task_id}")
                result = ReflectionResult(
                    task_id=task.task_id,
                    success=False,
                    summary="No strategy available for this reflection type",
                    confidence=0.0,
                    processing_time=time.time() - start_time
                )
            else:
                # Use the first available strategy (in a more advanced implementation,
                # we might select the best strategy or combine multiple)
                strategy = strategies[0]

                # Execute the strategy
                result = strategy.reflect(task)

                # Ensure task_id is set correctly
                result.task_id = task.task_id

                # Set processing time
                result.processing_time = time.time() - start_time

                # Process recommendations if successful
                if result.success and result.recommendations:
                    self._process_recommendations(result.recommendations)

                # Extract lessons if successful
                if result.success and task.reflection_type == ReflectionType.LESSON_EXTRACTION:
                    self._store_lessons(result.insights)

        except Exception as e:
            self.logger.error(f"Error executing reflection task {task.task_id}: {str(e)}")
            result = ReflectionResult(
                task_id=task.task_id,
                success=False,
                summary=f"Error during reflection: {str(e)}",
                confidence=0.0,
                processing_time=time.time() - start_time
            )

        finally:
            # Add to completed tasks and remove from current tasks
            with self.lock:
                if result:
                    self.completed_tasks.append(result)

                if task in self.current_tasks:
                    self.current_tasks.remove(task)

    def _process_recommendations(self, recommendations: List[Dict[str, Any]]) -> None:
        """
        Process recommendations from reflection results.

        Args:
            recommendations: List of recommendation dictionaries
        """
        for recommendation in recommendations:
            # Process different types of recommendations
            rec_type = recommendation.get("type", "")

            if rec_type == "parameter_adjustment":
                # Recommendation to adjust a parameter
                param_name = recommendation.get("parameter")
                new_value = recommendation.get("value")
                reason = recommendation.get("reason", "Based on reflection")
                confidence = recommendation.get("confidence", 0.5)

                if param_name and new_value is not None and self.adaptation_engine:
                    try:
                        # Only apply high-confidence recommendations automatically
                        if confidence >= 0.8:
                            self.adaptation_engine.set_parameter(
                                param_name,
                                new_value,
                                reason=reason
                            )
                            self.logger.info(f"Applied parameter adjustment: {param_name} = {new_value}")
                        else:
                            # Log lower confidence recommendations without applying
                            self.logger.info(f"Suggested parameter adjustment: {param_name} = {new_value} (confidence: {confidence})")
                    except Exception as e:
                        self.logger.error(f"Error applying parameter adjustment: {str(e)}")

            elif rec_type == "knowledge_update":
                # Recommendation to update semantic knowledge
                concept = recommendation.get("concept")
                update = recommendation.get("update")
                confidence = recommendation.get("confidence", 0.5)

                if concept and update and self.semantic_memory and confidence >= 0.7:
                    try:
                        # Apply knowledge update
                        # Implementation depends on semantic memory interface
                        self.logger.info(f"Applied knowledge update for concept: {concept}")
                    except Exception as e:
                        self.logger.error(f"Error applying knowledge update: {str(e)}")

            elif rec_type == "strategy_change":
                # Recommendation to change cognitive strategy
                module = recommendation.get("module")
                strategy = recommendation.get("strategy")
                reason = recommendation.get("reason", "Based on reflection")

                if module and strategy and self.nexus_core:
                    try:
                        # Notify the target module about the strategy change
                        target_module = self.nexus_core.get_module(module)
                        if target_module and hasattr(target_module, "set_strategy"):
                            target_module.set_strategy(strategy, reason=reason)
                            self.logger.info(f"Changed strategy for {module} to {strategy}")
                    except Exception as e:
                        self.logger.error(f"Error changing strategy: {str(e)}")

            elif rec_type == "resource_allocation":
                # Recommendation to adjust resource allocation
                resource = recommendation.get("resource")
                allocation = recommendation.get("allocation")
                reason = recommendation.get("reason", "Based on reflection")

                # Implementation would depend on resource management interface
                self.logger.info(f"Resource allocation recommendation: {resource} = {allocation}")

            elif rec_type == "further_reflection":
                # Recommendation for additional reflection
                reflection_type = recommendation.get("reflection_type")
                target_data = recommendation.get("target_data")
                priority = recommendation.get("priority", 0.5)

                if reflection_type and target_data:
                    try:
                        # Create a new reflection task
                        task = ReflectionTask(
                            target_type=ReflectionTarget[reflection_type.upper()],
                            reflection_type=ReflectionType[reflection_type.upper()],
                            target_data=target_data,
                            priority=priority,
                            metadata={"source": "recommendation"}
                        )

                        self.submit_reflection_task(task)
                        self.logger.info(f"Scheduled further reflection: {reflection_type}")
                    except Exception as e:
                        self.logger.error(f"Error scheduling further reflection: {str(e)}")

            else:
                # Unknown recommendation type
                self.logger.warning(f"Unknown recommendation type: {rec_type}")

    def _store_lessons(self, insights: List[Dict[str, Any]]) -> None:
        """
        Store lessons learned from reflection.

        Args:
            insights: List of insight dictionaries containing lessons
        """
        for insight in insights:
            if "lesson" in insight:
                lesson = {
                    "lesson": insight["lesson"],
                    "category": insight.get("category", "general"),
                    "context": insight.get("context", {}),
                    "importance": insight.get("importance", 0.5),
                    "confidence": insight.get("confidence", 0.5),
                    "timestamp": time.time(),
                    "source_task_id": insight.get("source_task_id")
                }

                # Add to lessons learned
                with self.lock:
                    self.lessons_learned.append(lesson)

                # Store in semantic memory if available
                if self.semantic_memory:
                    try:
                        # Implementation depends on semantic memory interface
                        self.logger.info(f"Stored lesson in semantic memory: {lesson['lesson']}")
                    except Exception as e:
                        self.logger.error(f"Error storing lesson in semantic memory: {str(e)}")

    def _generate_llm_reflection(self,
                                task: ReflectionTask,
                                prompt_template: str) -> Optional[Dict[str, Any]]:
        """
        Use LLM to generate reflection on a task.

        Args:
            task: The reflection task
            prompt_template: Template for LLM prompt

        Returns:
            Dictionary with LLM-generated reflection or None if failed
        """
        if not self.llm_available or not self.llm_module:
            return None

        try:
            # Format the prompt
            prompt = self._format_reflection_prompt(prompt_template, task)

            # Call LLM module
            response = self.llm_module.generate(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.7
            )

            # Parse the response
            result = self._parse_llm_response(response)

            return result
        except Exception as e:
            self.logger.error(f"Error generating LLM reflection: {str(e)}")
            return None

    def _format_reflection_prompt(self, template: str, task: ReflectionTask) -> str:
        """
        Format a prompt template for LLM reflection.

        Args:
            template: The prompt template
            task: The reflection task

        Returns:
            Formatted prompt string
        """
        # Extract task data as string representation
        data_str = self._format_data_for_prompt(task.target_data)

        # Format the template with task data
        prompt = template.format(
            task_type=task.reflection_type.name,
            target_type=task.target_type.name,
            data=data_str,
            context=str(task.context)
        )

        return prompt

    def _format_data_for_prompt(self, data: Any) -> str:
        """
        Format task data for inclusion in a prompt.

        Args:
            data: The data to format

        Returns:
            Formatted string representation
        """
        if isinstance(data, dict):
            # Format dictionary as JSON string
            return json.dumps(data, indent=2)
        elif isinstance(data, list):
            # Format list items
            return "\n".join([f"- {self._format_data_for_prompt(item)}" for item in data])
        else:
            # Convert to string
            return str(data)

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured reflection result.

        Args:
            response: The raw LLM response

        Returns:
            Dictionary with parsed reflection
        """
        # This is a simple implementation that assumes a particular format
        # A more robust implementation would use a structured response format (e.g., JSON)

        result = {
            "summary": "",
            "insights": [],
            "recommendations": []
        }

        # Extract summary (everything up to first heading)
        match = re.search(r'^(.*?)(?:#|$)', response, re.DOTALL)
        if match:
            result["summary"] = match.group(1).strip()

        # Extract insights
        insights_match = re.search(r'(?:#+ *Insights? *\n)(.*?)(?:#|$)', response, re.DOTALL)
        if insights_match:
            insights_text = insights_match.group(1).strip()
            insight_items = re.findall(r'- (.*?)(?:\n-|\n#|$)', insights_text + "\n#", re.DOTALL)

            for item in insight_items:
                result["insights"].append({"description": item.strip()})

        # Extract recommendations
        recommendations_match = re.search(r'(?:#+ *Recommendations? *\n)(.*?)(?:#|$)', response, re.DOTALL)
        if recommendations_match:
            recommendations_text = recommendations_match.group(1).strip()
            recommendation_items = re.findall(r'- (.*?)(?:\n-|\n#|$)', recommendations_text + "\n#", re.DOTALL)

            for item in recommendation_items:
                result["recommendations"].append({"description": item.strip()})

        return result


# Concrete Reflection Strategy Implementations

class ProcessTraceReflectionStrategy(ReflectionStrategy):
    """Strategy for analyzing cognitive process traces"""

    def __init__(self):
        super().__init__("ProcessTraceAnalysis", ReflectionType.PROCESS_TRACE)

    def reflect(self, task: ReflectionTask) -> ReflectionResult:
        """
        Analyze a reasoning process trace.

        Args:
            task: The reflection task

        Returns:
            ReflectionResult containing insights and recommendations
        """
        process_data = task.target_data

        # Validate input
        if not isinstance(process_data, dict) or "steps" not in process_data:
            return ReflectionResult(
                task_id=task.task_id,
                success=False,
                summary="Invalid process data format. Expected dictionary with 'steps' key.",
                confidence=0.0
            )

        # Initialize result
        result = ReflectionResult(
            task_id=task.task_id,
            success=True,
            confidence=0.8
        )

        # Extract process steps
        steps = process_data.get("steps", [])

        # Analyze step coherence
        insights = []
        recommendations = []

        # Check for logical gaps between steps
        has_gaps = False
        gap_indices = []

        for i in range(1, len(steps)):
            prev_step = steps[i-1]
            curr_step = steps[i]

            # Simple heuristic: check if current step references previous step outputs
            if "outputs" in prev_step and "inputs" in curr_step:
                # Check for overlap between previous outputs and current inputs
                prev_outputs = set(str(o) for o in prev_step.get("outputs", []))
                curr_inputs = set(str(i) for i in curr_step.get("inputs", []))

                if not prev_outputs or not curr_inputs or not any(o in s for o in prev_outputs for s in curr_inputs):
                    has_gaps = True
                    gap_indices.append(i)

        if has_gaps:
            insights.append({
                "type": "coherence_issue",
                "description": f"Logical gaps detected between steps: {gap_indices}",
                "confidence": 0.7
            })

            recommendations.append({
                "type": "process_improvement",
                "description": "Add intermediate steps to bridge logical gaps",
                "steps": gap_indices
            })

        # Check for redundant steps
        redundant_indices = []
        for i in range(len(steps)):
            for j in range(i+1, len(steps)):
                # Simple heuristic: check for similar operation types and inputs
                if steps[i].get("operation") == steps[j].get("operation"):
                    if self._are_similar(steps[i].get("inputs", []), steps[j].get("inputs", [])):
                        redundant_indices.append((i, j))

        if redundant_indices:
            insights.append({
                "type": "efficiency_issue",
                "description": f"Potentially redundant steps: {redundant_indices}",
                "confidence": 0.6
            })

            recommendations.append({
                "type": "process_improvement",
                "description": "Consolidate redundant steps to improve efficiency",
                "steps": redundant_indices
            })

        # Check for error patterns
        error_steps = []
        for i, step in enumerate(steps):
            if step.get("error") or step.get("status") == "error":
                error_steps.append(i)

        if error_steps:
            insights.append({
                "type": "error_pattern",
                "description": f"Errors detected in steps: {error_steps}",
                "confidence": 0.9
            })

            recommendations.append({
                "type": "error_handling",
                "description": "Implement error recovery strategies for error-prone steps",
                "steps": error_steps
            })

        # Generate summary
        if insights:
            summary = f"Analysis of {len(steps)} reasoning steps revealed {len(insights)} insights."
            if has_gaps:
                summary += f" Logical gaps found between {len(gap_indices)} step pairs."
            if redundant_indices:
                summary += f" {len(redundant_indices)} potentially redundant steps identified."
            if error_steps:
                summary += f" Errors detected in {len(error_steps)} steps."
        else:
            summary = f"Analysis of {len(steps)} reasoning steps found no significant issues."

        # Update result
        result.summary = summary
        result.insights = insights
        result.recommendations = recommendations

        return result

    def _are_similar(self, list1, list2, threshold=0.7):
        """Check if two lists contain similar elements"""
        if not list1 or not list2:
            return False

        # Convert elements to strings for comparison
        set1 = set(str(x) for x in list1)
        set2 = set(str(x) for x in list2)

        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        if union == 0:
            return False

        similarity = intersection / union
        return similarity >= threshold


class DecisionAnalysisStrategy(ReflectionStrategy):
    """Strategy for analyzing decision-making processes"""

    def __init__(self):
        super().__init__("DecisionAnalysis", ReflectionType.DECISION_ANALYSIS)

    def reflect(self, task: ReflectionTask) -> ReflectionResult:
        """
        Analyze a decision-making process.

        Args:
            task: The reflection task

        Returns:
            ReflectionResult containing insights and recommendations
        """
        decision_data = task.target_data

        # Validate input
        if not isinstance(decision_data, dict):
            return ReflectionResult(
                task_id=task.task_id,
                success=False,
                summary="Invalid decision data format. Expected dictionary.",
                confidence=0.0
            )

        # Initialize result
        result = ReflectionResult(
            task_id=task.task_id,
            success=True,
            confidence=0.8
        )

        # Extract decision components
        options = decision_data.get("options", [])
        criteria = decision_data.get("criteria", [])
        weights = decision_data.get("weights", {})
        scores = decision_data.get("scores", {})
        chosen_option = decision_data.get("chosen", None)
        confidence_score = decision_data.get("confidence", 0.5)

        # Analyze decision components
        insights = []
        recommendations = []

        # Check if all criteria were used
        unused_criteria = []
        for criterion in criteria:
            if criterion not in weights:
                unused_criteria.append(criterion)

        if unused_criteria:
            insights.append({
                "type": "unused_criteria",
                "description": f"Decision criteria not used: {unused_criteria}",
                "confidence": 0.9
            })

            recommendations.append({
                "type": "decision_improvement",
                "description": "Include all defined criteria in the decision process",
                "criteria": unused_criteria
            })

        # Check for balanced criteria weighting
        if weights:
            max_weight = max(weights.values())
            min_weight = min(weights.values())
            weight_range = max_weight - min_weight

            if weight_range > 0.7:  # Arbitrary threshold
                insights.append({
                    "type": "imbalanced_weights",
                    "description": "Large disparity in criteria weights may oversimplify decision",
                    "confidence": 0.6,
                    "max_weight": max_weight,
                    "min_weight": min_weight
                })

        # Check if chosen option was actually highest scored
        if chosen_option and scores:
            option_scores = {}
            for option in options:
                if option in scores:
                    option_scores[option] = sum(scores[option].get(c, 0) * weights.get(c, 1) for c in criteria)

            if option_scores:
                highest_scored = max(option_scores.items(), key=lambda x: x[1])[0]

                if highest_scored != chosen_option:
                    insights.append({
                        "type": "inconsistent_choice",
                        "description": f"Chosen option ({chosen_option}) was not the highest scored ({highest_scored})",
                        "confidence": 0.9,
                        "scores": option_scores
                    })

                    recommendations.append({
                        "type": "decision_improvement",
                        "description": "Align chosen option with highest scored option or document override reasoning",
                        "highest_scored": highest_scored,
                        "chosen": chosen_option
                    })

        # Check for low confidence
        if confidence_score < 0.5:
            insights.append({
                "type": "low_confidence",
                "description": f"Low confidence in decision ({confidence_score})",
                "confidence": 0.9
            })

            recommendations.append({
                "type": "confidence_improvement",
                "description": "Gather more information before finalizing decision",
                "confidence_score": confidence_score
            })

        # Generate summary
        summary = f"Analysis of decision with {len(options)} options and {len(criteria)} criteria."
        if unused_criteria:
            summary += f" {len(unused_criteria)} criteria were unused."
        if chosen_option and scores and highest_scored != chosen_option:
            summary += f" Chosen option ({chosen_option}) differs from highest scored ({highest_scored})."
        if confidence_score < 0.5:
            summary += f" Decision confidence is low ({confidence_score})."

        # Update result
        result.summary = summary
        result.insights = insights
        result.recommendations = recommendations

        return result


class ErrorAnalysisStrategy(ReflectionStrategy):
    """Strategy for analyzing errors"""

    def __init__(self):
        super().__init__("ErrorAnalysis", ReflectionType.ERROR_ANALYSIS)

    def reflect(self, task: ReflectionTask) -> ReflectionResult:
        """
        Analyze an error.

        Args:
            task: The reflection task

        Returns:
            ReflectionResult containing insights and recommendations
        """
        error_data = task.target_data

        # Validate input
        if not isinstance(error_data, dict):
            return ReflectionResult(
                task_id=task.task_id,
                success=False,
                summary="Invalid error data format. Expected dictionary.",
                confidence=0.0
            )

        # Initialize result
        result = ReflectionResult(
            task_id=task.task_id,
            success=True,
            confidence=0.8
        )

        # Extract error components
        error_type = error_data.get("type", "unknown")
        error_message = error_data.get("message", "")
        module = error_data.get("module", "")
        context = error_data.get("context", {})
        stack_trace = error_data.get("stack_trace", "")

        # Analyze error
        insights = []
        recommendations = []

        # Check for known error patterns
        known_pattern = self._match_known_error_pattern(error_type, error_message)
        if known_pattern:
            insights.append({
                "type": "known_error_pattern",
                "description": f"Error matches known pattern: {known_pattern['pattern']}",
                "confidence": 0.9,
                "pattern": known_pattern
            })

            recommendations.append({
                "type": "error_handling",
                "description": known_pattern.get("recommendation", "Apply standard fix for this error"),
                "error_type": error_type
            })

        # Check for resource-related errors
        if any(term in error_message.lower() for term in ["memory", "timeout", "resource", "capacity"]):
            insights.append({
                "type": "resource_issue",
                "description": "Error may be related to resource constraints",
                "confidence": 0.7
            })

            recommendations.append({
                "type": "resource_allocation",
                "description": "Consider adjusting resource allocation",
                "resource": "memory" if "memory" in error_message.lower() else "processing",
                "allocation": 1.2  # Arbitrary increase factor
            })

        # Check for input validation errors
        if any(term in error_message.lower() for term in ["invalid", "expected", "type error", "value error"]):
            insights.append({
                "type": "input_validation",
                "description": "Error suggests input validation issues",
                "confidence": 0.8
            })

            recommendations.append({
                "type": "parameter_adjustment",
                "description": "Strengthen input validation",
                "parameter": "system.validation_strictness",
                "value": 0.8,
                "confidence": 0.7
            })

        # Generate summary
        summary = f"Analysis of {error_type} error in {module}: {error_message}"
        if known_pattern:
            summary += f" This is a known error pattern: {known_pattern['pattern']}."

        # Update result
        result.summary = summary
        result.insights = insights
        result.recommendations = recommendations

        return result

    def _match_known_error_pattern(self, error_type, error_message):
        """Match error against known patterns"""
        # This would be a database or dictionary of known error patterns
        # For demonstration, we'll use a simple list

        known_patterns = [
            {
                "pattern": "memory_allocation_failure",
                "type_match": ["MemoryError", "OutOfMemoryError"],
                "message_regex": r"(out of memory|allocation failed|memory exceeded)",
                "recommendation": "Increase memory allocation or implement memory optimization"
            },
            {
                "pattern": "timeout_error",
                "type_match": ["TimeoutError", "DeadlineExceededError"],
                "message_regex": r"(timeout|deadline exceeded|took too long)",
                "recommendation": "Increase timeout threshold or optimize processing speed"
            },
            {
                "pattern": "dependency_failure",
                "type_match": ["ImportError", "ModuleNotFoundError"],
                "message_regex": r"(not found|cannot import|no module named)",
                "recommendation": "Verify module dependencies and availability"
            }
        ]

        # Check each pattern for matches
        for pattern in known_patterns:
            # Check type match
            if error_type in pattern["type_match"]:
                # Check message regex
                if re.search(pattern["message_regex"], error_message.lower()):
                    return pattern

        return None


class KnowledgeGapStrategy(ReflectionStrategy):
    """Strategy for identifying knowledge gaps"""

    def __init__(self):
        super().__init__("KnowledgeGapDetection", ReflectionType.KNOWLEDGE_GAP)

    def reflect(self, task: ReflectionTask) -> ReflectionResult:
        """
        Identify knowledge gaps.

        Args:
            task: The reflection task

        Returns:
            ReflectionResult containing insights and recommendations
        """
        knowledge_data = task.target_data

        # Validate input
        if not isinstance(knowledge_data, dict):
            return ReflectionResult(
                task_id=task.task_id,
                success=False,
                summary="Invalid knowledge data format. Expected dictionary.",
                confidence=0.0
            )

        # Initialize result
        result = ReflectionResult(
            task_id=task.task_id,
            success=True,
            confidence=0.7
        )

        # Extract knowledge components
        query = knowledge_data.get("query", "")
        available_knowledge = knowledge_data.get("available_knowledge", [])
        missing_knowledge = knowledge_data.get("missing_knowledge", [])
        confidence_scores = knowledge_data.get("confidence_scores", {})

        # Analyze knowledge gaps
        insights = []
        recommendations = []

        # Check for explicit missing knowledge
        if missing_knowledge:
            insights.append({
                "type": "explicit_knowledge_gap",
                "description": f"Explicitly identified missing knowledge: {missing_knowledge}",
                "confidence": 0.9,
                "missing": missing_knowledge
            })

            recommendations.append({
                "type": "knowledge_acquisition",
                "description": "Prioritize acquisition of missing knowledge",
                "targets": missing_knowledge
            })

        # Check for low confidence areas
        low_confidence_areas = []
        for area, score in confidence_scores.items():
            if score < 0.4:  # Arbitrary threshold
                low_confidence_areas.append((area, score))

        if low_confidence_areas:
            insights.append({
                "type": "low_confidence_knowledge",
                "description": f"Knowledge areas with low confidence: {[a[0] for a in low_confidence_areas]}",
                "confidence": 0.8,
                "areas": low_confidence_areas
            })

            recommendations.append({
                "type": "knowledge_verification",
                "description": "Verify knowledge in low confidence areas",
                "areas": [a[0] for a in low_confidence_areas]
            })

        # Check for knowledge relevance
        if query and available_knowledge:
            # In a real implementation, this would use more sophisticated
            # relevance assessment, possibly using semantic similarity

            # For demonstration, we'll use a simple keyword matching
            query_keywords = set(query.lower().split())
            relevant_count = 0

            for knowledge_item in available_knowledge:
                item_text = str(knowledge_item).lower()
                if any(keyword in item_text for keyword in query_keywords):
                    relevant_count += 1

            relevance_ratio = relevant_count / len(available_knowledge) if available_knowledge else 0

            if relevance_ratio < 0.3:  # Arbitrary threshold
                insights.append({
                    "type": "low_knowledge_relevance",
                    "description": "Available knowledge has low relevance to the query",
                    "confidence": 0.6,
                    "relevance_ratio": relevance_ratio
                })

                recommendations.append({
                    "type": "knowledge_organization",
                    "description": "Improve knowledge retrieval relevance",
                    "query": query
                })

        # Generate summary
        summary = f"Knowledge gap analysis for query: '{query}'."
        if missing_knowledge:
            summary += f" {len(missing_knowledge)} missing knowledge items identified."
        if low_confidence_areas:
            summary += f" {len(low_confidence_areas)} knowledge areas with low confidence."

        # Update result
        result.summary = summary
        result.insights = insights
        result.recommendations = recommendations

        return result


class ExplanationGenerationStrategy(ReflectionStrategy):
    """Strategy for generating explanations"""

    def __init__(self):
        super().__init__("ExplanationGeneration", ReflectionType.EXPLANATION_GENERATION)

    def reflect(self, task: ReflectionTask) -> ReflectionResult:
        """
        Generate an explanation.

        Args:
            task: The reflection task

        Returns:
            ReflectionResult containing the explanation
        """
        explanandum = task.target_data
        context = task.context

        # Initialize result
        result = ReflectionResult(
            task_id=task.task_id,
            success=True,
            confidence=0.8
        )

        # Generate explanation
        explanation = self._generate_explanation(explanandum, context)

        # Generate summary
        summary = f"Explanation generated for: {str(explanandum)[:100]}"

        # Create insights with the explanation
        insights = [{
            "type": "explanation",
            "description": explanation,
            "confidence": 0.8
        }]

        # Update result
        result.summary = summary
        result.insights = insights

        return result

    def _generate_explanation(self, explanandum, context):
        """
        Generate an explanation for the given item.

        This is a placeholder implementation. In a real system, this would use
        more sophisticated techniques, possibly involving LLMs, causal inference,
        or domain-specific explanation methods.
        """
        # For demonstration, we'll create a simple template-based explanation
        explanation = f"The system produced this result based on the following factors:\n\n"

        if isinstance(explanandum, dict):
            for key, value in explanandum.items():
                explanation += f"- {key}: {value}\n"
        else:
            explanation += f"- Input: {explanandum}\n"

        if context:
            explanation += "\nContextual factors that influenced this result:\n\n"

            for key, value in context.items():
                explanation += f"- {key}: {value}\n"

        explanation += "\nThe reasoning process involved evaluating the available information "
        explanation += "against established patterns and applying relevant rules or models."

        return explanation


class LessonExtractionStrategy(ReflectionStrategy):
    """Strategy for extracting lessons from experiences"""

    def __init__(self):
        super().__init__("LessonExtraction", ReflectionType.LESSON_EXTRACTION)

    def reflect(self, task: ReflectionTask) -> ReflectionResult:
        """
        Extract lessons from an experience.

        Args:
            task: The reflection task

        Returns:
            ReflectionResult containing extracted lessons
        """
        experience_data = task.target_data

        # Validate input
        if not isinstance(experience_data, dict):
            return ReflectionResult(
                task_id=task.task_id,
                success=False,
                summary="Invalid experience data format. Expected dictionary.",
                confidence=0.0
            )

        # Initialize result
        result = ReflectionResult(
            task_id=task.task_id,
            success=True,
            confidence=0.7
        )

        # Extract experience components
        experience_type = experience_data.get("type", "general")
        outcome = experience_data.get("outcome", "")
        success = experience_data.get("success", False)
        performance = experience_data.get("performance", {})
        steps = experience_data.get("steps", [])
        failures = experience_data.get("failures", [])
        context = experience_data.get("context", {})

        # Extract lessons
        insights = []
        recommendations = []

        # Lesson from outcome
        if outcome:
            insights.append({
                "type": "outcome_lesson",
                "description": f"Experience resulted in: {outcome}",
                "confidence": 0.9,
                "lesson": f"In {experience_type} scenarios, outcomes tend to be: {outcome}",
                "category": experience_type,
                "importance": 0.7 if success else 0.8  # Failures are slightly more important to learn from
            })

        # Lessons from performance
        if performance:
            # Identify top and bottom performance areas
            if len(performance) > 0:
                top_areas = sorted(performance.items(), key=lambda x: x[1], reverse=True)[:2]
                bottom_areas = sorted(performance.items(), key=lambda x: x[1])[:2]

                if top_areas:
                    insights.append({
                        "type": "strength_lesson",
                        "description": f"Strong performance in: {[a[0] for a in top_areas]}",
                        "confidence": 0.8,
                        "lesson": f"Strategies used in {[a[0] for a in top_areas]} are effective and should be maintained",
                        "category": "strengths",
                        "importance": 0.6
                    })

                if bottom_areas:
                    insights.append({
                        "type": "weakness_lesson",
                        "description": f"Weak performance in: {[a[0] for a in bottom_areas]}",
                        "confidence": 0.8,
                        "lesson": f"Improvement needed in {[a[0] for a in bottom_areas]} areas with adjusted strategies",
                        "category": "weaknesses",
                        "importance": 0.8
                    })

                    recommendations.append({
                        "type": "performance_improvement",
                        "description": f"Focus improvement efforts on {[a[0] for a in bottom_areas]}",
                        "areas": [a[0] for a in bottom_areas]
                    })

        # Lessons from failures
        if failures:
            failure_categories = {}

            # Group failures by type
            for failure in failures:
                failure_type = failure.get("type", "unknown")
                if failure_type not in failure_categories:
                    failure_categories[failure_type] = []

                failure_categories[failure_type].append(failure)

            # Extract lessons for each failure category
            for f_type, f_list in failure_categories.items():
                insights.append({
                    "type": "failure_lesson",
                    "description": f"{len(f_list)} failures of type {f_type}",
                    "confidence": 0.9,
                    "lesson": f"Prevent {f_type} failures by ensuring proper validation and handling",
                    "category": "failures",
                    "importance": 0.9  # Failures are high importance for learning
                })

                # If multiple failures of same type, recommend specific handling
                if len(f_list) >= 2:
                    recommendations.append({
                        "type": "error_handling",
                        "description": f"Implement specific handling for {f_type} failures",
                        "failure_type": f_type
                    })

        # Lessons from steps (if process-oriented experience)
        if steps and len(steps) > 0:
            # Identify bottlenecks or high-latency steps
            latency_steps = []
            for step in steps:
                if "latency" in step and step["latency"] > 0.5:  # Arbitrary threshold
                    latency_steps.append(step)

            if latency_steps:
                insights.append({
                    "type": "process_lesson",
                    "description": f"High latency identified in {len(latency_steps)} steps",
                    "confidence": 0.7,
                    "lesson": f"Process efficiency can be improved by optimizing high-latency steps",
                    "category": "process",
                    "importance": 0.7
                })

                recommendations.append({
                    "type": "process_improvement",
                    "description": "Optimize high-latency steps",
                    "steps": [s.get("name", i) for i, s in enumerate(latency_steps)]
                })

            # Identify optimal step sequencing
            if success and len(steps) > 3:
                insights.append({
                    "type": "sequence_lesson",
                    "description": "Successful process sequence identified",
                    "confidence": 0.6,
                    "lesson": "This step sequence can be applied to similar problems",
                    "category": "process",
                    "importance": 0.7
                })

        # Contextual lessons
        if context:
            # Identify key contextual factors
            insights.append({
                "type": "context_lesson",
                "description": f"Key contextual factors: {list(context.keys())}",
                "confidence": 0.7,
                "lesson": f"In {experience_type} scenarios, pay attention to these contextual factors: {list(context.keys())}",
                "category": "context",
                "importance": 0.6
            })

        # Generate summary
        if success:
            summary = f"Lessons extracted from successful {experience_type} experience."
        else:
            summary = f"Lessons extracted from unsuccessful {experience_type} experience."

        summary += f" {len(insights)} lessons identified across {len(set(i.get('category', 'general') for i in insights))} categories."

        # Update result
        result.summary = summary
        result.insights = insights
        result.recommendations = recommendations

        return result