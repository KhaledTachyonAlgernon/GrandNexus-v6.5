"""
planning.py - Planning and Goal Decomposition Module for GrandNexus

This module provides planning capabilities for GrandNexus, enabling the system to:
- Decompose high-level goals into actionable sub-goals and tasks
- Create and manage execution plans with dependencies
- Monitor plan execution and handle contingencies
- Adapt plans based on feedback and changing contexts

The planning approach combines:
- Hierarchical Task Networks (HTN) for goal decomposition
- STRIPS-like operators for action representation
- Dynamic re-planning capabilities for adaptation
- Belief-Desire-Intention (BDI) inspired goal management

Design principles:
- Flexible: Plans can be modified during execution
- Contextual: Planning takes into account the current state and context
- Emergent: Complex plans can emerge from simpler building blocks
- Adaptive: Plans adjust to feedback and changing circumstances
"""

import logging
import threading
import uuid
import time
import heapq
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Union
from collections import deque, defaultdict

# Import from GrandNexus core modules
from grandnexus.core.nexus_core import NexusCore
from grandnexus.memory.working import WorkingMemory
from grandnexus.memory.semantic import SemanticMemory
from grandnexus.memory.episodic import EpisodicMemory
from grandnexus.reasoning.reasoning_core import ReasoningCore

# Define goal and plan states
class GoalState(Enum):
    """States a goal can be in"""
    PENDING = auto()    # Not yet activated
    ACTIVE = auto()     # Currently being pursued
    ACHIEVED = auto()   # Successfully completed
    FAILED = auto()     # Failed to achieve
    SUSPENDED = auto()  # Temporarily suspended
    ABANDONED = auto()  # Permanently abandoned

class TaskState(Enum):
    """States a task/action can be in"""
    PENDING = auto()    # Not yet started
    READY = auto()      # Ready to execute (all preconditions met)
    EXECUTING = auto()  # Currently executing
    COMPLETED = auto()  # Successfully completed
    FAILED = auto()     # Failed to complete
    BLOCKED = auto()    # Waiting for preconditions

class PlanningApproach(Enum):
    """Different planning approaches that can be selected"""
    HIERARCHICAL = auto()  # Hierarchical Task Network planning
    FORWARD = auto()       # Forward state-space planning
    BACKWARD = auto()      # Backward state-space planning
    PARTIAL_ORDER = auto() # Partial-order planning
    REACTIVE = auto()      # Reactive planning (minimal lookahead)
    ADAPTIVE = auto()      # Adaptive planning (context-dependent strategy)

@dataclass
class Condition:
    """Represents a condition that must be true for a task to execute or a goal to be achieved"""
    condition_type: str  # e.g., "state_check", "resource_check", "capability_check"
    parameters: Dict[str, Any] = field(default_factory=dict)
    negated: bool = False

    def check(self, state: Dict[str, Any]) -> bool:
        """
        Check if this condition is satisfied in the given state.

        Args:
            state: Current state of the world

        Returns:
            True if condition is satisfied, False otherwise
        """
        # This is a simplified implementation
        # A real implementation would have logic for different condition types

        if self.condition_type == "state_check":
            # Check if a state variable has the expected value
            var_name = self.parameters.get("variable", "")
            expected_value = self.parameters.get("value")

            if var_name in state:
                result = state[var_name] == expected_value
                return not result if self.negated else result
            return False

        elif self.condition_type == "capability_check":
            # Check if a capability is available
            capability = self.parameters.get("capability", "")
            result = capability in state.get("capabilities", [])
            return not result if self.negated else result

        elif self.condition_type == "resource_check":
            # Check if a resource level meets the requirement
            resource = self.parameters.get("resource", "")
            min_level = self.parameters.get("min_level", 0)

            if resource in state.get("resources", {}):
                result = state["resources"][resource] >= min_level
                return not result if self.negated else result
            return False

        # Add more condition types as needed

        return False

@dataclass
class Effect:
    """Represents an effect of executing a task or achieving a goal"""
    effect_type: str  # e.g., "state_change", "resource_change"
    parameters: Dict[str, Any] = field(default_factory=dict)

    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply this effect to the given state.

        Args:
            state: Current state of the world

        Returns:
            Updated state with effect applied
        """
        # This is a simplified implementation
        # A real implementation would have logic for different effect types

        # Make a copy of the state to avoid modifying the original
        new_state = state.copy()

        if self.effect_type == "state_change":
            # Change a state variable to a new value
            var_name = self.parameters.get("variable", "")
            new_value = self.parameters.get("value")

            if var_name:
                new_state[var_name] = new_value

        elif self.effect_type == "resource_change":
            # Change a resource level
            resource = self.parameters.get("resource", "")
            change = self.parameters.get("change", 0)

            if resource:
                if "resources" not in new_state:
                    new_state["resources"] = {}
                if resource not in new_state["resources"]:
                    new_state["resources"][resource] = 0

                new_state["resources"][resource] += change

        # Add more effect types as needed

        return new_state

@dataclass
class Task:
    """
    Represents an executable task in a plan.

    Tasks are the basic building blocks of plans. They have preconditions
    that must be met before execution, effects that occur after execution,
    and a handler function that performs the actual execution.
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    preconditions: List[Condition] = field(default_factory=list)
    effects: List[Effect] = field(default_factory=list)
    handler: Optional[Callable[[Dict[str, Any]], Tuple[bool, Dict[str, Any]]]] = None
    estimated_duration: float = 0.0
    priority: float = 0.5
    state: TaskState = TaskState.PENDING
    retry_count: int = 0
    max_retries: int = 3
    failure_handlers: List[Callable[[Dict[str, Any]], None]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def can_execute(self, state: Dict[str, Any]) -> bool:
        """
        Check if all preconditions are satisfied and the task can be executed.

        Args:
            state: Current state of the world

        Returns:
            True if the task can be executed, False otherwise
        """
        for condition in self.preconditions:
            if not condition.check(state):
                return False
        return True

    def predict_effects(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the state after executing this task.

        Args:
            state: Current state of the world

        Returns:
            Predicted state after task execution
        """
        new_state = state.copy()

        for effect in self.effects:
            new_state = effect.apply(new_state)

        return new_state

    def execute(self, state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute the task with the given state.

        Args:
            state: Current state of the world

        Returns:
            Tuple of (success, new_state)
        """
        if not self.can_execute(state):
            return False, state

        self.state = TaskState.EXECUTING

        try:
            if self.handler:
                # Call the handler function
                success, new_state = self.handler(state)
            else:
                # No handler, just apply the effects
                success = True
                new_state = self.predict_effects(state)

            if success:
                self.state = TaskState.COMPLETED
            else:
                self.state = TaskState.FAILED
                self.retry_count += 1

                # Try to handle the failure
                for handler in self.failure_handlers:
                    handler(state)

                # Check if we can retry
                if self.retry_count <= self.max_retries:
                    self.state = TaskState.PENDING

            return success, new_state

        except Exception as e:
            self.state = TaskState.FAILED
            self.retry_count += 1

            # Try to handle the failure
            for handler in self.failure_handlers:
                try:
                    handler(state)
                except:
                    pass

            return False, state

@dataclass
class Goal:
    """
    Represents a goal to be achieved.

    Goals are high-level objectives that can be decomposed into sub-goals
    and eventually into tasks. They have conditions that define when the
    goal is achieved.
    """
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    achievement_conditions: List[Condition] = field(default_factory=list)
    parent_goal_id: Optional[str] = None
    sub_goals: List[str] = field(default_factory=list)
    tasks: List[str] = field(default_factory=list)
    state: GoalState = GoalState.PENDING
    priority: float = 0.5
    deadline: Optional[float] = None
    creation_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    achievement_time: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_achieved(self, state: Dict[str, Any]) -> bool:
        """
        Check if the goal is achieved in the given state.

        Args:
            state: Current state of the world

        Returns:
            True if the goal is achieved, False otherwise
        """
        for condition in self.achievement_conditions:
            if not condition.check(state):
                return False
        return True

    def has_expired(self, current_time: float) -> bool:
        """
        Check if the goal has expired (past its deadline).

        Args:
            current_time: Current time

        Returns:
            True if the goal has expired, False otherwise
        """
        return self.deadline is not None and current_time > self.deadline

    def update_state(self, new_state: GoalState, current_time: float) -> None:
        """
        Update the state of the goal.

        Args:
            new_state: New goal state
            current_time: Current time
        """
        self.state = new_state
        self.last_updated = current_time

        if new_state == GoalState.ACHIEVED:
            self.achievement_time = current_time

@dataclass
class Plan:
    """
    Represents a plan for achieving a goal.

    A plan consists of a collection of tasks (and potentially sub-plans)
    with dependencies between them. It specifies the order in which tasks
    should be executed to achieve a goal.
    """
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal_id: str = ""
    tasks: Dict[str, Task] = field(default_factory=dict)
    task_dependencies: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    sub_plans: Dict[str, "Plan"] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    current_task_id: Optional[str] = None
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    creation_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    completion_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_ready_tasks(self) -> List[str]:
        """
        Get tasks that are ready to be executed (all dependencies completed).

        Returns:
            List of task IDs
        """
        ready_tasks = []

        for task_id, task in self.tasks.items():
            if task.state != TaskState.PENDING:
                continue

            dependencies = self.task_dependencies.get(task_id, [])
            all_deps_completed = True

            for dep_id in dependencies:
                dep_task = self.tasks.get(dep_id)
                if dep_task is None or dep_task.state != TaskState.COMPLETED:
                    all_deps_completed = False
                    break

            if all_deps_completed and task.can_execute(self.state):
                ready_tasks.append(task_id)

        return ready_tasks

    def is_completed(self) -> bool:
        """
        Check if the plan is completed (all tasks completed).

        Returns:
            True if the plan is completed, False otherwise
        """
        return all(task.state == TaskState.COMPLETED for task in self.tasks.values())

    def is_failed(self) -> bool:
        """
        Check if the plan has failed (any task failed with no retries left).

        Returns:
            True if the plan has failed, False otherwise
        """
        return any(task.state == TaskState.FAILED and task.retry_count > task.max_retries
                   for task in self.tasks.values())

    def get_progress(self) -> float:
        """
        Get the progress of the plan (percentage of completed tasks).

        Returns:
            Progress as a float between 0 and 1
        """
        if not self.tasks:
            return 0.0

        completed = sum(1 for task in self.tasks.values() if task.state == TaskState.COMPLETED)
        return completed / len(self.tasks)

    def update_state(self, new_state: Dict[str, Any]) -> None:
        """
        Update the state of the plan.

        Args:
            new_state: New state
        """
        self.state.update(new_state)
        self.last_updated = time.time()

class PlanningModule:
    """
    Planning module for GrandNexus.

    This module is responsible for:
    - Goal management and decomposition
    - Plan generation and execution
    - Task scheduling and monitoring
    - Adaptive re-planning

    It integrates with other cognitive modules to create, execute,
    and adapt plans based on the system's goals and context.
    """

    def __init__(self, nexus_core: Optional[NexusCore] = None,
                 working_memory: Optional[WorkingMemory] = None,
                 semantic_memory: Optional[SemanticMemory] = None,
                 episodic_memory: Optional[EpisodicMemory] = None,
                 reasoning_core: Optional[ReasoningCore] = None):
        """
        Initialize the planning module.

        Args:
            nexus_core: Reference to the NexusCore for system integration
            working_memory: Reference to working memory for context
            semantic_memory: Reference to semantic memory for knowledge
            episodic_memory: Reference to episodic memory for past experiences
            reasoning_core: Reference to reasoning core for inference
        """
        self.logger = logging.getLogger("GrandNexus.Reasoning.Planning")
        self.nexus_core = nexus_core
        self.working_memory = working_memory
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory
        self.reasoning_core = reasoning_core

        # Internal state
        self.instance_id = str(uuid.uuid4())[:8]
        self.running = False
        self.lock = threading.RLock()

        # Goals and plans
        self.goals: Dict[str, Goal] = {}
        self.plans: Dict[str, Plan] = {}

        # Task library (reusable task templates)
        self.task_library: Dict[str, Task] = {}

        # Goal decomposers (functions that decompose goals into sub-goals and tasks)
        self.goal_decomposers: Dict[str, Callable[[Goal, Dict[str, Any]], Tuple[List[Goal], List[Task]]]] = {}

        # Planning algorithms
        self.planning_algorithms: Dict[PlanningApproach, Callable[[Goal, Dict[str, Any]], Plan]] = {}

        # Active execution
        self.executing_plans: Set[str] = set()
        self.execution_thread: Optional[threading.Thread] = None
        self.stop_execution = threading.Event()

        # Performance metrics
        self.metrics: Dict[str, List[float]] = defaultdict(list)

        self.logger.info(f"PlanningModule initialized with ID={self.instance_id}")

    def start(self) -> bool:
        """Start the planning module"""
        with self.lock:
            if self.running:
                return True

            # Initialize planning algorithms
            self._register_planning_algorithms()

            # Initialize task library
            self._initialize_task_library()

            # Initialize goal decomposers
            self._register_goal_decomposers()

            # Start execution thread
            self.stop_execution.clear()
            self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
            self.execution_thread.start()

            self.running = True
            self.logger.info("PlanningModule started")
            return True

    def stop(self) -> bool:
        """Stop the planning module"""
        with self.lock:
            if not self.running:
                return True

            # Stop execution thread
            self.stop_execution.set()
            if self.execution_thread:
                self.execution_thread.join(timeout=5.0)
                self.execution_thread = None

            self.running = False
            self.logger.info("PlanningModule stopped")
            return True

    def add_goal(self, goal: Goal) -> str:
        """
        Add a new goal to be planned for.

        Args:
            goal: The goal to add

        Returns:
            Goal ID
        """
        with self.lock:
            self.goals[goal.goal_id] = goal
            self.logger.info(f"Added goal {goal.goal_id}: {goal.name}")
            return goal.goal_id

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """
        Get a goal by ID.

        Args:
            goal_id: Goal ID

        Returns:
            Goal object, or None if not found
        """
        with self.lock:
            return self.goals.get(goal_id)

    def update_goal(self, goal_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a goal's properties.

        Args:
            goal_id: Goal ID
            updates: Dictionary of properties to update

        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            goal = self.goals.get(goal_id)
            if not goal:
                return False

            # Update goal properties
            for key, value in updates.items():
                if hasattr(goal, key):
                    setattr(goal, key, value)

            goal.last_updated = time.time()
            return True

    def remove_goal(self, goal_id: str) -> bool:
        """
        Remove a goal.

        Args:
            goal_id: Goal ID

        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if goal_id not in self.goals:
                return False

            # Remove associated plans
            for plan_id, plan in list(self.plans.items()):
                if plan.goal_id == goal_id:
                    if plan_id in self.executing_plans:
                        self.executing_plans.remove(plan_id)
                    del self.plans[plan_id]

            del self.goals[goal_id]
            return True

    def create_plan_for_goal(self, goal_id: str, approach: PlanningApproach = PlanningApproach.HIERARCHICAL,
                            context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Create a plan for achieving a goal.

        Args:
            goal_id: Goal ID
            approach: Planning approach to use
            context: Additional context for planning

        Returns:
            Plan ID if successful, None otherwise
        """
        with self.lock:
            goal = self.goals.get(goal_id)
            if not goal:
                self.logger.warning(f"Goal {goal_id} not found")
                return None

            # Check if there's already an active plan for this goal
            for plan in self.plans.values():
                if plan.goal_id == goal_id and plan.goal_id in self.executing_plans:
                    self.logger.warning(f"Already have an executing plan for goal {goal_id}")
                    return plan.plan_id

            # Prepare planning context
            if context is None:
                context = {}

            # Get the current state
            current_state = self._get_current_state()

            # Choose planning algorithm
            planning_algo = self.planning_algorithms.get(approach)
            if not planning_algo:
                self.logger.error(f"No planning algorithm available for approach {approach}")
                return None

            try:
                # Generate plan
                plan = planning_algo(goal, {**current_state, **context})

                # Store plan
                self.plans[plan.plan_id] = plan

                self.logger.info(f"Created plan {plan.plan_id} for goal {goal_id}")
                return plan.plan_id
            except Exception as e:
                self.logger.error(f"Error creating plan for goal {goal_id}: {str(e)}")
                return None

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """
        Get a plan by ID.

        Args:
            plan_id: Plan ID

        Returns:
            Plan object, or None if not found
        """
        with self.lock:
            return self.plans.get(plan_id)

    def execute_plan(self, plan_id: str) -> bool:
        """
        Start executing a plan.

        Args:
            plan_id: Plan ID

        Returns:
            True if execution started, False otherwise
        """
        with self.lock:
            plan = self.plans.get(plan_id)
            if not plan:
                self.logger.warning(f"Plan {plan_id} not found")
                return False

            # Add to executing plans
            self.executing_plans.add(plan_id)

            # Update goal state
            goal = self.goals.get(plan.goal_id)
            if goal:
                goal.update_state(GoalState.ACTIVE, time.time())

            self.logger.info(f"Started execution of plan {plan_id}")
            return True

    def pause_plan(self, plan_id: str) -> bool:
        """
        Pause execution of a plan.

        Args:
            plan_id: Plan ID

        Returns:
            True if paused, False otherwise
        """
        with self.lock:
            if plan_id not in self.executing_plans:
                return False

            self.executing_plans.remove(plan_id)

            plan = self.plans.get(plan_id)
            if plan and plan.current_task_id:
                # Mark current task as pending
                task = plan.tasks.get(plan.current_task_id)
                if task and task.state == TaskState.EXECUTING:
                    task.state = TaskState.PENDING

            # Update goal state
            if plan:
                goal = self.goals.get(plan.goal_id)
                if goal:
                    goal.update_state(GoalState.SUSPENDED, time.time())

            self.logger.info(f"Paused execution of plan {plan_id}")
            return True

    def cancel_plan(self, plan_id: str) -> bool:
        """
        Cancel execution of a plan.

        Args:
            plan_id: Plan ID

        Returns:
            True if cancelled, False otherwise
        """
        with self.lock:
            if plan_id in self.executing_plans:
                self.executing_plans.remove(plan_id)

            plan = self.plans.get(plan_id)
            if not plan:
                return False

            # Update goal state
            goal = self.goals.get(plan.goal_id)
            if goal:
                goal.update_state(GoalState.ABANDONED, time.time())

            # Remove plan
            del self.plans[plan_id]

            self.logger.info(f"Cancelled plan {plan_id}")
            return True

    def get_active_goals(self) -> List[Goal]:
        """
        Get all active goals.

        Returns:
            List of active goals
        """
        with self.lock:
            return [goal for goal in self.goals.values() if goal.state == GoalState.ACTIVE]

    def get_executing_plans(self) -> List[Plan]:
        """
        Get all currently executing plans.

        Returns:
            List of executing plans
        """
        with self.lock:
            return [self.plans[plan_id] for plan_id in self.executing_plans if plan_id in self.plans]

    def add_task_to_library(self, task_template: Task) -> str:
        """
        Add a task template to the task library.

        Args:
            task_template: Task template to add

        Returns:
            Task ID
        """
        with self.lock:
            self.task_library[task_template.task_id] = task_template
            return task_template.task_id

    def get_task_from_library(self, task_id: str) -> Optional[Task]:
        """
        Get a task template from the library.

        Args:
            task_id: Task ID

        Returns:
            Task template, or None if not found
        """
        with self.lock:
            return self.task_library.get(task_id)

    def register_goal_decomposer(self, goal_type: str,
                                decomposer: Callable[[Goal, Dict[str, Any]], Tuple[List[Goal], List[Task]]]) -> None:
        """
        Register a goal decomposer function.

        Args:
            goal_type: Type of goal this decomposer handles
            decomposer: Function that decomposes goals into sub-goals and tasks
        """
        with self.lock:
            self.goal_decomposers[goal_type] = decomposer
            self.logger.info(f"Registered goal decomposer for {goal_type}")

    def get_planning_metrics(self) -> Dict[str, Any]:
        """
        Get planning performance metrics.

        Returns:
            Dictionary of metrics
        """
        with self.lock:
            metrics = {}
            for key, values in self.metrics.items():
                if values:
                    metrics[key] = {
                        "current": values[-1],
                        "average": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values)
                    }
            return metrics

    # -----------------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------------

    def _register_planning_algorithms(self) -> None:
        """Register available planning algorithms"""
        self.planning_algorithms[PlanningApproach.HIERARCHICAL] = self._hierarchical_planning
        self.planning_algorithms[PlanningApproach.FORWARD] = self._forward_planning
        self.planning_algorithms[PlanningApproach.BACKWARD] = self._backward_planning
        self.planning_algorithms[PlanningApproach.PARTIAL_ORDER] = self._partial_order_planning
        self.planning_algorithms[PlanningApproach.REACTIVE] = self._reactive_planning
        self.planning_algorithms[PlanningApproach.ADAPTIVE] = self._adaptive_planning

    def _initialize_task_library(self) -> None:
        """Initialize the task library with basic task templates"""
        # This would be populated with domain-specific tasks
        # Here we just provide a simple example

        # Example: Wait task
        wait_task = Task(
            task_id="task_wait",
            name="Wait",
            description="Wait for a specified duration",
            handler=lambda state: (True, state),  # Just returns success
            estimated_duration=1.0
        )
        self.task_library[wait_task.task_id] = wait_task

        # Additional tasks would be defined here

    def _register_goal_decomposers(self) -> None:
        """Register goal decomposers for different goal types"""
        # This would be populated with domain-specific decomposers
        # Here we just provide a simple example

        # Example: Generic goal decomposer
        self.register_goal_decomposer("generic", self._generic_goal_decomposer)

        # Additional decomposers would be registered here

    def _generic_goal_decomposer(self, goal: Goal, context: Dict[str, Any]) -> Tuple[List[Goal], List[Task]]:
        """
        Generic goal decomposer that handles simple goals.

        Args:
            goal: Goal to decompose
            context: Planning context

        Returns:
            Tuple of (sub-goals, tasks)
        """
        # This is a placeholder implementation
        # In a real system, this would have more sophisticated decomposition logic

        # Example: Create a simple task to achieve the goal
        task = Task(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            name=f"Achieve {goal.name}",
            description=f"Generic task to achieve goal: {goal.description}",
            effects=[
                # Assume the task will achieve all goal conditions
                Effect(
                    effect_type="state_change",
                    parameters={
                        "variable": f"goal_{goal.goal_id}_achieved",
                        "value": True
                    }
                )
            ]
        )

        return [], [task]

    def _get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of the world for planning.

        Returns:
            Dictionary representing the current state
        """
        state = {}

        # Get basic system state
        state["current_time"] = time.time()
        state["capabilities"] = ["reasoning", "perception", "learning"]  # Basic capabilities

        # Get state from working memory if available
        if self.working_memory:
            try:
                items = self.working_memory.get_all_items()

                # Extract relevant state information from working memory
                for item in items:
                    if item.content and isinstance(item.content, dict) and "state" in item.content:
                        state.update(item.content["state"])
            except Exception as e:
                self.logger.warning(f"Error getting state from working memory: {str(e)}")

        # Include information about active goals
        active_goals = [goal.goal_id for goal in self.goals.values() if goal.state == GoalState.ACTIVE]
        state["active_goals"] = active_goals

        return state

    def _hierarchical_planning(self, goal: Goal, context: Dict[str, Any]) -> Plan:
        """
        Hierarchical Task Network (HTN) planning algorithm.

        Args:
            goal: Goal to plan for
            context: Planning context

        Returns:
            A Plan to achieve the goal
        """
        self.logger.info(f"Creating hierarchical plan for goal {goal.goal_id}: {goal.name}")

        # Create a new plan
        plan = Plan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            goal_id=goal.goal_id,
            state=context.copy()
        )

        # Determine goal type
        goal_type = goal.metadata.get("type", "generic")

        # Get appropriate decomposer
        decomposer = self.goal_decomposers.get(goal_type, self.goal_decomposers.get("generic"))
        if not decomposer:
            raise ValueError(f"No decomposer available for goal type: {goal_type}")

        # Decompose goal into sub-goals and tasks
        sub_goals, tasks = decomposer(goal, context)

        # Add tasks to the plan
        for task in tasks:
            plan.tasks[task.task_id] = task

        # Process sub-goals recursively
        for sub_goal in sub_goals:
            # Store the sub-goal
            self.goals[sub_goal.goal_id] = sub_goal

            # Link to parent goal
            goal.sub_goals.append(sub_goal.goal_id)

            # Create a sub-plan for the sub-goal
            sub_plan = self._hierarchical_planning(sub_goal, context)

            # Add sub-plan to the main plan
            plan.sub_plans[sub_plan.plan_id] = sub_plan

            # Create a task to execute the sub-plan
            sub_plan_task = Task(
                task_id=f"task_exec_plan_{sub_plan.plan_id}",
                name=f"Execute plan for {sub_goal.name}",
                description=f"Execute sub-plan to achieve {sub_goal.name}",
                handler=lambda state, plan_id=sub_plan.plan_id: self._execute_sub_plan(plan_id, state)
            )

            plan.tasks[sub_plan_task.task_id] = sub_plan_task

        # Set up task dependencies based on effects and preconditions
        self._setup_task_dependencies(plan)

        return plan

    def _forward_planning(self, goal: Goal, context: Dict[str, Any]) -> Plan:
        """
        Forward state-space planning algorithm.

        Args:
            goal: Goal to plan for
            context: Planning context

        Returns:
            A Plan to achieve the goal
        """
        # This is a simplified implementation of forward planning
        self.logger.info(f"Creating forward plan for goal {goal.goal_id}: {goal.name}")

        # Create a new plan
        plan = Plan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            goal_id=goal.goal_id,
            state=context.copy()
        )

        # Get all applicable task templates from the library
        applicable_tasks = []
        for task_id, task_template in self.task_library.items():
            # Create a copy of the template
            task = Task(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                name=task_template.name,
                description=task_template.description,
                preconditions=task_template.preconditions.copy(),
                effects=task_template.effects.copy(),
                handler=task_template.handler,
                estimated_duration=task_template.estimated_duration,
                priority=task_template.priority
            )
            applicable_tasks.append(task)

        # Start from the initial state
        current_state = context.copy()

        # Keep adding tasks until the goal is achieved or no more progress can be made
        while not goal.is_achieved(current_state):
            # Find a task that can be executed in the current state
            next_task = None
            next_state = None

            for task in applicable_tasks:
                if task.can_execute(current_state):
                    # Predict the effects of executing this task
                    predicted_state = task.predict_effects(current_state)

                    # Check if this gets us closer to the goal
                    # (Simple heuristic: count how many goal conditions are satisfied)
                    current_satisfied = sum(1 for cond in goal.achievement_conditions if cond.check(current_state))
                    predicted_satisfied = sum(1 for cond in goal.achievement_conditions if cond.check(predicted_state))

                    if predicted_satisfied > current_satisfied:
                        next_task = task
                        next_state = predicted_state
                        break

            if next_task is None:
                # No task can get us closer to the goal
                self.logger.warning(f"Could not find a path to achieve goal {goal.goal_id}")
                break

            # Add the task to the plan
            plan.tasks[next_task.task_id] = next_task

            # Update the current state
            current_state = next_state

            # Remove the task from the applicable list to avoid adding it again
            applicable_tasks.remove(next_task)

        # Set up task dependencies (sequential for forward planning)
        task_ids = list(plan.tasks.keys())
        for i in range(1, len(task_ids)):
            plan.task_dependencies[task_ids[i]].append(task_ids[i-1])

        return plan

    def _backward_planning(self, goal: Goal, context: Dict[str, Any]) -> Plan:
        """
        Backward state-space planning algorithm.

        Args:
            goal: Goal to plan for
            context: Planning context

        Returns:
            A Plan to achieve the goal
        """
        # This is a simplified implementation of backward planning
        self.logger.info(f"Creating backward plan for goal {goal.goal_id}: {goal.name}")

        # Create a new plan
        plan = Plan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            goal_id=goal.goal_id,
            state=context.copy()
        )

        # Start with the goal conditions as the current sub-goals
        subgoals = goal.achievement_conditions.copy()

        # Keep track of the planned tasks in reverse order
        planned_tasks = []

        # Keep adding tasks until all sub-goals are achieved in the initial state
        while subgoals:
            subgoal = subgoals.pop(0)

            # Check if the sub-goal is already satisfied in the initial state
            if subgoal.check(context):
                continue

            # Find a task that can achieve this sub-goal
            achieving_task = None

            for task_id, task_template in self.task_library.items():
                # Check if any of the task's effects would achieve the sub-goal
                for effect in task_template.effects:
                    # This is a simplified check - in a real system,
                    # we would need a more sophisticated check
                    if (effect.effect_type == "state_change" and
                        effect.parameters.get("variable") == subgoal.parameters.get("variable") and
                        effect.parameters.get("value") == subgoal.parameters.get("value")):

                        # Create a copy of the template
                        achieving_task = Task(
                            task_id=f"task_{uuid.uuid4().hex[:8]}",
                            name=task_template.name,
                            description=task_template.description,
                            preconditions=task_template.preconditions.copy(),
                            effects=task_template.effects.copy(),
                            handler=task_template.handler,
                            estimated_duration=task_template.estimated_duration,
                            priority=task_template.priority
                        )
                        break

                if achieving_task:
                    break

            if achieving_task is None:
                # No task can achieve this sub-goal
                self.logger.warning(f"Could not find a task to achieve sub-goal: {subgoal}")
                raise ValueError(f"No task can achieve sub-goal: {subgoal}")

            # Add the task's preconditions as new sub-goals
            subgoals.extend(achieving_task.preconditions)

            # Add the task to the planned tasks
            planned_tasks.append(achieving_task)

        # Reverse the list of tasks to get the correct order
        planned_tasks.reverse()

        # Add tasks to the plan
        for task in planned_tasks:
            plan.tasks[task.task_id] = task

        # Set up task dependencies (sequential for backward planning)
        task_ids = list(plan.tasks.keys())
        for i in range(1, len(task_ids)):
            plan.task_dependencies[task_ids[i]].append(task_ids[i-1])

        return plan

    def _partial_order_planning(self, goal: Goal, context: Dict[str, Any]) -> Plan:
        """
        Partial-Order Planning (POP) algorithm.

        Args:
            goal: Goal to plan for
            context: Planning context

        Returns:
            A Plan to achieve the goal
        """
        # This would be a more complex implementation of POP
        # For simplicity, we'll delegate to hierarchical planning for now
        self.logger.info(f"Delegating partial-order planning to hierarchical planning for goal {goal.goal_id}")
        return self._hierarchical_planning(goal, context)

    def _reactive_planning(self, goal: Goal, context: Dict[str, Any]) -> Plan:
        """
        Reactive planning algorithm (minimal lookahead).

        Args:
            goal: Goal to plan for
            context: Planning context

        Returns:
            A Plan to achieve the goal
        """
        # This is a simplified implementation of reactive planning
        self.logger.info(f"Creating reactive plan for goal {goal.goal_id}: {goal.name}")

        # Create a new plan
        plan = Plan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            goal_id=goal.goal_id,
            state=context.copy()
        )

        # For reactive planning, we just create a single "monitor and react" task
        react_task = Task(
            task_id=f"task_react_{uuid.uuid4().hex[:8]}",
            name=f"Monitor and react for {goal.name}",
            description=f"Continuously monitor the environment and react to achieve {goal.name}",
            handler=lambda state: self._reactive_handler(goal, state)
        )

        plan.tasks[react_task.task_id] = react_task

        return plan

    def _adaptive_planning(self, goal: Goal, context: Dict[str, Any]) -> Plan:
        """
        Adaptive planning algorithm (selects best approach based on context).

        Args:
            goal: Goal to plan for
            context: Planning context

        Returns:
            A Plan to achieve the goal
        """
        # Analyze the goal and context to select the best planning approach
        self.logger.info(f"Selecting best planning approach for goal {goal.goal_id}: {goal.name}")

        # Check if the goal has a specific planning approach specified
        if "preferred_approach" in goal.metadata:
            approach_name = goal.metadata["preferred_approach"]
            if hasattr(PlanningApproach, approach_name.upper()):
                approach = getattr(PlanningApproach, approach_name.upper())
                self.logger.info(f"Using preferred approach: {approach}")
                return self.planning_algorithms[approach](goal, context)

        # Criteria for selecting an approach

        # If the goal is complex (has many achievement conditions)
        if len(goal.achievement_conditions) > 3:
            self.logger.info("Complex goal detected, using hierarchical planning")
            return self._hierarchical_planning(goal, context)

        # If the goal is time-critical (has a tight deadline)
        if goal.deadline and goal.deadline - time.time() < 60:  # less than 60 seconds
            self.logger.info("Time-critical goal detected, using reactive planning")
            return self._reactive_planning(goal, context)

        # If the goal requires looking ahead
        if "requires_lookahead" in goal.metadata and goal.metadata["requires_lookahead"]:
            self.logger.info("Goal requires lookahead, using forward planning")
            return self._forward_planning(goal, context)

        # Default to hierarchical planning
        self.logger.info("Using default hierarchical planning")
        return self._hierarchical_planning(goal, context)

    def _setup_task_dependencies(self, plan: Plan) -> None:
        """
        Set up dependencies between tasks based on preconditions and effects.

        Args:
            plan: The plan to set up dependencies for
        """
        # Clear existing dependencies
        plan.task_dependencies = defaultdict(list)

        # For each task
        for task_id, task in plan.tasks.items():
            # Check which other tasks need to come before this one
            for other_id, other_task in plan.tasks.items():
                if task_id == other_id:
                    continue

                # Check if any of other_task's effects satisfy task's preconditions
                for precond in task.preconditions:
                    for effect in other_task.effects:
                        if (effect.effect_type == "state_change" and
                            effect.parameters.get("variable") == precond.parameters.get("variable") and
                            effect.parameters.get("value") == precond.parameters.get("value")):

                            # other_task needs to come before task
                            plan.task_dependencies[task_id].append(other_id)
                            break

    def _execution_loop(self) -> None:
        """Background thread for executing plans"""
        self.logger.info("Plan execution loop started")

        while not self.stop_execution.is_set():
            try:
                # Process each executing plan
                with self.lock:
                    executing_plans = list(self.executing_plans)

                for plan_id in executing_plans:
                    try:
                        self._process_plan(plan_id)
                    except Exception as e:
                        self.logger.error(f"Error processing plan {plan_id}: {str(e)}")

                # Sleep briefly to avoid hogging the CPU
                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error in execution loop: {str(e)}")

        self.logger.info("Plan execution loop stopped")

    def _process_plan(self, plan_id: str) -> None:
        """
        Process a single plan, executing ready tasks.

        Args:
            plan_id: ID of the plan to process
        """
        with self.lock:
            plan = self.plans.get(plan_id)
            if not plan:
                if plan_id in self.executing_plans:
                    self.executing_plans.remove(plan_id)
                return

            # Check if the plan is completed or failed
            if plan.is_completed():
                self.logger.info(f"Plan {plan_id} completed successfully")
                self.executing_plans.remove(plan_id)
                plan.completion_time = time.time()

                # Update goal state
                goal = self.goals.get(plan.goal_id)
                if goal:
                    goal.update_state(GoalState.ACHIEVED, time.time())

                # Record metrics
                self.metrics["plan_success_rate"].append(1.0)
                return

            if plan.is_failed():
                self.logger.warning(f"Plan {plan_id} failed")
                self.executing_plans.remove(plan_id)

                # Update goal state
                goal = self.goals.get(plan.goal_id)
                if goal:
                    goal.update_state(GoalState.FAILED, time.time())

                # Record metrics
                self.metrics["plan_success_rate"].append(0.0)
                return

            # Get ready tasks
            ready_tasks = plan.get_ready_tasks()

            if not ready_tasks and not plan.current_task_id:
                self.logger.warning(f"Plan {plan_id} has no ready tasks")
                return

            # If there's a current task, check its status
            if plan.current_task_id:
                current_task = plan.tasks.get(plan.current_task_id)

                if current_task and current_task.state == TaskState.EXECUTING:
                    # Task is still executing, nothing to do
                    return

                elif current_task and current_task.state == TaskState.COMPLETED:
                    # Task completed, clear current task
                    plan.completed_tasks.append(plan.current_task_id)
                    plan.current_task_id = None

                elif current_task and current_task.state == TaskState.FAILED:
                    # Task failed
                    plan.failed_tasks.append(plan.current_task_id)
                    plan.current_task_id = None

                    # Check if we can retry
                    if current_task.retry_count <= current_task.max_retries:
                        # Reset task state
                        current_task.state = TaskState.PENDING
                    else:
                        # Can't retry, mark as failed
                        self.logger.warning(f"Task {current_task.task_id} failed with no retries left")
                else:
                    # Invalid current task
                    plan.current_task_id = None

            # If no current task and there are ready tasks, start one
            if not plan.current_task_id and ready_tasks:
                # Sort ready tasks by priority (highest first)
                ready_tasks.sort(key=lambda task_id: plan.tasks[task_id].priority, reverse=True)

                # Get the highest priority task
                next_task_id = ready_tasks[0]
                next_task = plan.tasks[next_task_id]

                self.logger.info(f"Executing task {next_task_id}: {next_task.name}")

                # Set as current task
                plan.current_task_id = next_task_id

                # Execute the task
                success, new_state = next_task.execute(plan.state)

                # Update plan state
                plan.update_state(new_state)

                # Record metrics
                self.metrics["task_success_rate"].append(1.0 if success else 0.0)

    def _execute_sub_plan(self, plan_id: str, state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute a sub-plan as part of a task handler.

        Args:
            plan_id: ID of the sub-plan to execute
            state: Current state

        Returns:
            Tuple of (success, new_state)
        """
        with self.lock:
            sub_plan = self.plans.get(plan_id)
            if not sub_plan:
                return False, state

            # Initialize sub-plan with current state
            sub_plan.state = state.copy()

            # Execute tasks in the sub-plan
            completed_tasks = []

            # Keep executing tasks until the plan is complete or failed
            while not sub_plan.is_completed() and not sub_plan.is_failed():
                # Get ready tasks
                ready_tasks = sub_plan.get_ready_tasks()

                if not ready_tasks:
                    break

                # Sort ready tasks by priority (highest first)
                ready_tasks.sort(key=lambda task_id: sub_plan.tasks[task_id].priority, reverse=True)

                # Execute each ready task
                for task_id in ready_tasks:
                    task = sub_plan.tasks[task_id]

                    self.logger.info(f"Executing sub-plan task {task_id}: {task.name}")

                    # Execute the task
                    success, new_state = task.execute(sub_plan.state)

                    # Update sub-plan state
                    sub_plan.update_state(new_state)

                    if success:
                        task.state = TaskState.COMPLETED
                        completed_tasks.append(task_id)
                    else:
                        task.state = TaskState.FAILED
                        return False, state

            return sub_plan.is_completed(), sub_plan.state

    def _reactive_handler(self, goal: Goal, state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Handler for reactive planning tasks.

        Args:
            goal: Goal to achieve
            state: Current state

        Returns:
            Tuple of (success, new_state)
        """
        # This is a simplified implementation
        # In a real system, this would continuously monitor and react

        # Check if the goal is already achieved
        if goal.is_achieved(state):
            return True, state

        # Try to find a task that can get us closer to the goal
        for task_id, task_template in self.task_library.items():
            # Create a copy of the template
            task = Task(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                name=task_template.name,
                description=task_template.description,
                preconditions=task_template.preconditions.copy(),
                effects=task_template.effects.copy(),
                handler=task_template.handler,
                estimated_duration=task_template.estimated_duration,
                priority=task_template.priority
            )

            # Check if the task can be executed
            if task.can_execute(state):
                # Predict effects
                new_state = task.predict_effects(state)

                # Check if this gets us closer to the goal
                current_satisfied = sum(1 for cond in goal.achievement_conditions if cond.check(state))
                predicted_satisfied = sum(1 for cond in goal.achievement_conditions if cond.check(new_state))

                if predicted_satisfied > current_satisfied:
                    # Execute the task
                    self.logger.info(f"Reactively executing task: {task.name}")
                    success, actual_state = task.execute(state)

                    if success:
                        return False, actual_state  # Return False to indicate we need to continue reacting

        # No suitable task found
        self.logger.warning(f"No suitable task found for reactive planning for goal {goal.goal_id}")
        return False, state