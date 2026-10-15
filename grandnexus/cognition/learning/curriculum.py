"""
curriculum.py - Curriculum Learning Module for GrandNexus

This module manages structured learning sequences by organizing tasks based on
their complexity, prerequisites, and learning objectives. It enables GrandNexus
to learn in a progressive manner, mastering foundational skills before tackling
more complex ones, similar to how educational curricula are designed.

Key features:
- Task difficulty estimation and progression management
- Prerequisite relationship tracking between skills/concepts
- Adaptive task selection based on current performance
- Integration with memory systems for skill mastery tracking
- Learning path generation for specific objectives
- Dynamic curriculum adjustment based on learning outcomes
"""

import logging
import threading
import time
import uuid
import random
import math
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Callable, Tuple, Union, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict, Counter
import copy
import heapq
import numpy as np

# Assuming these exist in the architecture
from grandnexus.core.nexus_core import NexusCore
from grandnexus.memory.episodic import EpisodicMemory
from grandnexus.memory.working import WorkingMemory
from grandnexus.memory.semantic import SemanticMemory
from grandnexus.learning.learning_core import LearningCore
from grandnexus.learning.intrinsic_motivation import IntrinsicMotivation

# Task status enum
class TaskStatus(Enum):
    """Status of a learning task in the curriculum"""
    NOT_STARTED = auto()    # Task has not been started yet
    IN_PROGRESS = auto()    # Task is currently being learned
    COMPLETED = auto()      # Task has been successfully completed
    MASTERED = auto()       # Task has been mastered (high proficiency)
    FAILED = auto()         # Task was attempted but not completed successfully
    SKIPPED = auto()        # Task was skipped (e.g., too difficult, not relevant)
    EXPIRED = auto()        # Task is no longer relevant

# Difficulty level enum
class DifficultyLevel(Enum):
    """Difficulty levels for tasks"""
    TRIVIAL = 0
    EASY = 1
    MEDIUM = 2
    HARD = 3
    VERY_HARD = 4
    EXPERT = 5

@dataclass
class LearningTask:
    """
    Represents a specific learning task within a curriculum.
    Tasks are the building blocks of curricula.
    """
    task_id: str
    name: str
    description: str = ""
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    prerequisites: List[str] = field(default_factory=list)  # List of prerequisite task IDs
    estimated_time: float = 0.0  # Estimated time to complete in minutes
    status: TaskStatus = TaskStatus.NOT_STARTED
    proficiency: float = 0.0  # Current proficiency level (0.0 to 1.0)
    attempts: int = 0  # Number of attempts made
    last_attempted: Optional[float] = None  # Timestamp of last attempt
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Function to execute this task (if executable)
    task_function: Optional[Callable] = None

@dataclass
class LearningUnit:
    """
    A group of related learning tasks that form a cohesive unit.
    Similar to a course module or chapter.
    """
    unit_id: str
    name: str
    description: str = ""
    tasks: List[str] = field(default_factory=list)  # List of task IDs
    dependencies: List[str] = field(default_factory=list)  # List of unit IDs this depends on
    completion_threshold: float = 0.8  # Proportion of tasks needed to consider unit complete
    progress: float = 0.0  # Current progress (0.0 to 1.0)
    status: TaskStatus = TaskStatus.NOT_STARTED
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Curriculum:
    """
    A structured learning program composed of learning units and tasks.
    Defines a complete learning path for a subject or skill area.
    """
    curriculum_id: str
    name: str
    description: str = ""
    units: List[str] = field(default_factory=list)  # List of unit IDs
    learning_objectives: List[str] = field(default_factory=list)
    target_audience: str = ""
    estimated_duration: float = 0.0  # Total estimated time in hours
    progress: float = 0.0  # Overall curriculum progress (0.0 to 1.0)
    status: TaskStatus = TaskStatus.NOT_STARTED
    creation_date: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceRecord:
    """
    Records performance on a specific task attempt.
    Used to track learning progress over time.
    """
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    timestamp: float = field(default_factory=time.time)
    score: float = 0.0  # Performance score (0.0 to 1.0)
    time_spent: float = 0.0  # Time spent in minutes
    mistakes: int = 0  # Number of mistakes or errors
    completed: bool = False  # Whether the task was completed
    feedback: str = ""  # Feedback on performance
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CurriculumConfig:
    """Configuration settings for the curriculum module"""
    # Mastery thresholds
    mastery_threshold: float = 0.9  # Score needed to consider a task mastered
    proficiency_decay_rate: float = 0.005  # Daily decay rate for unused skills

    # Task selection parameters
    difficulty_weight: float = 0.3  # How much difficulty influences task selection
    novelty_weight: float = 0.2  # Preference for tasks not attempted recently
    relevance_weight: float = 0.4  # Importance of task to current goals

    # Adaptive curriculum parameters
    adaptation_enabled: bool = True  # Whether to adapt curriculum based on performance
    max_consecutive_failures: int = 3  # Maximum failures before suggesting easier tasks
    review_interval_days: int = 14  # Days before suggesting review of previously mastered tasks

    # Integration parameters
    use_intrinsic_motivation: bool = True  # Use intrinsic motivation for task selection
    use_episodic_memory: bool = True  # Record performance in episodic memory
    use_semantic_memory: bool = True  # Store curriculum structure in semantic memory

    # Custom parameters
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

class CurriculumManager:
    """
    Core class for managing learning curricula within GrandNexus.

    This module organizes learning into structured sequences, tracks progress,
    and selects appropriate tasks based on the system's current capabilities
    and learning goals.
    """

    def __init__(self, config: Optional[CurriculumConfig] = None,
                nexus_core: Optional[NexusCore] = None,
                episodic_memory: Optional[EpisodicMemory] = None,
                working_memory: Optional[WorkingMemory] = None,
                semantic_memory: Optional[SemanticMemory] = None,
                learning_core: Optional[LearningCore] = None,
                intrinsic_motivation: Optional[IntrinsicMotivation] = None):
        """
        Initialize the curriculum manager.

        Args:
            config: Configuration settings
            nexus_core: Reference to NexusCore
            episodic_memory: Reference to episodic memory
            working_memory: Reference to working memory
            semantic_memory: Reference to semantic memory
            learning_core: Reference to learning core
            intrinsic_motivation: Reference to intrinsic motivation module
        """
        self.logger = logging.getLogger("GrandNexus.Learning.Curriculum")

        # Store configuration
        self.config = config or CurriculumConfig()

        # Store module references
        self.nexus_core = nexus_core
        self.episodic_memory = episodic_memory
        self.working_memory = working_memory
        self.semantic_memory = semantic_memory
        self.learning_core = learning_core
        self.intrinsic_motivation = intrinsic_motivation

        # Internal state
        self.instance_id = str(uuid.uuid4())[:8]
        self.lock = threading.RLock()
        self.initialized = False
        self.running = False

        # Storage for curricula, units, tasks, and performance records
        self.curricula = {}
        self.learning_units = {}
        self.learning_tasks = {}
        self.performance_records = {}

        # Current learning state
        self.active_curriculum = None
        self.active_unit = None
        self.active_task = None
        self.current_focus = []  # List of tasks currently in focus

        # Skill mastery tracking
        self.skill_mastery = {}  # Mapping of skill IDs to mastery levels

        # Learning path generation
        self.learning_paths = {}  # Stored learning paths for different objectives

        # Performance metrics
        self.metrics = defaultdict(lambda: deque(maxlen=100))

        self.logger.info(f"CurriculumManager initialized with ID={self.instance_id}")

    def initialize(self) -> bool:
        """
        Initialize the curriculum manager.

        Returns:
            True if initialization was successful, False otherwise
        """
        if self.initialized:
            return True

        with self.lock:
            try:
                # Register with NexusCore if available
                if self.nexus_core:
                    self._register_with_nexus_core()

                # Load any persistent curricula
                self._load_from_storage()

                self.initialized = True
                self.logger.info("CurriculumManager initialization complete")
                return True
            except Exception as e:
                self.logger.error(f"CurriculumManager initialization failed: {str(e)}")
                return False

    def create_curriculum(self, name: str, description: str,
                         learning_objectives: List[str],
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new curriculum.

        Args:
            name: Name of the curriculum
            description: Description of what this curriculum teaches
            learning_objectives: List of learning objectives
            metadata: Additional information about the curriculum

        Returns:
            The ID of the created curriculum
        """
        with self.lock:
            # Generate curriculum ID
            curriculum_id = f"curriculum_{str(uuid.uuid4())[:8]}"

            # Create curriculum
            curriculum = Curriculum(
                curriculum_id=curriculum_id,
                name=name,
                description=description,
                learning_objectives=learning_objectives,
                metadata=metadata or {}
            )

            # Store curriculum
            self.curricula[curriculum_id] = curriculum

            self.logger.info(f"Created curriculum: {name} ({curriculum_id})")
            return curriculum_id

    def create_learning_unit(self, curriculum_id: str, name: str,
                           description: str, dependencies: Optional[List[str]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new learning unit within a curriculum.

        Args:
            curriculum_id: ID of the curriculum this unit belongs to
            name: Name of the learning unit
            description: Description of the unit's content
            dependencies: IDs of other units this one depends on
            metadata: Additional information about the unit

        Returns:
            The ID of the created learning unit
        """
        with self.lock:
            if curriculum_id not in self.curricula:
                raise ValueError(f"Curriculum {curriculum_id} not found")

            # Generate unit ID
            unit_id = f"unit_{str(uuid.uuid4())[:8]}"

            # Create learning unit
            unit = LearningUnit(
                unit_id=unit_id,
                name=name,
                description=description,
                dependencies=dependencies or [],
                metadata=metadata or {}
            )

            # Store unit
            self.learning_units[unit_id] = unit

            # Add to curriculum
            self.curricula[curriculum_id].units.append(unit_id)
            self.curricula[curriculum_id].last_modified = time.time()

            self.logger.info(f"Created learning unit: {name} ({unit_id}) in curriculum {curriculum_id}")
            return unit_id

    def create_learning_task(self, unit_id: str, name: str,
                           description: str, difficulty: DifficultyLevel,
                           prerequisites: Optional[List[str]] = None,
                           estimated_time: float = 30.0,
                           task_function: Optional[Callable] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new learning task within a unit.

        Args:
            unit_id: ID of the unit this task belongs to
            name: Name of the task
            description: Description of what the task involves
            difficulty: Difficulty level of the task
            prerequisites: IDs of tasks that should be completed before this one
            estimated_time: Estimated time to complete in minutes
            task_function: Optional function to execute this task
            metadata: Additional information about the task

        Returns:
            The ID of the created task
        """
        with self.lock:
            if unit_id not in self.learning_units:
                raise ValueError(f"Learning unit {unit_id} not found")

            # Generate task ID
            task_id = f"task_{str(uuid.uuid4())[:8]}"

            # Create learning task
            task = LearningTask(
                task_id=task_id,
                name=name,
                description=description,
                difficulty=difficulty,
                prerequisites=prerequisites or [],
                estimated_time=estimated_time,
                task_function=task_function,
                metadata=metadata or {}
            )

            # Store task
            self.learning_tasks[task_id] = task

            # Add to unit
            self.learning_units[unit_id].tasks.append(task_id)

            self.logger.info(f"Created learning task: {name} ({task_id}) in unit {unit_id}")
            return task_id

    def get_curriculum(self, curriculum_id: str) -> Optional[Curriculum]:
        """
        Get a curriculum by ID.

        Args:
            curriculum_id: ID of the curriculum to retrieve

        Returns:
            The curriculum if found, None otherwise
        """
        with self.lock:
            return self.curricula.get(curriculum_id)

    def get_learning_unit(self, unit_id: str) -> Optional[LearningUnit]:
        """
        Get a learning unit by ID.

        Args:
            unit_id: ID of the learning unit to retrieve

        Returns:
            The learning unit if found, None otherwise
        """
        with self.lock:
            return self.learning_units.get(unit_id)

    def get_learning_task(self, task_id: str) -> Optional[LearningTask]:
        """
        Get a learning task by ID.

        Args:
            task_id: ID of the learning task to retrieve

        Returns:
            The learning task if found, None otherwise
        """
        with self.lock:
            return self.learning_tasks.get(task_id)

    def set_active_curriculum(self, curriculum_id: str) -> bool:
        """
        Set the active curriculum being worked on.

        Args:
            curriculum_id: ID of the curriculum to activate

        Returns:
            True if successful, False if curriculum not found
        """
        with self.lock:
            if curriculum_id not in self.curricula:
                self.logger.warning(f"Curriculum {curriculum_id} not found")
                return False

            self.active_curriculum = curriculum_id

            # Mark as in progress if not already
            if self.curricula[curriculum_id].status == TaskStatus.NOT_STARTED:
                self.curricula[curriculum_id].status = TaskStatus.IN_PROGRESS

            self.logger.info(f"Set active curriculum to: {self.curricula[curriculum_id].name}")
            return True

    def set_active_unit(self, unit_id: str) -> bool:
        """
        Set the active learning unit being worked on.

        Args:
            unit_id: ID of the learning unit to activate

        Returns:
            True if successful, False if unit not found
        """
        with self.lock:
            if unit_id not in self.learning_units:
                self.logger.warning(f"Learning unit {unit_id} not found")
                return False

            self.active_unit = unit_id

            # Mark as in progress if not already
            if self.learning_units[unit_id].status == TaskStatus.NOT_STARTED:
                self.learning_units[unit_id].status = TaskStatus.IN_PROGRESS

            self.logger.info(f"Set active unit to: {self.learning_units[unit_id].name}")
            return True

    def get_next_task(self) -> Optional[LearningTask]:
        """
        Get the next recommended learning task from the active curriculum.

        Returns:
            The next recommended task, or None if no suitable task is found
        """
        with self.lock:
            if not self.active_curriculum:
                self.logger.warning("No active curriculum set")
                return None

            if not self.active_unit:
                # Recommend next unit if none active
                next_unit_id = self._get_next_unit_id()
                if next_unit_id:
                    self.set_active_unit(next_unit_id)
                else:
                    self.logger.warning("No suitable unit found")
                    return None

            # Get recommended task from active unit
            next_task_id = self._get_next_task_id()
            if not next_task_id:
                # Try finding a task from another unit
                next_unit_id = self._get_next_unit_id()
                if next_unit_id:
                    self.set_active_unit(next_unit_id)
                    next_task_id = self._get_next_task_id()

                if not next_task_id:
                    self.logger.warning("No suitable task found")
                    return None

            # Set as active task
            self.active_task = next_task_id

            # Mark as in progress
            task = self.learning_tasks[next_task_id]
            if task.status == TaskStatus.NOT_STARTED:
                task.status = TaskStatus.IN_PROGRESS

            # Add to current focus
            if next_task_id not in self.current_focus:
                self.current_focus.append(next_task_id)
                # Limit focus size
                if len(self.current_focus) > 5:
                    self.current_focus.pop(0)

            return task

    def record_performance(self, task_id: str, score: float, time_spent: float,
                          completed: bool, mistakes: int = 0,
                          feedback: str = "", metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Record performance on a learning task.

        Args:
            task_id: ID of the task
            score: Performance score (0.0 to 1.0)
            time_spent: Time spent on the task in minutes
            completed: Whether the task was completed
            mistakes: Number of mistakes made
            feedback: Feedback on the performance
            metadata: Additional data about the performance

        Returns:
            ID of the created performance record
        """
        with self.lock:
            if task_id not in self.learning_tasks:
                raise ValueError(f"Learning task {task_id} not found")

            # Create performance record
            record_id = f"perf_{str(uuid.uuid4())[:8]}"
            record = PerformanceRecord(
                record_id=record_id,
                task_id=task_id,
                score=score,
                time_spent=time_spent,
                completed=completed,
                mistakes=mistakes,
                feedback=feedback,
                metadata=metadata or {}
            )

            # Store record
            self.performance_records[record_id] = record

            # Update task status and proficiency
            task = self.learning_tasks[task_id]
            task.attempts += 1
            task.last_attempted = time.time()

            # Update proficiency (weighted average with previous proficiency)
            if task.attempts == 1:
                task.proficiency = score
            else:
                # Higher weight for most recent attempt
                task.proficiency = 0.3 * task.proficiency + 0.7 * score

            # Update status based on performance
            if completed:
                if score >= self.config.mastery_threshold:
                    task.status = TaskStatus.MASTERED
                else:
                    task.status = TaskStatus.COMPLETED
            else:
                if task.attempts >= self.config.max_consecutive_failures:
                    task.status = TaskStatus.FAILED
                else:
                    task.status = TaskStatus.IN_PROGRESS

            # Update unit progress
            if task_id in self.learning_tasks and self.active_unit:
                unit = self.learning_units[self.active_unit]
                completed_tasks = sum(1 for tid in unit.tasks
                                     if tid in self.learning_tasks and
                                     self.learning_tasks[tid].status in
                                     [TaskStatus.COMPLETED, TaskStatus.MASTERED])
                unit.progress = completed_tasks / max(1, len(unit.tasks))

                # Update unit status if all tasks completed
                if unit.progress >= unit.completion_threshold:
                    unit.status = TaskStatus.COMPLETED

            # Update curriculum progress
            if self.active_curriculum:
                curriculum = self.curricula[self.active_curriculum]
                completed_units = sum(1 for uid in curriculum.units
                                     if uid in self.learning_units and
                                     self.learning_units[uid].status in
                                     [TaskStatus.COMPLETED, TaskStatus.MASTERED])
                curriculum.progress = completed_units / max(1, len(curriculum.units))

                # Update curriculum status if all units completed
                if curriculum.progress >= 1.0:
                    curriculum.status = TaskStatus.COMPLETED

            # Update skill mastery
            self._update_skill_mastery(task_id, score, completed)

            # Store in memory systems
            self._store_performance_in_memory(record)

            # Generate learning feedback
            if self.config.adaptation_enabled:
                self._adapt_curriculum(task_id, score, completed)

            self.logger.info(f"Recorded performance for task {task.name}: "
                           f"score={score:.2f}, completed={completed}")

            # Notify intrinsic motivation if available
            if (self.intrinsic_motivation and self.config.use_intrinsic_motivation and
                not getattr(record, 'from_intrinsic_motivation', False)):
                try:
                    self.intrinsic_motivation.update_competence(
                        skill_area=task.metadata.get('skill_area', task.name),
                        performance=score,
                        observation=self._create_observation_data(task, record)
                    )
                except Exception as e:
                    self.logger.warning(f"Error updating intrinsic motivation: {str(e)}")

            # Update metrics
            self.metrics["task_scores"].append(score)
            self.metrics["task_completion"].append(1.0 if completed else 0.0)

            return record_id

    def get_learning_path(self, objective: str) -> List[str]:
        """
        Get a recommended learning path for a specific objective.

        Args:
            objective: Learning objective to achieve

        Returns:
            List of task IDs forming a learning path
        """
        with self.lock:
            # Check if we already have a computed path
            if objective in self.learning_paths:
                return self.learning_paths[objective]

            # Find tasks related to this objective
            relevant_tasks = []
            for task_id, task in self.learning_tasks.items():
                if (objective.lower() in task.name.lower() or
                    objective.lower() in task.description.lower() or
                    objective.lower() in str(task.metadata).lower()):
                    relevant_tasks.append(task_id)

            if not relevant_tasks:
                self.logger.warning(f"No tasks found for objective: {objective}")
                return []

            # Build dependency graph
            graph = self._build_dependency_graph()

            # Find path to each relevant task
            all_paths = []
            for task_id in relevant_tasks:
                path = self._find_path_to_task(task_id, graph)
                if path:
                    all_paths.append(path)

            # Combine paths and remove duplicates
            combined_path = []
            for path in all_paths:
                for task_id in path:
                    if task_id not in combined_path:
                        combined_path.append(task_id)

            # Sort by difficulty and prerequisites
            sorted_path = self._topological_sort(combined_path, graph)

            # Store for future reference
            self.learning_paths[objective] = sorted_path

            return sorted_path

    def get_skill_mastery(self, skill_id: str) -> float:
        """
        Get the current mastery level for a skill.

        Args:
            skill_id: ID of the skill

        Returns:
            Mastery level from 0.0 to 1.0
        """
        with self.lock:
            return self.skill_mastery.get(skill_id, 0.0)

    def get_performance_history(self, task_id: str) -> List[PerformanceRecord]:
        """
        Get performance history for a specific task.

        Args:
            task_id: ID of the task

        Returns:
            List of performance records for the task
        """
        with self.lock:
            return [record for record in self.performance_records.values()
                   if record.task_id == task_id]

    def get_curriculum_progress(self, curriculum_id: str) -> Dict[str, Any]:
        """
        Get detailed progress information for a curriculum.

        Args:
            curriculum_id: ID of the curriculum

        Returns:
            Dictionary with progress information
        """
        with self.lock:
            if curriculum_id not in self.curricula:
                raise ValueError(f"Curriculum {curriculum_id} not found")

            curriculum = self.curricula[curriculum_id]

            # Collect unit progress
            units_progress = {}
            for unit_id in curriculum.units:
                if unit_id in self.learning_units:
                    unit = self.learning_units[unit_id]

                    # Collect task progress
                    tasks_progress = {}
                    for task_id in unit.tasks:
                        if task_id in self.learning_tasks:
                            task = self.learning_tasks[task_id]
                            tasks_progress[task_id] = {
                                "name": task.name,
                                "status": task.status.name,
                                "proficiency": task.proficiency,
                                "attempts": task.attempts
                            }

                    units_progress[unit_id] = {
                        "name": unit.name,
                        "progress": unit.progress,
                        "status": unit.status.name,
                        "tasks": tasks_progress
                    }

            return {
                "name": curriculum.name,
                "progress": curriculum.progress,
                "status": curriculum.status.name,
                "units": units_progress
            }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the curriculum manager.

        Returns:
            Dictionary with metrics
        """
        with self.lock:
            metrics = {}

            # Process metric queues
            for key, values in self.metrics.items():
                if values:
                    metrics[key] = {
                        "current": values[-1],
                        "mean": float(np.mean(values)),
                        "max": float(np.max(values)),
                        "min": float(np.min(values))
                    }

            # Add curriculum stats
            metrics["curricula_count"] = len(self.curricula)
            metrics["units_count"] = len(self.learning_units)
            metrics["tasks_count"] = len(self.learning_tasks)
            metrics["performance_records_count"] = len(self.performance_records)

            if self.active_curriculum:
                metrics["active_curriculum"] = {
                    "id": self.active_curriculum,
                    "name": self.curricula[self.active_curriculum].name,
                    "progress": self.curricula[self.active_curriculum].progress
                }

            return metrics

    def reset_progress(self, curriculum_id: Optional[str] = None) -> None:
        """
        Reset progress for a curriculum or all curricula.

        Args:
            curriculum_id: ID of curriculum to reset, or None for all
        """
        with self.lock:
            if curriculum_id:
                if curriculum_id not in self.curricula:
                    raise ValueError(f"Curriculum {curriculum_id} not found")

                # Reset curriculum progress
                self.curricula[curriculum_id].progress = 0.0
                self.curricula[curriculum_id].status = TaskStatus.NOT_STARTED

                # Reset units in this curriculum
                for unit_id in self.curricula[curriculum_id].units:
                    if unit_id in self.learning_units:
                        self.learning_units[unit_id].progress = 0.0
                        self.learning_units[unit_id].status = TaskStatus.NOT_STARTED

                        # Reset tasks in this unit
                        for task_id in self.learning_units[unit_id].tasks:
                            if task_id in self.learning_tasks:
                                self.learning_tasks[task_id].status = TaskStatus.NOT_STARTED
                                self.learning_tasks[task_id].proficiency = 0.0
                                self.learning_tasks[task_id].attempts = 0
                                self.learning_tasks[task_id].last_attempted = None

                self.logger.info(f"Reset progress for curriculum: {self.curricula[curriculum_id].name}")
            else:
                # Reset all curricula
                for curr_id, curriculum in self.curricula.items():
                    curriculum.progress = 0.0
                    curriculum.status = TaskStatus.NOT_STARTED

                # Reset all units
                for unit_id, unit in self.learning_units.items():
                    unit.progress = 0.0
                    unit.status = TaskStatus.NOT_STARTED

                # Reset all tasks
                for task_id, task in self.learning_tasks.items():
                    task.status = TaskStatus.NOT_STARTED
                    task.proficiency = 0.0
                    task.attempts = 0
                    task.last_attempted = None

                self.logger.info(f"Reset progress for all curricula")

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
            if self.episodic_memory:
                dependencies.append("episodic_memory")
            if self.working_memory:
                dependencies.append("working_memory")
            if self.semantic_memory:
                dependencies.append("semantic_memory")
            if self.learning_core:
                dependencies.append("learning_core")
            if self.intrinsic_motivation:
                dependencies.append("intrinsic_motivation")

            self.nexus_core.register_module(
                name="curriculum",
                module=self,
                dependencies=dependencies
            )

            self.logger.info("Successfully registered with NexusCore")
        except Exception as e:
            self.logger.error(f"Failed to register with NexusCore: {str(e)}")

    def _load_from_storage(self) -> None:
        """Load curricula from persistent storage"""
        # In a real implementation, this would load from disk or database
        # For simplicity, we'll just log a message
        self.logger.info("No persistent curricula to load")

    def _get_next_unit_id(self) -> Optional[str]:
        """
        Get the ID of the next recommended unit in the active curriculum.

        Returns:
            The ID of the next unit, or None if no suitable unit is found
        """
        if not self.active_curriculum:
            return None

        curriculum = self.curricula[self.active_curriculum]

        # Find incomplete units
        candidate_units = []
        for unit_id in curriculum.units:
            if unit_id in self.learning_units:
                unit = self.learning_units[unit_id]
                if unit.status in [TaskStatus.NOT_STARTED, TaskStatus.IN_PROGRESS]:
                    # Check if dependencies are satisfied
                    dependencies_met = True
                    for dep_id in unit.dependencies:
                        if dep_id in self.learning_units:
                            dep_unit = self.learning_units[dep_id]
                            if dep_unit.status != TaskStatus.COMPLETED:
                                dependencies_met = False
                                break

                    if dependencies_met:
                        candidate_units.append(unit_id)

        if not candidate_units:
            return None

        # Sort by progress (prefer units already in progress)
        candidate_units.sort(key=lambda uid:
                             1 if self.learning_units[uid].status == TaskStatus.IN_PROGRESS else 0,
                             reverse=True)

        return candidate_units[0]

    def _get_next_task_id(self) -> Optional[str]:
        """
        Get the ID of the next recommended task in the active unit.

        Returns:
            The ID of the next task, or None if no suitable task is found
        """
        if not self.active_unit:
            return None

        unit = self.learning_units[self.active_unit]

        # Find incomplete tasks
        candidate_tasks = []
        for task_id in unit.tasks:
            if task_id in self.learning_tasks:
                task = self.learning_tasks[task_id]
                if task.status in [TaskStatus.NOT_STARTED, TaskStatus.IN_PROGRESS]:
                    # Check if prerequisites are satisfied
                    prerequisites_met = True
                    for prereq_id in task.prerequisites:
                        if prereq_id in self.learning_tasks:
                            prereq_task = self.learning_tasks[prereq_id]
                            if prereq_task.status not in [TaskStatus.COMPLETED, TaskStatus.MASTERED]:
                                prerequisites_met = False
                                break

                    if prerequisites_met:
                        candidate_tasks.append(task_id)

        if not candidate_tasks:
            return None

        # If intrinsic motivation is available, use it to influence task selection
        if self.intrinsic_motivation and self.config.use_intrinsic_motivation:
            try:
                # Get curiosity state
                curiosity_state = self.intrinsic_motivation.get_curiosity_state()

                # Calculate task scores based on novelty, difficulty, and interest
                task_scores = []
                for task_id in candidate_tasks:
                    task = self.learning_tasks[task_id]

                    # Base score
                    score = 0.5

                    # Adjust for novelty preference
                    novelty_factor = 0.0
                    if task.last_attempted is None:
                        # Never attempted
                        novelty_factor = 1.0
                    else:
                        # Inversely proportional to recency
                        days_since = (time.time() - task.last_attempted) / (24 * 3600)
                        novelty_factor = min(1.0, days_since / 7.0)  # Max after a week

                    score += curiosity_state.novelty_seeking * novelty_factor * self.config.novelty_weight

                    # Adjust for difficulty preference
                    difficulty_factor = task.difficulty.value / DifficultyLevel.EXPERT.value
                    # If exploration is high, prefer harder tasks
                    if curiosity_state.exploration_rate > 0.7:
                        score += difficulty_factor * self.config.difficulty_weight
                    else:
                        # Otherwise prefer tasks of moderate difficulty
                        moderate_difficulty = 1.0 - abs(difficulty_factor - 0.5) * 2
                        score += moderate_difficulty * self.config.difficulty_weight

                    # Adjust for relevance to current focus
                    if curiosity_state.attention_focus:
                        relevance = 0.0
                        focus_str = str(curiosity_state.attention_focus).lower()

                        if focus_str in task.name.lower() or focus_str in task.description.lower():
                            relevance = 1.0
                        elif any(focus_str in str(meta_val).lower()
                               for meta_val in task.metadata.values()):
                            relevance = 0.8

                        score += relevance * self.config.relevance_weight * curiosity_state.focus_strength

                    task_scores.append((task_id, score))

                # Sort by score (highest first)
                task_scores.sort(key=lambda x: x[1], reverse=True)

                if task_scores:
                    self.logger.debug(f"Selected task {task_scores[0][0]} with score {task_scores[0][1]:.2f}")
                    return task_scores[0][0]

            except Exception as e:
                self.logger.warning(f"Error using intrinsic motivation for task selection: {str(e)}")

        # Fallback: sort by difficulty (easiest first)
        candidate_tasks.sort(key=lambda tid: self.learning_tasks[tid].difficulty.value)

        return candidate_tasks[0]

    def _update_skill_mastery(self, task_id: str, score: float, completed: bool) -> None:
        """
        Update skill mastery based on task performance.

        Args:
            task_id: ID of the task
            score: Performance score
            completed: Whether the task was completed
        """
        if task_id not in self.learning_tasks:
            return

        task = self.learning_tasks[task_id]

        # Get skill areas from task metadata
        skill_areas = []
        if 'skill_area' in task.metadata:
            if isinstance(task.metadata['skill_area'], list):
                skill_areas.extend(task.metadata['skill_area'])
            else:
                skill_areas.append(task.metadata['skill_area'])

        if 'skills' in task.metadata:
            if isinstance(task.metadata['skills'], list):
                skill_areas.extend(task.metadata['skills'])
            elif isinstance(task.metadata['skills'], dict):
                skill_areas.extend(task.metadata['skills'].keys())
            else:
                skill_areas.append(task.metadata['skills'])

        # If no skills specified, use task name
        if not skill_areas:
            skill_areas.append(task.name)

        # Update mastery for each skill
        for skill in skill_areas:
            skill_id = str(skill)

            # Get current mastery
            current_mastery = self.skill_mastery.get(skill_id, 0.0)

            # If task was completed, update mastery
            if completed:
                # Weight based on task difficulty
                difficulty_weight = (task.difficulty.value + 1) / (DifficultyLevel.EXPERT.value + 1)

                # Higher difficulty gives more mastery gain
                mastery_gain = score * difficulty_weight * 0.2

                # New mastery is weighted average of current and gain
                new_mastery = current_mastery * 0.8 + mastery_gain

                # Cap at 1.0
                new_mastery = min(1.0, new_mastery)

                # Update mastery
                self.skill_mastery[skill_id] = new_mastery

                if new_mastery > current_mastery + 0.1:
                    self.logger.info(f"Skill mastery increased for {skill_id}: "
                                    f"{current_mastery:.2f} -> {new_mastery:.2f}")

    def _adapt_curriculum(self, task_id: str, score: float, completed: bool) -> None:
        """
        Adapt curriculum based on learning performance.

        Args:
            task_id: ID of the task
            score: Performance score
            completed: Whether the task was completed
        """
        if not self.config.adaptation_enabled:
            return

        task = self.learning_tasks[task_id]

        # Handle failed tasks
        if not completed and score < 0.3:
            consecutive_failures = 0

            # Check if there are multiple recent failures
            for record in reversed(list(self.performance_records.values())):
                if record.task_id == task_id and record.score < 0.3:
                    consecutive_failures += 1
                else:
                    break

            if consecutive_failures >= self.config.max_consecutive_failures:
                # Suggest easier alternative tasks
                self._suggest_easier_alternatives(task_id)

        # Handle completed tasks
        elif completed and score > 0.7:
            # Check if mastery review is needed for any related skills
            self._check_for_review_needed(task)

    def _suggest_easier_alternatives(self, task_id: str) -> List[str]:
        """
        Suggest easier alternative tasks for a failed task.

        Args:
            task_id: ID of the failed task

        Returns:
            List of suggested alternative task IDs
        """
        if task_id not in self.learning_tasks:
            return []

        task = self.learning_tasks[task_id]

        # Find easier tasks with similar skills
        alternatives = []

        # Get task skills
        task_skills = set()
        if 'skill_area' in task.metadata:
            if isinstance(task.metadata['skill_area'], list):
                task_skills.update(task.metadata['skill_area'])
            else:
                task_skills.add(task.metadata['skill_area'])

        if 'skills' in task.metadata:
            if isinstance(task.metadata['skills'], list):
                task_skills.update(task.metadata['skills'])
            elif isinstance(task.metadata['skills'], dict):
                task_skills.update(task.metadata['skills'].keys())
            else:
                task_skills.add(task.metadata['skills'])

        # If no skills, use task name
        if not task_skills:
            task_skills.add(task.name)

        # Find all tasks with similar skills but lower difficulty
        for tid, t in self.learning_tasks.items():
            if tid == task_id:
                continue

            # Check if easier
            if t.difficulty.value >= task.difficulty.value:
                continue

            # Check for skill overlap
            t_skills = set()
            if 'skill_area' in t.metadata:
                if isinstance(t.metadata['skill_area'], list):
                    t_skills.update(t.metadata['skill_area'])
                else:
                    t_skills.add(t.metadata['skill_area'])

            if 'skills' in t.metadata:
                if isinstance(t.metadata['skills'], list):
                    t_skills.update(t.metadata['skills'])
                elif isinstance(t.metadata['skills'], dict):
                    t_skills.update(t.metadata['skills'].keys())
                else:
                    t_skills.add(t.metadata['skills'])

            # If no skills, use task name
            if not t_skills:
                t_skills.add(t.name)

            # Check overlap
            if task_skills.intersection(t_skills):
                alternatives.append(tid)

        # Log suggestion
        if alternatives:
            alt_names = [self.learning_tasks[tid].name for tid in alternatives[:3]]
            self.logger.info(f"Suggesting easier alternatives for {task.name}: {', '.join(alt_names)}")

        return alternatives

    def _check_for_review_needed(self, task: LearningTask) -> List[str]:
        """
        Check if any related skills need review.

        Args:
            task: The current task

        Returns:
            List of task IDs recommended for review
        """
        # Get current task skills
        task_skills = set()
        if 'skill_area' in task.metadata:
            if isinstance(task.metadata['skill_area'], list):
                task_skills.update(task.metadata['skill_area'])
            else:
                task_skills.add(task.metadata['skill_area'])

        if 'skills' in task.metadata:
            if isinstance(task.metadata['skills'], list):
                task_skills.update(task.metadata['skills'])
            elif isinstance(task.metadata['skills'], dict):
                task_skills.update(task.metadata['skills'].keys())
            else:
                task_skills.add(task.metadata['skills'])

        # Find tasks that haven't been practiced recently
        now = time.time()
        review_candidates = []

        for tid, t in self.learning_tasks.items():
            if tid == task.task_id:
                continue

            # Check if previously mastered
            if t.status != TaskStatus.MASTERED:
                continue

            # Check if not practiced recently
            if t.last_attempted:
                days_since = (now - t.last_attempted) / (24 * 3600)
                if days_since < self.config.review_interval_days:
                    continue

            # Check for skill overlap
            t_skills = set()
            if 'skill_area' in t.metadata:
                if isinstance(t.metadata['skill_area'], list):
                    t_skills.update(t.metadata['skill_area'])
                else:
                    t_skills.add(t.metadata['skill_area'])

            if 'skills' in t.metadata:
                if isinstance(t.metadata['skills'], list):
                    t_skills.update(t.metadata['skills'])
                elif isinstance(t.metadata['skills'], dict):
                    t_skills.update(t.metadata['skills'].keys())
                else:
                    t_skills.add(t.metadata['skills'])

            # If skills overlap, add to review candidates
            if task_skills.intersection(t_skills):
                review_candidates.append(tid)

        # Log recommendation
        if review_candidates:
            review_names = [self.learning_tasks[tid].name for tid in review_candidates[:3]]
            self.logger.info(f"Recommending review of: {', '.join(review_names)}")

        return review_candidates

    def _store_performance_in_memory(self, record: PerformanceRecord) -> None:
        """
        Store performance record in memory systems.

        Args:
            record: The performance record to store
        """
        # Store in episodic memory if available
        if self.episodic_memory and self.config.use_episodic_memory:
            try:
                task_id = record.task_id
                task = self.learning_tasks.get(task_id)

                if task:
                    # Create episodic memory event
                    self.episodic_memory.add_event(
                        episode_id="curriculum_learning",  # This should be a real episode ID
                        content={
                            "type": "task_performance",
                            "task_name": task.name,
                            "score": record.score,
                            "completed": record.completed,
                            "feedback": record.feedback
                        },
                        metadata={
                            "source": "curriculum",
                            "task_id": task_id,
                            "importance": min(1.0, record.score + 0.3) if record.completed else 0.3
                        }
                    )
            except Exception as e:
                self.logger.warning(f"Error storing in episodic memory: {str(e)}")

        # Store in semantic memory if available
        if self.semantic_memory and self.config.use_semantic_memory:
            try:
                task_id = record.task_id
                task = self.learning_tasks.get(task_id)

                if task:
                    # Get skill areas from task
                    skill_areas = []
                    if 'skill_area' in task.metadata:
                        if isinstance(task.metadata['skill_area'], list):
                            skill_areas.extend(task.metadata['skill_area'])
                        else:
                            skill_areas.append(task.metadata['skill_area'])

                    # Create skill concepts or update them
                    for skill in skill_areas:
                        # This is simplified - actual usage would depend on semantic memory API
                        skill_concept = {
                            "name": str(skill),
                            "description": f"Skill area related to task {task.name}",
                            "properties": {
                                "mastery": self.skill_mastery.get(str(skill), 0.0),
                                "last_practiced": time.time()
                            }
                        }

                        self.semantic_memory.add_concept(skill_concept)
            except Exception as e:
                self.logger.warning(f"Error storing in semantic memory: {str(e)}")

    def _create_observation_data(self, task: LearningTask,
                               record: PerformanceRecord) -> Any:
        """
        Create observation data for intrinsic motivation.

        Args:
            task: The task
            record: The performance record

        Returns:
            Observation data object
        """
        if not self.intrinsic_motivation:
            return None

        # Format depends on intrinsic motivation module's expectations
        # This is a simplified approach
        try:
            from grandnexus.learning.intrinsic_motivation import ObservationData

            observation = ObservationData(
                content=f"Task performance: {task.name}, score={record.score:.2f}",
                context={
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "difficulty": task.difficulty.value,
                    "completed": record.completed,
                    "skill_area": task.metadata.get("skill_area", task.name)
                },
                source="curriculum",
                metadata={
                    "from_curriculum": True,
                    "performance_record_id": record.record_id
                }
            )

            return observation
        except (ImportError, AttributeError):
            # Fall back to a simple dictionary
            return {
                "content": f"Task performance: {task.name}, score={record.score:.2f}",
                "context": {
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "difficulty": task.difficulty.value,
                    "completed": record.completed
                },
                "source": "curriculum",
                "timestamp": time.time()
            }

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Build a dependency graph for tasks and units.

        Returns:
            Dictionary mapping task IDs to their dependencies
        """
        graph = {}

        # Add task prerequisites
        for task_id, task in self.learning_tasks.items():
            graph[task_id] = task.prerequisites.copy()

        # Add unit dependencies (all tasks in dependent units are prerequisites)
        for unit_id, unit in self.learning_units.items():
            for task_id in unit.tasks:
                if task_id not in graph:
                    graph[task_id] = []

                # Add prerequisites from dependent units
                for dep_unit_id in unit.dependencies:
                    if dep_unit_id in self.learning_units:
                        dep_unit = self.learning_units[dep_unit_id]
                        for dep_task_id in dep_unit.tasks:
                            if dep_task_id != task_id and dep_task_id not in graph[task_id]:
                                graph[task_id].append(dep_task_id)

        return graph

    def _find_path_to_task(self, task_id: str,
                         graph: Dict[str, List[str]]) -> List[str]:
        """
        Find a path to a task including all prerequisites.

        Args:
            task_id: Target task ID
            graph: Dependency graph

        Returns:
            List of task IDs forming a path to the target
        """
        if task_id not in graph:
            return []

        # Use DFS to find all dependencies
        visited = set()
        path = []

        def dfs(node):
            if node in visited:
                return

            visited.add(node)

            # Visit dependencies first
            for dep in graph.get(node, []):
                dfs(dep)

            # Add node to path
            path.append(node)

        dfs(task_id)

        return path

    def _topological_sort(self, tasks: List[str],
                        graph: Dict[str, List[str]]) -> List[str]:
        """
        Sort tasks topologically (prerequisites before dependents).

        Args:
            tasks: List of task IDs to sort
            graph: Dependency graph

        Returns:
            Topologically sorted list of task IDs
        """
        # Filter tasks to include only those in the graph
        valid_tasks = [t for t in tasks if t in graph]

        # Create subgraph
        subgraph = {t: [d for d in graph[t] if d in valid_tasks] for t in valid_tasks}

        # Count incoming edges
        in_degree = {t: 0 for t in valid_tasks}
        for t in valid_tasks:
            for dep in subgraph[t]:
                in_degree[dep] = in_degree.get(dep, 0) + 1

        # Start with nodes with no dependencies
        queue = [t for t in valid_tasks if in_degree.get(t, 0) == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # Remove edges
            for t in valid_tasks:
                if node in subgraph[t]:
                    in_degree[t] -= 1
                    if in_degree[t] == 0:
                        queue.append(t)

        # Check for cycles
        if len(result) != len(valid_tasks):
            # Fall back to difficulty-based sorting
            return sorted(valid_tasks,
                         key=lambda t: self.learning_tasks[t].difficulty.value if t in self.learning_tasks else 0)

        return result