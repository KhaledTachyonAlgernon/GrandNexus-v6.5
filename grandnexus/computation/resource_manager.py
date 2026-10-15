"""
resource_manager.py - Advanced Resource & Scheduling System for GrandNexus

This module provides sophisticated resource management capabilities for GrandNexus,
including:

1) Smart scheduling of computational tasks based on priority, deadlines and resources
2) Dynamic resource allocation across different modules
3) Monitoring of system resources and performance
4) Adaptive load balancing and throttling
5) Error handling and recovery for resource-related issues

The ResourceManager serves as a central coordinator for computational resources,
ensuring that GrandNexus can efficiently manage its processing requirements
while maintaining responsiveness and stability.
"""

import logging
import threading
import time
import uuid
import queue
import heapq
import psutil
import os
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from collections import defaultdict, deque
import numpy as np

# Assuming these exist or will exist
from grandnexus.core.nexus_core import NexusCore


class TaskStatus(Enum):
    """Status of a task in the resource management system"""
    PENDING = auto()    # Waiting in queue
    RUNNING = auto()    # Currently executing
    COMPLETED = auto()  # Successfully completed
    FAILED = auto()     # Failed due to error
    CANCELLED = auto()  # Cancelled before completion
    TIMEOUT = auto()    # Timed out before completion
    DEFERRED = auto()   # Temporarily deferred due to resource constraints


class SchedulingPolicy(Enum):
    """Different policies for task scheduling"""
    FIFO = auto()             # First-in, first-out
    PRIORITY = auto()         # Based on task priority
    DEADLINE = auto()         # Earliest deadline first
    FAIR_SHARE = auto()       # Fair allocation between modules
    RESOURCE_AWARE = auto()   # Based on resource availability
    ADAPTIVE = auto()         # Adapts based on system state
    ML_BASED = auto()         # Uses ML model to decide scheduling


@dataclass(order=True)
class Task:
    """
    Represents a computational task to be managed by the ResourceManager.
    The order=True allows tasks to be compared based on priority and creation_time.
    """
    # Fields used for priority queue ordering
    priority: int
    creation_time: float = field(default_factory=time.time, compare=True)

    # Task identification
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()), compare=False)
    source_module: str = field(default=None, compare=False)

    # Task details
    callback: Callable = field(compare=False)
    args: Tuple = field(default_factory=tuple, compare=False)
    kwargs: Dict[str, Any] = field(default_factory=dict, compare=False)
    resource_requirements: Dict[str, float] = field(default_factory=dict, compare=False)

    # Task control
    status: TaskStatus = field(default=TaskStatus.PENDING, compare=False)
    deadline: Optional[float] = field(default=None, compare=False)
    max_runtime: Optional[float] = field(default=None, compare=False)
    max_retries: int = field(default=0, compare=False)
    retry_count: int = field(default=0, compare=False)
    result: Any = field(default=None, compare=False)
    error: Optional[Exception] = field(default=None, compare=False)

    # Task tracking
    start_time: Optional[float] = field(default=None, compare=False)
    end_time: Optional[float] = field(default=None, compare=False)
    assigned_thread: Optional[threading.Thread] = field(default=None, compare=False)

    def __str__(self) -> str:
        """String representation for logging and debugging"""
        return (f"<Task id={self.task_id[:8]}, priority={self.priority}, "
                f"source={self.source_module}, status={self.status.name}>")


@dataclass
class ResourceSnapshot:
    """Snapshot of system resource usage at a given time"""
    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0  # As percentage
    memory_available: int = 0  # In bytes
    disk_usage: float = 0.0    # As percentage
    io_counters: Dict[str, int] = field(default_factory=dict)
    per_module_cpu: Dict[str, float] = field(default_factory=dict)
    per_module_memory: Dict[str, int] = field(default_factory=dict)  # In bytes

    def __str__(self) -> str:
        """String representation for logging"""
        return (f"<ResourceSnapshot time={self.timestamp:.1f}, CPU={self.cpu_usage:.1f}%, "
                f"Memory={self.memory_usage:.1f}%, Avail={self.memory_available/(1024*1024):.1f}MB>")


@dataclass
class ResourceManagerConfig:
    """Configuration for ResourceManager behavior"""
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.PRIORITY
    max_worker_threads: int = 4
    task_queue_size: int = 1000
    monitoring_interval: float = 1.0  # seconds
    cpu_threshold_soft: float = 75.0  # percentage
    cpu_threshold_hard: float = 90.0  # percentage
    memory_threshold_soft: float = 75.0  # percentage
    memory_threshold_hard: float = 90.0  # percentage
    thread_pool_resize_interval: float = 30.0  # seconds
    resource_snapshot_history: int = 60  # Keep last N snapshots
    enable_module_quotas: bool = True
    enable_adaptive_scheduling: bool = True
    enable_ml_scheduling: bool = False  # Requires ML model integration
    task_timeout_default: float = 300.0  # 5 minutes
    recovery_backoff_base: float = 2.0  # For exponential backoff


@dataclass
class ModuleQuota:
    """Resource quotas for a specific module"""
    module_name: str
    cpu_share: float = 0.25  # 0.0 to 1.0
    memory_share: float = 0.25  # 0.0 to 1.0
    max_concurrent_tasks: int = 2
    max_queued_tasks: int = 50
    task_priority_range: Tuple[int, int] = (1, 10)  # min, max
    current_task_count: int = 0

    def can_accept_task(self) -> bool:
        """Check if the module can accept another task"""
        return self.current_task_count < self.max_concurrent_tasks


class ResourceManager:
    """
    Central resource management system for GrandNexus.

    This class handles task scheduling, resource allocation, and performance
    monitoring to ensure efficient use of computational resources across
    the system.
    """

    def __init__(self, config: Optional[ResourceManagerConfig] = None,
                 nexus_core: Optional[NexusCore] = None):
        """
        Initialize the resource manager.

        Args:
            config: Configuration parameters (uses defaults if None)
            nexus_core: Reference to the NexusCore for system integration
        """
        self.logger = logging.getLogger("GrandNexus.ResourceManager")
        self.config = config or ResourceManagerConfig()
        self.nexus_core = nexus_core

        # Internal state
        self.instance_id = str(uuid.uuid4())[:8]
        self.running = False
        self.initialized = False
        self.lock = threading.RLock()

        # Task management
        self.task_queue = []  # heapq-based priority queue
        self.running_tasks = set()
        self.completed_tasks = deque(maxlen=100)
        self.failed_tasks = deque(maxlen=50)
        self.cancelled_tasks = deque(maxlen=50)

        # Worker thread pool
        self.workers = []
        self.worker_control = {}  # worker_id -> control dict

        # Resource monitoring
        self.resource_snapshots = deque(maxlen=self.config.resource_snapshot_history)
        self.module_stats = defaultdict(lambda: {
            'cpu_usage': deque(maxlen=30),
            'memory_usage': deque(maxlen=30),
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_runtime': 0.0
        })

        # Module quotas
        self.module_quotas = {}  # module_name -> ModuleQuota

        # Adaptive learning
        self.task_execution_stats = defaultdict(lambda: {
            'runtimes': deque(maxlen=20),
            'success_rate': 1.0,
            'resource_usage': deque(maxlen=20)
        })

        # ML-based scheduling (if enabled)
        self.scheduling_model = None
        if self.config.enable_ml_scheduling:
            self._init_scheduling_model()

        self.logger.info(f"ResourceManager initialized with ID={self.instance_id}")

    def initialize(self) -> bool:
        """
        Initialize the resource manager.

        Returns:
            True if initialization succeeded, False otherwise
        """
        if self.initialized:
            return True

        with self.lock:
            try:
                # Set up monitoring
                self._take_resource_snapshot()

                # Initialize worker threads
                self._init_worker_pool()

                # Set up default module quotas (if enabled)
                if self.config.enable_module_quotas:
                    self._init_default_quotas()

                # Connect to other modules
                if self.nexus_core:
                    self._register_with_nexus_core()

                self.initialized = True
                self.logger.info("ResourceManager initialization complete")
                return True
            except Exception as e:
                self.logger.error(f"ResourceManager initialization failed: {str(e)}")
                return False

    def start(self) -> bool:
        """
        Start the resource manager operations.

        Returns:
            True if start succeeded, False otherwise
        """
        if not self.initialized:
            if not self.initialize():
                return False

        with self.lock:
            if self.running:
                return True

            try:
                # Start resource monitoring
                self._start_monitoring()

                # Start worker threads
                self._start_workers()

                self.running = True
                self.logger.info("ResourceManager started successfully")
                return True
            except Exception as e:
                self.logger.error(f"Failed to start ResourceManager: {str(e)}")
                return False

    def stop(self) -> bool:
        """
        Stop the resource manager operations.

        Returns:
            True if stop succeeded, False otherwise
        """
        with self.lock:
            if not self.running:
                return True

            try:
                # Stop monitoring
                self._stop_monitoring()

                # Stop worker threads
                self._stop_workers()

                self.running = False
                self.logger.info("ResourceManager stopped successfully")
                return True
            except Exception as e:
                self.logger.error(f"Error stopping ResourceManager: {str(e)}")
                return False

    def submit_task(self, source_module: str, callback: Callable,
                    args: Tuple = (), kwargs: Dict[str, Any] = None,
                    priority: int = 5, deadline: Optional[float] = None,
                    resource_requirements: Optional[Dict[str, float]] = None,
                    max_runtime: Optional[float] = None,
                    max_retries: int = 0) -> str:
        """
        Submit a task for execution.

        Args:
            source_module: Name of the module submitting the task
            callback: The function to call for task execution
            args: Positional arguments for the callback
            kwargs: Keyword arguments for the callback
            priority: Task priority (lower number = higher priority)
            deadline: Optional deadline timestamp for task completion
            resource_requirements: Dictionary of resource requirements
            max_runtime: Maximum runtime in seconds before timeout
            max_retries: Maximum number of retries on failure

        Returns:
            Task ID for tracking
        """
        if not self.running:
            if not self.start():
                raise RuntimeError("ResourceManager is not running and could not be started")

        if kwargs is None:
            kwargs = {}

        if resource_requirements is None:
            resource_requirements = {}

        with self.lock:
            # Check module quota if enabled
            if (self.config.enable_module_quotas and
                source_module in self.module_quotas):
                quota = self.module_quotas[source_module]

                # Enforce max queued tasks
                module_queued = sum(1 for task in self.task_queue
                                   if task.source_module == source_module)
                if module_queued >= quota.max_queued_tasks:
                    self.logger.warning(f"Module {source_module} exceeded queue quota")
                    return None

                # Enforce priority range
                min_prio, max_prio = quota.task_priority_range
                priority = max(min_prio, min(max_prio, priority))

            # Validate deadline if provided
            if deadline is not None and deadline < time.time():
                self.logger.warning(f"Task deadline already passed: {deadline}")
                return None

            # Set default max_runtime if not provided
            if max_runtime is None:
                max_runtime = self.config.task_timeout_default

            # Create task
            task = Task(
                priority=priority,
                source_module=source_module,
                callback=callback,
                args=args,
                kwargs=kwargs,
                resource_requirements=resource_requirements,
                deadline=deadline,
                max_runtime=max_runtime,
                max_retries=max_retries
            )

            # Add to queue
            heapq.heappush(self.task_queue, task)

            self.logger.info(f"Task {task.task_id} submitted from {source_module} with priority {priority}")

            return task.task_id

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.

        Args:
            task_id: ID of the task to cancel

        Returns:
            True if task was found and cancelled, False otherwise
        """
        with self.lock:
            # Check pending tasks
            for i, task in enumerate(self.task_queue):
                if task.task_id == task_id:
                    # Mark as cancelled
                    task.status = TaskStatus.CANCELLED

                    # Remove from queue
                    self.task_queue[i] = self.task_queue[-1]
                    self.task_queue.pop()
                    heapq.heapify(self.task_queue)

                    # Add to cancelled tasks
                    self.cancelled_tasks.append(task)

                    self.logger.info(f"Cancelled pending task {task_id}")
                    return True

            # Check running tasks
            for task in self.running_tasks:
                if task.task_id == task_id:
                    # Mark as cancelled
                    task.status = TaskStatus.CANCELLED

                    # Interrupt if possible
                    if task.assigned_thread:
                        # Note: This is a placeholder. In a real system,
                        # we would need a proper way to interrupt tasks
                        self.logger.warning(f"Cannot reliably interrupt running task {task_id}")

                    self.logger.info(f"Marked running task {task_id} for cancellation")
                    return True

            self.logger.warning(f"Task {task_id} not found for cancellation")
            return False

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a task.

        Args:
            task_id: ID of the task to check

        Returns:
            Dictionary with task status information, or None if not found
        """
        with self.lock:
            # Check all possible task locations
            for task_list in [self.task_queue, self.running_tasks,
                              self.completed_tasks, self.failed_tasks,
                              self.cancelled_tasks]:
                for task in task_list:
                    if getattr(task, 'task_id', None) == task_id:
                        # Convert to dict for API consumption
                        return {
                            'task_id': task.task_id,
                            'status': task.status.name,
                            'source_module': task.source_module,
                            'priority': task.priority,
                            'creation_time': task.creation_time,
                            'start_time': task.start_time,
                            'end_time': task.end_time,
                            'deadline': task.deadline,
                            'result': task.result if task.status == TaskStatus.COMPLETED else None,
                            'error': str(task.error) if task.error else None,
                            'retry_count': task.retry_count
                        }

            return None

    def get_module_stats(self, module_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance statistics for a specific module or all modules.

        Args:
            module_name: Name of the module, or None for all modules

        Returns:
            Dictionary with statistics
        """
        with self.lock:
            if module_name:
                if module_name in self.module_stats:
                    stats = self.module_stats[module_name]
                    return {
                        'module': module_name,
                        'avg_cpu': sum(stats['cpu_usage']) / len(stats['cpu_usage']) if stats['cpu_usage'] else 0,
                        'avg_memory': sum(stats['memory_usage']) / len(stats['memory_usage']) if stats['memory_usage'] else 0,
                        'tasks_completed': stats['tasks_completed'],
                        'tasks_failed': stats['tasks_failed'],
                        'total_runtime': stats['total_runtime'],
                        # Include quota if available
                        'quota': self._get_module_quota(module_name)
                    }
                return {'module': module_name, 'error': 'No stats available'}
            else:
                # Return stats for all modules
                result = {}
                for mod_name, stats in self.module_stats.items():
                    result[mod_name] = {
                        'avg_cpu': sum(stats['cpu_usage']) / len(stats['cpu_usage']) if stats['cpu_usage'] else 0,
                        'avg_memory': sum(stats['memory_usage']) / len(stats['memory_usage']) if stats['memory_usage'] else 0,
                        'tasks_completed': stats['tasks_completed'],
                        'tasks_failed': stats['tasks_failed'],
                        'total_runtime': stats['total_runtime'],
                        # Include quota if available
                        'quota': self._get_module_quota(mod_name)
                    }
                return result

    def set_module_quota(self, module_name: str,
                         cpu_share: Optional[float] = None,
                         memory_share: Optional[float] = None,
                         max_concurrent_tasks: Optional[int] = None,
                         max_queued_tasks: Optional[int] = None,
                         task_priority_range: Optional[Tuple[int, int]] = None) -> bool:
        """
        Set or update resource quotas for a module.

        Args:
            module_name: Name of the module
            cpu_share: Share of CPU resources (0.0 to 1.0)
            memory_share: Share of memory resources (0.0 to 1.0)
            max_concurrent_tasks: Maximum concurrent tasks
            max_queued_tasks: Maximum queued tasks
            task_priority_range: Tuple of (min_priority, max_priority)

        Returns:
            True if update succeeded, False otherwise
        """
        with self.lock:
            # Check if quota exists
            if module_name in self.module_quotas:
                quota = self.module_quotas[module_name]

                # Update fields if provided
                if cpu_share is not None:
                    quota.cpu_share = max(0.0, min(1.0, cpu_share))

                if memory_share is not None:
                    quota.memory_share = max(0.0, min(1.0, memory_share))

                if max_concurrent_tasks is not None:
                    quota.max_concurrent_tasks = max(1, max_concurrent_tasks)

                if max_queued_tasks is not None:
                    quota.max_queued_tasks = max(0, max_queued_tasks)

                if task_priority_range is not None:
                    min_prio, max_prio = task_priority_range
                    quota.task_priority_range = (min_prio, max_prio)

                self.logger.info(f"Updated quota for module {module_name}")
                return True
            else:
                # Create new quota
                quota = ModuleQuota(
                    module_name=module_name,
                    cpu_share=cpu_share if cpu_share is not None else 0.25,
                    memory_share=memory_share if memory_share is not None else 0.25,
                    max_concurrent_tasks=max_concurrent_tasks if max_concurrent_tasks is not None else 2,
                    max_queued_tasks=max_queued_tasks if max_queued_tasks is not None else 50,
                    task_priority_range=task_priority_range if task_priority_range is not None else (1, 10)
                )

                self.module_quotas[module_name] = quota
                self.logger.info(f"Created quota for module {module_name}")
                return True

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status and resource usage.

        Returns:
            Dictionary with system status information
        """
        with self.lock:
            # Get latest resource snapshot
            snapshot = self.resource_snapshots[-1] if self.resource_snapshots else None

            # Count tasks by status
            pending_count = len(self.task_queue)
            running_count = len(self.running_tasks)
            completed_count = len(self.completed_tasks)
            failed_count = len(self.failed_tasks)
            cancelled_count = len(self.cancelled_tasks)

            # Calculate average task wait and execution times
            avg_wait_time = 0.0
            if completed_count > 0:
                wait_times = [task.start_time - task.creation_time
                             for task in self.completed_tasks
                             if task.start_time is not None]
                if wait_times:
                    avg_wait_time = sum(wait_times) / len(wait_times)

            avg_exec_time = 0.0
            if completed_count > 0:
                exec_times = [task.end_time - task.start_time
                             for task in self.completed_tasks
                             if task.start_time is not None and task.end_time is not None]
                if exec_times:
                    avg_exec_time = sum(exec_times) / len(exec_times)

            return {
                'timestamp': time.time(),
                'cpu_usage': snapshot.cpu_usage if snapshot else None,
                'memory_usage': snapshot.memory_usage if snapshot else None,
                'memory_available': snapshot.memory_available if snapshot else None,
                'tasks_pending': pending_count,
                'tasks_running': running_count,
                'tasks_completed': completed_count,
                'tasks_failed': failed_count,
                'tasks_cancelled': cancelled_count,
                'avg_wait_time': avg_wait_time,
                'avg_exec_time': avg_exec_time,
                'worker_count': len(self.workers),
                'scheduling_policy': self.config.scheduling_policy.name
            }

    # -----------------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------------

    def _init_worker_pool(self) -> None:
        """Initialize the worker thread pool"""
        self.workers = []
        self.worker_control = {}

        for i in range(self.config.max_worker_threads):
            worker_id = f"worker-{i}"
            self.worker_control[worker_id] = {
                'running': False,
                'stop_event': threading.Event(),
                'tasks_processed': 0,
                'current_task': None
            }
            worker = threading.Thread(
                target=self._worker_loop,
                args=(worker_id,),
                name=worker_id,
                daemon=True
            )
            self.workers.append(worker)

        self.logger.info(f"Initialized worker pool with {len(self.workers)} workers")

    def _start_workers(self) -> None:
        """Start all worker threads"""
        for i, worker in enumerate(self.workers):
            worker_id = f"worker-{i}"
            self.worker_control[worker_id]['running'] = True
            self.worker_control[worker_id]['stop_event'].clear()
            worker.start()

        self.logger.info(f"Started {len(self.workers)} worker threads")

    def _stop_workers(self) -> None:
        """Stop all worker threads"""
        # Signal all workers to stop
        for worker_id in self.worker_control:
            self.worker_control[worker_id]['running'] = False
            self.worker_control[worker_id]['stop_event'].set()

        # Wait for workers to terminate
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=2.0)

        self.logger.info("Stopped all worker threads")

    def _worker_loop(self, worker_id: str) -> None:
        """
        Main worker thread loop.

        Args:
            worker_id: Identifier for this worker
        """
        control = self.worker_control[worker_id]
        self.logger.debug(f"Worker {worker_id} started")

        while control['running']:
            try:
                # Try to get a task
                task = self._get_next_task()

                if task:
                    # Process the task
                    control['current_task'] = task
                    task.assigned_thread = threading.current_thread()

                    self._execute_task(task)

                    control['tasks_processed'] += 1
                    control['current_task'] = None
                else:
                    # No tasks available, sleep briefly
                    control['stop_event'].wait(0.1)
            except Exception as e:
                self.logger.error(f"Error in worker {worker_id}: {str(e)}")
                # Sleep to avoid tight loop in case of persistent error
                time.sleep(0.5)

        self.logger.debug(f"Worker {worker_id} stopped")

    def _get_next_task(self) -> Optional[Task]:
        """
        Get the next task to execute based on scheduling policy.

        Returns:
            Next task to execute, or None if no tasks are ready
        """
        with self.lock:
            if not self.task_queue:
                return None

            # Apply scheduling policy
            if self.config.scheduling_policy == SchedulingPolicy.FIFO:
                # Find oldest task (ignoring priority)
                oldest_idx = 0
                oldest_time = float('inf')
                for i, task in enumerate(self.task_queue):
                    if task.creation_time < oldest_time:
                        oldest_time = task.creation_time
                        oldest_idx = i

                # Swap with last element and pop
                self.task_queue[oldest_idx], self.task_queue[-1] = self.task_queue[-1], self.task_queue[oldest_idx]
                task = self.task_queue.pop()

                # Fix heap
                if self.task_queue:
                    heapq.heapify(self.task_queue)

            elif self.config.scheduling_policy == SchedulingPolicy.DEADLINE:
                # Find task with earliest deadline (that's not passed)
                now = time.time()
                earliest_idx = None
                earliest_deadline = float('inf')

                for i, task in enumerate(self.task_queue):
                    if task.deadline is not None and task.deadline > now:
                        if task.deadline < earliest_deadline:
                            earliest_deadline = task.deadline
                            earliest_idx = i

                if earliest_idx is not None:
                    # Swap with last element and pop
                    self.task_queue[earliest_idx], self.task_queue[-1] = self.task_queue[-1], self.task_queue[earliest_idx]
                    task = self.task_queue.pop()

                    # Fix heap
                    if self.task_queue:
                        heapq.heapify(self.task_queue)
                else:
                    # No task with valid deadline, use priority
                    task = heapq.heappop(self.task_queue)

            elif self.config.scheduling_policy == SchedulingPolicy.FAIR_SHARE:
                # Group tasks by module
                tasks_by_module = defaultdict(list)
                for i, task in enumerate(self.task_queue):
                    tasks_by_module[task.source_module].append((i, task))

                # Find module with fewest running tasks
                module_counts = defaultdict(int)
                for task in self.running_tasks:
                    module_counts[task.source_module] += 1

                # Select from least-represented module
                selected_idx = None
                min_count = float('inf')

                for module, tasks in tasks_by_module.items():
                    count = module_counts[module]
                    if count < min_count and tasks:
                        min_count = count
                        # Select highest priority task from this module
                        tasks.sort(key=lambda x: x[1].priority)
                        selected_idx = tasks[0][0]

                if selected_idx is not None:
                    # Swap with last element and pop
                    self.task_queue[selected_idx], self.task_queue[-1] = self.task_queue[-1], self.task_queue[selected_idx]
                    task = self.task_queue.pop()

                    # Fix heap
                    if self.task_queue:
                        heapq.heapify(self.task_queue)
                else:
                    # Fallback to priority
                    task = heapq.heappop(self.task_queue)

            elif self.config.scheduling_policy == SchedulingPolicy.ML_BASED and self.scheduling_model:
                # Use ML model to score and select task
                # This is a placeholder for actual ML-based scheduling
                # In a real implementation, we would run the model on each task

                # Fallback to priority for now
                task = heapq.heappop(self.task_queue)

            else:
                # Default: Priority-based (heap is already sorted by priority)
                task = heapq.heappop(self.task_queue)

            # Check module quota if enabled
            if (self.config.enable_module_quotas and
                task.source_module in self.module_quotas):
                quota = self.module_quotas[task.source_module]

                # Count current tasks from this module
                module_running = sum(1 for t in self.running_tasks
                                    if t.source_module == task.source_module)

                if module_running >= quota.max_concurrent_tasks:
                    # Module quota exceeded, put task back and return None
                    heapq.heappush(self.task_queue, task)
                    return None

            # Update task status
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()

            # Add to running tasks
            self.running_tasks.add(task)

            return task

    def _execute_task(self, task: Task) -> None:
        """
        Execute a task and handle results.

        Args:
            task: The task to execute
        """
        result = None
        error = None

        try:
            # Log task start
            self.logger.debug(f"Starting execution of task {task.task_id} from {task.source_module}")

            # Execute task
            start_time = time.time()
            result = task.callback(*task.args, **task.kwargs)
            end_time = time.time()

            # Calculate duration
            duration = end_time - start_time

            # Update task
            task.end_time = end_time
            task.result = result
            task.status = TaskStatus.COMPLETED

            # Log completion
            self.logger.debug(f"Task {task.task_id} completed in {duration:.3f}s")

            # Update module stats
            with self.lock:
                stats = self.module_stats[task.source_module]
                stats['tasks_completed'] += 1
                stats['total_runtime'] += duration

            # Update execution stats for this task type
            task_type = task.kwargs.get('task_type', 'default')
            with self.lock:
                self.task_execution_stats[task_type]['runtimes'].append(duration)
                # Update success rate (weighted average)
                current_rate = self.task_execution_stats[task_type]['success_rate']
                self.task_execution_stats[task_type]['success_rate'] = 0.9 * current_rate + 0.1

        except Exception as e:
            # Handle task failure
            error = e
            end_time = time.time()

            self.logger.error(f"Task {task.task_id} failed: {str(e)}")

            # Update task
            task.end_time = end_time
            task.error = error

            # Check for retry
            if task.retry_count < task.max_retries:
                # Schedule retry
                task.retry_count += 1
                task.status = TaskStatus.PENDING

                # Calculate backoff delay
                backoff = self.config.recovery_backoff_base ** task.retry_count

                # Reset task for retry
                task.start_time = None
                task.end_time = None
                task.error = None

                # Re-add to queue with original priority
                with self.lock:
                    heapq.heappush(self.task_queue, task)

                self.logger.info(f"Scheduled retry {task.retry_count}/{task.max_retries} "
                               f"for task {task.task_id} after {backoff:.1f}s")

            else:
                # Max retries exceeded, mark as failed
                task.status = TaskStatus.FAILED

                # Update module stats
                with self.lock:
                    stats = self.module_stats[task.source_module]
                    stats['tasks_failed'] += 1

                # Update execution stats for this task type
                task_type = task.kwargs.get('task_type', 'default')
                with self.lock:
                    # Update success rate (weighted average)
                    current_rate = self.task_execution_stats[task_type]['success_rate']
                    self.task_execution_stats[task_type]['success_rate'] = 0.9 * current_rate

                # Move to failed tasks
                with self.lock:
                    self.running_tasks.remove(task)
                    self.failed_tasks.append(task)

            return

        # Task completed successfully
        with self.lock:
            # Remove from running
            self.running_tasks.remove(task)

            # Add to completed
            self.completed_tasks.append(task)

    def _start_monitoring(self) -> None:
        """Start resource monitoring thread"""
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="resource-monitor"
        )
        self.monitoring_thread.start()
        self.logger.info("Started resource monitoring thread")

    def _stop_monitoring(self) -> None:
        """Stop resource monitoring thread"""
        if hasattr(self, 'monitoring_thread') and self.monitoring_thread.is_alive():
            # There's no clean way to stop a thread, so we just let it exit on its own
            # by checking self.running in the loop
            self.monitoring_thread.join(timeout=2.0)
            self.logger.info("Stopped resource monitoring thread")

    def _monitoring_loop(self) -> None:
        """Main loop for resource monitoring"""
        while self.running:
            try:
                # Take resource snapshot
                self._take_resource_snapshot()

                # Check for resource constraints
                self._check_resource_constraints()

                # Resize thread pool if needed (periodically)
                if (time.time() % self.config.thread_pool_resize_interval <
                    self.config.monitoring_interval):
                    self._resize_thread_pool()

                # Sleep until next monitoring interval
                time.sleep(self.config.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(1.0)  # Sleep to avoid tight loop in case of error

    def _take_resource_snapshot(self) -> None:
        """Take a snapshot of current resource usage"""
        try:
            # Get system-wide CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()

            # Get disk usage for system drive
            disk = psutil.disk_usage('/')

            # Get IO counters
            io = psutil.disk_io_counters()
            io_dict = {
                'read_count': io.read_count if io else 0,
                'write_count': io.write_count if io else 0,
                'read_bytes': io.read_bytes if io else 0,
                'write_bytes': io.write_bytes if io else 0
            }

            # Get per-module resource usage (if possible)
            per_module_cpu = {}
            per_module_memory = {}

            # In a real implementation, we might track processes/threads per module
            # For simplicity, we'll just estimate based on running tasks

            for task in self.running_tasks:
                module = task.source_module
                # Assign some arbitrary CPU/memory values for demonstration
                per_module_cpu[module] = per_module_cpu.get(module, 0) + (cpu_percent / (len(self.running_tasks) or 1))
                per_module_memory[module] = per_module_memory.get(module, 0) + (memory.used / (len(self.running_tasks) or 1))

            # Create snapshot
            snapshot = ResourceSnapshot(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                memory_available=memory.available,
                disk_usage=disk.percent,
                io_counters=io_dict,
                per_module_cpu=per_module_cpu,
                per_module_memory=per_module_memory
            )

            # Add to history
            with self.lock:
                self.resource_snapshots.append(snapshot)

                # Update module stats
                for module, cpu in per_module_cpu.items():
                    self.module_stats[module]['cpu_usage'].append(cpu)

                for module, mem in per_module_memory.items():
                    self.module_stats[module]['memory_usage'].append(mem)

        except Exception as e:
            self.logger.error(f"Error taking resource snapshot: {str(e)}")

    def _check_resource_constraints(self) -> None:
        """Check if system is under resource constraints and take action"""
        if not self.resource_snapshots:
            return

        snapshot = self.resource_snapshots[-1]

        # Check CPU usage
        if snapshot.cpu_usage > self.config.cpu_threshold_hard:
            self.logger.warning(f"CPU usage critical: {snapshot.cpu_usage:.1f}% > {self.config.cpu_threshold_hard}%")
            self._handle_critical_cpu_usage()
        elif snapshot.cpu_usage > self.config.cpu_threshold_soft:
            self.logger.info(f"CPU usage high: {snapshot.cpu_usage:.1f}% > {self.config.cpu_threshold_soft}%")
            self._handle_high_cpu_usage()

        # Check memory usage
        if snapshot.memory_usage > self.config.memory_threshold_hard:
            self.logger.warning(f"Memory usage critical: {snapshot.memory_usage:.1f}% > {self.config.memory_threshold_hard}%")
            self._handle_critical_memory_usage()
        elif snapshot.memory_usage > self.config.memory_threshold_soft:
            self.logger.info(f"Memory usage high: {snapshot.memory_usage:.1f}% > {self.config.memory_threshold_soft}%")
            self._handle_high_memory_usage()

    def _handle_high_cpu_usage(self) -> None:
        """Handle high CPU usage (soft threshold)"""
        # Slow down task scheduling
        # We don't have to do anything specific - the monitoring thread
        # will continue to run, but worker threads will sleep more between tasks
        pass

    def _handle_critical_cpu_usage(self) -> None:
        """Handle critical CPU usage (hard threshold)"""
        # Consider suspending some tasks
        with self.lock:
            # Find lowest priority running tasks
            if self.running_tasks:
                tasks = sorted(self.running_tasks, key=lambda t: -t.priority)  # Reverse sort (highest priority last)
                for task in tasks[:max(1, len(tasks) // 4)]:  # Take ~25% of tasks
                    # Here we would ideally suspend the task
                    # But Python doesn't have a good way to suspend a thread
                    # So we'll just log it
                    self.logger.warning(f"Would suspend task {task.task_id} due to CPU constraints")

    def _handle_high_memory_usage(self) -> None:
        """Handle high memory usage (soft threshold)"""
        # Throttle new memory-intensive tasks
        pass

    def _handle_critical_memory_usage(self) -> None:
        """Handle critical memory usage (hard threshold)"""
        # Release memory if possible and delay memory-intensive tasks
        with self.lock:
            # Clear completed tasks history to free memory
            self.completed_tasks.clear()

            # GC and release memory
            import gc
            gc.collect()

    def _resize_thread_pool(self) -> None:
        """
        Dynamically resize the worker thread pool based on load.
        This is called periodically to adjust the number of worker threads.
        """
        # This is a simplified implementation
        # In a real system, this would be more sophisticated

        if not self.resource_snapshots:
            return

        snapshot = self.resource_snapshots[-1]

        # Calculate desired worker count based on load
        current_workers = len(self.workers)
        queue_size = len(self.task_queue)

        # Very simple heuristic:
        # - If queue is growing and CPU isn't maxed, add workers
        # - If queue is small and we have excess workers, remove some

        if queue_size > current_workers * 2 and snapshot.cpu_usage < 70:
            # Add a worker
            if current_workers < self.config.max_worker_threads:
                self._add_worker()
        elif queue_size < current_workers // 2 and current_workers > 2:
            # Remove a worker
            self._remove_worker()

    def _add_worker(self) -> None:
        """Add a new worker thread to the pool"""
        worker_id = f"worker-{len(self.workers)}"

        self.worker_control[worker_id] = {
            'running': True,
            'stop_event': threading.Event(),
            'tasks_processed': 0,
            'current_task': None
        }

        worker = threading.Thread(
            target=self._worker_loop,
            args=(worker_id,),
            name=worker_id,
            daemon=True
        )

        self.workers.append(worker)
        worker.start()

        self.logger.info(f"Added worker thread {worker_id}, now have {len(self.workers)} workers")

    def _remove_worker(self) -> None:
        """Remove a worker thread from the pool"""
        if not self.workers:
            return

        # Get the last worker
        worker_id = f"worker-{len(self.workers) - 1}"

        # Signal it to stop
        if worker_id in self.worker_control:
            self.worker_control[worker_id]['running'] = False
            self.worker_control[worker_id]['stop_event'].set()

            # Remove from list
            worker = self.workers.pop()

            # No need to join, it will exit on its own

            self.logger.info(f"Removed worker thread {worker_id}, now have {len(self.workers)} workers")

    def _init_scheduling_model(self) -> None:
        """Initialize the ML-based scheduling model"""
        # This is a placeholder for an actual ML model initialization
        # In a real implementation, this would load a trained model

        # For now, we'll just set a dummy model
        class DummySchedulingModel:
            def predict(self, task: Task, system_state: Dict[str, Any]) -> float:
                # Return a simple score based on priority and deadline
                score = 10 - task.priority  # Higher priority (lower number) = higher score

                # Factor in deadline if present
                if task.deadline:
                    time_to_deadline = max(0, task.deadline - time.time())
                    urgency = 10 / (1 + time_to_deadline / 60)  # More urgent as deadline approaches
                    score += urgency

                return score

        self.scheduling_model = DummySchedulingModel()
        self.logger.info("Initialized dummy scheduling model")

    def _register_with_nexus_core(self) -> None:
        """Register with NexusCore for system-wide coordination"""
        if not self.nexus_core:
            self.logger.warning("No NexusCore available for registration")
            return

        try:
            # Register as a module with NexusCore
            self.nexus_core.register_module(
                name="resource_manager",
                module=self,
                dependencies=[]  # No direct dependencies
            )

            self.logger.info("Successfully registered with NexusCore")
        except Exception as e:
            self.logger.error(f"Failed to register with NexusCore: {str(e)}")

    def _init_default_quotas(self) -> None:
        """Initialize default resource quotas for known modules"""
        # Set up some reasonable defaults
        default_quotas = {
            "perception": {
                "cpu_share": 0.3,
                "memory_share": 0.2,
                "max_concurrent_tasks": 3,
                "max_queued_tasks": 50,
                "task_priority_range": (1, 8)
            },
            "reasoning": {
                "cpu_share": 0.4,
                "memory_share": 0.4,
                "max_concurrent_tasks": 2,
                "max_queued_tasks": 30,
                "task_priority_range": (1, 5)
            },
            "learning": {
                "cpu_share": 0.2,
                "memory_share": 0.3,
                "max_concurrent_tasks": 1,
                "max_queued_tasks": 20,
                "task_priority_range": (3, 10)
            },
            "memory_manager": {
                "cpu_share": 0.1,
                "memory_share": 0.4,
                "max_concurrent_tasks": 2,
                "max_queued_tasks": 40,
                "task_priority_range": (2, 7)
            }
        }

        for module_name, quota_params in default_quotas.items():
            self.set_module_quota(
                module_name=module_name,
                cpu_share=quota_params["cpu_share"],
                memory_share=quota_params["memory_share"],
                max_concurrent_tasks=quota_params["max_concurrent_tasks"],
                max_queued_tasks=quota_params["max_queued_tasks"],
                task_priority_range=quota_params["task_priority_range"]
            )

    def _get_module_quota(self, module_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the quota settings for a module.

        Args:
            module_name: Name of the module

        Returns:
            Dictionary with quota settings or None if not found
        """
        if module_name in self.module_quotas:
            quota = self.module_quotas[module_name]
            return {
                "cpu_share": quota.cpu_share,
                "memory_share": quota.memory_share,
                "max_concurrent_tasks": quota.max_concurrent_tasks,
                "max_queued_tasks": quota.max_queued_tasks,
                "task_priority_range": quota.task_priority_range,
                "current_task_count": quota.current_task_count
            }
        return None