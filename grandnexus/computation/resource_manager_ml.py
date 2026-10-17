"""
resource_manager.py - Advanced Resource & Scheduling System for GrandNexus

Key Features:
    1) Multi-node / HPC integration hooks:
       - Optional stubs for connecting to orchestration frameworks (e.g., Kubernetes, Ray, SLURM).
    2) Extended scheduling policies:
       - Priority-based, FIFO, Deadline-based, or ML-based (placeholders for integrating a
         learning model or offline RL approach).
    3) Deadlines, timeouts, and max waiting times:
       - Each TaskRequest can specify a deadline or max_waiting_time. The scheduler attempts
         to respect these constraints (Earliest-Deadline-First logic).
    4) Enhanced DRAS (Dynamic Resource Allocation Strategy):
       - Per-module resource partitions that can be updated dynamically.
       - Optionally load a learned model to adjust resource shares based on usage metrics.
    5) Anomaly detection stubs:
       - Hooks for detecting out-of-range CPU or memory usage that suggests a module is
         misbehaving or in a spinlock.
    6) Comprehensive logging, docstrings, type hints, error handling, and concurrency controls.
    7) Task interruption/cancellation:
       - More detailed placeholders for stopping tasks that are in progress, possibly across nodes.
    8) HPC/Cluster references:
       - If cluster_mode is True in the ResourceManagerConfig, one can integrate with real cluster
         managers (not implemented here, but the stubs are in place).
    9) ML-based scheduling placeholders:
       - A minimal “policy_model” interface is used to demonstrate how a learned model might
         advise on scheduling decisions.
    10) Preservation of advanced structures from previous versions (heapq, data classes, etc.)

"""

import logging
import psutil
import time
import queue
import threading
import heapq
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Callable, Union, Deque, Tuple, Set
)


# ---------------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------------

@dataclass(order=True, kw_only=True)
class TaskRequest:
    """
    Represents a task request. The 'order=True' property enables comparison
    based on (priority, creation_time) to support heapq-based scheduling.

    Attributes:
        priority (int): Lower number = higher scheduling priority (if using priority-based).
        creation_time (float): Timestamp used for FIFO tie-break or deadline-based logic.
        task_id (str): Unique ID. By default, it's a UUID4 string.
        module_name (str): The module or origin that submitted this task.
        required_resources (Dict[str, float]): E.g. {"cpu":0.2, "memory":512} meaning 20% CPU, 512MB memory.
        callback (Callable): The function to execute for the task.
        callback_args (Tuple[Any, ...]): Positional arguments for callback.
        callback_kwargs (Dict[str, Any]): Keyword arguments for callback.
        max_runtime (Optional[float]): The maximum allowed runtime (seconds).
        retries (int): How many times the task can be retried if it fails.
        retry_delay (float): The delay (seconds) between retries.
        state (str): "pending", "running", "completed", "failed", "cancelled", etc.
        deadline (Optional[float]): A timestamp or None. If set, the scheduler might use EDF logic.
        max_waiting_time (Optional[float]): If the task waits in the queue too long, it can time out.
    """
    priority: int
    creation_time: float = field(default_factory=time.time, compare=True)
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()), compare=False)
    module_name: str = field(compare=False)
    required_resources: Dict[str, float] = field(default_factory=dict, compare=False)
    callback: Callable = field(compare=False)
    callback_args: Tuple[Any, ...] = field(default_factory=tuple, compare=False)
    callback_kwargs: Dict[str, Any] = field(default_factory=dict, compare=False)
    max_runtime: Optional[float] = field(default=None, compare=False)
    retries: int = field(default=0, compare=False)
    retry_delay: float = field(default=1.0, compare=False)
    state: str = field(default='pending', compare=False)
    deadline: Optional[float] = field(default=None, compare=False)
    max_waiting_time: Optional[float] = field(default=None, compare=False)

    def __repr__(self) -> str:
        return (f"<TaskRequest id={self.task_id[-8:]}, module={self.module_name}, "
                f"priority={self.priority}, state={self.state}>")

@dataclass
class ResourceUsageSnapshot:
    """
    Captures system or cluster resource usage at a specific time.

    Attributes:
        timestamp (float): The time the snapshot was taken.
        cpu_percent (float): Overall CPU usage.
        mem_percent (float): Overall memory usage.
        available_memory (float): Bytes of memory available.
        used_memory (float): Bytes of memory in use.
        process_usage (Dict[str, float]): {module: CPU usage} or {pid: usage}, up to the user approach.
        per_core_cpu_percent (List[float]): Detailed usage per CPU core.
        anomaly_score (float): A placeholder for anomaly detection logic (0=normal, >1=very anomalous).
    """
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    mem_percent: float = 0.0
    available_memory: float = 0.0
    used_memory: float = 0.0
    process_usage: Dict[str, float] = field(default_factory=dict)
    per_core_cpu_percent: List[float] = field(default_factory=list)
    anomaly_score: float = 0.0

    def __repr__(self) -> str:
        return (f"<ResourceUsageSnapshot time={self.timestamp:.1f}, CPU={self.cpu_percent:.1f}%, "
                f"MEM={self.mem_percent:.1f}%, anomaly={self.anomaly_score:.2f}>")

@dataclass
class ResourceManagerConfig:
    """
    Configuration object for the ResourceManager.

    Attributes:
        monitor_interval (float): Frequency of resource monitoring, in seconds.
        scheduler_interval (float): Frequency of scheduling loop, in seconds.
        cpu_soft_limit (float): Soft CPU usage limit (percentage).
        cpu_hard_limit (float): Hard CPU usage limit (percentage).
        mem_soft_limit (float): Soft memory usage limit (percentage).
        mem_hard_limit (float): Hard memory usage limit (percentage).
        max_task_queue_size (int): Max number of tasks in the queue before rejecting new ones.
        history_size (int): Max number of ResourceUsageSnapshots to keep.
        scheduling_policy (str): Could be "priority", "fifo", "deadline", or "ml".
        dynamic_resource_allocation (bool): If True, we track per-module resource partitions (DRAS).
        cluster_mode (bool): If True, we assume multi-node HPC context (stubs for integration with HPC managers).
        anomaly_detection (bool): If True, we attempt to compute an anomaly_score on each snapshot (stub).
        ml_policy_enabled (bool): If True, we attempt to consult an ML-based scheduling policy.
    """
    monitor_interval: float = 2.0
    scheduler_interval: float = 1.0
    cpu_soft_limit: float = 80.0
    cpu_hard_limit: float = 95.0
    mem_soft_limit: float = 80.0
    mem_hard_limit: float = 95.0
    max_task_queue_size: int = 200
    history_size: int = 100
    scheduling_policy: str = "priority"
    dynamic_resource_allocation: bool = True
    cluster_mode: bool = False
    anomaly_detection: bool = False
    ml_policy_enabled: bool = False


# ---------------------------------------------------------------------------------
# ML POLICY INTERFACE (Stub)
# ---------------------------------------------------------------------------------

class SchedulingPolicyModel:
    """
    A placeholder interface for an ML-based scheduling policy. A real model
    might implement .predict() to decide the next task to schedule or define
    a recommended 'priority' override.
    """

    def __init__(self, model_path: str = ""):
        """
        model_path: location of a trained ML or RL model. Implementation omitted.
        """
        self.model_path = model_path
        # load model from disk or initialize in memory, etc.

    def advise_task_start(self,
                          task: TaskRequest,
                          snapshot: ResourceUsageSnapshot) -> bool:
        """
        Decide if we can start 'task' given the system snapshot. Return True if yes,
        or False if no, or implement a more complex numeric score.
        """
        # Placeholder: Always return True. Real logic would incorporate e.g. neural net or RL.
        return True

    def advise_task_priority(self, task: TaskRequest) -> int:
        """
        Possibly override the task's priority. For demonstration, we do not override.
        """
        return task.priority


# ---------------------------------------------------------------------------------
# RESOURCE MANAGER CLASS
# ---------------------------------------------------------------------------------

class ResourceManager:
    """
    The advanced ResourceManager for GrandNexus, featuring HPC integration stubs,
    advanced scheduling logic, DRAS, anomaly detection hooks, and ML-based policy stubs.
    """

    def __init__(self, config: ResourceManagerConfig,
                 nexus_core: Optional[Any] = None,
                 ml_policy_model: Optional[SchedulingPolicyModel] = None):
        """
        Args:
            config: ResourceManagerConfig with intervals, limits, etc.
            nexus_core: Optional reference to the GrandNexus core or registry.
            ml_policy_model: If ml_policy_enabled, we can pass a loaded model here.
        """
        self.logger = logging.getLogger("ResourceManager")
        self.config = config
        self.nexus_core = nexus_core

        self.resource_snapshots: Deque[ResourceUsageSnapshot] = deque(maxlen=self.config.history_size)
        self.lock = threading.RLock()
        self.monitor_thread: Optional[threading.Thread] = None
        self.scheduler_thread: Optional[threading.Thread] = None
        self.running = False

        # Priority queue or store for tasks
        # We'll store (sort_key, creation_time, TaskRequest). Sort key depends on scheduling_policy.
        self.task_queue: List[Tuple[Any, float, TaskRequest]] = []
        self.running_tasks: Dict[str, TaskRequest] = {}
        self.completed_tasks: Dict[str, TaskRequest] = {}
        self.failed_tasks: Dict[str, TaskRequest] = {}

        # DRAS
        self._dras_enabled = self.config.dynamic_resource_allocation
        # The dictionary: module_name -> {"cpu": float, "memory": float, ...}
        # representing the allocated resources. Adjusted dynamically or by user calls.
        self._resource_allocation: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"cpu": 0.1, "memory": 128.0}  # default minimal allocation
        )

        self._ml_policy = ml_policy_model if ml_policy_model else SchedulingPolicyModel()

    # -----------------------------------------------------------------------------
    # PUBLIC METHODS
    # -----------------------------------------------------------------------------

    def start(self) -> None:
        """Begin the resource monitoring and scheduling loops."""
        with self.lock:
            if self.running:
                self.logger.warning("ResourceManager is already running.")
                return
            self.running = True

        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="ResourceMonitor",
            daemon=True
        )
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="ResourceScheduler",
            daemon=True
        )
        self.monitor_thread.start()
        self.scheduler_thread.start()
        self.logger.info("ResourceManager started with advanced features enabled.")

    def stop(self) -> None:
        """Cleanly stop the manager, halting threads and preventing new tasks."""
        with self.lock:
            if not self.running:
                self.logger.warning("ResourceManager is not running.")
                return
            self.running = False

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=2.0)

        self.logger.info("ResourceManager stopped. No more scheduling or monitoring will occur.")

    def submit_task(self, task: TaskRequest) -> bool:
        """
        Submits a new task to the resource manager, returning True if accepted, False otherwise.
        We may incorporate scheduling_policy logic to adjust the 'sort_key' at insertion time.
        """
        with self.lock:
            if len(self.task_queue) + len(self.running_tasks) >= self.config.max_task_queue_size:
                self.logger.error(f"Task queue limit reached; cannot accept task {task.task_id}.")
                return False

            # Possibly override priority using ML policy
            if self.config.ml_policy_enabled:
                new_prio = self._ml_policy.advise_task_priority(task)
                if new_prio != task.priority:
                    self.logger.debug(f"ML policy overriding priority from {task.priority} to {new_prio} for {task.task_id}.")
                    task.priority = new_prio

            sort_key = self._compute_sort_key(task)
            heapq.heappush(self.task_queue, (sort_key, task.creation_time, task))
            self.logger.info(f"Task {task.task_id} submitted. (Queue size: {len(self.task_queue)})")

        return True

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancels a task in the queue or running. If running, calls _interrupt_task stub.
        Returns True if found and cancelled, False if not found.
        """
        with self.lock:
            # 1) Check if running
            if task_id in self.running_tasks:
                t = self.running_tasks[task_id]
                t.state = "cancelled"
                self._interrupt_task(t)
                del self.running_tasks[task_id]
                self.logger.info(f"Cancelled running task {task_id}.")
                return True

            # 2) Check queue
            for idx, (sort_key, ctime, t) in enumerate(self.task_queue):
                if t.task_id == task_id:
                    t.state = "cancelled"
                    # remove from heap
                    del self.task_queue[idx]
                    heapq.heapify(self.task_queue)
                    self.logger.info(f"Cancelled queued task {task_id}.")
                    return True

            self.logger.warning(f"Task {task_id} not found for cancellation.")
            return False

    def clear_tasks(self, task_state: Optional[str] = None) -> int:
        """
        Clears tasks from queue/running/completed based on their state. If task_state is None, clears all.
        Returns the number of tasks cleared.
        """
        count = 0
        with self.lock:
            if task_state is None or task_state == "pending":
                count += len(self.task_queue)
                self.task_queue = []
            if task_state is None or task_state == "running":
                for rt_id in list(self.running_tasks.keys()):
                    t = self.running_tasks[rt_id]
                    if task_state is None or t.state == task_state:
                        t.state = "cancelled"
                        self._interrupt_task(t)
                        del self.running_tasks[rt_id]
                        count += 1
            if task_state is None or task_state in ["completed", "failed", "cancelled"]:
                for store_name in ["completed_tasks", "failed_tasks"]:
                    store_dict = getattr(self, store_name)
                    for key in list(store_dict.keys()):
                        if task_state is None or store_dict[key].state == task_state:
                            del store_dict[key]
                            count += 1

        self.logger.info(f"Cleared {count} tasks with state {task_state or 'all'}.")
        return count

    def update_dras_allocations(self, allocations: Dict[str, Dict[str, float]]) -> None:
        """
        Updates the dynamic resource allocations for modules. E.g.:
        {
          "ModuleA": {"cpu": 0.4, "memory": 1024},
          "ModuleB": {"cpu": 0.6, "memory": 2048}
        }
        """
        with self.lock:
            for mod, res in allocations.items():
                cpu_val = res.get("cpu", None)
                if cpu_val is not None:
                    if 0.0 <= cpu_val <= 1.0:
                        self._resource_allocation[mod]["cpu"] = cpu_val
                    else:
                        self.logger.warning(f"Invalid CPU for {mod}: {cpu_val}. Must be in [0,1].")
                mem_val = res.get("memory", None)
                if mem_val is not None and mem_val > 0:
                    self._resource_allocation[mod]["memory"] = mem_val
                self.logger.debug(f"DRAS allocation updated for {mod}: {self._resource_allocation[mod]}")

    def get_dras_allocations(self) -> Dict[str, Dict[str, float]]:
        """Returns a copy of the current resource allocation dictionary."""
        with self.lock:
            return dict((k, dict(v)) for k, v in self._resource_allocation.items())

    def adjust_task_priority(self, task_id: str, new_priority: int) -> bool:
        """
        Adjusts the priority of a queued task. Returns True if found, else False.
        """
        with self.lock:
            found = False
            for i, (skey, ctime, task) in enumerate(self.task_queue):
                if task.task_id == task_id:
                    found = True
                    task.priority = new_priority
                    # remove & re-insert
                    del self.task_queue[i]
                    heapq.heapify(self.task_queue)
                    skey = self._compute_sort_key(task)
                    heapq.heappush(self.task_queue, (skey, task.creation_time, task))
                    self.logger.info(f"Task {task_id} priority updated to {new_priority}.")
                    break

            if not found:
                self.logger.warning(f"Task {task_id} not found in queue for priority update.")
            return found

    def get_usage_history(self, limit: int = -1) -> List[ResourceUsageSnapshot]:
        """Returns up to 'limit' usage snapshots (or all if limit < 0)."""
        with self.lock:
            if limit < 0:
                return list(self.resource_snapshots)
            else:
                return list(self.resource_snapshots)[-limit:]

    def get_queued_tasks(self) -> List[TaskRequest]:
        """Returns all pending tasks in sorted order (like the queue sees them)."""
        with self.lock:
            sorted_entries = sorted(self.task_queue, key=lambda x: (x[0], x[1]))
            return [te[2] for te in sorted_entries]

    def get_running_tasks(self) -> List[TaskRequest]:
        """Returns a list of currently running tasks."""
        with self.lock:
            return list(self.running_tasks.values())

    def get_completed_tasks(self) -> List[TaskRequest]:
        """Returns a list of completed tasks."""
        with self.lock:
            return list(self.completed_tasks.values())

    def get_failed_tasks(self) -> List[TaskRequest]:
        """Returns a list of failed tasks."""
        with self.lock:
            return list(self.failed_tasks.values())

    # -----------------------------------------------------------------------------
    # INTERNAL / PROTECTED METHODS
    # -----------------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """
        Periodically collects resource usage stats. Optionally does anomaly detection,
        HPC calls, or distributed cluster checks.
        """
        self.logger.info("Monitor loop started.")
        while True:
            with self.lock:
                if not self.running:
                    break

            # Single-node usage
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                per_core = psutil.cpu_percent(interval=None, percpu=True)
                mem_stats = psutil.virtual_memory()

                # Optional: gather per-process usage if we want
                process_usage = {}
                # HPC/cluster logic stubs could go here if cluster_mode is True

                snap = ResourceUsageSnapshot(
                    cpu_percent=cpu_percent,
                    mem_percent=mem_stats.percent,
                    available_memory=mem_stats.available,
                    used_memory=mem_stats.used,
                    process_usage=process_usage,
                    per_core_cpu_percent=per_core
                )
                if self.config.anomaly_detection:
                    snap.anomaly_score = self._compute_anomaly_score(snap)

                with self.lock:
                    self.resource_snapshots.append(snap)
            except Exception as e:
                self.logger.exception(f"Exception in monitor loop: {e}", exc_info=True)

            time.sleep(self.config.monitor_interval)
        self.logger.info("Monitor loop ended.")

    def _scheduler_loop(self) -> None:
        """Periodically attempts to schedule tasks and react to resource limits."""
        self.logger.info("Scheduler loop started.")
        while True:
            with self.lock:
                if not self.running:
                    break
            self._perform_scheduling_cycle()
            time.sleep(self.config.scheduler_interval)
        self.logger.info("Scheduler loop ended.")

    def _perform_scheduling_cycle(self) -> None:
        """
        The main scheduling logic. Checks resource usage vs. soft/hard limits,
        attempts to start tasks from the queue if feasible. Also handles
        HPC expansions or job submissions if cluster_mode is True.
        """
        snapshot = None
        with self.lock:
            if self.resource_snapshots:
                snapshot = self.resource_snapshots[-1]

        if not snapshot:
            return

        # Hard-limit enforcement
        if snapshot.cpu_percent > self.config.cpu_hard_limit or snapshot.mem_percent > self.config.mem_hard_limit:
            self._handle_hard_limit_exceed(snapshot)
            # We might choose to skip scheduling new tasks this cycle
            return

        # If we are below soft-limits, attempt to schedule
        can_run_more = True
        if snapshot.cpu_percent >= self.config.cpu_soft_limit:
            can_run_more = False
        if snapshot.mem_percent >= self.config.mem_soft_limit:
            can_run_more = False

        if not can_run_more:
            self.logger.debug("Soft limit reached, deferring new tasks.")
            return

        # Actually pick tasks from the queue
        self._attempt_task_starts(snapshot)

        # HPC expansions or synergy: If cluster_mode is True, we might call external HPC APIs
        if self.config.cluster_mode:
            # e.g., self._submit_to_cluster(...)
            pass

    def _attempt_task_starts(self, snapshot: ResourceUsageSnapshot) -> None:
        """
        Pop tasks from the queue according to the scheduling policy,
        check feasibility, and start them if possible.
        """
        started_tasks = []
        # We'll try to start as many tasks as we can until we reach resource constraints
        # or run out of tasks.
        attempts = 0

        while True:
            with self.lock:
                if not self.task_queue:
                    break  # no more tasks
                # Peek the top
                top_entry = self.task_queue[0]
                sort_key, creation_time, task = top_entry

                # If the task is expired by max_waiting_time or deadline, skip it
                if self._is_task_expired(task):
                    # Remove it from queue
                    heapq.heappop(self.task_queue)
                    task.state = "failed"
                    task_str = "deadline" if (task.deadline and time.time() >= task.deadline) else "max_waiting"
                    self.logger.warning(f"Task {task.task_id} expired due to {task_str} constraint.")
                    self.failed_tasks[task.task_id] = task
                    continue

                # If we can't run the task (lack resources or ML policy says no), break out
                if not self._can_run_task(task, snapshot):
                    break

                # We can start the task. Remove from queue
                heapq.heappop(self.task_queue)
                task.state = "running"
                self.running_tasks[task.task_id] = task
                started_tasks.append(task)

            attempts += 1
            if attempts > 50:
                # Safety guard: don't start too many tasks in one cycle
                break

        # Outside lock: actually run the tasks
        for t in started_tasks:
            self._launch_task(t)

    def _launch_task(self, task: TaskRequest) -> None:
        """
        Actually run the task's callback. We do it in a new thread so scheduling loop can continue.
        For HPC or multi-node, we might do a cluster job submission instead.
        """
        def runner():
            self.logger.info(f"Starting task {task.task_id} from module {task.module_name} with priority {task.priority}.")
            start_time = time.time()
            try:
                # Possibly set a timer for max_runtime
                # or we rely on the callback to be well-behaved
                # or we run in a separate process we can kill if needed
                task.callback(*task.callback_args, **task.callback_kwargs)
                duration = time.time() - start_time
                with self.lock:
                    task.state = "completed"
                    self.completed_tasks[task.task_id] = task
                    if task.task_id in self.running_tasks:
                        del self.running_tasks[task.task_id]
                self.logger.info(f"Task {task.task_id} completed in {duration:.2f}s")
            except Exception as e:
                with self.lock:
                    self.logger.error(f"Task {task.task_id} failed: {e}", exc_info=True)
                    task.retries -= 1
                    if task.retries >= 0:
                        task.state = "pending"
                        # Re-queue it with a small delay
                        time.sleep(task.retry_delay)
                        skey = self._compute_sort_key(task)
                        heapq.heappush(self.task_queue, (skey, task.creation_time, task))
                    else:
                        task.state = "failed"
                        self.failed_tasks[task.task_id] = task
                    if task.task_id in self.running_tasks:
                        del self.running_tasks[task.task_id]

        threading.Thread(target=runner, daemon=True).start()

    def _can_run_task(self, task: TaskRequest, snapshot: ResourceUsageSnapshot) -> bool:
        """
        Check if we can run the given task right now. Takes into account:
            - Current system usage vs. soft limits
            - DRAS allocations if enabled
            - ML policy if config.ml_policy_enabled
        """
        if self.config.ml_policy_enabled:
            # If the policy returns false, we skip this task for now
            if not self._ml_policy.advise_task_start(task, snapshot):
                return False

        if self._dras_enabled:
            # Basic approach: we see if the module's CPU + required CPU would push us over some threshold
            module_alloc = self._resource_allocation[task.module_name]
            # e.g. if the module is allocated 0.3 CPU (30%), does task require 0.2? Then 0.5 total vs. 1.0
            # This is simplistic. Real DRAS might be more advanced
            hypothetical_cpu_usage = snapshot.cpu_percent + (task.required_resources.get("cpu", 0.0) * 100)
            if hypothetical_cpu_usage > self.config.cpu_soft_limit:
                return False
            # memory check
            # we interpret "memory" as MB
            needed_mem_bytes = task.required_resources.get("memory", 0) * 1024 * 1024
            if snapshot.available_memory < needed_mem_bytes:
                return False

            # If passed the checks, we also "account" for the new usage (a naive approach)
            # In a real system, you might store real-time usage per module
            # For demonstration, we'll just note the allocated CPU in the dictionary
            module_alloc["cpu"] += task.required_resources.get("cpu", 0)
            module_alloc["memory"] += task.required_resources.get("memory", 0)

        else:
            # If not using DRAS, we do a simpler check
            hypothetical_cpu_usage = snapshot.cpu_percent + (task.required_resources.get("cpu", 0.0) * 100)
            if hypothetical_cpu_usage > self.config.cpu_soft_limit:
                return False
            needed_mem_bytes = task.required_resources.get("memory", 0) * 1024 * 1024
            if snapshot.available_memory < needed_mem_bytes:
                return False

        return True

    def _compute_sort_key(self, task: TaskRequest) -> Any:
        """
        Decide how to sort tasks in the queue, based on self.config.scheduling_policy.
        Could produce a multi-component key. Lower is popped first from the min-heap.
        """
        if self.config.scheduling_policy == "fifo":
            return task.creation_time
        elif self.config.scheduling_policy == "deadline":
            # Earliest Deadline First: If no deadline, fallback to a large number
            dd = task.deadline if task.deadline else (time.time() + 9999999)
            return dd
        elif self.config.scheduling_policy == "ml":
            # We can do a more advanced approach, e.g. combining the priority with
            # a model-based score. For demonstration, let's just keep the 'priority' for now.
            return task.priority
        else:
            # "priority" or default
            return task.priority

    def _is_task_expired(self, task: TaskRequest) -> bool:
        """
        Check if a task has missed its deadline or exceeded max_waiting_time in the queue.
        """
        now = time.time()
        # Deadline
        if task.deadline and now >= task.deadline:
            return True
        # Max waiting time
        if task.max_waiting_time and (now - task.creation_time) >= task.max_waiting_time:
            return True
        return False

    def _interrupt_task(self, task: TaskRequest) -> None:
        """
        Actual method to forcibly interrupt a running task. The logic is heavily environment-dependent:
          - If tasks run in local threads, we might set a stop flag that tasks check.
          - If tasks run in separate processes, we might do an os.kill with a signal.
          - If tasks run in HPC or a container, we might issue a job cancel to the cluster manager.
        Here, we stub it out as a no-op besides logging.
        """
        self.logger.warning(f"Task {task.task_id} is being interrupted. Implementation needed.")

    def _compute_anomaly_score(self, snapshot: ResourceUsageSnapshot) -> float:
        """
        Placeholder for an anomaly detection approach. For example, we might track the
        standard deviation of CPU usage or memory usage vs. historical data, or an advanced
        ML outlier detection. Here we do something trivial:
        """
        # e.g. if CPU usage is 120% (which is impossible?), or memory usage is > 99%, that is suspicious
        # We'll do a naive approach: any usage above 90% => anomaly_score = usage - 90
        # so 95% => 5.0 anomaly, etc.
        anomaly = 0.0
        if snapshot.cpu_percent > 90:
            anomaly += (snapshot.cpu_percent - 90)
        if snapshot.mem_percent > 90:
            anomaly += (snapshot.mem_percent - 90)
        return anomaly

    def _handle_hard_limit_exceed(self, snapshot: ResourceUsageSnapshot) -> None:
        """
        If the system usage is above the hard limit, take immediate steps
        such as forcibly pausing or killing tasks, or request cluster expansion if cluster_mode = True.
        """
        self.logger.warning(f"Hard limit exceeded (CPU={snapshot.cpu_percent:.1f}% / MEM={snapshot.mem_percent:.1f}%).")
        # We can throttle tasks, forcibly reduce their priority, or kill the biggest resource hog
        # For demonstration, let's forcibly throttle everything
        for t_id, t in self.running_tasks.items():
            self.logger.warning(f"Forcibly throttling {t_id} due to hard limit exceed.")
            t.priority += 2  # degrade priority
        # HPC logic could expand cluster or request more memory from a container orchestrator, etc.

        # We might also skip scheduling new tasks until usage returns to normal

# ---------------------------------------------------------------------------------
# EXAMPLE USAGE (if run standalone)
# ---------------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    config = ResourceManagerConfig(
        monitor_interval=1.0,
        scheduler_interval=1.0,
        cpu_soft_limit=70.0,
        cpu_hard_limit=90.0,
        mem_soft_limit=75.0,
        mem_hard_limit=90.0,
        scheduling_policy="deadline",
        ml_policy_enabled=False,  # toggle for demonstration
        cluster_mode=False,
        anomaly_detection=True
    )

    # Optionally pass a trained ML policy here:
    # ml_policy = SchedulingPolicyModel("some_model_path")
    ml_policy = None

    manager = ResourceManager(config, nexus_core=None, ml_policy_model=ml_policy)
    manager.start()

    # Example tasks
    def example_task_callback(task: TaskRequest, msg: str, delay_s: float):
        print(f"[TASK {task.task_id}] {msg} - sleeping {delay_s}s ...")
        time.sleep(delay_s)
        print(f"[TASK {task.task_id}] Completed.")

    # Create a few tasks with deadlines
    t1 = TaskRequest(
        priority=1,
        module_name="ModuleA",
        required_resources={"cpu": 0.2, "memory": 128},
        callback=example_task_callback,
        callback_args=("Hello from T1", 3.0),
        deadline=time.time() + 15.0,  # 15s from now
    )
    t2 = TaskRequest(
        priority=0,
        module_name="ModuleB",
        required_resources={"cpu": 0.3, "memory": 256},
        callback=example_task_callback,
        callback_args=("Greetings from T2", 2.0),
        retries=1
    )
    t3 = TaskRequest(
        priority=2,
        module_name="ModuleC",
        required_resources={"cpu": 0.1, "memory": 64},
        callback=example_task_callback,
        callback_args=("Third task", 4.0),
        max_waiting_time=10.0,  # won't wait more than 10s in queue
    )

    manager.submit_task(t1)
    manager.submit_task(t2)
    manager.submit_task(t3)

    time.sleep(8)
    manager.cancel_task(t3.task_id)  # demonstration of cancelling

    time.sleep(10)
    print("\nUsage history so far:")
    for snap in manager.get_usage_history():
        print(snap)

    manager.stop()
    print("\n--- ResourceManager has stopped ---")
    print("Queued tasks:", manager.get_queued_tasks())
    print("Running tasks:", manager.get_running_tasks())
    print("Completed tasks:", manager.get_completed_tasks())
    print("Failed tasks:", manager.get_failed_tasks())