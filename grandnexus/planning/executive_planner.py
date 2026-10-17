# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 59873-60165
# grandnexus/planning/executive_planner.py
# ──────────────────────────────────────────────────────────────────────────
"""
ExecutivePlanner — Hierarchical Task Network (HTN)-based Action Scheduler
========================================================================

Rôle
────
• Réceptionne _CognitiveTask_ et _Goal_ (voir `cognition_manager.py`) :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
• Décompose en primitives d’action via opérateurs HTN (« methods ») paramétrables.
• Maintient un _agenda_ FIFO pondéré : priorité, deadline, drive urgency.
• Publie des `action_command` vers `ActuatorHub` et gère le feedback.
• S’ajuste dynamiquement selon les drives *Homeostasis* (FATIGUE, HUNGER…) :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}
• Expose `health_check()` et journaux d’exécution (succès, latence, taux d’abandon).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
import heapq

from grandnexus.core.nexus_core import NexusCore

# Utilise la structure CognitiveTask déjà définie par CognitionManager
from grandnexus.cognition.cognition_manager import CognitiveTask, CognitiveEventType  # type: ignore


# ──────────────────────────────────────────────────────────────────────────
# 1) Structures locales
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class PlanStep:
    """Une primitive envoyée à ActuatorHub."""
    driver_id: str
    command: Dict[str, Any]
    expected_latency: float = 0.05
    retry: int = 0


@dataclass
class HTNMethod:
    """Décomposition HTN : goal_type → steps ou sous-goals."""
    goal_type: str
    expand: Callable[[Dict[str, Any]], List[PlanStep]]


@dataclass(order=True)
class AgendaEntry:
    """File de priorités (heapq)."""
    priority: float
    deadline: float
    created_at: float = field(compare=False)
    task: CognitiveTask = field(compare=False)
    plan: List[PlanStep] = field(compare=False, default_factory=list)


# ──────────────────────────────────────────────────────────────────────────
# 2) Configuration
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class PlannerConfig:
    default_priority: float = 0.5
    max_parallel_steps: int = 2
    feedback_timeout: float = 5.0
    drive_weight: float = 0.3                # influence des drives [0-1]
    tick_channel: str = "medium"
    drive_source: str = "homeostasis_manager"


# ──────────────────────────────────────────────────────────────────────────
# 3) ExecutivePlanner
# ──────────────────────────────────────────────────────────────────────────
class ExecutivePlanner:
    MODULE_NAME = "executive_planner"

    def __init__(
        self,
        nexus_core: NexusCore,
        config: Optional[PlannerConfig] = None,
    ):
        self.logger = logging.getLogger("GrandNexus.Planning.ExecutivePlanner")
        self.nexus_core = nexus_core
        self.config = config or PlannerConfig()

        # Runtime
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._instance_id = str(uuid.uuid4())[:8]

        # Agenda (heap)
        self._agenda: List[AgendaEntry] = []

        # HTN repository
        self._methods: Dict[str, List[HTNMethod]] = {}

        # Drives
        self._drives: Dict[str, float] = {}

        # Stats
        self._executed = 0
        self._failed = 0

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────
    def start(self) -> bool:
        with self._lock:
            if self._running:
                return True
            self._running = True

        # Register to NexusCore
        self.nexus_core.register_module(
            name=self.MODULE_NAME,
            module=self,
            dependencies=["cognitive_clock", "actuator_hub", self.config.drive_source],
        )

        # Subscribe ticks
        clock = self.nexus_core.get_module("cognitive_clock")
        if clock:
            clock.subscribe(self.config.tick_channel, self._on_tick)

        self._thread = threading.Thread(target=self._planner_loop, daemon=True)
        self._thread.start()

        self.logger.info(f"ExecutivePlanner {self._instance_id} started")
        return True

    def stop(self) -> bool:
        with self._lock:
            if not self._running:
                return True
            self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.logger.info("ExecutivePlanner stopped")
        return True

    # ──────────────────────────────────────────────────────────────────────
    # Message handling (tasks & drives & feedback)
    # ──────────────────────────────────────────────────────────────────────
    def handle_message(self, msg: Dict[str, Any]) -> None:
        """Route selon type."""
        mtype = msg.get("type")
        if mtype == "cognitive_task":
            self._ingest_task(msg["content"])
        elif mtype == "homeostasis_drive":
            self._update_drives(msg["content"])
        elif mtype == "action_feedback":
            self._process_feedback(msg["content"])

    # — tasks
    def _ingest_task(self, task_dict: Dict[str, Any]) -> None:
        task = CognitiveTask(**task_dict)       # reconstruction rapide
        plan = self._decompose(task)
        priority = task.priority or self.config.default_priority
        # modulation par drive (ex : température élevée ⇒ priorité basse)
        stress = self._drives.get("THERMAL", 0.0)
        priority += stress * self.config.drive_weight
        entry = AgendaEntry(
            priority=priority,
            deadline=task.deadline or (time.time() + 30),
            created_at=time.time(),
            task=task,
            plan=plan,
        )
        with self._lock:
            heapq.heappush(self._agenda, entry)
        self.logger.debug(f"Ingested task {task.task_id} with {len(plan)} steps")

    # — drives
    def _update_drives(self, drives: Dict[str, float]) -> None:
        with self._lock:
            self._drives = drives

    # — feedback
    def _process_feedback(self, fb: Dict[str, Any]) -> None:
        # Simple success counting
        if fb["result"].get("success"):
            self._executed += 1
        else:
            self._failed += 1

    # ──────────────────────────────────────────────────────────────────────
    # HTN registration
    # ──────────────────────────────────────────────────────────────────────
    def register_method(self, method: HTNMethod) -> None:
        self._methods.setdefault(method.goal_type, []).append(method)
        self.logger.info(f"Registered HTN method for {method.goal_type}")

    # Simple decompose (first-match)
    def _decompose(self, task: CognitiveTask) -> List[PlanStep]:
        methods = self._methods.get(task.task_type, [])
        if not methods:
            self.logger.warning(f"No HTN method for {task.task_type}")
            return []
        try:
            return methods[0].expand(task.parameters)
        except Exception as exc:
            self.logger.error(f"HTN expansion error {exc}")
            return []

    # ──────────────────────────────────────────────────────────────────────
    # Core loop
    # ──────────────────────────────────────────────────────────────────────
    def _planner_loop(self) -> None:
        while self._running:
            entry = None
            with self._lock:
                if self._agenda:
                    entry = heapq.heappop(self._agenda)

            if not entry:
                time.sleep(0.005)
                continue

            now = time.time()
            if now > entry.deadline:
                self.logger.warning(f"Task {entry.task.task_id} expired")
                self._failed += 1
                continue

            # Execute steps sequentially (could parallelize later)
            for step in entry.plan:
                self._dispatch_step(step)
                # Wait for feedback or timeout
                time.sleep(step.expected_latency)

    def _dispatch_step(self, step: PlanStep) -> None:
        self.nexus_core.send_message(
            source=self.MODULE_NAME,
            target="actuator_hub",
            message_type="action_command",
            content={
                "driver_id": step.driver_id,
                "command": step.command,
                "priority": 2,
                "expire_at": time.time() + self.config.feedback_timeout,
            },
            priority=2,
        )
        self.logger.debug(f"Dispatched step to {step.driver_id}: {step.command}")

    # ──────────────────────────────────────────────────────────────────────
    # Tick watchdog
    # ──────────────────────────────────────────────────────────────────────
    def _on_tick(self, payload: Dict[str, Any]) -> None:
        # backlog monitoring
        q_len = len(self._agenda)
        if q_len > 50:
            self.logger.warning(f"Planner backlog {q_len} tasks")

    # ──────────────────────────────────────────────────────────────────────
    # Health / status
    # ──────────────────────────────────────────────────────────────────────
    def health_check(self) -> Dict[str, Any]:
        with self._lock:
            fail_ratio = self._failed / max(1, self._executed + self._failed)
            healthy = fail_ratio < 0.1 and len(self._agenda) < 100
            return {
                "executed": self._executed,
                "failed": self._failed,
                "agenda_len": len(self._agenda),
                "healthy": healthy,
            }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "agenda_len": len(self._agenda),
                "drives": {k: round(v, 2) for k, v in self._drives.items()},
            }

if __name__ == "__main__":  # démo d'origine, ne tourne plus à l'import
    from grandnexus.planning.executive_planner import HTNMethod, PlanStep

    def move_arm_expand(params):
        return [
            PlanStep("virt_gripper", {"action": "open"}),
            PlanStep("robot_arm", {"move_to": params["target"]}),
            PlanStep("virt_gripper", {"action": "close"}),
        ]

    planner = nexus.get_module("executive_planner")
    planner.register_method(HTNMethod("move_object", move_arm_expand))
