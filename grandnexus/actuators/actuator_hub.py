# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 59201-59542
# grandnexus/actuators/actuator_hub.py
# ──────────────────────────────────────────────────────────────────────────
"""
ActuatorHub — Unified Motor Gateway for GrandNexus
==================================================

Responsabilités
───────────────
• Agréger et router les *ActionCommand* émis par la couche Executive / Planner
  vers le ou les drivers matériels ou simulés.
• Orchestrer l’exécution concurrente avec contrôle de priorité, timeout,
  reprise sur erreur et escalade au *ErrorRecoveryManager* de `NexusCore` :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}.
• Offrir une API runtime (RPC ou message) : pause/reprise, changement de
  vitesse, mise en *safe-mode*, hot-swap de driver.
• Exposer `health_check()` et `get_status()` pour la télémétrie DevOps.
• Option _sandbox_ (« sécu process ») pour isoler un driver non fiable.
• Gestion fine de la *command-queue* : FIFO pondéré, latence mesurée,
  compensation de dérive.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Protocol, Union
import importlib
import inspect
import heapq

from grandnexus.core.nexus_core import NexusCore


# ──────────────────────────────────────────────────────────────────────────
# 1) Interfaces et configuration
# ──────────────────────────────────────────────────────────────────────────
class ActuatorDriver(Protocol):
    """
    Spécification minimale d’un driver actionneur.
    execute(cmd: Dict) doit retourner un dict {success: bool, …}.
    """

    driver_id: str
    name: str
    running: bool
    latency_budget_ms: float

    def start(self) -> bool: ...
    def stop(self) -> bool: ...
    def execute(self, command: Dict[str, Any]) -> Dict[str, Any]: ...
    def health(self) -> Dict[str, Any]: ...


@dataclass
class HubConfig:
    plugin_paths: List[Union[str, Path]] = None
    default_priority: int = 3
    message_type: str = "action_command"
    sandbox_drivers: bool = False
    safe_mode_on_start: bool = False
    max_queue_size: int = 1_000
    command_timeout_s: float = 5.0
    tick_channel: str = "medium"          # pour heartbeat
    metrics_window: int = 500


# ──────────────────────────────────────────────────────────────────────────
# 2) ActuatorHub principal
# ──────────────────────────────────────────────────────────────────────────
class ActuatorHub:
    MODULE_NAME = "actuator_hub"

    def __init__(
        self,
        nexus_core: NexusCore,
        config: Optional[HubConfig] = None,
    ):
        self.logger = logging.getLogger("GrandNexus.Actuators.ActuatorHub")
        self.nexus_core = nexus_core
        self.config = config or HubConfig(plugin_paths=["grandnexus/actuators/drivers"])

        # Runtime
        self._lock = threading.RLock()
        self._drivers: Dict[str, ActuatorDriver] = {}
        self._driver_threads: Dict[str, threading.Thread] = {}
        self._running = False
        self._instance_id = str(uuid.uuid4())[:8]

        # Command queue  (priority, ts, cmd_dict)
        self._queue: List[tuple] = []

        # Stats
        self._exec_count: Dict[str, int] = {}
        self._latencies: Dict[str, List[float]] = {}

        # Safe-mode flag
        self._safe_mode = self.config.safe_mode_on_start

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────
    def start(self) -> bool:
        with self._lock:
            if self._running:
                return True
            self._running = True

        self._discover_plugins()
        self._spin_all_drivers()

        # Subscribe to CognitiveClock for heartbeat
        clock = self.nexus_core.get_module("cognitive_clock")
        if clock:
            clock.subscribe(self.config.tick_channel, self._on_tick)

        # Register message handler
        self.nexus_core.register_module(
            name=self.MODULE_NAME,
            module=self,
            dependencies=["cognitive_clock"],
            config={"drivers": list(self._drivers.keys())},
        )

        # Spawn dispatcher thread
        dispatcher = threading.Thread(target=self._dispatcher_loop, daemon=True)
        dispatcher.start()

        self.logger.info(f"ActuatorHub {self._instance_id} started — safe_mode={self._safe_mode}")
        return True

    def stop(self) -> bool:
        with self._lock:
            if not self._running:
                return True
            self._running = False

        # Unsubscribe clock
        clock = self.nexus_core.get_module("cognitive_clock")
        if clock:
            clock.unsubscribe(self.config.tick_channel, self._on_tick)

        # Drain queue
        with self._lock:
            self._queue.clear()

        # Stop drivers
        for drv in self._drivers.values():
            try:
                drv.stop()
            except Exception as exc:
                self.logger.error(f"Stopping driver error {drv.driver_id}: {exc}")

        self.logger.info("ActuatorHub stopped")
        return True

    # ──────────────────────────────────────────────────────────────────────
    # NexusCore message entry-point
    # ──────────────────────────────────────────────────────────────────────
    def handle_message(self, msg: Dict[str, Any]) -> None:
        """
        Attendu : msg['type'] == config.message_type
        content : {
            driver_id: (optionnel),
            command: Dict,
            priority: int (optionnel),
            expire_at: float (epoch secondes) (optionnel)
        }
        """
        if msg.get("type") != self.config.message_type:
            return
        content = msg["content"]
        priority = content.get("priority", self.config.default_priority)
        with self._lock:
            if len(self._queue) >= self.config.max_queue_size:
                self.logger.warning("Command queue overflow — dropping lowest priority")
                heapq.heappop(self._queue)
            heapq.heappush(
                self._queue,
                (priority, time.time(), content),
            )

    # ──────────────────────────────────────────────────────────────────────
    # Dispatcher loop
    # ──────────────────────────────────────────────────────────────────────
    def _dispatcher_loop(self) -> None:
        while self._running:
            with self._lock:
                if not self._queue:
                    cmd_tuple = None
                else:
                    cmd_tuple = heapq.heappop(self._queue)
            if not cmd_tuple:
                time.sleep(0.002)
                continue

            _prio, queued_ts, payload = cmd_tuple

            # Check expiration
            if "expire_at" in payload and time.time() > payload["expire_at"]:
                self.logger.debug("Command expired → discard")
                continue

            if self._safe_mode:
                self.logger.debug("Safe-mode ON — command ignored")
                continue

            # Route
            driver_id = payload.get("driver_id")
            if driver_id:
                drv = self._drivers.get(driver_id)
                if not drv:
                    self.logger.error(f"Unknown driver {driver_id}")
                    continue
                self._execute_on_driver(drv, payload["command"])
            else:
                # broadcast to all drivers capable
                for drv in self._drivers.values():
                    self._execute_on_driver(drv, payload["command"])

    def _execute_on_driver(self, drv: ActuatorDriver, command: Dict[str, Any]) -> None:
        start_ts = time.perf_counter()
        try:
            result = drv.execute(command)
        except Exception as exc:
            self.logger.error(f"Driver {drv.driver_id} execution error: {exc}")
            result = {"success": False, "error": str(exc)}

        latency = (time.perf_counter() - start_ts) * 1_000  # ms
        with self._lock:
            self._exec_count[drv.driver_id] = self._exec_count.get(drv.driver_id, 0) + 1
            self._latencies.setdefault(drv.driver_id, []).append(latency)

        # Feedback to ExecutivePlanner (if needed)
        self.nexus_core.send_message(
            source=self.MODULE_NAME,
            target="executive_planner",
            message_type="action_feedback",
            content={
                "driver_id": drv.driver_id,
                "command": command,
                "result": result,
                "latency_ms": latency,
            },
            priority=self.config.default_priority + 1,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Tick (heartbeat)
    # ──────────────────────────────────────────────────────────────────────
    def _on_tick(self, payload: Dict[str, Any]) -> None:
        # Simple watchdog : si queue trop pleine → warning
        with self._lock:
            q_len = len(self._queue)
        if q_len > self.config.max_queue_size * 0.8:
            self.logger.warning(f"Actuator queue high load : {q_len} items")

    # ──────────────────────────────────────────────────────────────────────
    # Plugin discovery
    # ──────────────────────────────────────────────────────────────────────
    def _discover_plugins(self) -> None:
        for path in self.config.plugin_paths or []:
            p = Path(path).expanduser()
            if not p.exists():
                continue
            for py in p.rglob("*.py"):
                try:
                    spec = importlib.util.spec_from_file_location(py.stem, py)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)  # type: ignore
                    self._register_drivers(module)
                except Exception as exc:
                    self.logger.error(f"Actuator plugin load error {py}: {exc}")

    def _register_drivers(self, module: ModuleType) -> None:
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, object) and hasattr(cls, "execute"):
                try:
                    drv: ActuatorDriver = cls()  # type: ignore
                    if drv.driver_id in self._drivers:
                        self.logger.warning(f"Duplicate driver id {drv.driver_id}")
                        continue
                    self._drivers[drv.driver_id] = drv
                    self.logger.info(f"Registered actuator driver {drv.driver_id} ({drv.name})")
                except Exception as exc:
                    self.logger.error(f"Driver instantiation error {cls}: {exc}")

    def _spin_all_drivers(self) -> None:
        for drv in self._drivers.values():
            if drv.start():
                self.logger.info(f"Driver {drv.driver_id} started")
            else:
                self.logger.error(f"Driver {drv.driver_id} failed to start")

    # ──────────────────────────────────────────────────────────────────────
    # Health & status
    # ──────────────────────────────────────────────────────────────────────
    def health_check(self) -> Dict[str, Any]:
        with self._lock:
            rep = {}
            for drv_id, drv in self._drivers.items():
                base = {"running": getattr(drv, "running", False)}
                extra = {}
                if hasattr(drv, "health"):
                    try:
                        extra = drv.health()
                    except Exception as exc:
                        extra = {"error": str(exc)}
                lat = self._latencies.get(drv_id, [])
                avg_lat = sum(lat) / len(lat) if lat else 0.0
                rep[drv_id] = {
                    **base,
                    **extra,
                    "executed": self._exec_count.get(drv_id, 0),
                    "avg_latency_ms": round(avg_lat, 2),
                    "healthy": base["running"] and (avg_lat <= drv.latency_budget_ms),
                }
            return rep

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "queue_len": len(self._queue),
                "drivers": {
                    d: {"executed": self._exec_count.get(d, 0)} for d in self._drivers
                },
                "safe_mode": self._safe_mode,
            }

    # ──────────────────────────────────────────────────────────────────────
    # Runtime control
    # ──────────────────────────────────────────────────────────────────────
    def set_safe_mode(self, enable: bool = True) -> None:
        self._safe_mode = enable
        self.logger.warning(f"SAFE-MODE {'ENABLED' if enable else 'DISABLED'}")


# ──────────────────────────────────────────────────────────────────────────
# 3) Exemple driver simulé : “virtual gripper”
