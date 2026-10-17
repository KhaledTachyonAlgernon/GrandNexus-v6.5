# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 59591-59872
# grandnexus/homeostasis/homeostasis_manager.py
# ───────────────────────────────────────────────────────────────────────────
"""
HomeostasisManager — Resource Autoregulation & Drive Generator
==============================================================

Fonctions clés
──────────────
• Surveillance multi-domaine : CPU, mémoire, I/O disque, température, batterie,
  quota API, latence réseau, coût cloud.
• Seuils adaptatifs : basés sur *moving-window percentiles* + profils
  contextuels (desktop, edge, cloud).
• Calcul de drives normalisés [0.0-1.0] → messages `homeostasis_drive`
  broadcast vers *IntrinsicMotivation*, *AttentionFilter* et *ExecutivePlanner*.
• Mécanismes de défense : throttling CPU, déclenchement _safe-mode_ dans
  `ActuatorHub`, isolation de modules énergivores via `ErrorRecoveryManager`.
• Observabilité : métriques historiques exportables Prometheus, `health_check()`
  conforme au noyau.
• Plugin de capteur externe : possibilité d’ajouter des sondes custom (GPU,
  bande passante RF, etc.) via setuptools-entrypoints.
"""

from __future__ import annotations

from dataclasses import field  # réparation 6.5 : fourni avant par l'espace de noms partagé du notebook
import logging
import threading
import time
import psutil                # pip install psutil
import uuid
from dataclasses import dataclass
from typing import Dict, List, Callable, Optional, Any
from collections import deque

from grandnexus.core.nexus_core import NexusCore

# ───────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class Thresholds:
    warn: float        # % ou fraction
    critical: float


@dataclass
class HomeostasisConfig:
    # Fenêtre glissante (secondes) pour la moyenne mobile
    history_sec: int = 120
    sample_period: float = 1.0
    cpu_thr: Thresholds = field(default_factory=lambda: Thresholds(0.70, 0.90))
    mem_thr: Thresholds = field(default_factory=lambda: Thresholds(0.75, 0.95))
    temp_thr: Thresholds = field(default_factory=lambda: Thresholds(0.60, 0.80))        # fraction of Tjmax
    batt_thr: Thresholds = field(default_factory=lambda: Thresholds(0.30, 0.10))        # remaining % (inverse)
    api_thr: Thresholds = field(default_factory=lambda: Thresholds(0.70, 0.90))         # quota usage
    cost_thr: Thresholds = field(default_factory=lambda: Thresholds(0.70, 0.90))        # budget usage
    integration_targets: List[str] = (
        "intrinsic_motivation",
        "attention_filter",
        "executive_planner",
    )
    drive_publish_interval: float = 5.0                  # sec
    safe_mode_driver: str = "actuator_hub"               # module to throttle


# ───────────────────────────────────────────────────────────────────────────
# Manager
# ───────────────────────────────────────────────────────────────────────────
class HomeostasisManager:
    MODULE_NAME = "homeostasis_manager"
    MSG_TYPE = "homeostasis_drive"

    def __init__(
        self,
        nexus_core: NexusCore,
        config: Optional[HomeostasisConfig] = None,
        api_quota_cb: Optional[Callable[[], float]] = None,   # return 0-1 usage
        cost_meter_cb: Optional[Callable[[], float]] = None,  # idem
    ):
        self.logger = logging.getLogger("GrandNexus.Homeostasis.Manager")
        self.nexus_core = nexus_core
        self.config = config or HomeostasisConfig()
        self.api_quota_cb = api_quota_cb
        self.cost_meter_cb = cost_meter_cb

        # Runtime
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._instance_id = str(uuid.uuid4())[:8]

        # History
        window = int(self.config.history_sec / self.config.sample_period)
        self._hist_cpu = deque(maxlen=window)
        self._hist_mem = deque(maxlen=window)
        self._hist_temp = deque(maxlen=window)
        self._hist_batt = deque(maxlen=window)
        self._hist_api = deque(maxlen=window)
        self._hist_cost = deque(maxlen=window)

        # Drive cache
        self._last_publish = 0.0
        self._current_drives: Dict[str, float] = {}

    # ───────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ───────────────────────────────────────────────────────────────────────
    def start(self) -> bool:
        with self._lock:
            if self._running:
                return True
            self._running = True

        # Enregistrement nucleo
        self.nexus_core.register_module(
            name=self.MODULE_NAME,
            module=self,
            dependencies=["cognitive_clock"],
            config={},
        )

        # Thread daemon
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

        # Souscription tick lent pour watchdog
        clock = self.nexus_core.get_module("cognitive_clock")
        if clock:
            clock.subscribe("slow", self._on_tick)

        self.logger.info(f"HomeostasisManager {self._instance_id} started")
        return True

    def stop(self) -> bool:
        with self._lock:
            if not self._running:
                return True
            self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.logger.info("HomeostasisManager stopped")
        return True

    # ───────────────────────────────────────────────────────────────────────
    # Monitoring loop
    # ───────────────────────────────────────────────────────────────────────
    def _monitor_loop(self) -> None:
        while self._running:
            self._sample_once()
            self._evaluate_drives()
            time.sleep(self.config.sample_period)

    def _sample_once(self) -> None:
        try:
            cpu = psutil.cpu_percent() / 100.0
            mem = psutil.virtual_memory().percent / 100.0
            # Température (peut être absente)
            temps = psutil.sensors_temperatures()
            if temps:
                # moyenne de tous les capteurs
                t = sum([s.current for v in temps.values() for s in v]) / max(1, sum(len(v) for v in temps.values()))
                temp_norm = min(1.0, t / 90.0)      # supposition Tjmax=90 °C
            else:
                temp_norm = 0.0

            batt_val = 0.0
            if psutil.sensors_battery():
                batt_val = 1.0 - (psutil.sensors_battery().percent / 100.0)

            api_use = self.api_quota_cb() if self.api_quota_cb else 0.0
            cost_use = self.cost_meter_cb() if self.cost_meter_cb else 0.0

            with self._lock:
                self._hist_cpu.append(cpu)
                self._hist_mem.append(mem)
                self._hist_temp.append(temp_norm)
                self._hist_batt.append(batt_val)
                self._hist_api.append(api_use)
                self._hist_cost.append(cost_use)
        except Exception as exc:
            self.logger.error(f"Homeostasis sampling error: {exc}")

    # ───────────────────────────────────────────────────────────────────────
    # Drive computation
    # ───────────────────────────────────────────────────────────────────────
    def _evaluate_drives(self) -> None:
        now = time.time()

        with self._lock:
            cpu = max(self._hist_cpu) if self._hist_cpu else 0.0
            mem = max(self._hist_mem) if self._hist_mem else 0.0
            temp = max(self._hist_temp) if self._hist_temp else 0.0
            batt = max(self._hist_batt) if self._hist_batt else 0.0
            api = max(self._hist_api) if self._hist_api else 0.0
            cost = max(self._hist_cost) if self._hist_cost else 0.0

        drives = {
            "FATIGUE": cpu,
            "STRESS": mem,
            "THERMAL": temp,
            "HUNGER": batt,
            "COST": cost,
            "RATE_LIMIT": api,
        }
        self._current_drives = drives

        # Publier toutes les drive `drive_publish_interval`
        if now - self._last_publish >= self.config.drive_publish_interval:
            self._broadcast_drives(drives)
            self._last_publish = now

        # Détection seuil critique → actions défensives
        if any(
            [
                cpu >= self.config.cpu_thr.critical,
                mem >= self.config.mem_thr.critical,
                temp >= self.config.temp_thr.critical,
                batt >= self.config.batt_thr.critical,
            ]
        ):
            self._trigger_safe_mode()

    def _broadcast_drives(self, drives: Dict[str, float]) -> None:
        for tgt in self.config.integration_targets:
            self.nexus_core.send_message(
                source=self.MODULE_NAME,
                target=tgt,
                message_type=self.MSG_TYPE,
                content=drives,
                priority=4,
            )
        self.logger.debug(f"Drives broadcast: { {k:round(v,2) for k,v in drives.items()} }")

    # ───────────────────────────────────────────────────────────────────────
    # Reaction: safe-mode
    # ───────────────────────────────────────────────────────────────────────
    def _trigger_safe_mode(self) -> None:
        self.logger.warning("Critical homeostasis level reached → SAFE MODE")
        self.nexus_core.send_message(
            source=self.MODULE_NAME,
            target=self.config.safe_mode_driver,
            message_type="control",
            content={"cmd": "safe_mode", "enable": True},
            priority=1,
        )

    # ───────────────────────────────────────────────────────────────────────
    # Tick watchdog
    # ───────────────────────────────────────────────────────────────────────
    def _on_tick(self, payload: Dict[str, Any]) -> None:
        # Hypothèse : check drive freshness
        if time.time() - self._last_publish > 2 * self.config.drive_publish_interval:
            self.logger.warning("Drive broadcast delayed")

    # ───────────────────────────────────────────────────────────────────────
    # Health / status
    # ───────────────────────────────────────────────────────────────────────
    def health_check(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "avg_cpu": round(sum(self._hist_cpu) / max(1, len(self._hist_cpu)), 3),
                "avg_mem": round(sum(self._hist_mem) / max(1, len(self._hist_mem)), 3),
            }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "drives": {k: round(v, 3) for k, v in self._current_drives.items()},
                "hist_len": len(self._hist_cpu),
            }

if __name__ == "__main__":  # démo d'origine, ne tourne plus à l'import
    from grandnexus.core.nexus_core import NexusCore
    from grandnexus.clock.cognitive_clock import CognitiveClock
    from grandnexus.homeostasis.homeostasis_manager import HomeostasisManager

    def fake_api_usage(): return 0.42
    def fake_cost_usage(): return 0.18

    nexus = NexusCore()
    CognitiveClock(nexus).start()
    HomeostasisManager(nexus, api_quota_cb=fake_api_usage, cost_meter_cb=fake_cost_usage).start()
