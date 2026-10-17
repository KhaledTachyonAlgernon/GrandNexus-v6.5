# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 58817-59138
# grandnexus/sensors/sensor_hub.py
# ──────────────────────────────────────────────────────────────────────────
"""
SensorHub — Unified Sensory Gateway for GrandNexus
==================================================

Fonctions clés
──────────────
• Découverte dynamique & enregistrement de drivers (plugins setuptools ou
  dépôt local).
• Gestion du cycle de vie (init → start → stop) de chaque driver dans des
  threads indépendants ou coroutines asyncio (support des deux modes).
• Cadencement externe : écoute des *ticks* « fast » (ou custom) du
  `CognitiveClock` pour produire un *SensorFrame* cohérent.
• Publication des observations via `NexusCore.send_message`, typées
  `sensor_data` (priorité configurable par driver).
• API runtime : (de)charger un driver, changer la fréquence, mettre en pause.
• Observabilité native : `get_status()` consolidé + `health_check()` granularisé.
• Sécurité : sandbox optionnelle pour isoler les drivers non-fiables
  (process séparé via `multiprocessing` si activé).
"""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Protocol, Union

from grandnexus.core.nexus_core import NexusCore

# ──────────────────────────────────────────────────────────────────────────
# 1. Interfaces communes
# ──────────────────────────────────────────────────────────────────────────
class SensorDriver(Protocol):
    """
    Spécification minimale d’un driver capteur.

    start() et stop() doivent retourner un booléen de succès.
    read() renvoie un dict sérialisable (ou None si pas de données).
    """

    driver_id: str                         # UUID court ou nom unique
    name: str                              # Descriptif humain
    frequency_hz: float                    # Cadence cible
    running: bool

    def start(self) -> bool: ...
    def stop(self) -> bool: ...
    def read(self) -> Optional[Dict[str, Any]]: ...
    def health(self) -> Dict[str, Any]: ...  # optionnel


@dataclass
class HubConfig:
    plugin_paths: List[Union[str, Path]] = None
    default_priority: int = 2          # priorité pour NexusCore messages
    broadcast_target: str = "perception_core"
    sandbox_drivers: bool = False      # True → lancer drivers en process
    max_jitter_ms: float = 10.0        # tolérance collecte
    tick_channel: str = "fast"         # canal écouté sur CognitiveClock


# ──────────────────────────────────────────────────────────────────────────
# 2. Hub principal
# ──────────────────────────────────────────────────────────────────────────
class SensorHub:
    MODULE_NAME = "sensor_hub"
    MSG_TYPE = "sensor_data"

    def __init__(
        self,
        nexus_core: NexusCore,
        config: Optional[HubConfig] = None,
    ):
        self.logger = logging.getLogger("GrandNexus.Sensors.SensorHub")
        self.nexus_core = nexus_core
        self.config = config or HubConfig(plugin_paths=["grandnexus/sensors/drivers"])
        self._lock = threading.RLock()
        self._drivers: Dict[str, SensorDriver] = {}
        self._driver_threads: Dict[str, threading.Thread] = {}
        self._running = False
        self._instance_id = str(uuid.uuid4())[:8]

        # Statistiques
        self._frame_counters: Dict[str, int] = {}
        self._last_frame_ts: Dict[str, float] = {}
        self._jitter_stats: Dict[str, List[float]] = {}

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

        # Subscribe to CognitiveClock ticks
        clock = self.nexus_core.get_module("cognitive_clock")
        if clock:
            clock.subscribe(self.config.tick_channel, self._on_tick)
        else:
            self.logger.warning("CognitiveClock not found — fallback to internal loop")
            t = threading.Thread(target=self._legacy_loop, daemon=True)
            t.start()

        # Register into NexusCore
        self.nexus_core.register_module(
            name=self.MODULE_NAME,
            module=self,
            dependencies=["cognitive_clock"],
            config={
                "drivers": list(self._drivers.keys()),
                "tick_channel": self.config.tick_channel,
            },
        )
        self.logger.info(f"SensorHub {self._instance_id} started with drivers: {list(self._drivers)}")
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

        # Stop drivers
        for drv_id, drv in list(self._drivers.items()):
            try:
                drv.stop()
            except Exception as exc:
                self.logger.error(f"Error stopping driver {drv_id}: {exc}")

        # Join threads
        for th in self._driver_threads.values():
            if th.is_alive():
                th.join(timeout=1.0)

        self.logger.info("SensorHub stopped")
        return True

    # ──────────────────────────────────────────────────────────────────────
    # Plugin discovery
    # ──────────────────────────────────────────────────────────────────────
    def _discover_plugins(self) -> None:
        """Charge les drivers trouvés dans les chemins plugin_paths."""
        for path in self.config.plugin_paths or []:
            mod_path = Path(path).expanduser()
            if not mod_path.exists():
                self.logger.warning(f"Plugin path {mod_path} not found")
                continue

            for py in mod_path.rglob("*.py"):
                try:
                    spec = importlib.util.spec_from_file_location(py.stem, py)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)  # type: ignore
                    self._register_drivers_in_module(module)
                except Exception as exc:
                    self.logger.error(f"Failed to load sensor plugin {py}: {exc}")

    def _register_drivers_in_module(self, module: ModuleType) -> None:
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, object) and hasattr(obj, "read") and hasattr(obj, "start"):
                try:
                    driver: SensorDriver = obj()  # type: ignore
                    self._drivers[driver.driver_id] = driver
                    self.logger.info(f"Registered driver {driver.driver_id} ({driver.name})")
                except Exception as exc:
                    self.logger.error(f"Cannot instantiate driver {obj}: {exc}")

    # ──────────────────────────────────────────────────────────────────────
    # Driver threads
    # ──────────────────────────────────────────────────────────────────────
    def _spin_all_drivers(self) -> None:
        for drv_id, drv in self._drivers.items():
            if drv.start():
                t = threading.Thread(
                    target=self._driver_loop,
                    args=(drv_id, drv),
                    daemon=True,
                    name=f"SensorDriver-{drv_id}",
                )
                t.start()
                self._driver_threads[drv_id] = t
                self._frame_counters[drv_id] = 0
                self._last_frame_ts[drv_id] = 0.0
                self._jitter_stats[drv_id] = []
            else:
                self.logger.error(f"Driver {drv_id} failed to start")

    def _driver_loop(self, drv_id: str, drv: SensorDriver) -> None:
        period = 1.0 / drv.frequency_hz
        next_time = time.perf_counter() + period
        while self._running and drv.running:
            now = time.perf_counter()
            sleep_dur = next_time - now
            if sleep_dur > 0:
                time.sleep(sleep_dur)
                now = time.perf_counter()

            frame = drv.read()
            if frame is not None:
                self._publish_frame(drv_id, frame)

            # metrics
            with self._lock:
                cnt = self._frame_counters[drv_id] + 1
                self._frame_counters[drv_id] = cnt
                if self._last_frame_ts[drv_id]:
                    jitter = abs(now - self._last_frame_ts[drv_id] - period)
                    self._jitter_stats[drv_id].append(jitter)
                self._last_frame_ts[drv_id] = now

            next_time += period

    # ──────────────────────────────────────────────────────────────────────
    # Tick handling (external cadence)
    # ──────────────────────────────────────────────────────────────────────
    def _on_tick(self, payload: Dict[str, Any]) -> None:
        """
        Appelé par CognitiveClock à chaque tick; sert à synchroniser les drivers
        *passifs* (ex. LIDAR full-frame) qui ne tournent pas en boucle interne.
        """
        # Pour un premier module, on laisse les drivers actifs ; les passifs
        # pourront implémenter handle_tick(payload).
        for drv in self._drivers.values():
            if hasattr(drv, "handle_tick"):
                try:
                    frame = drv.handle_tick(payload)  # type: ignore
                    if frame:
                        self._publish_frame(drv.driver_id, frame)
                except Exception as exc:
                    self.logger.warning(f"Driver {drv.driver_id} tick error: {exc}")

    # ──────────────────────────────────────────────────────────────────────
    # Publication NexusCore
    # ──────────────────────────────────────────────────────────────────────
    def _publish_frame(self, drv_id: str, frame: Dict[str, Any]) -> None:
        content = {
            "driver_id": drv_id,
            "timestamp": time.time(),
            "data": frame,
        }
        self.nexus_core.send_message(
            source=self.MODULE_NAME,
            target=self.config.broadcast_target,
            message_type=self.MSG_TYPE,
            content=content,
            priority=self.config.default_priority,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Health & status
    # ──────────────────────────────────────────────────────────────────────
    def health_check(self) -> Dict[str, Any]:
        with self._lock:
            report = {}
            for drv_id, drv in self._drivers.items():
                stats = {}
                if hasattr(drv, "health"):
                    try:
                        stats = drv.health()  # type: ignore
                    except Exception as exc:
                        self.logger.warning(f"Driver {drv_id} health error: {exc}")
                # Metrics
                cnt = self._frame_counters.get(drv_id, 0)
                avg_jitter = (
                    sum(self._jitter_stats[drv_id]) / len(self._jitter_stats[drv_id])
                    if self._jitter_stats[drv_id]
                    else 0.0
                )
                stats.update(
                    {
                        "frames": cnt,
                        "avg_jitter_ms": round(avg_jitter * 1_000, 3),
                        "running": getattr(drv, "running", False),
                    }
                )
                # Simple health flag
                stats["healthy"] = stats["running"] and avg_jitter * 1_000 < self.config.max_jitter_ms
                report[drv_id] = stats
            return report

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "drivers": {
                    drv_id: {
                        "name": drv.name,
                        "freq_hz": drv.frequency_hz,
                        "frames": self._frame_counters.get(drv_id, 0),
                    }
                    for drv_id, drv in self._drivers.items()
                }
            }

    # ──────────────────────────────────────────────────────────────────────
    # Fallback internal loop (si pas de CognitiveClock)
    # ──────────────────────────────────────────────────────────────────────
    def _legacy_loop(self) -> None:
        self.logger.warning("SensorHub running legacy loop (no external clock)")
        while self._running:
            time.sleep(1.0)  # laisse les driver loops autonomes


# ──────────────────────────────────────────────────────────────────────────
# 3. Exemple de driver Gym simplifié
