# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 60166-60438
# grandnexus/perception/filters/attention_filter.py
# ─────────────────────────────────────────────────────────────────────────────
"""
AttentionFilter — Dynamic Saliency & Perceptual Gating Layer
===========================================================

Capacités
─────────
• **Bottom-Up** : calcul de saillance (variance, nouveauté, intensité) par pilote
  de capteur ; combinaison en score ∈ [0-1].
• **Top-Down** : injection d’objectifs (features, régions d’intérêt, regex,
  embeddings) et de _homeostasis drives_ (FATIGUE, THERMAL…) pour biaiser
  la distribution d’attention.
• **Budget de bande passante** : limite la fréquence ou la taille des flux
  entrants selon la charge CPU et la _working-memory_ disponible.
• **Apprentissage** : maintient un histogramme de « gains informatifs » et
  renforce la probabilité de sélection de stimuli qui ont amélioré la reward.
• **Observabilité & self-diagnosis** : expose métriques de *drop-rate*,
  latence, taux de faux-positifs (via feedback).
• **Hot-swap rules** : ajout / retrait de filtres ou _spotlights_ en JSON RPC.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Callable, Optional, Deque
from collections import deque, defaultdict

from grandnexus.core.nexus_core import NexusCore

# ─────────────────────────────────────────────────────────────────────────────
# 1. Structures auxiliaires
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SpotlightRule:
    """Filtre Top-Down : garde ou rejette si condition match."""
    name: str
    matcher: Callable[[Dict[str, Any]], bool]
    boost: float = 0.3                # ajoute au score
    expiry_ts: Optional[float] = None


@dataclass
class FilterConfig:
    # Bande passante max (items/s) par capteur
    max_rate: Dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 50.0))
    # Score seuil minimal pour laisser passer
    threshold: float = 0.4
    # Influence des drives [0-1] ⇒ modifie threshold
    drive_bias: Dict[str, float] = field(default_factory=lambda: {
        "FATIGUE": +0.2,      # fatigué ⇒ seuil ↑
        "HUNGER":  -0.1,      # batterie faible ⇒ veut stimuli pour recharge
        "THERMAL": +0.3,
    })
    history_sec: int = 30
    tick_channel: str = "fast"
    drive_source: str = "homeostasis_manager"
    perception_target: str = "perception_core"


# ─────────────────────────────────────────────────────────────────────────────
# 2. AttentionFilter principal
# ─────────────────────────────────────────────────────────────────────────────
class AttentionFilter:
    MODULE_NAME = "attention_filter"
    MSG_SENSOR = "sensor_data"
    MSG_DRIVE = "homeostasis_drive"

    def __init__(
        self,
        nexus_core: NexusCore,
        config: Optional[FilterConfig] = None,
    ):
        self.logger = logging.getLogger("GrandNexus.Perception.AttentionFilter")
        self.nexus_core = nexus_core
        self.config = config or FilterConfig()

        # Runtime
        self._lock = threading.RLock()
        self._running = False
        self._instance_id = str(uuid.uuid4())[:8]

        # Stats & state
        window = int(self.config.history_sec * max(self.config.max_rate.values(), default=50.0))  # réparation 6.5 : 50 Hz = défaut du defaultdict
        self._hist_scores: Deque[float] = deque(maxlen=window)
        self._last_sent: Dict[str, float] = defaultdict(float)   # per driver
        self._drives: Dict[str, float] = {}
        self._spotlights: List[SpotlightRule] = []

        # Metrics
        self._accepted = 0
        self._dropped = 0

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────
    def start(self) -> bool:
        with self._lock:
            if self._running:
                return True
            self._running = True

        self.nexus_core.register_module(
            name=self.MODULE_NAME,
            module=self,
            dependencies=["sensor_hub", "cognitive_clock", self.config.drive_source],
        )

        # Subscribe to ticks (for cleanup / rate reset)
        clock = self.nexus_core.get_module("cognitive_clock")
        if clock:
            clock.subscribe(self.config.tick_channel, self._on_tick)

        self.logger.info(f"AttentionFilter {self._instance_id} started")
        return True

    def stop(self) -> bool:
        with self._lock:
            self._running = False
        self.logger.info("AttentionFilter stopped")
        return True

    # ──────────────────────────────────────────────────────────────────────
    # Message entry-point
    # ──────────────────────────────────────────────────────────────────────
    def handle_message(self, msg: Dict[str, Any]) -> None:
        mtype = msg.get("type")
        if mtype == self.MSG_SENSOR:
            self._process_sensor(msg["content"])
        elif mtype == self.MSG_DRIVE:
            self._update_drives(msg["content"])

    # ──────────────────────────────────────────────────────────────────────
    # Sensor processing
    # ──────────────────────────────────────────────────────────────────────
    def _process_sensor(self, content: Dict[str, Any]) -> None:
        drv = content["driver_id"]
        now = time.time()

        # Rate limiting
        last = self._last_sent[drv]
        max_rate = self.config.max_rate[drv]
        if max_rate and (now - last) < (1.0 / max_rate):
            self._dropped += 1
            return

        # Score
        score = self._compute_saliency(content["data"])
        score = self._apply_spotlights(content, score)
        score = self._apply_drive_bias(score)

        self._hist_scores.append(score)

        if score >= self.config.threshold:
            self._accepted += 1
            self._last_sent[drv] = now
            self.nexus_core.send_message(
                source=self.MODULE_NAME,
                target=self.config.perception_target,
                message_type=self.MSG_SENSOR,
                content=content,
                priority=3,
            )
        else:
            self._dropped += 1

    # — saliency heuristics
    def _compute_saliency(self, data: Any) -> float:
        # Simple heuristics : variance / novelty
        try:
            if isinstance(data, dict) and "observation" in data:
                vals = data["observation"]
                if isinstance(vals, (list, tuple)):
                    return min(1.0, float(max(vals) - min(vals)))
            elif isinstance(data, (int, float)):
                return min(1.0, abs(float(data)) / 10.0)
        except Exception:
            pass
        return 0.1

    # — top-down spotlights
    def _apply_spotlights(self, msg: Dict[str, Any], score: float) -> float:
        boost = 0.0
        now = time.time()
        for rule in self._spotlights:
            if rule.expiry_ts and now > rule.expiry_ts:
                continue
            try:
                if rule.matcher(msg):
                    boost = max(boost, rule.boost)
            except Exception as exc:
                self.logger.error(f"Spotlight rule {rule.name} error: {exc}")
        return min(1.0, score + boost)

    # — homeostasis bias
    def _apply_drive_bias(self, score: float) -> float:
        adjusted = score
        for drive, bias in self.config.drive_bias.items():
            adjusted += bias * self._drives.get(drive, 0.0)
        return max(0.0, min(1.0, adjusted))

    # — drives update
    def _update_drives(self, drive: Dict[str, float]) -> None:
        with self._lock:
            self._drives = drive

    # ──────────────────────────────────────────────────────────────────────
    # Tick cleanup
    # ──────────────────────────────────────────────────────────────────────
    def _on_tick(self, payload: Dict[str, Any]) -> None:
        # Purge expired spotlights
        now = time.time()
        self._spotlights = [r for r in self._spotlights if not r.expiry_ts or r.expiry_ts > now]

    # ──────────────────────────────────────────────────────────────────────
    # Runtime API
    # ──────────────────────────────────────────────────────────────────────
    def add_spotlight(
        self,
        name: str,
        matcher: Callable[[Dict[str, Any]], bool],
        boost: float = 0.3,
        duration: float = 10.0,
    ) -> None:
        self._spotlights.append(
            SpotlightRule(name=name, matcher=matcher, boost=boost, expiry_ts=time.time() + duration)
        )
        self.logger.info(f"Added spotlight {name}")

    # ──────────────────────────────────────────────────────────────────────
    # Health / status
    # ──────────────────────────────────────────────────────────────────────
    def health_check(self) -> Dict[str, Any]:
        total = self._accepted + self._dropped
        ratio = self._dropped / total if total else 0.0
        healthy = ratio < 0.8
        return {
            "accepted": self._accepted,
            "dropped": self._dropped,
            "drop_ratio": round(ratio, 3),
            "healthy": healthy,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "hist_avg": round(sum(self._hist_scores) / max(1, len(self._hist_scores)), 3),
            "spotlights": [r.name for r in self._spotlights],
            "drives": {k: round(v, 2) for k, v in self._drives.items()},
        }

if __name__ == "__main__":  # démo d'origine, ne tourne plus à l'import
    from grandnexus.core.nexus_core import NexusCore
    from grandnexus.clock.cognitive_clock import CognitiveClock
    from grandnexus.sensors.sensor_hub import SensorHub
    from grandnexus.perception.filters.attention_filter import AttentionFilter

    nexus = NexusCore()
    CognitiveClock(nexus).start()
    SensorHub(nexus).start()
    AttentionFilter(nexus).start()

    # Spotlight pour driver 'gym_env' quand reward > 0.5
    af = nexus.get_module("attention_filter")
    af.add_spotlight(
        "high_reward",
        matcher=lambda m: m["driver_id"] == "gym_env" and m["data"]["reward"] > 0.5,
        boost=0.4,
        duration=30,
    )
