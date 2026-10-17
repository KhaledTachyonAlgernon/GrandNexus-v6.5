# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 62049-62255
# grandnexus/metacognition/affect_regulator.py
# ─────────────────────────────────────────────────────────────────────────
"""
AffectRegulator — Valence-Arousal Modulator for GrandNexus
==========================================================

Entrées
───────
• Drives Homeostasie  : FATIGUE, HUNGER, THERMAL, COST, RATE_LIMIT …
• Feedback Exécutif   : action_feedback (success ↗ valence, failure ↘)
• Évènements Externes : user_emotion, reward_signal (optionnel)

Sorties
───────
• Message affect_drive {valence:-1..1, arousal:0..1, tag:str}
  Direction : intrinsic_motivation, attention_filter, executive_planner
• Ajustement runtime  : appel direct à WorkingMemory / Planner pour
  modifier paramètres (decay, exploration_rate).
"""

from __future__ import annotations

import logging, time, math, threading, uuid
from dataclasses import dataclass
from typing import Dict, Any, Optional

from grandnexus.core.nexus_core import NexusCore

# ─────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class AffectConfig:
    publish_interval: float = 2.0          # s
    homeo_weights: Dict[str, float] = None # mapping drive -> (Δval, Δaru)
    feedback_gain: float = 0.1             # valence boost par success
    decay_half_life: float = 20.0          # temps retour à baseline
    baseline_valence: float = 0.0
    baseline_arousal: float = 0.3
    tick_channel: str = "slow"
    targets: tuple = (
        "intrinsic_motivation",
        "attention_filter",
        "executive_planner",
        "working_memory_v2",  # réparation 6.5 : renommage
    )

    def __post_init__(self):
        if self.homeo_weights is None:
            # map drive -> (Δvalence per unit, Δarousal per unit)
            self.homeo_weights = {
                "FATIGUE": (-0.6, -0.3),
                "HUNGER":  (-0.5, +0.2),
                "THERMAL": (-0.4, +0.4),
                "COST":    (-0.3, +0.1),
                "RATE_LIMIT": (-0.2, +0.3),
                "MOOD_EXT": (+0.5, +0.5),  # exemple d’évènement externe positif
            }

# ─────────────────────────────────────────────────────────────────────────
# 2. AffectRegulator
# ─────────────────────────────────────────────────────────────────────────
class AffectRegulator:
    MODULE_NAME = "affect_regulator"
    MSG_AFFECT = "affect_drive"

    def __init__(self, nexus: NexusCore, cfg: Optional[AffectConfig] = None):
        self.logger = logging.getLogger("GrandNexus.Meta.AffectReg")
        self.nexus = nexus
        self.cfg = cfg or AffectConfig()

        # Current state
        self._valence = self.cfg.baseline_valence
        self._arousal = self.cfg.baseline_arousal

        # Runtime
        self._lock = threading.RLock()
        self._running = False
        self._last_pub = 0.0
        self._instance = str(uuid.uuid4())[:8]

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────
    def start(self):
        if self._running: return
        self._running = True

        self.nexus.register_module(
            name=self.MODULE_NAME,
            module=self,
            dependencies=["cognitive_clock", "homeostasis_manager"],
        )

        clk = self.nexus.get_module("cognitive_clock")
        if clk:
            clk.subscribe(self.cfg.tick_channel, self._on_tick)

        self.logger.info(f"AffectRegulator {self._instance} started")

    def stop(self): self._running = False

    # ─────────────────────────────────────────────────────────────────────
    # Message handling
    # ─────────────────────────────────────────────────────────────────────
    def handle_message(self, msg: Dict[str, Any]) -> None:
        tp = msg.get("type")
        if tp == "homeostasis_drive":
            self._apply_drives(msg["content"])
        elif tp == "action_feedback":
            self._apply_feedback(msg["content"])
        elif tp == "external_emotion":
            self._apply_external(msg["content"])

    # ─────────────────────────────────────────────────────────────────────
    # Internal affect updates
    # ─────────────────────────────────────────────────────────────────────
    def _apply_drives(self, drives: Dict[str, float]):
        with self._lock:
            for d, v in drives.items():
                if d in self.cfg.homeo_weights:
                    dv, da = self.cfg.homeo_weights[d]
                    self._valence += dv * v
                    self._arousal += da * v
            self._clip()

    def _apply_feedback(self, fb: Dict[str, Any]):
        success = fb.get("result", {}).get("success", False)
        with self._lock:
            self._valence += self.cfg.feedback_gain if success else -self.cfg.feedback_gain
            self._arousal += 0.05 if success else 0.1
            self._clip()

    def _apply_external(self, ext: Dict[str, Any]):
        # ext: {"valence": float, "arousal": float}
        with self._lock:
            self._valence += ext.get("valence", 0)
            self._arousal += ext.get("arousal", 0)
            self._clip()

    def _decay_towards_baseline(self, dt: float):
        half = self.cfg.decay_half_life
        if half <= 0: return
        decay = math.pow(0.5, dt / half)
        self._valence = self.cfg.baseline_valence + (self._valence - self.cfg.baseline_valence) * decay
        self._arousal = self.cfg.baseline_arousal + (self._arousal - self.cfg.baseline_arousal) * decay

    def _clip(self):
        self._valence = max(-1.0, min(1.0, self._valence))
        self._arousal = max(0.0, min(1.0, self._arousal))

    # ─────────────────────────────────────────────────────────────────────
    # Tick publish
    # ─────────────────────────────────────────────────────────────────────
    def _on_tick(self, _payload):
        now = time.time()
        dt = now - self._last_pub
        with self._lock:
            self._decay_towards_baseline(dt)
            if now - self._last_pub >= self.cfg.publish_interval:
                self._publish_affect()
                self._last_pub = now

    def _publish_affect(self):
        payload = {"valence": round(self._valence, 3), "arousal": round(self._arousal, 3)}
        # Broadcasting
        for tgt in self.cfg.targets:
            self.nexus.send_message(
                source=self.MODULE_NAME,
                target=tgt,
                message_type=self.MSG_AFFECT,
                content=payload,
                priority=4,
            )
        # Direct parameter tuning examples
        wm = self.nexus.get_module("working_memory_v2")  # réparation 6.5 : seule la v2 a decay_half_life
        if wm and hasattr(wm, "config"):
            # Higher arousal ⇒ slower decay (retain more)
            wm.config.decay_half_life = max(5.0, 30.0 * (1.0 - self._arousal))

        self.logger.debug(f"Affect broadcast {payload}")

    # ─────────────────────────────────────────────────────────────────────
    # Health / status
    # ─────────────────────────────────────────────────────────────────────
    def health_check(self) -> Dict[str, Any]:
        return {"valence": self._valence, "arousal": self._arousal, "healthy": True}

    def get_status(self) -> Dict[str, Any]:
        return {"valence": round(self._valence, 3), "arousal": round(self._arousal, 3)}

if __name__ == "__main__":  # démo d'origine, ne tourne plus à l'import
    from grandnexus.core.nexus_core import NexusCore
    from grandnexus.clock.cognitive_clock import CognitiveClock
    from grandnexus.metacognition.affect_regulator import AffectRegulator

    nexus = NexusCore()
    CognitiveClock(nexus).start()
    AffectRegulator(nexus).start()

    # Simuler drive FATIGUE élevé
    nexus.send_message(
        source="homeostasis_manager",
        target="affect_regulator",
        message_type="homeostasis_drive",
        content={"FATIGUE": 0.9},
    )
