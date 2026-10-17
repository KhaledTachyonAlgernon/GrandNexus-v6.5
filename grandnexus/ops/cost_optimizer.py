# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 62256-62452
# grandnexus/ops/cost_optimizer.py
# ─────────────────────────────────────────────────────────────────────────
"""
CostOptimizer — FinOps Governor for GrandNexus
==============================================

Fonctions clés
──────────────
• Mesure coût courant via callbacks (hébergeurs cloud ou mock).
• Projette dépense fin de mois, compare au `monthly_budget`.
• Trois états : *NORMAL*, *WARN*, *CRITICAL* ; actions correctives graduelles.
• Publie drive `COST` à HomeostasisManager + métriques Prometheus.
• Table d’adaptation configurable par état (frequences, sampling, safe-mode).
"""

from __future__ import annotations

import logging, time, threading, uuid, math
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional

from grandnexus.core.nexus_core import NexusCore

try:
    from prometheus_client import Gauge
    PROM_ENABLED = True
except ImportError:
    PROM_ENABLED = False

# ─────────────────────────────────────────────────────────────────────────
# 1. Config structures
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class AdaptationRule:
    """Action exécutée lors d’un changement d’état budgétaire."""
    set_clock_fast_hz: Optional[float] = None
    sensor_sampling_factor: Optional[float] = None    # multiplicateur <1.0
    actuator_safe_mode: Optional[bool] = None
    freeze_embeddings: Optional[bool] = None          # désactive SemanticMemory upserts


@dataclass
class CostCfg:
    monthly_budget_usd: float = 300.0
    alert_threshold: float = 0.8          # 80 % du budget → WARN
    critical_threshold: float = 1.0       # 100 % du budget → CRIT
    sample_period: float = 1800.0         # 30 min
    cost_callback: Optional[Callable[[], float]] = None   # doit renvoyer dépense cumulée USD
    tick_channel: str = "slow"
    homeostasis_target: str = "homeostasis_manager"
    adaptation_table: Dict[str, AdaptationRule] = field(default_factory=lambda: {
        "WARN": AdaptationRule(set_clock_fast_hz=30, sensor_sampling_factor=0.5),
        "CRITICAL": AdaptationRule(set_clock_fast_hz=10,
                                   sensor_sampling_factor=0.2,
                                   actuator_safe_mode=True,
                                   freeze_embeddings=True),
    })

# ─────────────────────────────────────────────────────────────────────────
# 2. CostOptimizer module
# ─────────────────────────────────────────────────────────────────────────
class CostOptimizer:
    MODULE_NAME = "cost_optimizer"
    MSG_DRIVE = "homeostasis_drive"

    def __init__(self, nexus: NexusCore, cfg: Optional[CostCfg] = None):
        self.logger = logging.getLogger("GrandNexus.Ops.CostOptimizer")
        self.nexus = nexus
        self.cfg = cfg or CostCfg()

        # Runtime
        self._lock = threading.RLock()
        self._running = False
        self._last_sample = 0.0
        self._state = "NORMAL"     # NORMAL / WARN / CRITICAL
        self._instance = str(uuid.uuid4())[:8]

        # Prometheus gauges
        if PROM_ENABLED:
            self.g_cost_curr = Gauge("gnx_cost_current_usd", "Current month cost USD")
            self.g_cost_proj = Gauge("gnx_cost_projection_usd", "Projected month-end cost USD")
            self.g_cost_budget = Gauge("gnx_cost_budget_usd", "Budget USD")
            self.g_cost_state = Gauge("gnx_cost_state", "0=NORMAL 1=WARN 2=CRIT")

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────
    def start(self):
        if self._running: return True
        self._running = True

        self.nexus.register_module(
            name=self.MODULE_NAME,
            module=self,
            dependencies=["cognitive_clock", "homeostasis_manager"],
        )

        clk = self.nexus.get_module("cognitive_clock")
        if clk:
            clk.subscribe(self.cfg.tick_channel, self._on_tick)

        # Init gauges
        if PROM_ENABLED:
            self.g_cost_budget.set(self.cfg.monthly_budget_usd)

        self.logger.info(f"CostOptimizer {self._instance} started")
        return True

    def stop(self): self._running = False

    # ─────────────────────────────────────────────────────────────────────
    # Tick sampling
    # ─────────────────────────────────────────────────────────────────────
    def _on_tick(self, _):
        now = time.time()
        if now - self._last_sample < self.cfg.sample_period:
            return
        self._last_sample = now
        self._sample_cost()

    def _sample_cost(self):
        try:
            spent = self.cfg.cost_callback() if self.cfg.cost_callback else 0.0
        except Exception as exc:
            self.logger.error(f"Cost callback error: {exc}")
            return

        days_in_month = 30
        day_of_month = time.localtime().tm_mday
        proj = spent / max(1, day_of_month) * days_in_month

        if PROM_ENABLED:
            self.g_cost_curr.set(spent)
            self.g_cost_proj.set(proj)
            self.g_cost_state.set({"NORMAL":0,"WARN":1,"CRITICAL":2}[self._state])

        # State machine
        budget = self.cfg.monthly_budget_usd
        ratio = proj / budget
        new_state = self._state
        if ratio >= self.cfg.critical_threshold:
            new_state = "CRITICAL"
        elif ratio >= self.cfg.alert_threshold:
            new_state = "WARN"
        else:
            new_state = "NORMAL"

        if new_state != self._state:
            self.logger.warning(f"Cost state transition {self._state} ➜ {new_state} (proj {proj:.1f}$)")
            self._apply_adaptation(new_state)
            self._state = new_state

        # Publish drive COST (0-1)
        cost_drive = min(1.0, ratio)
        self.nexus.send_message(
            source=self.MODULE_NAME,
            target=self.cfg.homeostasis_target,
            message_type=self.MSG_DRIVE,
            content={"COST": cost_drive},
            priority=3,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Adaptation actions
    # ─────────────────────────────────────────────────────────────────────
    def _apply_adaptation(self, state: str):
        rule = self.cfg.adaptation_table.get(state)
        if not rule: return
        # Clock
        if rule.set_clock_fast_hz:
            clk = self.nexus.get_module("cognitive_clock")
            if clk: clk.set_frequency("fast", rule.set_clock_fast_hz)
        # Sensor sampling
        if rule.sensor_sampling_factor:
            sh = self.nexus.get_module("sensor_hub")
            if sh:
                for k in sh.config.max_rate:
                    sh.config.max_rate[k] *= rule.sensor_sampling_factor
                self.logger.info(f"Sensor sampling reduced ×{rule.sensor_sampling_factor}")
        # Actuator safe-mode
        if rule.actuator_safe_mode is not None:
            ah = self.nexus.get_module("actuator_hub")
            if ah: ah.set_safe_mode(rule.actuator_safe_mode)
        # Freeze embeddings
        if rule.freeze_embeddings:
            sm = self.nexus.get_module("semantic_memory")
            if sm: setattr(sm, "freeze_upsert", True)

    # ─────────────────────────────────────────────────────────────────────
    # Health & status
    # ─────────────────────────────────────────────────────────────────────
    def health_check(self) -> Dict[str, Any]:
        return {"state": self._state, "healthy": True}

    def get_status(self) -> Dict[str, Any]:
        return {"state": self._state}
