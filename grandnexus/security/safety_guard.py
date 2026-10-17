# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 61111-61351
# grandnexus/security/safety_guard.py
# ──────────────────────────────────────────────────────────────────────────
"""
SafetyGuard — Alignment, Compliance & Action Safety Layer
=========================================================

Dépendances externes (optionnelles) :
    pip install pyopa pydantic[dotenv] prometheus-client
"""

from __future__ import annotations

import logging, re, time, uuid, hashlib, json, threading, os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Callable, Optional, Tuple

from grandnexus.core.nexus_core import NexusCore

# OPA optional
try:
    import pyopa                                 # type: ignore
    OPA_AVAILABLE = True
except ImportError:
    OPA_AVAILABLE = False

# Prometheus
try:
    from prometheus_client import Counter, Gauge
    METRIC_BLOCKED = Counter("gnx_guard_blocked", "Number of blocked items", ["reason"])
    METRIC_ALLOWED = Counter("gnx_guard_allowed", "Number of allowed items")
    METRIC_RISK    = Gauge("gnx_guard_risk", "Latest risk score")
except ImportError:
    METRIC_BLOCKED = METRIC_ALLOWED = METRIC_RISK = None


# ──────────────────────────────────────────────────────────────────────────
# 1. Enum & Config
# ──────────────────────────────────────────────────────────────────────────
class Risk(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class GuardConfig:
    policy_file: Optional[str] = None               # .rego ou .yaml
    log_file: str = "safety_audit.jsonl"
    block_threshold: Risk = Risk.HIGH
    anonymise_patterns: List[str] = field(default_factory=lambda: [
        r"\b([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\b",   # emails
        r"\b(\+?\d[\d \-\(\)]{6,})\b",                    # phone
    ])
    max_torque_nm: float = 50.0                   # exemple contrainte action
    max_velocity_m_s: float = 1.5
    opa_entrypoint: str = "policy/allow"

# ──────────────────────────────────────────────────────────────────────────
# 2. SafetyGuard principale
# ──────────────────────────────────────────────────────────────────────────
class SafetyGuard:
    MODULE_NAME = "safety_guard"

    def __init__(self, nexus_core: NexusCore, cfg: Optional[GuardConfig] = None):
        self.logger = logging.getLogger("GrandNexus.Security.SafetyGuard")
        self.nexus = nexus_core
        self.cfg = cfg or GuardConfig()

        # Runtime
        self._lock = threading.RLock()
        self._running = False

        # Audit log file handle
        self._log_fh = open(self.cfg.log_file, "a", encoding="utf8")

        # OPA
        self._opa_ctx = None
        if self.cfg.policy_file and OPA_AVAILABLE and self.cfg.policy_file.endswith(".rego"):
            with open(self.cfg.policy_file, "r", encoding="utf8") as f:
                rego_code = f.read()
            self._opa_ctx = pyopa.load(rego_code)

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────
    def start(self):
        if self._running:
            return
        self._running = True
        self.nexus.register_module(
            name=self.MODULE_NAME,
            module=self,
            dependencies=[],
        )
        self.logger.info("SafetyGuard started")

    def stop(self):
        self._running = False
        self._log_fh.close()

    # ─────────────────────────────────────────────────────────────────────
    # Main public API
    # ─────────────────────────────────────────────────────────────────────
    # 1) Content (text / doc) --------------------------------------------------
    def is_allowed(self, text: str, metadata: Dict[str, Any] | None = None) -> bool:
        """
        Retourne True si la conservation / diffusion est autorisée.
        Si bloqué : journalise et incrémente métriques.
        """
        meta = metadata or {}
        risk = self._assess_risk_text(text, meta)

        if METRIC_RISK:
            METRIC_RISK.set(risk.value)

        if risk.value >= self.cfg.block_threshold.value:
            self._audit("BLOCK", risk, text=text, meta=meta)
            if METRIC_BLOCKED:
                METRIC_BLOCKED.labels(reason=risk.name).inc()
            return False

        # anonymisation soft
        text_an = self._anonymise(text) if risk == Risk.HIGH else text
        self._audit("ALLOW", risk, text=text_an, meta=meta)
        if METRIC_ALLOWED:
            METRIC_ALLOWED.inc()
        return True

    # 2) Action command --------------------------------------------------------
    def validate_action(self, command: Dict[str, Any]) -> bool:
        """
        Vérifie contraintes de sûreté avant dispatch à ActuatorHub.
        """
        risk = self._assess_risk_action(command)
        if risk.value >= self.cfg.block_threshold.value:
            self._audit("BLOCK_ACTION", risk, action=command)
            if METRIC_BLOCKED:
                METRIC_BLOCKED.labels(reason="ACTION").inc()
            return False
        self._audit("ALLOW_ACTION", risk, action=command)
        return True

    # ─────────────────────────────────────────────────────────────────────
    # Risk assessment helpers
    # ─────────────────────────────────────────────────────────────────────
    def _assess_risk_text(self, text: str, meta: Dict[str, Any]) -> Risk:
        # 1) OPA policy
        if self._opa_ctx:
            decision = pyopa.evaluate(self._opa_ctx, self.cfg.opa_entrypoint, {"text": text, "meta": meta})
            if decision is False:
                return Risk.CRITICAL
            elif isinstance(decision, (int, float)):
                return self._map_score(decision)

        # 2) simple heuristics
        if len(text) > 10_000:
            return Risk.HIGH
        if any(re.search(pat, text, re.I) for pat in self.cfg.anonymise_patterns):
            return Risk.MEDIUM
        return Risk.LOW

    def _assess_risk_action(self, cmd: Dict[str, Any]) -> Risk:
        # Example kinematic check
        torque = cmd.get("torque_nm", 0.0)
        speed = cmd.get("speed_m_s", 0.0)
        if torque > self.cfg.max_torque_nm or speed > self.cfg.max_velocity_m_s:
            return Risk.CRITICAL
        return Risk.LOW

    @staticmethod
    def _map_score(score: float) -> Risk:
        if score >= 0.9:
            return Risk.CRITICAL
        if score >= 0.7:
            return Risk.HIGH
        if score >= 0.4:
            return Risk.MEDIUM
        return Risk.LOW

    # ─────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────
    def _anonymise(self, text: str) -> str:
        out = text
        for pat in self.cfg.anonymise_patterns:
            out = re.sub(pat, "[REDACTED]", out, flags=re.I)
        return out

    def _audit(self, verdict: str, risk: Risk, **payload):
        record = {
            "ts": time.time(),
            "verdict": verdict,
            "risk": risk.name,
            "payload": payload,
            "id": str(uuid.uuid4())[:8],
        }
        with self._lock:
            self._log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._log_fh.flush()
        self.logger.debug(f"Audit: {verdict} risk={risk.name}")

    # ─────────────────────────────────────────────────────────────────────
    # Message interface
    # ─────────────────────────────────────────────────────────────────────
    def handle_message(self, msg: Dict[str, Any]) -> None:
        mtype = msg.get("type")
        if mtype == "check_action":
            allowed = self.validate_action(msg["content"])
            self.nexus.send_message(
                source=self.MODULE_NAME,
                target=msg["source"],
                message_type="check_action_response",
                content={"allowed": allowed},
            )

    # ─────────────────────────────────────────────────────────────────────
    # Health / status
    # ─────────────────────────────────────────────────────────────────────
    def health_check(self) -> Dict[str, Any]:
        return {
            "policy_source": self.cfg.policy_file or "internal",
            "opa": bool(self._opa_ctx),
            "log_file": self.cfg.log_file,
            "healthy": True,
        }

    def get_status(self) -> Dict[str, Any]:
        return {"block_threshold": self.cfg.block_threshold.name}

if __name__ == "__main__":  # démo d'origine, ne tourne plus à l'import
    from grandnexus.security.safety_guard import SafetyGuard, GuardConfig
    from grandnexus.core.nexus_core import NexusCore

    nexus = NexusCore()
    guard = SafetyGuard(nexus, GuardConfig(block_threshold=Risk.MEDIUM))
    guard.start()

    print(guard.is_allowed("Contact me at john.doe@example.com", {}))   # False (PII)
    print(guard.validate_action({"torque_nm": 10, "speed_m_s": 0.5}))   # True
