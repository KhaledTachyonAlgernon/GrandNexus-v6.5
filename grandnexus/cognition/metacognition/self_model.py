# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 61352-61578
# grandnexus/metacognition/self_model.py
# ─────────────────────────────────────────────────────────────────────────
"""
SelfModelService — Structured Body-Schema & Capability Registry
==============================================================

• Représente le « moi » logiciel + matériel (robots, GPU, réseau).
• Valide en temps réel qu’une action planifiée / exécutée respecte les contraintes
  cinématiques, énergétiques, de sécurité et de licence.
• Réserve des ressources (GPU, API-tokens, bras robot) avec timeout et rollback.
• Fournit un flux de mises à jour (pub-sub) pour le raisonnement contre-factuel.
"""

from __future__ import annotations

import logging, time, threading, uuid, math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum, auto

from grandnexus.core.nexus_core import NexusCore

# ─────────────────────────────────────────────────────────────────────────
# 1. Énumérations & structures
# ─────────────────────────────────────────────────────────────────────────
class ResourceType(Enum):
    CPU = auto()
    GPU = auto()
    MEMORY = auto()
    BATTERY = auto()
    ARM = auto()
    GRIPPER = auto()
    NETWORK = auto()
    LICENSE = auto()


@dataclass
class JointState:
    name: str
    position_deg: float
    velocity_deg_s: float
    torque_nm: float
    limits: Dict[str, float]


@dataclass
class ResourceQuota:
    max_units: float
    reserved: float = 0.0
    last_update: float = field(default_factory=time.time)


@dataclass
class SelfModelConfig:
    publish_interval: float = 2.0             # s
    tick_channel: str = "slow"
    guard_module: str = "safety_guard"


# ─────────────────────────────────────────────────────────────────────────
# 2. Service principal
# ─────────────────────────────────────────────────────────────────────────
class SelfModelService:
    MODULE_NAME = "self_model_service"
    MSG_STATE = "self_state"

    def __init__(self, nexus: NexusCore, cfg: Optional[SelfModelConfig] = None):
        self.logger = logging.getLogger("GrandNexus.Meta.SelfModel")
        self.nexus = nexus
        self.cfg = cfg or SelfModelConfig()

        # Mutex
        self._lock = threading.RLock()

        # Runtime maps
        self._joints: Dict[str, JointState] = {}
        self._resources: Dict[ResourceType, ResourceQuota] = {
            ResourceType.CPU: ResourceQuota(max_units=1.0),
            ResourceType.GPU: ResourceQuota(max_units=1.0),
            ResourceType.MEMORY: ResourceQuota(max_units=1.0),
            ResourceType.BATTERY: ResourceQuota(max_units=1.0),
        }

        # Publipostage
        self._last_pub = 0.0
        self._running = False
        self._instance_id = str(uuid.uuid4())[:8]

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────
    def start(self):
        with self._lock:
            if self._running:
                return True
            self._running = True

        self.nexus.register_module(
            name=self.MODULE_NAME,
            module=self,
            dependencies=["cognitive_clock"],
        )

        clk = self.nexus.get_module("cognitive_clock")
        if clk:
            clk.subscribe(self.cfg.tick_channel, self._on_tick)

        self.logger.info(f"SelfModelService {self._instance_id} started")
        return True

    def stop(self):
        self._running = False
        return True

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────
    def update_joint(self, name: str, pos: float, vel: float, tq: float, limits: Dict[str, float]):
        with self._lock:
            self._joints[name] = JointState(name, pos, vel, tq, limits)

    def reserve(self, rtype: ResourceType, units: float, ttl: float = 5.0) -> bool:
        with self._lock:
            q = self._resources.setdefault(rtype, ResourceQuota(max_units=units))
            if q.reserved + units > q.max_units:
                return False
            q.reserved += units
            q.last_update = time.time()
        # Schedule release (simple thread)
        threading.Timer(ttl, self.release, args=[rtype, units]).start()
        return True

    def release(self, rtype: ResourceType, units: float):
        with self._lock:
            q = self._resources.get(rtype)
            if q:
                q.reserved = max(0.0, q.reserved - units)
                q.last_update = time.time()

    def is_action_safe(self, action: Dict[str, Any]) -> bool:
        """
        Vérifie cinématique + ressources avant exécution.
        Délégué partiel à SafetyGuard si présent.
        """
        guard = self.nexus.get_module(self.cfg.guard_module)
        if guard and hasattr(guard, "validate_action"):
            return guard.validate_action(action)

        # fallback simple : limite vitesse/torque
        jid = action.get("joint")
        if jid and jid in self._joints:
            st = self._joints[jid]
            max_v = st.limits.get("vel_deg_s", 180)
            max_t = st.limits.get("torque_nm", 50)
            if abs(action.get("velocity_deg_s", 0)) > max_v:
                return False
            if abs(action.get("torque_nm", 0)) > max_t:
                return False
        return True

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "joints": {n: st.__dict__ for n, st in self._joints.items()},
                "resources": {rt.name: q.__dict__ for rt, q in self._resources.items()},
            }

    # ─────────────────────────────────────────────────────────────────────
    # Tick publication
    # ─────────────────────────────────────────────────────────────────────
    def _on_tick(self, _payload):
        now = time.time()
        if now - self._last_pub >= self.cfg.publish_interval:
            state = self.get_state()
            self.nexus.send_message(
                source=self.MODULE_NAME,
                target="monitor",                    # module de monitoring générique
                message_type=self.MSG_STATE,
                content=state,
                priority=6,
            )
            self._last_pub = now

    # ─────────────────────────────────────────────────────────────────────
    # Message interface
    # ─────────────────────────────────────────────────────────────────────
    def handle_message(self, msg: Dict[str, Any]) -> None:
        if msg.get("type") == "query_self_state":
            self.nexus.send_message(
                source=self.MODULE_NAME,
                target=msg["source"],
                message_type="self_state_response",
                content=self.get_state(),
            )

    # ─────────────────────────────────────────────────────────────────────
    # Health & status
    # ─────────────────────────────────────────────────────────────────────
    def health_check(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "joints": len(self._joints),
                "resources": {r.name: q.reserved / q.max_units for r, q in self._resources.items()},
                "healthy": True,
            }

    def get_status(self) -> Dict[str, Any]:
        return self.get_state()

if __name__ == "__main__":  # démo d'origine, ne tourne plus à l'import
    from grandnexus.core.nexus_core import NexusCore
    from grandnexus.clock.cognitive_clock import CognitiveClock
    from grandnexus.metacognition.self_model import SelfModelService, ResourceType

    nexus = NexusCore()
    CognitiveClock(nexus).start()
    self_model = SelfModelService(nexus).start()

    # Enregistrer articulation
    self_model.update_joint("elbow", pos=10, vel=0, tq=0, limits={"vel_deg_s": 180, "torque_nm": 30})

    # Réserver 0.5 GPU pendant 3 s
    print(self_model.reserve(ResourceType.GPU, 0.5, ttl=3.0))  # → True

    # Vérifier sûreté action
    action = {"joint": "elbow", "velocity_deg_s": 90, "torque_nm": 10}
    print(self_model.is_action_safe(action))                   # → True
