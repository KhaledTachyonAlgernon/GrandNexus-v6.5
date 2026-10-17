# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 64047-64223
# grandnexus/distributed/federated_learning_coordinator.py
# ───────────────────────────────────────────────────────────────────────────
"""
FederatedLearningCoordinator — Federated Model Aggregation Service for GrandNexus
================================================================================

Dependencies :
    pip install torch numpy cryptography
"""

from __future__ import annotations
import logging, threading, time, uuid
from dataclasses import dataclass
from typing import Dict, Any, List

import torch
import numpy as np
from cryptography.fernet import Fernet  # For lightweight encryption of updates

from grandnexus.core.nexus_core import NexusCore

# ───────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class FedConfig:
    aggregation_interval: float = 600.0   # secondes entre agrégations
    tick_channel: str = "slow"
    encryption_key: bytes = Fernet.generate_key()  # clé symétrique pour chiffrer mises à jour


# ───────────────────────────────────────────────────────────────────────────
# 2. FederatedLearningCoordinator
# ───────────────────────────────────────────────────────────────────────────
class FederatedLearningCoordinator:
    MODULE_NAME = "federated_learning_coordinator"
    MSG_LOCAL_UPDATE = "local_model_update"     # attend : {"client_id", "update_encrypted"}
    MSG_GLOBAL_UPDATE = "global_model_update"   # envoie : {"round_id", "state_dict"}

    def __init__(self, nexus: NexusCore, cfg: FedConfig = FedConfig()):
        self.logger = logging.getLogger("GrandNexus.FedLearn")
        self.nexus = nexus
        self.cfg = cfg
        self._lock = threading.RLock()
        self._running = False

        # Registre des mises à jour reçues par round
        self._pending_updates: Dict[str, List[Dict[str, Any]]] = {}
        self._round = 0
        self._fernet = Fernet(self.cfg.encryption_key)

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────
    def start(self):
        if self._running: return
        self._running = True
        self.nexus.register_module(
            name=self.MODULE_NAME,
            module=self,
            dependencies=["cognitive_clock", "multi_agent_coordinator"],
        )
        clk = self.nexus.get_module("cognitive_clock")
        if clk:
            clk.subscribe(self.cfg.tick_channel, self._on_tick)
        self.logger.info("FederatedLearningCoordinator started")

    def stop(self):
        self._running = False

    # ──────────────────────────────────────────────────────────────────────
    # Message handling — réception des updates chiffrés
    # ──────────────────────────────────────────────────────────────────────
    def handle_message(self, msg: Dict[str, Any]):
        if msg.get("type") == self.MSG_LOCAL_UPDATE:
            content = msg["content"]
            round_id = content["round_id"]
            encrypted = content["update_encrypted"].encode()
            try:
                raw = self._fernet.decrypt(encrypted)
                state_dict_np = torch.load(torch.io.BytesIO(raw), map_location="cpu")
                with self._lock:
                    self._pending_updates.setdefault(round_id, []).append(state_dict_np)
                self.logger.debug(f"Received local update from {content['client_id']} for round {round_id}")
            except Exception as e:
                self.logger.error(f"Failed to decrypt or deserialize update: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Tick callback — agrégation périodique
    # ──────────────────────────────────────────────────────────────────────
    def _on_tick(self, _):
        now = time.time()
        # Lancer une agrégation à chaque intervalle
        if not hasattr(self, "_last_agg"): self._last_agg = now
        if now - self._last_agg < self.cfg.aggregation_interval:
            return
        self._aggregate_round()
        self._last_agg = now

    # ──────────────────────────────────────────────────────────────────────
    # Agrégation fédérée (FedAvg)
    # ──────────────────────────────────────────────────────────────────────
    def _aggregate_round(self):
        with self._lock:
            r = f"round_{self._round}"
            updates = self._pending_updates.pop(r, [])
        if not updates:
            self.logger.info(f"No updates to aggregate for {r}")
            self._round += 1
            return

        # FedAvg : moyenne pondérée uniforme
        avg_state = {}
        for key in updates[0].keys():
            stacked = torch.stack([u[key] for u in updates], dim=0)
            avg_state[key] = torch.mean(stacked, dim=0)

        # Préparer la diffusion du modèle global
        payload = {
            "round_id": r,
            "state_dict": avg_state
        }
        # Diffuser via MultiAgentCoordinator
        mac = self.nexus.get_module("multi_agent_coordinator")
        if mac:
            mac.broadcast({"topic":"model_update","data":payload})
        # Mettre à jour localement le WorldModelLearner
        wml = self.nexus.get_module("world_model_learner")
        if wml and hasattr(wml.model, "load_state_dict"):
            wml.model.load_state_dict(avg_state)
            self.logger.info(f"Applied aggregated model for {r} locally")

        # Notifier les participants (via NexusCore message)
        self.nexus.send_message(
            source=self.MODULE_NAME,
            target="all",  # convention pour diffusion globale
            message_type=self.MSG_GLOBAL_UPDATE,
            content=payload,
            priority=5
        )
        self.logger.info(f"FedAvg aggregation completed for {r} with {len(updates)} updates")
        self._round += 1

    # ──────────────────────────────────────────────────────────────────────
    # Méthodes utilitaires pour clients — appelées par WorldModelLearner ou ContinuousLearner
    # ──────────────────────────────────────────────────────────────────────
    def submit_local_update(self, client_id: str, state_dict: Dict[str, torch.Tensor]):
        """
        Sérialiser et chiffrer state_dict, puis envoyer à l'orchestrateur.
        """
        buf = torch.io.BytesIO()
        torch.save(state_dict, buf)
        raw = buf.getvalue()
        encrypted = self._fernet.encrypt(raw).decode()
        payload = {
            "round_id": f"round_{self._round}",
            "client_id": client_id,
            "update_encrypted": encrypted
        }
        self.nexus.send_message(
            source=self.MODULE_NAME,
            target=self.MODULE_NAME,
            message_type=self.MSG_LOCAL_UPDATE,
            content=payload
        )

    # ──────────────────────────────────────────────────────────────────────
    # Health & status
    # ──────────────────────────────────────────────────────────────────────
    def health_check(self):
        with self._lock:
            pending = len(self._pending_updates.get(f"round_{self._round}", []))
        return {"current_round": self._round, "pending_updates": pending, "healthy": True}

    def get_status(self):
        return {"round": self._round, "pending": len(self._pending_updates.get(f"round_{self._round}", []))}
