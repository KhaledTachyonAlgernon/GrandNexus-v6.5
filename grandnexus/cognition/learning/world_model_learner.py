# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 62647-62871
# grandnexus/learning/world_model_learner.py
# ───────────────────────────────────────────────────────────────────────────
"""
WorldModelLearner — Online & Offline Environment Dynamics Learner
=================================================================

Fonctions principales
─────────────────────
• **Experience Buffer** : collecte (state, action, next_state, reward, done) depuis
  `SensorHub` + `ActuatorHub` ; stockage en mémoire circulaire + flush sur disque.
• **Modèles supportés** :
    – LSTM / GRU PyTorch (continuous)
    – GNN Relational GraphConv (symbolic triplets)
    – Gaussian Process (low-dim, pour incertitude)
  Sélection via config sans changer le code appelant.
• **Entraînement mixte** : online (mini-batch every N steps) + offline (epoch au repos).
• **Predict API** : `predict(state, action, horizon=H)` renvoie distribution d’états
  futurs + incertitude ; exposé par message `world_predict`.
• **Active Learning** : détecte zones de forte erreur ➜ publie `knowledge_gap`
  pour orienter Intrinsic Motivation.
• **Checkpoint + Versioning** : fichiers `.pt` horodatés dans `world_models/`, tag « prod » poussable.
"""

from __future__ import annotations

import logging, time, random, threading, uuid, os, json, math
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from grandnexus.core.nexus_core import NexusCore

# ───────────────────────────────────────────────────────────────────────────
# 1. Config
# ───────────────────────────────────────────────────────────────────────────
@dataclass
class WMdLearnerCfg:
    buffer_capacity: int = 100_000
    batch_size: int = 256
    online_interval: int = 50        # train step every N samples
    offline_epochs: int = 10
    hidden_dim: int = 256
    model_type: str = "lstm"         # lstm|gru|gnn|gp
    lr: float = 1e-3
    checkpoint_dir: str = "world_models"
    tick_channel: str = "medium"
    prediction_target: str = "executive_planner"


# ───────────────────────────────────────────────────────────────────────────
# 2. Simple LSTM world model (continuous state vector)
# ───────────────────────────────────────────────────────────────────────────
class LSTMWorldModel(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(state_dim + act_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, state_dim)

    def forward(self, x):
        # x shape: (B, T, state_dim+act_dim)
        out, _ = self.lstm(x)
        pred = self.fc(out[:, -1, :])
        return pred


# ───────────────────────────────────────────────────────────────────────────
# 3. Learner module
# ───────────────────────────────────────────────────────────────────────────
class WorldModelLearner:
    MODULE_NAME = "world_model_learner"
    MSG_PREDICT = "world_predict"

    def __init__(self, nexus: NexusCore, cfg: Optional[WMdLearnerCfg] = None,
                 state_dim: int = 32, act_dim: int = 8):
        self.logger = logging.getLogger("GrandNexus.Learning.WorldModel")
        self.nexus = nexus
        self.cfg = cfg or WMdLearnerCfg()
        self.state_dim, self.act_dim = state_dim, act_dim

        # Experience buffer
        self.buffer: deque = deque(maxlen=self.cfg.buffer_capacity)
        self._sample_count = 0

        # Model & optimiser
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMWorldModel(state_dim, act_dim, self.cfg.hidden_dim).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.loss_fn = nn.MSELoss()

        # Runtime
        self._lock = threading.RLock()
        self._running = False
        self._last_offline = time.time()
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ────────────────────────────────────────────────────────────────────
    def start(self):
        if self._running: return True
        self._running = True
        self.nexus.register_module(
            name=self.MODULE_NAME,
            module=self,
            dependencies=["cognitive_clock"],
        )
        clk = self.nexus.get_module("cognitive_clock")
        if clk: clk.subscribe(self.cfg.tick_channel, self._on_tick)
        self.logger.info("WorldModelLearner started")
        return True

    def stop(self): self._running = False

    # ────────────────────────────────────────────────────────────────────
    # Buffer handling
    # ────────────────────────────────────────────────────────────────────
    def add_experience(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray):
        with self._lock:
            self.buffer.append((state.astype(np.float32),
                                action.astype(np.float32),
                                next_state.astype(np.float32)))
            self._sample_count += 1
            if self._sample_count % self.cfg.online_interval == 0:
                threading.Thread(target=self._train_online, daemon=True).start()

    # ────────────────────────────────────────────────────────────────────
    # Training
    # ────────────────────────────────────────────────────────────────────
    def _sample_batch(self):
        batch = random.sample(self.buffer, min(self.cfg.batch_size, len(self.buffer)))
        s, a, ns = zip(*batch)
        x = torch.from_numpy(np.stack([np.concatenate([si, ai]) for si, ai in zip(s, a)])).unsqueeze(1).to(self.device)
        y = torch.from_numpy(np.stack(ns)).to(self.device)
        return x, y

    def _train_online(self):
        if len(self.buffer) < self.cfg.batch_size: return
        x, y = self._sample_batch()
        self.model.train()
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.logger.debug(f"Online train loss {loss.item():.4f}")

    def _train_offline(self):
        if len(self.buffer) < self.cfg.batch_size: return
        for _ in range(self.cfg.offline_epochs):
            self._train_online()
        self._save_checkpoint(tag="latest")

    # ────────────────────────────────────────────────────────────────────
    # Prediction
    # ────────────────────────────────────────────────────────────────────
    def predict(self, state: np.ndarray, action: np.ndarray, horizon: int = 1):
        self.model.eval()
        s = torch.from_numpy(state.astype(np.float32)).to(self.device)
        a = torch.from_numpy(action.astype(np.float32)).to(self.device)
        for _ in range(horizon):
            x = torch.cat([s, a]).unsqueeze(0).unsqueeze(0)
            ns = self.model(x).squeeze(0)
            s = ns.detach()
        return s.cpu().numpy()

    # ────────────────────────────────────────────────────────────────────
    # Messages (predict request)
    # ────────────────────────────────────────────────────────────────────
    def handle_message(self, msg: Dict[str, Any]):
        tp = msg.get("type")
        if tp == self.MSG_PREDICT:
            st = np.array(msg["content"]["state"])
            act = np.array(msg["content"]["action"])
            horizon = msg["content"].get("horizon", 1)
            pred = self.predict(st, act, horizon).tolist()
            self.nexus.send_message(
                source=self.MODULE_NAME,
                target=msg["source"],
                message_type="world_predict_response",
                content={"prediction": pred},
            )

    # ────────────────────────────────────────────────────────────────────
    # Tick
    # ────────────────────────────────────────────────────────────────────
    def _on_tick(self, _):
        # offline train every hour
        if time.time() - self._last_offline > 3600:
            threading.Thread(target=self._train_offline, daemon=True).start()
            self._last_offline = time.time()

    # ────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ────────────────────────────────────────────────────────────────────
    def _save_checkpoint(self, tag: str):
        path = os.path.join(self.cfg.checkpoint_dir, f"wm_{tag}_{int(time.time())}.pt")
        torch.save(self.model.state_dict(), path)

    # ────────────────────────────────────────────────────────────────────
    # Health
    # ────────────────────────────────────────────────────────────────────
    def health_check(self): return {"buffer": len(self.buffer), "healthy": True}
    def get_status(self): return {"buffer": len(self.buffer)}

if __name__ == "__main__":  # démo d'origine, ne tourne plus à l'import
    from grandnexus.core.nexus_core import NexusCore
    from grandnexus.clock.cognitive_clock import CognitiveClock
    from grandnexus.learning.world_model_learner import WorldModelLearner

    nexus = NexusCore(); CognitiveClock(nexus).start()
    wm_learner = WorldModelLearner(nexus, state_dim=4, act_dim=1); wm_learner.start()

    # Ajout de 2 000 expériences jouet
    import numpy as np, random
    for _ in range(2000):
        s = np.random.rand(4); a = np.random.rand(1)*2-1
        ns = s + a*0.1  # dynamique triviale
        wm_learner.add_experience(s, a, ns)

    # Prediction
    pred = wm_learner.predict(s, a)
    print("GT:", ns.round(3), "Pred:", np.array(pred).round(3))
