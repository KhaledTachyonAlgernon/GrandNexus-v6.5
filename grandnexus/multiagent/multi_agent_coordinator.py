# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 62872-63083
# grandnexus/multiagent/multi_agent_coordinator.py
# ──────────────────────────────────────────────────────────────────────────
"""
MultiAgentCoordinator — Lightweight P2P / Hub-and-Spoke synchronizer for GrandNexus
===================================================================================

Fonctions
─────────
• **Service Discovery**     : registre Redis « GNX:PEERS » (hash TTL) → heart-beat.
• **Pub/Sub Bus**           : canal† « GNX:BROADCAST » pour drives, alertes, etc.
• **Task Queue Stream**     : Redis Stream « GNX:TASKS » (XADD / XREADGROUP) → load-balancing FIFO.
• **Distributed Lock**      : Redlock style (« GNX:LOCK:{key} »).
• **Knowledge Sync**        : push delta triplets vers `GraphSyncService` pair.
• **Conflict Resolver**     : strat. “first-claim wins” + horloge logique Lamport.
"""

from __future__ import annotations

import logging, time, json, threading, uuid, os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

try:
    import redis               # pip install redis
except ImportError:
    raise RuntimeError("MultiAgentCoordinator requires 'redis' package")

from grandnexus.core.nexus_core import NexusCore

# ──────────────────────────────────────────────────────────────────────────
# 1. Config
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class MACConfig:
    redis_url: str = "redis://localhost:6379/0"
    peer_ttl: int = 30                        # s — disparition après TTL
    heartbeat_period: float = 5.0             # s
    stream_maxlen: int = 10_000
    group: str = "gnx_group"
    consumer_prefix: str = "node"
    tick_channel: str = "slow"
    broadcast_target: str = "homeostasis_manager"   # ex. pour drives

# ──────────────────────────────────────────────────────────────────────────
# 2. MultiAgentCoordinator
# ──────────────────────────────────────────────────────────────────────────
class MultiAgentCoordinator:
    MODULE_NAME = "multi_agent_coordinator"
    STREAM_TASKS = "GNX:TASKS"
    HASH_PEERS = "GNX:PEERS"
    CHANNEL_BROADCAST = "GNX:BROADCAST"

    def __init__(self, nexus: NexusCore, cfg: Optional[MACConfig] = None):
        self.logger = logging.getLogger("GrandNexus.Multi.MAC")
        self.nexus = nexus
        self.cfg = cfg or MACConfig()

        # Redis client
        self.redis = redis.Redis.from_url(self.cfg.redis_url, decode_responses=True)
        # Consumer identity
        self.node_id = f"{self.cfg.consumer_prefix}-{uuid.uuid4().hex[:6]}"
        self.start_ms = int(time.time() * 1000)

        # Create consumer-group idempotently
        with self.redis.pipeline() as pipe:
            try:
                pipe.xgroup_create(self.STREAM_TASKS, self.cfg.group, id="0-0", mkstream=True)
            except redis.ResponseError as e:
                if "BUSYGROUP" not in str(e): raise
            pipe.execute()

        # Threads
        self._running = False
        self._thread_hb = None
        self._thread_tasks = None
        self._thread_sub = None

    # ────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ────────────────────────────────────────────────────────────────────
    def start(self):
        if self._running: return
        self._running = True

        self.nexus.register_module(name=self.MODULE_NAME,
                                   module=self,
                                   dependencies=["cognitive_clock"])
        # Heart-beat, tasks poller, broadcast sub
        self._thread_hb = threading.Thread(target=self._heartbeat_loop, daemon=True); self._thread_hb.start()
        self._thread_tasks = threading.Thread(target=self._task_loop, daemon=True);    self._thread_tasks.start()
        self._thread_sub = threading.Thread(target=self._broadcast_loop, daemon=True); self._thread_sub.start()

        self.logger.info(f"MAC node {self.node_id} connected to Redis @ {self.cfg.redis_url}")

    def stop(self): self._running = False

    # ────────────────────────────────────────────────────────────────────
    # Heart-beat & peer registry
    # ────────────────────────────────────────────────────────────────────
    def _heartbeat_loop(self):
        ttl = self.cfg.peer_ttl
        while self._running:
            self.redis.hset(self.HASH_PEERS, self.node_id, int(time.time()))
            self.redis.expire(self.HASH_PEERS, ttl*2)
            time.sleep(self.cfg.heartbeat_period)

    def list_peers(self) -> List[str]:
        peers = self.redis.hgetall(self.HASH_PEERS)
        now = time.time()
        return [p for p, ts in peers.items() if now - int(ts) < self.cfg.peer_ttl]

    # ────────────────────────────────────────────────────────────────────
    # Distributed lock
    # ────────────────────────────────────────────────────────────────────
    def acquire_lock(self, key: str, ttl: int = 10) -> bool:
        return self.redis.set(f"GNX:LOCK:{key}", self.node_id, nx=True, ex=ttl)

    def release_lock(self, key: str):
        if self.redis.get(f"GNX:LOCK:{key}") == self.node_id:
            self.redis.delete(f"GNX:LOCK:{key}")

    # ────────────────────────────────────────────────────────────────────
    # Task queue (Redis Streams)
    # ────────────────────────────────────────────────────────────────────
    def enqueue_task(self, task: Dict[str, Any]) -> str:
        tid = self.redis.xadd(self.STREAM_TASKS, {"json": json.dumps(task)},
                              maxlen=self.cfg.stream_maxlen)
        return tid

    def _task_loop(self):
        while self._running:
            try:
                resp = self.redis.xreadgroup(self.cfg.group, self.node_id,
                                             {self.STREAM_TASKS: ">"}, count=10, block=1000)
                if not resp: continue
                for _stream, msgs in resp:
                    for msg_id, fields in msgs:
                        task = json.loads(fields["json"])
                        self._process_task(task)
                        # ACK
                        self.redis.xack(self.STREAM_TASKS, self.cfg.group, msg_id)
            except Exception as e:
                self.logger.error(f"Task loop error: {e}")
                time.sleep(1)

    def _process_task(self, task: Dict[str, Any]):
        """
        Exemple task:
        {"type":"broadcast_drive","drive":{"RATE_LIMIT":0.9}}
        """
        typ = task.get("type")
        if typ == "broadcast_drive":
            self.nexus.send_message(
                source=self.MODULE_NAME,
                target=self.cfg.broadcast_target,
                message_type="homeostasis_drive",
                content=task["drive"],
                priority=3,
            )
        elif typ == "triplet_sync":
            self.nexus.send_message(
                source=self.MODULE_NAME,
                target="graph_sync_service",
                message_type="semantic_upsert",
                content=task["triplet_doc"],
                priority=6,
            )
        # … autres types

    # ────────────────────────────────────────────────────────────────────
    # Broadcast channel
    # ────────────────────────────────────────────────────────────────────
    def _broadcast_loop(self):
        pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(self.CHANNEL_BROADCAST)
        for msg in pubsub.listen():
            if not self._running: break
            try:
                payload = json.loads(msg["data"])
                self._process_broadcast(payload)
            except Exception as exc:
                self.logger.error(f"Broadcast error {exc}")

    def broadcast(self, payload: Dict[str, Any]):
        self.redis.publish(self.CHANNEL_BROADCAST, json.dumps(payload))

    def _process_broadcast(self, payload: Dict[str, Any]):
        """
        Exemple broadcast: {"topic":"alert","data":"Node X high TEMP"}
        """
        topic = payload.get("topic")
        if topic == "alert":
            self.logger.warning(f"[Cluster Alert] {payload['data']}")
        elif topic == "drive":
            self.nexus.send_message(
                source=self.MODULE_NAME,
                target=self.cfg.broadcast_target,
                message_type="homeostasis_drive",
                content=payload["data"],
                priority=4,
            )
        # …

    # ────────────────────────────────────────────────────────────────────
    # Health & status
    # ────────────────────────────────────────────────────────────────────
    def health_check(self):
        return {"peers": len(self.list_peers()), "healthy": True}

    def get_status(self):
        return {"peers": self.list_peers()}
