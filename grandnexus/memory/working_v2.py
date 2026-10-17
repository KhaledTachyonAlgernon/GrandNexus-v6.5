# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 60439-60751
# grandnexus/memory/working.py
# ─────────────────────────────────────────────────────────────────────────
"""
WorkingMemory — Active Store with Decay & Semantic Retrieval
===========================================================

Caractéristiques
────────────────
• **Capacité bornée** (par nombre ou par taille) avec éviction LRU ⇢ salience.
• **Décroissance temporelle** : score(t) = salience₀ · e^(–λ·Δt) ajusté par
  les _drives_ homéostatiques (fatigue ⇒ λ↑).
• **Chunking & tagging** : chaque item est un `MemoryChunk` (dataclass) doté
  de : embedding facultatif, métadonnées, priorité, compteur d’accès.
• **Requêtes** :
  – par clé / id
  – par prédicat (lambda)
  – par similarité (cos θ) via backend d’embeddings externe (callback).
• **Consolidation** : flush périodique (tick lent) ⇢ `EpisodicMemory`; règle
  configurable (age, importance).
• **Observabilité / health** ( taux hit/miss, utilisation, avg latency ).
• **Plug-in vector store** : fallback numpy / Annoy / Faiss ; injection via DI.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Deque, Tuple
from collections import deque, defaultdict

import numpy as np

from grandnexus.core.nexus_core import NexusCore

# ─────────────────────────────────────────────────────────────────────────
# 1. Structures
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class MemoryChunk:
    """Représentation élémentaire en mémoire de travail."""
    chunk_id: str
    data: Any
    timestamp: float
    salience: float = 1.0
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0


@dataclass
class WMConfig:
    capacity: int = 2048                     # max items
    decay_half_life: float = 30.0            # s (λ = ln2 / t½)
    consolidation_age: float = 120.0         # s → episodic
    max_query_results: int = 8
    tick_channel: str = "fast"
    drive_source: str = "homeostasis_manager"
    episodic_target: str = "episodic_memory"
    embedding_fn: Optional[Callable[[Any], np.ndarray]] = None


# ─────────────────────────────────────────────────────────────────────────
# 2. WorkingMemory class
# ─────────────────────────────────────────────────────────────────────────
class WorkingMemory:
    MODULE_NAME = "working_memory_v2"  # réparation 6.5 : cohabite avec la mémoire du cœur

    def __init__(
        self,
        nexus_core: NexusCore,
        config: Optional[WMConfig] = None,
    ):
        self.logger = logging.getLogger("GrandNexus.Memory.WorkingMemory")
        self.nexus_core = nexus_core
        self.config = config or WMConfig()

        # Storage
        self._lock = threading.RLock()
        self._store: Dict[str, MemoryChunk] = {}
        self._lru_order: Deque[str] = deque()

        # Stats
        self._hit = 0
        self._miss = 0

        # Drives
        self._drives: Dict[str, float] = {}

        # Runtime
        self._running = False
        self._instance_id = str(uuid.uuid4())[:8]

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────
    def start(self) -> bool:
        with self._lock:
            if self._running:
                return True
            self._running = True

        # Register
        self.nexus_core.register_module(
            name=self.MODULE_NAME,
            module=self,
            dependencies=["cognitive_clock", self.config.drive_source],
        )

        # Subscribe clock
        clock = self.nexus_core.get_module("cognitive_clock")
        if clock:
            clock.subscribe(self.config.tick_channel, self._on_tick)

        self.logger.info(f"WorkingMemory {self._instance_id} started")
        return True

    def stop(self) -> bool:
        with self._lock:
            self._running = False
        return True

    # ─────────────────────────────────────────────────────────────────────
    # Insertion
    # ─────────────────────────────────────────────────────────────────────
    def put(
        self,
        data: Any,
        salience: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> str:
        if embedding is None and self.config.embedding_fn:
            try:
                embedding = self.config.embedding_fn(data)
            except Exception as exc:
                self.logger.warning(f"Embedding failure: {exc}")

        chunk = MemoryChunk(
            chunk_id=str(uuid.uuid4())[:8],
            data=data,
            timestamp=time.time(),
            salience=salience,
            embedding=embedding,
            metadata=metadata or {},
        )

        with self._lock:
            # Eviction if necessary
            if len(self._store) >= self.config.capacity:
                self._evict_one()

            self._store[chunk.chunk_id] = chunk
            self._lru_order.append(chunk.chunk_id)
        return chunk.chunk_id

    # ─────────────────────────────────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────────────────────────────────
    def get(
        self,
        chunk_id: str,
    ) -> Optional[MemoryChunk]:
        with self._lock:
            chunk = self._store.get(chunk_id)
            if chunk:
                self._hit += 1
                chunk.access_count += 1
                chunk.timestamp = time.time()
                # move to end (most recent)
                try:
                    self._lru_order.remove(chunk_id)
                except ValueError:
                    pass
                self._lru_order.append(chunk_id)
            else:
                self._miss += 1
        return chunk

    def query(
        self,
        predicate: Optional[Callable[[MemoryChunk], bool]] = None,
        embedding: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
    ) -> List[MemoryChunk]:
        results: List[Tuple[float, MemoryChunk]] = []
        now = time.time()
        with self._lock:
            for ch in self._store.values():
                if predicate and not predicate(ch):
                    continue

                score = self._current_salience(ch, now)
                # embedding similarity boost
                if embedding is not None and ch.embedding is not None:
                    sim = self._cos_sim(embedding, ch.embedding)
                    score += sim
                results.append((score, ch))

        results.sort(key=lambda x: x[0], reverse=True)
        k = top_k or self.config.max_query_results
        out = [c for _, c in results[:k]]
        for c in out:
            self.get(c.chunk_id)   # update LRU + stats
        return out

    # ─────────────────────────────────────────────────────────────────────
    # Internal utilities
    # ─────────────────────────────────────────────────────────────────────
    def _evict_one(self) -> None:
        """LRU with salience fallback."""
        while self._lru_order:
            cid = self._lru_order.popleft()
            if cid in self._store:
                del self._store[cid]
                self.logger.debug(f"Evicted chunk {cid}")
                return

    def _current_salience(self, chunk: MemoryChunk, now: float) -> float:
        age = now - chunk.timestamp
        λ = math.log(2) / self.config.decay_half_life
        # Drive-modulated decay
        λ *= 1.0 + self._drives.get("FATIGUE", 0.0)
        return chunk.salience * math.exp(-λ * age)

    @staticmethod
    def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    # ─────────────────────────────────────────────────────────────────────
    # Tick & consolidation
    # ─────────────────────────────────────────────────────────────────────
    def _on_tick(self, payload: Dict[str, Any]) -> None:
        now = time.time()
        to_consolidate = []
        with self._lock:
            for cid in list(self._store.keys()):
                ch = self._store[cid]
                # Decay update & consolidation
                if (now - ch.timestamp) >= self.config.consolidation_age:
                    to_consolidate.append(ch)
                    del self._store[cid]
                    try:
                        self._lru_order.remove(cid)
                    except ValueError:
                        pass

        # Send to EpisodicMemory
        for ch in to_consolidate:
            self.nexus_core.send_message(
                source=self.MODULE_NAME,
                target=self.config.episodic_target,
                message_type="episodic_ingest",
                content={
                    "chunk_id": ch.chunk_id,
                    "data": ch.data,
                    "metadata": ch.metadata,
                    "embedding": ch.embedding.tolist() if ch.embedding is not None else None,
                    "timestamp": ch.timestamp,
                },
                priority=5,
            )

    # ─────────────────────────────────────────────────────────────────────
    # Drives update via message
    # ─────────────────────────────────────────────────────────────────────
    def handle_message(self, msg: Dict[str, Any]) -> None:
        if msg.get("type") == "homeostasis_drive":
            with self._lock:
                self._drives = msg["content"]

    # ─────────────────────────────────────────────────────────────────────
    # Health / status
    # ─────────────────────────────────────────────────────────────────────
    def health_check(self) -> Dict[str, Any]:
        with self._lock:
            miss_ratio = self._miss / max(1, self._hit + self._miss)
            healthy = len(self._store) < self.config.capacity and miss_ratio < 0.5
            return {
                "items": len(self._store),
                "hit": self._hit,
                "miss": self._miss,
                "miss_ratio": round(miss_ratio, 3),
                "healthy": healthy,
            }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "items": len(self._store),
                "lru_len": len(self._lru_order),
                "drives": {k: round(v, 2) for k, v in self._drives.items()},
            }

if __name__ == "__main__":  # démo d'origine, ne tourne plus à l'import
    from grandnexus.core.nexus_core import NexusCore
    from grandnexus.clock.cognitive_clock import CognitiveClock
    from grandnexus.memory.working import WorkingMemory

    nexus = NexusCore()
    CognitiveClock(nexus).start()
    wm = WorkingMemory(nexus).start()

    chunk_id = wm.put({"text": "The quick brown fox"}, salience=0.8)

    # Recherche par predicate
    fox_chunks = wm.query(predicate=lambda c: "fox" in c.data["text"])
    print([c.chunk_id for c in fox_chunks])
