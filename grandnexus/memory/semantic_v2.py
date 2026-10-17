# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 60752-61110
# grandnexus/memory/semantic.py
# ─────────────────────────────────────────────────────────────────────────
"""
SemanticMemory — Hybrid Vector + Graph Knowledge Store for GrandNexus
=====================================================================

Dépendances externes :
    pip install pgvector psycopg2-binary neo4j annoy sentence-transformers
"""

from __future__ import annotations

import logging, threading, time, uuid, os, json, math, contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Sequence, Iterable
import numpy as np

# Embedding
from sentence_transformers import SentenceTransformer                  # type: ignore

# Vector back-end
with contextlib.suppress(ImportError):
    import psycopg2, psycopg2.extras                                    # type: ignore
    from pgvector.psycopg2 import register_vector                      # type: ignore
with contextlib.suppress(ImportError):
    from annoy import AnnoyIndex                                        # type: ignore

# Graph back-end
with contextlib.suppress(ImportError):
    from neo4j import GraphDatabase                                     # type: ignore

from grandnexus.core.nexus_core import NexusCore

# ─────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class Neo4jCfg:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"


@dataclass
class PGVectorCfg:
    dsn: str = "postgresql://user:pass@localhost:5432/gnx"
    table: str = "semantic_vectors"
    dim: int = 768
    metric: str = "cosine"                     # pgvector: cosine|l2|inner


@dataclass
class AnnoyCfg:
    dim: int = 768
    n_trees: int = 10
    metric: str = "angular"


@dataclass
class SemMemConfig:
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    pg: Optional[PGVectorCfg] = PGVectorCfg()
    neo4j: Optional[Neo4jCfg] = Neo4jCfg()
    annoy: Optional[AnnoyCfg] = AnnoyCfg()
    consolidate_every: float = 30.0                # s
    tick_channel: str = "slow"
    safety_guard: Optional[str] = "safety_guard"


# ─────────────────────────────────────────────────────────────────────────
# 2. SemanticMemory
# ─────────────────────────────────────────────────────────────────────────
class SemanticMemory:
    MODULE_NAME = "semantic_memory"
    MSG_QUERY = "semantic_query"
    MSG_UPSERT = "semantic_upsert"

    def __init__(self, nexus_core: NexusCore, cfg: Optional[SemMemConfig] = None):
        self.logger = logging.getLogger("GrandNexus.Memory.SemanticMemory")
        self.nexus = nexus_core
        self.cfg = cfg or SemMemConfig()

        # Embedding model (thread-safe)
        self._model = SentenceTransformer(self.cfg.embed_model)

        # Vector store
        self._pg_conn: Optional["psycopg2.extensions.connection"] = None
        self._ann: Optional["AnnoyIndex"] = None
        self._ann_map: Dict[int, str] = {}

        # Graph store
        self._neo4j = None

        # Runtime
        self._lock = threading.RLock()
        self._running = False
        self._last_consolidate = 0.0
        self._pending_ann: List[Tuple[str, np.ndarray]] = []  # chunk_id, emb
        self._instance_id = str(uuid.uuid4())[:8]

        # Stats
        self._q_latency: List[float] = []
        self._upserts = 0

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────
    def start(self) -> bool:
        with self._lock:
            if self._running:
                return True
            self._running = True

        self._init_pgvector()
        self._init_neo4j()
        self._init_annoy()

        # Enregistrement NexusCore
        self.nexus.register_module(
            name=self.MODULE_NAME,
            module=self,
            dependencies=["cognitive_clock"],
        )

        # Tick subscription pour consolidation ANN ↔ PG
        clk = self.nexus.get_module("cognitive_clock")
        if clk:
            clk.subscribe(self.cfg.tick_channel, self._on_tick)

        self.logger.info(f"SemanticMemory {self._instance_id} started")
        return True

    def stop(self) -> bool:
        self._running = False
        if self._pg_conn:
            self._pg_conn.close()
        if self._neo4j:
            self._neo4j.close()
        return True

    # ─────────────────────────────────────────────────────────────────────
    # Backend initialisation
    # ─────────────────────────────────────────────────────────────────────
    def _init_pgvector(self):
        if not self.cfg.pg:
            return
        try:
            self._pg_conn = psycopg2.connect(self.cfg.pg.dsn)
            register_vector(self._pg_conn)
            cur = self._pg_conn.cursor()
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.cfg.pg.table} (
                    id TEXT PRIMARY KEY,
                    embedding VECTOR({self.cfg.pg.dim}),
                    metadata JSONB
                );
                """
            )
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_vec ON {self.cfg.pg.table} USING ivfflat (embedding vector_{self.cfg.pg.metric});")
            self._pg_conn.commit()
        except Exception as exc:
            self.logger.error(f"PGVector init failed: {exc}")
            self._pg_conn = None

    def _init_annoy(self):
        if not AnnoyCfg:
            return
        self._ann = AnnoyIndex(self.cfg.annoy.dim, self.cfg.annoy.metric)
        # will rebuild periodically

    def _init_neo4j(self):
        if not self.cfg.neo4j:
            return
        try:
            self._neo4j = GraphDatabase.driver(
                self.cfg.neo4j.uri, auth=(self.cfg.neo4j.user, self.cfg.neo4j.password)
            )
            with self._neo4j.session() as sess:
                sess.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Knowledge) REQUIRE n.id IS UNIQUE")
        except Exception as exc:
            self.logger.error(f"Neo4j init failed: {exc}")
            self._neo4j = None

    # ─────────────────────────────────────────────────────────────────────
    # Public API – upsert & query
    # ─────────────────────────────────────────────────────────────────────
    def upsert_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        triplets: Optional[List[Tuple[str, str, str]]] = None,
    ) -> str:
        doc_id = str(uuid.uuid4())[:8]
        emb = self._model.encode(text)
        meta = metadata or {}

        # Safety guard pre-check
        if self.cfg.safety_guard:
            sg = self.nexus.get_module(self.cfg.safety_guard)
            if sg and hasattr(sg, "is_allowed") and not sg.is_allowed(text, meta):
                self.logger.warning("SafetyGuard blocked upsert")
                return ""

        # PGVector
        if self._pg_conn:
            try:
                cur = self._pg_conn.cursor()
                cur.execute(
                    f"INSERT INTO {self.cfg.pg.table} (id, embedding, metadata) VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING",
                    (doc_id, emb.tolist(), json.dumps(meta)),
                )
                self._pg_conn.commit()
            except Exception as exc:
                self.logger.error(f"PG upsert error: {exc}")

        # Annoy (lazy)
        if self._ann is not None:
            self._pending_ann.append((doc_id, emb))

        # Neo4j triplets
        if triplets and self._neo4j:
            with self._neo4j.session() as s:
                for (sub, pred, obj) in triplets:
                    s.run(
                        """
                        MERGE (s:Entity {name:$sub})
                        MERGE (o:Entity {name:$obj})
                        MERGE (s)-[:REL {type:$pred, doc_id:$doc}]->(o)
                        """,
                        sub=sub, obj=obj, pred=pred, doc=doc_id,
                    )
                s.run("MERGE (d:Knowledge {id:$id}) SET d.text=$txt", id=doc_id, txt=text)

        with self._lock:
            self._upserts += 1
        return doc_id

    def semantic_query(
        self,
        text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        start = time.perf_counter()
        emb = self._model.encode(text)

        results: List[Tuple[float, str, Dict[str, Any]]] = []

        # PGVector search
        if self._pg_conn:
            cur = self._pg_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cur.execute(
                f"""
                SELECT id, metadata, 1 - (embedding <=> %s) AS score
                FROM {self.cfg.pg.table}
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (emb.tolist(), emb.tolist(), top_k * 2),
            )
            for row in cur.fetchall():
                results.append((row["score"], row["id"], dict(row["metadata"])))

        # Annoy fallback
        elif self._ann and self._ann.get_n_items():
            idxs, dists = self._ann.get_nns_by_vector(emb, top_k * 2, include_distances=True)
            for dist, idx in zip(dists, idxs):
                cid = self._ann_map.get(idx)
                results.append((1 - dist, cid, {}))

        # Sort & truncate
        results.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, cid, meta in results[: top_k]:
            trip = self._fetch_triplets(cid)
            out.append(
                {"doc_id": cid, "score": round(float(score), 3), "metadata": meta, "triplets": trip}
            )

        self._q_latency.append(time.perf_counter() - start)
        return out

    # ─────────────────────────────────────────────────────────────────────
    # Triplet fetch helper
    # ─────────────────────────────────────────────────────────────────────
    def _fetch_triplets(self, doc_id: str) -> List[Tuple[str, str, str]]:
        if not self._neo4j:
            return []
        with self._neo4j.session() as s:
            res = s.run(
                """
                MATCH (s:Entity)-[r:REL {doc_id:$doc}]->(o:Entity)
                RETURN s.name AS sub, r.type AS pred, o.name AS obj
                """,
                doc=doc_id,
            )
            return [(r["sub"], r["pred"], r["obj"]) for r in res]

    # ─────────────────────────────────────────────────────────────────────
    # Message interface
    # ─────────────────────────────────────────────────────────────────────
    def handle_message(self, msg: Dict[str, Any]) -> None:
        mtype = msg.get("type")
        if mtype == self.MSG_UPSERT:
            self.upsert_document(**msg["content"])
        elif mtype == self.MSG_QUERY:
            res = self.semantic_query(**msg["content"])
            # Répondre à la source si demandé
            if "reply_to" in msg:
                self.nexus.send_message(
                    source=self.MODULE_NAME,
                    target=msg["reply_to"],
                    message_type="semantic_response",
                    content=res,
                )

    # ─────────────────────────────────────────────────────────────────────
    # Tick consolidation (annoy build)
    # ─────────────────────────────────────────────────────────────────────
    def _on_tick(self, _):
        now = time.time()
        if self._ann is None or not self._pending_ann:
            return
        if now - self._last_consolidate < self.cfg.consolidate_every:
            return

        with self._lock:
            pending = list(self._pending_ann)
            self._pending_ann.clear()

        for cid, emb in pending:
            idx = self._ann.get_n_items()
            self._ann.add_item(idx, emb)
            self._ann_map[idx] = cid
        self._ann.build(self.cfg.annoy.n_trees)
        self._last_consolidate = now
        self.logger.info(f"Annoy index rebuilt with {self._ann.get_n_items()} items")

    # ─────────────────────────────────────────────────────────────────────
    # Health / status
    # ─────────────────────────────────────────────────────────────────────
    def health_check(self) -> Dict[str, Any]:
        lat = sum(self._q_latency) / len(self._q_latency) if self._q_latency else 0.0
        healthy = (self._pg_conn or self._ann) and (self._neo4j is not None)
        return {
            "latency_avg_ms": round(lat * 1_000, 2),
            "upserts": self._upserts,
            "vector_backend": "pgvector" if self._pg_conn else ("annoy" if self._ann else "none"),
            "graph_backend": bool(self._neo4j),
            "healthy": healthy,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "items_vector": self._ann.get_n_items() if self._ann else "N/A",
            "upserts": self._upserts,
        }
