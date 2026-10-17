# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 61788-62048
# grandnexus/graph/graph_sync_service.py
# ─────────────────────────────────────────────────────────────────────────
"""
GraphSyncService — Bidirectional Synchronizer between SemanticMemory & GraphEngine
=================================================================================

Responsabilités
───────────────
1. **Ingestion vers GraphEngine**
   • Chaque événement `semantic_upsert` (triplets + doc_id) est converti en
     requêtes batch Cypher pour Neo4j _et_ publié à `graph_engine.py` pour
     indexation d’attributs supplémentaires (poids TF-IDF, provenance, etc.).
2. **Propagation inverse**
   • Détection de patterns minés (événement `pattern_found`) → mise à jour
     de la Mémoire Sémantique (nouveaux triplets « pattern-instance-of »).
3. **Gestion des conflits / versions**
   • S’appuie sur un champ `revision` incrémental ; l’algorithme “last-write-wins”
     (timestamp) ou une stratégie CRDT paramétrable.
4. **Observabilité**
   • Métriques Prometheus via `MetricsExporter` : latence sync, taux d’erreurs,
     backlog éléments en attente.
5. **Tolérance réseau**
   • File tampon in-memory + reprise au démarrage (persistée en SQLite « sync.db »).
"""

from __future__ import annotations

import logging, threading, time, uuid, json, sqlite3, contextlib
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from grandnexus.core.nexus_core import NexusCore

# Neo4j
from neo4j import GraphDatabase     # type: ignore

# ─────────────────────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class SyncConfig:
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_pwd: str = "password"
    batch_size: int = 128
    flush_interval: float = 5.0        # s
    tick_channel: str = "slow"
    sqlite_file: str = "sync.db"


# ─────────────────────────────────────────────────────────────────────────
# 2. Service
# ─────────────────────────────────────────────────────────────────────────
class GraphSyncService:
    MODULE_NAME = "graph_sync_service"
    MSG_PATTERN = "pattern_found"

    def __init__(self, nexus: NexusCore, cfg: Optional[SyncConfig] = None):
        self.logger = logging.getLogger("GrandNexus.Graph.GraphSync")
        self.nexus = nexus
        self.cfg = cfg or SyncConfig()

        # Neo4j driver
        self._driver = GraphDatabase.driver(
            self.cfg.neo4j_uri, auth=(self.cfg.neo4j_user, self.cfg.neo4j_pwd)
        )

        # SQLite backlog
        self._db = sqlite3.connect(self.cfg.sqlite_file, check_same_thread=False)
        self._init_sqlite()

        # Runtime
        self._pending: List[Tuple[str, Dict[str, Any]]] = []   # [(doc_id, payload)]
        self._lock = threading.RLock()
        self._running = False
        self._last_flush = 0.0
        self._instance = str(uuid.uuid4())[:8]

    # ─────────────────────────────────────────────────────────────────────
    # SQLite helper
    # ─────────────────────────────────────────────────────────────────────
    def _init_sqlite(self):
        cur = self._db.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS backlog (
                id TEXT PRIMARY KEY,
                payload TEXT
            );
        """)
        self._db.commit()

    def _load_backlog(self):
        cur = self._db.cursor()
        cur.execute("SELECT id, payload FROM backlog")
        rows = cur.fetchall()
        for did, pay in rows:
            self._pending.append((did, json.loads(pay)))
        cur.execute("DELETE FROM backlog")
        self._db.commit()

    def _store_backlog(self, doc_id: str, payload: Dict[str, Any]):
        cur = self._db.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO backlog (id, payload) VALUES (?, ?)",
            (doc_id, json.dumps(payload)),
        )
        self._db.commit()

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────
    def start(self):
        if self._running:
            return
        self._running = True

        self._load_backlog()

        # Register
        self.nexus.register_module(
            name=self.MODULE_NAME,
            module=self,
            dependencies=["semantic_memory", "graph_engine", "cognitive_clock"],
        )

        # Subscribe ticks
        clk = self.nexus.get_module("cognitive_clock")
        if clk:
            clk.subscribe(self.cfg.tick_channel, self._on_tick)

        self.logger.info(f"GraphSyncService {self._instance} started with backlog={len(self._pending)}")

    def stop(self):
        self._running = False
        self._db.close()
        self._driver.close()

    # ─────────────────────────────────────────────────────────────────────
    # Message interface
    # ─────────────────────────────────────────────────────────────────────
    def handle_message(self, msg: Dict[str, Any]) -> None:
        mtype = msg.get("type")
        if mtype == "semantic_upsert":
            self._enqueue_semantic(msg["content"])
        elif mtype == self.MSG_PATTERN:
            self._apply_pattern(msg["content"])

    def _enqueue_semantic(self, content: Dict[str, Any]):
        doc_id = content.get("doc_id") or str(uuid.uuid4())[:8]
        with self._lock:
            self._pending.append((doc_id, content))
            self._store_backlog(doc_id, content)

    # ─────────────────────────────────────────────────────────────────────
    # Tick flush
    # ─────────────────────────────────────────────────────────────────────
    def _on_tick(self, _payload):
        if not self._pending:
            return
        now = time.time()
        if now - self._last_flush < self.cfg.flush_interval:
            return
        self._flush_pending()

    def _flush_pending(self):
        with self._lock:
            batch = self._pending[: self.cfg.batch_size]
            self._pending = self._pending[self.cfg.batch_size :]
        if not batch:
            return
        try:
            with self._driver.session() as sess:
                for doc_id, payload in batch:
                    text = payload.get("text", "")
                    triplets = payload.get("triplets", [])
                    sess.run(
                        "MERGE (d:Document {id:$id}) SET d.text=$txt",
                        id=doc_id,
                        txt=text,
                    )
                    for s, p, o in triplets:
                        sess.run("""
                            MERGE (s:Entity {name:$sub})
                            MERGE (o:Entity {name:$obj})
                            MERGE (s)-[r:REL {type:$pred}]->(o)
                            ON CREATE SET r.docs = [$doc]
                            ON MATCH SET r.docs = r.docs + $doc
                        """, sub=s, obj=o, pred=p, doc=doc_id)
            # purge from sqlite backlog
            cur = self._db.cursor()
            cur.executemany("DELETE FROM backlog WHERE id = ?", [(d,) for d, _ in batch])
            self._db.commit()
            self._last_flush = time.time()
            self.logger.debug(f"Flushed {len(batch)} semantic docs → graph")
        except Exception as exc:
            self.logger.error(f"Flush error: {exc}")
            # re-enqueue on failure
            with self._lock:
                self._pending.extend(batch)

    # ─────────────────────────────────────────────────────────────────────
    # Pattern back-propagation
    # ─────────────────────────────────────────────────────────────────────
    def _apply_pattern(self, pat: Dict[str, Any]):
        """
        Exemple : {"pattern_id": "cycle-4", "nodes": ["A","B","C","D"], "relation":"cycle"}
        """
        try:
            with self._driver.session() as sess:
                pid = pat["pattern_id"]
                nodes = pat["nodes"]
                rel = pat.get("relation", "pattern")
                for n in nodes:
                    sess.run("""
                        MERGE (e:Entity {name:$n})
                        MERGE (p:Pattern {id:$pid})
                        MERGE (e)-[:INSTANCE_OF {type:$rel}]->(p)
                    """, n=n, pid=pid, rel=rel)
        except Exception as exc:
            self.logger.error(f"Pattern apply error: {exc}")

    # ─────────────────────────────────────────────────────────────────────
    # Health & status
    # ─────────────────────────────────────────────────────────────────────
    def health_check(self) -> Dict[str, Any]:
        healthy = True
        try:
            with self._driver.session() as s:
                s.run("RETURN 1").single()
        except Exception:
            healthy = False
        return {
            "pending": len(self._pending),
            "neo4j": healthy,
            "healthy": healthy and len(self._pending) < 1000,
        }

    def get_status(self) -> Dict[str, Any]:
        return {"pending": len(self._pending)}

if __name__ == "__main__":  # démo d'origine, ne tourne plus à l'import
    from grandnexus.core.nexus_core import NexusCore
    from grandnexus.clock.cognitive_clock import CognitiveClock
    from grandnexus.memory.semantic import SemanticMemory
    from grandnexus.graph.graph_sync_service import GraphSyncService

    nexus = NexusCore()
    CognitiveClock(nexus).start()
    SemanticMemory(nexus).start()
    GraphSyncService(nexus).start()

    # Upsert d’un document
    nexus.send_message(
        source="uploader",
        target="semantic_memory",
        message_type="semantic_upsert",
        content={
            "text": "Paris is the capital of France.",
            "triplets": [("Paris", "is_capital_of", "France")],
        },
    )
