# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 63690-63804
# grandnexus/ops/decision_history_archive.py
# ──────────────────────────────────────────────────────────────────────────
"""
DecisionHistoryArchive — Long-Term Decision Archival for GrandNexus
===================================================================

Dependencies :
    pip install pymongo fastapi uvicorn
"""

import logging, json, time, uuid, threading
from typing import Any, Dict
from dataclasses import dataclass
from pymongo import MongoClient, ASCENDING
from fastapi import FastAPI
from grandnexus.core.nexus_core import NexusCore
import uvicorn

# ──────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class HistoryConfig:
    mongodb_uri: str = "mongodb://localhost:27017/"
    database_name: str = "grandnexus_decisions"
    collection_name: str = "decision_history"
    retention_days: int = 730  # 2 ans
    api_port: int = 9000

# ──────────────────────────────────────────────────────────────────────────
# 2. DecisionHistoryArchive
# ──────────────────────────────────────────────────────────────────────────
class DecisionHistoryArchive:
    MODULE_NAME = "decision_history_archive"

    def __init__(self, nexus: NexusCore, cfg: HistoryConfig = HistoryConfig()):
        self.logger = logging.getLogger("GrandNexus.DecisionHistoryArchive")
        self.nexus = nexus
        self.cfg = cfg
        self.client = MongoClient(self.cfg.mongodb_uri)
        self.db = self.client[self.cfg.database_name]
        self.collection = self.db[self.cfg.collection_name]
        self._running = False
        self.app = FastAPI()

        # Setup indexes
        self.collection.create_index([("timestamp", ASCENDING)])
        self._configure_routes()

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle methods
    # ──────────────────────────────────────────────────────────────────────
    def start(self):
        if self._running:
            return
        self._running = True
        self.nexus.register_module(
            name=self.MODULE_NAME,
            module=self,
            dependencies=["executive_planner", "safety_guard", "semantic_memory", "metrics_exporter"],
        )
        threading.Thread(target=self._run_api, daemon=True).start()
        self.logger.info("DecisionHistoryArchive started")

    def stop(self):
        self._running = False
        self.client.close()

    # ──────────────────────────────────────────────────────────────────────
    # Capture & archive
    # ──────────────────────────────────────────────────────────────────────
    def archive_decision(self, decision: Dict[str, Any], context: Dict[str, Any]):
        record = {
            "_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "decision": decision,
            "context": context
        }
        self.collection.insert_one(record)
        self.logger.debug(f"Decision archived: {record['_id']}")

    # ──────────────────────────────────────────────────────────────────────
    # API routes configuration
    # ──────────────────────────────────────────────────────────────────────
    def _configure_routes(self):
        @self.app.get("/history/decisions/")
        def get_decisions(skip: int = 0, limit: int = 10):
            cursor = self.collection.find().sort("timestamp", -1).skip(skip).limit(limit)
            return list(cursor)

        @self.app.get("/history/decision/{decision_id}")
        def get_decision(decision_id: str):
            result = self.collection.find_one({"_id": decision_id})
            return result or {"error": "Decision not found"}

    def _run_api(self):
        uvicorn.run(self.app, host="0.0.0.0", port=self.cfg.api_port, log_level="warning")

    # ──────────────────────────────────────────────────────────────────────
    # Message handling
    # ──────────────────────────────────────────────────────────────────────
    def handle_message(self, msg: Dict[str, Any]):
        if msg["type"] == "archive_decision":
            self.archive_decision(msg["content"]["decision"], msg["content"]["context"])

    # ──────────────────────────────────────────────────────────────────────
    # Maintenance & Health
    # ──────────────────────────────────────────────────────────────────────
    def health_check(self):
        count = self.collection.count_documents({})
        return {"decision_count": count, "healthy": True}

    def get_status(self):
        return {"decisions_archived": self.collection.count_documents({})}
