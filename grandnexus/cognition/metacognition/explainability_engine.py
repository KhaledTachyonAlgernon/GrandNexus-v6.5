# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 63400-63540
# grandnexus/metacognition/explainability_engine.py
# ──────────────────────────────────────────────────────────────────────────
"""
ExplainabilityEngine — Real-Time Explainability Layer for GrandNexus
====================================================================

Dependencies :
    pip install openai tiktoken
"""

import os  # réparation 6.5 : fourni avant par l'espace de noms partagé du notebook
import logging, json, time, uuid, threading
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from grandnexus.core.nexus_core import NexusCore
from grandnexus.dialogue.dialogue_interface import DialogueInterface

try:
    import openai
except ImportError:  # réparation 6.5
    openai = None
# Assurez-vous d'avoir accès à OpenAI API ou équivalent local

# ──────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class ExplainabilityConfig:
    explanation_backend: str = "openai"
    openai_model: str = "gpt-4-turbo"
    api_key: Optional[str] = None
    max_tokens: int = 300
    explanation_log_file: str = "explanations.jsonl"
    tick_channel: str = "slow"

# ──────────────────────────────────────────────────────────────────────────
# 2. Explainability Engine
# ──────────────────────────────────────────────────────────────────────────
class ExplainabilityEngine:
    MODULE_NAME = "explainability_engine"
    MSG_REQUEST = "request_explanation"

    def __init__(self, nexus: NexusCore, cfg: Optional[ExplainabilityConfig] = None):
        self.logger = logging.getLogger("GrandNexus.ExplainabilityEngine")
        self.nexus = nexus
        self.cfg = cfg or ExplainabilityConfig()
        self._running = False
        self._lock = threading.RLock()
        self._explanation_log = open(self.cfg.explanation_log_file, "a", encoding="utf-8")

        if self.cfg.explanation_backend == "openai":
            openai.api_key = self.cfg.api_key or os.getenv("OPENAI_API_KEY")

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────
    def start(self):
        if self._running:
            return
        self._running = True
        self.nexus.register_module(
            name=self.MODULE_NAME,
            module=self,
            dependencies=["executive_planner", "semantic_memory", "dialogue_interface", "metrics_exporter"],
        )
        self.logger.info("ExplainabilityEngine started")

    def stop(self):
        self._running = False
        self._explanation_log.close()

    # ──────────────────────────────────────────────────────────────────────
    # Explanation generation
    # ──────────────────────────────────────────────────────────────────────
    def generate_explanation(self, context: Dict[str, Any], decision: Dict[str, Any]) -> str:
        prompt = self._build_prompt(context, decision)
        try:
            response = openai.ChatCompletion.create(
                model=self.cfg.openai_model,
                messages=[{"role": "system", "content": "You provide clear explanations."},
                          {"role": "user", "content": prompt}],
                max_tokens=self.cfg.max_tokens,
                temperature=0.3,
            )
            explanation = response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            explanation = "Explanation currently unavailable due to internal error."

        self._log_explanation(context, decision, explanation)
        return explanation

    def _build_prompt(self, context: Dict[str, Any], decision: Dict[str, Any]) -> str:
        return (f"Explain clearly why the following decision was made given the context.\n\n"
                f"Context:\n{json.dumps(context, indent=2)}\n\n"
                f"Decision:\n{json.dumps(decision, indent=2)}\n\n"
                "Provide a clear, concise explanation understandable by a non-technical person.")

    def _log_explanation(self, context, decision, explanation):
        record = {
            "timestamp": time.time(),
            "context": context,
            "decision": decision,
            "explanation": explanation,
            "id": str(uuid.uuid4())
        }
        with self._lock:
            self._explanation_log.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._explanation_log.flush()

    # ──────────────────────────────────────────────────────────────────────
    # Message handling interface
    # ──────────────────────────────────────────────────────────────────────
    def handle_message(self, msg: Dict[str, Any]):
        if msg.get("type") == self.MSG_REQUEST:
            context = msg["content"]["context"]
            decision = msg["content"]["decision"]
            explanation = self.generate_explanation(context, decision)

            self.nexus.send_message(
                source=self.MODULE_NAME,
                target=msg["source"],
                message_type="explanation_response",
                content={"explanation": explanation}
            )

    # ──────────────────────────────────────────────────────────────────────
    # Health & Metrics
    # ──────────────────────────────────────────────────────────────────────
    def health_check(self):
        return {"explanation_log_size": os.path.getsize(self.cfg.explanation_log_file), "healthy": True}

    def get_status(self):
        return {"running": self._running}

# Demande d'explication depuis un autre module :
if __name__ == "__main__":  # démo d'origine, ne tourne plus à l'import
    nexus.send_message(
        source="executive_planner",
        target="explainability_engine",
        message_type="request_explanation",
        content={
            "context": {"battery": 15, "temperature": 78, "mission": "survey"},
            "decision": {"action": "return_to_base", "reason": "low_battery"}
        }
    )
