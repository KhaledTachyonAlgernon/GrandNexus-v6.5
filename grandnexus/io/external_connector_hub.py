# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 63921-64046
# grandnexus/io/external_connector_hub.py
# ──────────────────────────────────────────────────────────────────────────
"""
ExternalConnectorHub — Universal External Integration Gateway for GrandNexus
===========================================================================

Dependencies:
    pip install fastapi uvicorn paho-mqtt requests confluent-kafka pymodbus sqlalchemy prometheus_client
"""

import logging, threading, json, time, uuid
from dataclasses import dataclass
from typing import Dict, Any
from fastapi import FastAPI
import uvicorn, requests, paho.mqtt.client as mqtt
from prometheus_client import Counter, start_http_server
from grandnexus.core.nexus_core import NexusCore

# ──────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class ConnectorConfig:
    api_port: int = 9020
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    prometheus_port: int = 8000

# ──────────────────────────────────────────────────────────────────────────
# 2. ExternalConnectorHub
# ──────────────────────────────────────────────────────────────────────────
class ExternalConnectorHub:
    MODULE_NAME = "external_connector_hub"

    def __init__(self, nexus: NexusCore, cfg: ConnectorConfig = ConnectorConfig()):
        self.logger = logging.getLogger("GrandNexus.ExternalConnectorHub")
        self.nexus = nexus
        self.cfg = cfg
        self._running = False
        self.app = FastAPI()

        # Prometheus metrics
        self.api_requests = Counter('api_requests_total', 'Total API Requests')
        self.mqtt_messages = Counter('mqtt_messages_total', 'Total MQTT Messages Received')

        self._configure_routes()
        self.mqtt_client = mqtt.Client()

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
            dependencies=["metrics_exporter"]
        )
        threading.Thread(target=self._run_api, daemon=True).start()
        threading.Thread(target=self._run_mqtt, daemon=True).start()
        start_http_server(self.cfg.prometheus_port)
        self.logger.info("ExternalConnectorHub started")

    def stop(self):
        self._running = False
        self.mqtt_client.disconnect()

    # ──────────────────────────────────────────────────────────────────────
    # API interface
    # ──────────────────────────────────────────────────────────────────────
    def _configure_routes(self):
        @self.app.post("/connectors/http/post")
        def send_http_request(payload: Dict[str, Any]):
            self.api_requests.inc()
            response = requests.post(payload["url"], json=payload["data"])
            return {"status": response.status_code, "content": response.json()}

    def _run_api(self):
        uvicorn.run(self.app, host="0.0.0.0", port=self.cfg.api_port, log_level="warning")

    # ──────────────────────────────────────────────────────────────────────
    # MQTT integration
    # ──────────────────────────────────────────────────────────────────────
    def _run_mqtt(self):
        def on_connect(client, userdata, flags, rc):
            client.subscribe("sensors/#")

        def on_message(client, userdata, msg):
            self.mqtt_messages.inc()
            payload = json.loads(msg.payload)
            self.nexus.send_message(
                source=self.MODULE_NAME,
                target="sensor_hub",
                message_type="sensor_update",
                content={"topic": msg.topic, "payload": payload}
            )

        self.mqtt_client.on_connect = on_connect
        self.mqtt_client.on_message = on_message
        self.mqtt_client.connect(self.cfg.mqtt_broker, self.cfg.mqtt_port)
        self.mqtt_client.loop_forever()

    # ──────────────────────────────────────────────────────────────────────
    # Message handling
    # ──────────────────────────────────────────────────────────────────────
    def handle_message(self, msg: Dict[str, Any]):
        if msg["type"] == "external_request":
            # Exemple de gestionnaire pour les requêtes sortantes
            url, data = msg["content"]["url"], msg["content"]["data"]
            try:
                response = requests.post(url, json=data)
                self.logger.info(f"External POST to {url} status: {response.status_code}")
            except Exception as e:
                self.logger.error(f"Error sending external request: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Health & Status
    # ──────────────────────────────────────────────────────────────────────
    def health_check(self):
        return {"api_running": self._running, "mqtt_connected": self.mqtt_client.is_connected(), "healthy": True}

    def get_status(self):
        return {"mqtt_messages_total": self.mqtt_messages._value.get()}
