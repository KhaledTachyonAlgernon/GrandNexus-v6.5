# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 61579-61787
# grandnexus/ops/metrics_exporter.py
# ─────────────────────────────────────────────────────────────────────────
"""
MetricsExporter — Prometheus & OTEL bridge for GrandNexus
=========================================================

Dépendances externes :
    pip install prometheus-client opentelemetry-api opentelemetry-sdk
"""

from __future__ import annotations

import logging, time, threading, socket
from dataclasses import dataclass
from typing import Dict, Any, Optional

from grandnexus.core.nexus_core import NexusCore

# Prometheus
from prometheus_client import (
    start_http_server,
    Gauge,
    Counter,
    Summary,
    CollectorRegistry,
    CONTENT_TYPE_LATEST,
    exposition,
)

# OpenTelemetry
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)

# ─────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class ExporterConfig:
    port: int = 9095                           # Port HTTP /metrics
    scrape_interval: float = 5.0               # s
    otel_enabled: bool = False
    otel_interval: float = 15.0                # s
    otel_export_console: bool = True
    tick_channel: str = "slow"


# ─────────────────────────────────────────────────────────────────────────
# 2. Exporter principal
# ─────────────────────────────────────────────────────────────────────────
class MetricsExporter:
    MODULE_NAME = "metrics_exporter"

    def __init__(self, nexus: NexusCore, cfg: Optional[ExporterConfig] = None):
        self.logger = logging.getLogger("GrandNexus.Ops.MetricsExporter")
        self.nexus = nexus
        self.cfg = cfg or ExporterConfig()

        # Prom registry
        self.prom_reg = CollectorRegistry(auto_describe=True)

        # Core metrics
        self.g_module_up = Gauge(
            "gnx_module_up",
            "1 if module registered, 0 otherwise",
            ["module"],
            registry=self.prom_reg,
        )
        self.g_module_health = Gauge(
            "gnx_module_health",
            "Health ratio per module (0-1) if available",
            ["module"],
            registry=self.prom_reg,
        )
        self.g_queue_size = Gauge(
            "gnx_core_queue_size",
            "Size of NexusCore message queue",
            registry=self.prom_reg,
        )
        self.s_scrape_latency = Summary(
            "gnx_metrics_scrape_seconds",
            "Latency of scraping health_check()",
            registry=self.prom_reg,
        )
        self.c_scrape_errors = Counter(
            "gnx_metrics_scrape_errors_total",
            "Errors while scraping metrics",
            registry=self.prom_reg,
        )

        # OTEL metrics
        if self.cfg.otel_enabled:
            self._init_otel()

        # Runtime
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────
    def start(self):
        if self._running:
            return True
        self._running = True

        # Demarrer HTTP /metrics
        start_http_server(self.cfg.port, registry=self.prom_reg)
        host = socket.gethostname()
        self.logger.info(f"Prometheus endpoint available at http://{host}:{self.cfg.port}/metrics")

        # Enregistrement NexusCore
        self.nexus.register_module(
            name=self.MODULE_NAME,
            module=self,
            dependencies=["cognitive_clock"],
        )

        # Tick souscription
        clk = self.nexus.get_module("cognitive_clock")
        if clk:
            clk.subscribe(self.cfg.tick_channel, self._on_tick)

        # Boucle OTEL (optionnelle)
        if self.cfg.otel_enabled:
            self._thread = threading.Thread(target=self._otel_loop, daemon=True)
            self._thread.start()

        return True

    def stop(self):
        self._running = False
        return True

    # ─────────────────────────────────────────────────────────────────────
    # Prometheus scrape
    # ─────────────────────────────────────────────────────────────────────
    def _on_tick(self, _payload):
        start = time.perf_counter()
        try:
            self._scrape_metrics()
        except Exception as exc:
            self.c_scrape_errors.inc()
            self.logger.error(f"Metrics scrape error: {exc}")
        finally:
            self.s_scrape_latency.observe(time.perf_counter() - start)

    def _scrape_metrics(self):
        modules = self.nexus.list_modules()
        self.g_queue_size.set(len(self.nexus.message_queue))

        # Refresh gauges
        for mod in modules:
            self.g_module_up.labels(mod).set(1)
            health = 1.0
            inst = self.nexus.get_module(mod)
            if hasattr(inst, "health_check"):
                try:
                    h = inst.health_check()
                    health = 1.0 if h.get("healthy", True) else 0.0
                    # If ratio exists, prefer it
                    if "miss_ratio" in h:
                        health = 1.0 - h["miss_ratio"]
                except Exception:
                    health = 0.0
            self.g_module_health.labels(mod).set(health)

    # ─────────────────────────────────────────────────────────────────────
    # OpenTelemetry exporter
    # ─────────────────────────────────────────────────────────────────────
    def _init_otel(self):
        reader = None
        if self.cfg.otel_export_console:
            reader = PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=int(self.cfg.otel_interval * 1000))
        metrics.set_meter_provider(MeterProvider(metric_readers=[reader] if reader else []))
        self._meter = metrics.get_meter("grandnexus_metrics")
        self._otel_queue_gauge = self._meter.create_observable_gauge(
            name="gnx.core.queue.size",
            callbacks=[lambda opts: [metrics.Observation(len(self.nexus.message_queue))]],
        )
        self.logger.info("OpenTelemetry metrics initialised")

    def _otel_loop(self):
        # OTEL handled by PeriodicExportingMetricReader; just keep thread alive
        while self._running:
            time.sleep(1.0)

    # ─────────────────────────────────────────────────────────────────────
    # Health & status
    # ─────────────────────────────────────────────────────────────────────
    def health_check(self) -> Dict[str, Any]:
        return {"prometheus_port": self.cfg.port, "otel": self.cfg.otel_enabled, "healthy": True}

    def get_status(self) -> Dict[str, Any]:
        return {"otel_enabled": self.cfg.otel_enabled}

if __name__ == "__main__":  # démo d'origine, ne tourne plus à l'import
    from grandnexus.core.nexus_core import NexusCore
    from grandnexus.clock.cognitive_clock import CognitiveClock
    from grandnexus.ops.metrics_exporter import MetricsExporter

    nexus = NexusCore()
    CognitiveClock(nexus).start()
    MetricsExporter(nexus).start()

    # Accéder à http://localhost:9095/metrics et vérifier les métriques gnx_*
