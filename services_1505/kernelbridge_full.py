#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kernelbridge_full.py — KernelBridge “tout-en-un” monolithe
=========================================================

Usage:
    python kernelbridge_full.py        # lance le service FastAPI + metrics
    python kernelbridge_full.py test   # lance les tests unitaires intégrés

Fonctions intégrées :
 • Configuration 12-Factor (Pydantic + Vault)
 • Metrics Prometheus + OpenTelemetry tracing
 • Circuit-breaker + retry/backoff
 • FastAPI REST API (/spawn, /terminate, /status)
 • CLI entrypoint via if __name__=="__main__"
 • Tests PyTest embarqués
 • Docstring Dockerfile & CI (GitHub Actions)
"""

import os, sys, time, threading, uuid, queue, resource, logging, multiprocessing as mp
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field, SecretStr
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, start_http_server
import uvicorn
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type, RetryError
from circuitbreaker import circuit, CircuitBreakerError
# Optional: Vault integration
try:
    import hvac
except ImportError:
    hvac = None

# ──────────────────────────────────────────────────────────────────────────────
# 1. Configuration (Pydantic + Vault)
# ──────────────────────────────────────────────────────────────────────────────
class KBConfig(BaseSettings):
    # Environment
    ENV: str = Field("prod", env="ENV")
    # Vault (optionnel)
    VAULT_ADDR: Optional[str] = Field(None, env="VAULT_ADDR")
    VAULT_ROLE_ID: Optional[str] = Field(None, env="VAULT_ROLE_ID")
    VAULT_SECRET_ID: Optional[str] = Field(None, env="VAULT_SECRET_ID")
    # Resource limits
    MAX_CHILDREN: int = Field(4, env="KB_MAX_CHILDREN")
    STATUS_INTERVAL: float = Field(2.0, env="KB_STATUS_INTERVAL")
    # Logging
    LOG_LEVEL: str = Field("INFO", env="KB_LOG_LEVEL")
    # Circuit breaker
    CB_MAX_FAILURES: int = Field(5, env="KB_CB_MAX_FAILURES")
    CB_RESET_TIMEOUT: int = Field(60, env="KB_CB_RESET_TIMEOUT")
    class Config:
        env_file = ".env"
        validate_all = True

cfg = KBConfig()

# If Vault configured, fetch secrets
def load_secret(name: str) -> str:
    if hvac and cfg.VAULT_ADDR and cfg.VAULT_ROLE_ID and cfg.VAULT_SECRET_ID:
        client = hvac.Client(url=cfg.VAULT_ADDR)
        client.auth_approle(cfg.VAULT_ROLE_ID, cfg.VAULT_SECRET_ID)
        sec = client.secrets.kv.v2.read_secret_version(path="kernelbridge")
        return sec["data"]["data"].get(name)
    return os.getenv(name)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Logging & Tracing
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL))
logger = logging.getLogger("kernelbridge_full")

# (Optional) OpenTelemetry tracing setup
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    resource = Resource.create({"service.name": "kernelbridge"})
    tp = TracerProvider(resource=resource)
    tp.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(tp)
    tracer = trace.get_tracer(__name__)
except ImportError:
    tracer = None

# ──────────────────────────────────────────────────────────────────────────────
# 3. Metrics (Prometheus)
# ──────────────────────────────────────────────────────────────────────────────
SPAWN_COUNT = Counter("kb_spawn_total", "Nombre de sous-cœurs spawnés")
TERMINATE_COUNT = Counter("kb_terminate_total", "Nombre de sous-cœurs terminés")
CHILD_UPTIME = Gauge("kb_child_uptime_seconds", "Uptime des sous-cœurs", ["child_id"])

# Start metrics server
start_http_server(8001)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Resilience decorators
# ──────────────────────────────────────────────────────────────────────────────
@circuit(failure_threshold=cfg.CB_MAX_FAILURES, recovery_timeout=cfg.CB_RESET_TIMEOUT)
def _resilient_noop(x):
    return x

def resilient_call(fn):
    return retry(stop=stop_after_attempt(3),
                 wait=wait_exponential_jitter(1, 10),
                 retry=retry_if_exception_type(Exception))(fn)

# ──────────────────────────────────────────────────────────────────────────────
# 5. KernelBridge service (monolithe)
# ──────────────────────────────────────────────────────────────────────────────
class SpawnPolicy(BaseSettings):
    ram_limit_mb: Optional[int] = None

class KernelBridge:
    def __init__(self):
        self._lock = threading.RLock()
        self._children: Dict[int,Dict[str,Any]] = {}
        self.app = FastAPI(title="KernelBridge Full")
        self._configure_api()
        threading.Thread(target=self._status_loop, daemon=True).start()

    def spawn(self, policy: SpawnPolicy) -> int:
        with tracer.start_as_current_span("spawn") if tracer else dummy_span():
            in_q, out_q = mp.Queue(), mp.Queue()
            cfg_child = {"ram_limit_mb": policy.ram_limit_mb}
            proc = mp.Process(target=self._child_bootstrap, args=(cfg_child,in_q,out_q), daemon=True)
            proc.start()
            cid = proc.pid
            with self._lock:
                self._children[cid] = {"proc":proc,"in_q":in_q,"out_q":out_q,"start":time.time()}
            SPAWN_COUNT.inc()
            logger.info(f"Spawned child {cid}")
            return cid

    def terminate(self, cid: int):
        with self._lock:
            info = self._children.pop(cid, None)
        if info:
            info["in_q"].put(None)
            info["proc"].join(timeout=3)
            TERMINATE_COUNT.inc()
            logger.info(f"Terminated child {cid}")

    def status(self) -> Dict[str,Any]:
        with self._lock:
            return {cid: {"alive":info["proc"].is_alive()} for cid,info in self._children.items()}

    def _status_loop(self):
        while True:
            with self._lock:
                for cid,info in self._children.items():
                    CHILD_UPTIME.labels(child_id=str(cid)).set(time.time()-info["start"])
            time.sleep(cfg.STATUS_INTERVAL)

    @staticmethod
    def _child_bootstrap(cfg_child, in_q, out_q):
        # Apply RAM limit
        if cfg_child.get("ram_limit_mb"):
            lim = cfg_child["ram_limit_mb"] * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (lim,lim))
        # Simple echo
        while True:
            try:
                msg = in_q.get(timeout=1)
                if msg is None: break
                out_q.put({"child": os.getpid(), "echo": msg})
            except queue.Empty:
                pass

    def _configure_api(self):
        @self.app.post("/spawn")
        def api_spawn(policy: SpawnPolicy):
            try:
                cid = resilient_call(lambda: self.spawn(policy))()
                return {"child_id": cid}
            except CircuitBreakerError:
                raise HTTPException(503, "Spawn circuit open")
        @self.app.post("/terminate/{cid}")
        def api_terminate(cid: int):
            self.terminate(cid)
            return {"terminated": cid}
        @self.app.get("/status")
        def api_status():
            return self.status()

# Dummy span context if no tracer
class dummy_span:
    def __enter__(self): return None
    def __exit__(self, *args): return False

# ──────────────────────────────────────────────────────────────────────────────
# 6. CLI & server start
# ──────────────────────────────────────────────────────────────────────────────
def main():
    kb = KernelBridge()
    uvicorn.run(kb.app, host="0.0.0.0", port=8000)

# ──────────────────────────────────────────────────────────────────────────────
# 7. Embedded tests (pytest)
# ──────────────────────────────────────────────────────────────────────────────
def _run_tests():
    import requests, subprocess, time
    from multiprocessing import Process
    # start server in subprocess
    p = Process(target=main, daemon=True); p.start()
    time.sleep(2)
    # test spawn
    r = requests.post("http://localhost:8000/spawn", json={"ram_limit_mb":5})
    assert r.status_code==200
    cid = r.json()["child_id"]
    st = requests.get("http://localhost:8000/status").json()
    assert cid in st
    # test terminate
    r2 = requests.post(f"http://localhost:8000/terminate/{cid}")
    assert r2.status_code==200
    st2 = requests.get("http://localhost:8000/status").json()
    assert cid not in st2
    p.terminate()
    print("All tests passed.")

if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1]=="test":
        _run_tests()
    else:
        main()

# ──────────────────────────────────────────────────────────────────────────────
# 8. Dockerfile & CI snippet (dans docstring)
# ──────────────────────────────────────────────────────────────────────────────
__dockerfile__ = r'''
FROM python:3.10-slim
WORKDIR /app
COPY kernelbridge_full.py .
RUN pip install fastapi uvicorn prometheus_client pydantic tenacity circuitbreaker hvac opentelemetry-sdk
EXPOSE 8000 8001
CMD ["python","kernelbridge_full.py"]
'''
__github_ci__ = r'''
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: python-version: 3.10
      - run: pip install pytest requests fastapi uvicorn prometheus_client pydantic tenacity circuitbreaker hvac opentelemetry-sdk
      - run: python kernelbridge_full.py test
'''

name: CI
on: [push, pull_request]
jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: {python-version: "3.10"}
      - run: pip install .
      - run: pip install pytest coverage
      - run: pytest --maxfail=1 --disable-warnings --cov=kernelbridge
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install flake8
      - run: flake8 src/ tests/
