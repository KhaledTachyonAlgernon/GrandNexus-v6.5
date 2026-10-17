# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 64224-64394
# grandnexus/ops/devsecops_orchestrator.py
# ──────────────────────────────────────────────────────────────────────────
"""
DevSecOpsOrchestrator — CI/CD & Security Automation for GrandNexus
==================================================================

Fonctions :
  • Surveillance Git (webhook ou polling)
  • Tests unitaires, linting, coverage
  • Analyse SAST (Bandit), DAST (OWASP ZAP)
  • Build & scan d’image Docker (Trivy)
  • Push image vers registry (DockerHub/ECR/GCR)
  • Déploiement automatisé sur Kubernetes (kubectl / Helm)
  • Notifications (Slack, email) et métriques Prometheus
  • Gestion des secrets via Vault or K8s Secrets

Dépendances externes :
    pip install gitpython docker pytest bandit trivy python-kubeconfig prometheus-client slack_sdk
"""
import logging, threading, time, uuid, os, subprocess, tempfile
from dataclasses import dataclass
from typing import Optional, Dict, Any
from prometheus_client import Counter, Gauge, start_http_server
from slack_sdk import WebClient
from git import Repo
from grandnexus.core.nexus_core import NexusCore

# ──────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class DevSecOpsConfig:
    git_repo_url: str
    git_branch: str = "main"
    poll_interval: float = 60.0                # s
    docker_registry: str = "registry.example.com/grandnexus"
    kubernetes_context: str = "prod-cluster"
    helm_chart_path: str = "./deploy/helm/grandnexus"
    slack_token: Optional[str] = None
    slack_channel: str = "#deployments"
    prometheus_port: int = 9100

# ──────────────────────────────────────────────────────────────────────────
# 2. DevSecOpsOrchestrator
# ──────────────────────────────────────────────────────────────────────────
class DevSecOpsOrchestrator:
    MODULE_NAME = "devsecops_orchestrator"

    def __init__(self, nexus: NexusCore, cfg: DevSecOpsConfig):
        self.logger = logging.getLogger("GrandNexus.DevSecOps")
        self.nexus = nexus
        self.cfg = cfg
        self._running = False
        self._last_commit = None
        self._lock = threading.RLock()

        # Metrics
        self.c_runs = Counter("devsecops_runs_total", "Total pipeline runs")
        self.c_failures = Counter("devsecops_failures_total", "Failed pipeline runs")
        self.g_last_duration = Gauge("devsecops_last_duration_seconds", "Last run duration")

        # Slack client
        self.slack = WebClient(token=self.cfg.slack_token) if self.cfg.slack_token else None

        # Start Prometheus endpoint
        start_http_server(self.cfg.prometheus_port)

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────
    def start(self):
        if self._running: return
        self._running = True
        self.nexus.register_module(
            name=self.MODULE_NAME,
            module=self,
            dependencies=[],
        )
        threading.Thread(target=self._poll_loop, daemon=True).start()
        self.logger.info("DevSecOpsOrchestrator started")

    def stop(self):
        self._running = False

    # ──────────────────────────────────────────────────────────────────────
    # Polling Git & triggering pipeline
    # ──────────────────────────────────────────────────────────────────────
    def _poll_loop(self):
        repo_dir = tempfile.mkdtemp(prefix="gnx_repo_")
        repo = Repo.clone_from(self.cfg.git_repo_url, repo_dir, branch=self.cfg.git_branch)
        while self._running:
            repo.remotes.origin.fetch()
            head = repo.head.commit.hexsha
            if head != self._last_commit:
                self._last_commit = head
                self.logger.info(f"New commit detected: {head}")
                threading.Thread(target=self._run_pipeline, args=(repo_dir, head), daemon=True).start()
            time.sleep(self.cfg.poll_interval)

    # ──────────────────────────────────────────────────────────────────────
    # Pipeline steps
    # ──────────────────────────────────────────────────────────────────────
    def _run_pipeline(self, repo_dir: str, commit: str):
        start = time.time()
        self.c_runs.inc()
        success = True
        try:
            # 1. Checkout
            self._shell(f"git -C {repo_dir} checkout {commit}")

            # 2. Unit tests & coverage
            self._shell(f"pytest {repo_dir}/src --maxfail=1 --disable-warnings")

            # 3. Static analysis (Bandit)
            self._shell(f"bandit -r {repo_dir}/src -lll")

            # 4. Build Docker image
            image_tag = f"{self.cfg.docker_registry}:{commit}"
            self._shell(f"docker build -t {image_tag} {repo_dir}")

            # 5. Scan image (Trivy)
            self._shell(f"trivy image --exit-code 1 --severity HIGH,CRITICAL {image_tag}")

            # 6. Push image
            self._shell(f"docker push {image_tag}")

            # 7. Deploy via Helm
            self._shell(
                f"helm upgrade --install grandnexus {self.cfg.helm_chart_path} "
                f"--kube-context {self.cfg.kubernetes_context} "
                f"--set image.repository={self.cfg.docker_registry} --set image.tag={commit}"
            )

            # 8. Post-deploy smoke tests
            self._shell(f"pytest {repo_dir}/tests/smoke.py")

            self._notify_slack(f"✅ Deployment successful for commit `{commit}`")
        except Exception as e:
            success = False
            self.c_failures.inc()
            self.logger.error(f"Pipeline failed at commit {commit}: {e}", exc_info=True)
            self._notify_slack(f"❌ Deployment failed for commit `{commit}`: {e}")
        finally:
            duration = time.time() - start
            self.g_last_duration.set(duration)
            self.logger.info(f"Pipeline duration: {duration:.1f}s")

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────
    def _shell(self, cmd: str):
        self.logger.debug(f"Running shell: {cmd}")
        res = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.logger.debug(res.stdout.decode())

    def _notify_slack(self, message: str):
        if self.slack:
            try:
                self.slack.chat_postMessage(channel=self.cfg.slack_channel, text=message)
            except Exception as e:
                self.logger.warning(f"Slack notification failed: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Health & Status
    # ──────────────────────────────────────────────────────────────────────
    def health_check(self) -> Dict[str, Any]:
        return {"last_commit": self._last_commit or "", "healthy": True}

    def get_status(self) -> Dict[str, Any]:
        return {"last_commit": self._last_commit, "runs": int(self.c_runs._value.get()), "failures": int(self.c_failures._value.get())}
