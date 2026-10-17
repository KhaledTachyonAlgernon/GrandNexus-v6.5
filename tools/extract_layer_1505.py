"""
Extraction de la couche expérimentale du .py du 15/05/2025 (lignes 54751-fin).
- Chaque module va au chemin indiqué par son en-tête « # grandnexus/... ».
- Collision avec un module du cœur (memory/working, memory/semantic,
  perception/filters/attention_filter) -> suffixe _v2 : le cœur n'est jamais écrasé.
- Doublons (orchestrateur x2, KernelBridge x3, Explainability x2) : tous conservés.
- Les démos exécutées à l'import (NexusCore() + start() au niveau module) sont
  placées sous `if __name__ == "__main__":` : elles restent, mais ne tournent plus
  à l'import.
- Les variantes « microservice » (FastAPI + Helm + pyproject mélangés) vont dans
  services_1505/ hors paquet : ce sont des déploiements, pas des modules en-process.
"""
import ast, os, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "archive" / "grandnexusv_6_4_15_05_2025.py"

SEGMENTS = [  # (début, fin, destination)
    (54751, 56536, "grandnexus/cognition/orchestration/cognitive_orchestrator_v1.py"),
    (56537, 58318, "grandnexus/cognition/orchestration/cognitive_orchestrator.py"),
    (58319, 58455, "archive/1505_generateur_tests_orchestrateur.py"),
    (58456, 58816, "grandnexus/clock/cognitive_clock.py"),
    (58817, 59138, "grandnexus/sensors/sensor_hub.py"),
    (59139, 59200, "grandnexus/sensors/drivers/gym_driver.py"),
    (59201, 59542, "grandnexus/actuators/actuator_hub.py"),
    (59543, 59590, "grandnexus/actuators/drivers/virtual_gripper.py"),
    (59591, 59872, "grandnexus/homeostasis/homeostasis_manager.py"),
    (59873, 60165, "grandnexus/planning/executive_planner.py"),
    (60166, 60438, "grandnexus/cognition/perception/filters/attention_filter_v2.py"),
    (60439, 60751, "grandnexus/memory/working_v2.py"),
    (60752, 61110, "grandnexus/memory/semantic_v2.py"),
    (61111, 61351, "grandnexus/security/safety_guard.py"),
    (61352, 61578, "grandnexus/cognition/metacognition/self_model.py"),
    (61579, 61787, "grandnexus/ops/metrics_exporter.py"),
    (61788, 62048, "grandnexus/graph/graph_sync_service.py"),
    (62049, 62255, "grandnexus/cognition/metacognition/affect_regulator.py"),
    (62256, 62452, "grandnexus/ops/cost_optimizer.py"),
    (62453, 62646, "grandnexus/io/dialogue_interface.py"),
    (62647, 62871, "grandnexus/cognition/learning/world_model_learner.py"),
    (62872, 63083, "grandnexus/multiagent/multi_agent_coordinator.py"),
    (63084, 63399, "grandnexus/kernel/kernel_bridge.py"),
    (63400, 63540, "grandnexus/cognition/metacognition/explainability_engine.py"),
    (63541, 63689, "grandnexus/evolution/evolution_manager.py"),
    (63690, 63804, "grandnexus/ops/decision_history_archive.py"),
    (63805, 63920, "grandnexus/simulation/digital_twin_service.py"),
    (63921, 64046, "grandnexus/io/external_connector_hub.py"),
    (64047, 64223, "grandnexus/distributed/federated_learning_coordinator.py"),
    (64224, 64394, "grandnexus/ops/devsecops_orchestrator.py"),
    (64395, 64579, "services_1505/explainability_service_bundle.txt"),
    (64580, 64775, "services_1505/kernelbridge_package_bundle.txt"),
    (64776, None, "services_1505/kernelbridge_full.py"),
]

def _is_exec(n):
    if isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    if isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant):
        return False
    if isinstance(n, ast.If) and "__main__" in ast.dump(n.test):
        return False
    if isinstance(n, (ast.Assign, ast.AnnAssign)) and not any(isinstance(x, ast.Call) for x in ast.walk(n)):
        return False
    return True

def guard_demos(src: str) -> str:
    """Après la dernière classe : le premier code exécutable (instanciation, appel)
    marque la démo ; les imports/fonctions/prose qui le précèdent immédiatement en
    font partie. Le tout passe sous `if __name__ == "__main__":`."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    body = tree.body
    last_cls = max((i for i, n in enumerate(body) if isinstance(n, ast.ClassDef)), default=None)
    if last_cls is None:
        return src
    e = next((i for i in range(last_cls + 1, len(body)) if _is_exec(body[i])), None)
    if e is None:
        return src
    k = e
    while k - 1 > last_cls and isinstance(body[k - 1], (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Expr)):
        k -= 1
    cut = body[k].lineno
    if getattr(body[k], "decorator_list", None):
        cut = min(d.lineno for d in body[k].decorator_list)
    lines = src.split("\n")
    head, tail = lines[:cut - 1], lines[cut - 1:]
    tail = ["    " + l if l.strip() else l for l in tail]
    return "\n".join(head + ['if __name__ == "__main__":  # démo d\'origine, ne tourne plus à l\'import'] + tail)

def main():
    L = SRC.read_text(encoding="utf-8").split("\n")
    for start, end, dest in SEGMENTS:
        chunk = "\n".join(L[start - 1:(end if end else len(L))])
        if dest.endswith(".py") and dest.startswith("grandnexus/"):
            chunk = (f"# Extrait de GrandNexusV6.4 du 15/05/2025, lignes {start}-{end or len(L)}\n" + chunk)
            chunk = guard_demos(chunk)
        out = ROOT / dest
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(chunk.rstrip() + "\n", encoding="utf-8")
        print(f"{start:>6}-{str(end or len(L)):<6} -> {dest}")
    for d in (ROOT / "grandnexus").rglob("*"):
        if d.is_dir() and not (d / "__init__.py").exists() and d.name != "__pycache__":
            (d / "__init__.py").write_text("")

if __name__ == "__main__":
    main()
