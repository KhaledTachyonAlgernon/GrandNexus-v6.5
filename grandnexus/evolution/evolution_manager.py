# Extrait de GrandNexusV6.4 du 15/05/2025, lignes 63541-63689
# grandnexus/evolution/evolution_manager.py
# ──────────────────────────────────────────────────────────────────────────
"""
EvolutionManager — Self-adaptive Evolutionary Optimization for GrandNexus
=========================================================================

Dependencies :
    pip install deap numpy
"""

import logging, json, time, uuid, threading, numpy as np
from deap import base, creator, tools, algorithms
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from grandnexus.core.nexus_core import NexusCore
from grandnexus.kernel.kernel_bridge import KernelBridge

# ──────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class EvolutionConfig:
    population_size: int = 20
    generations: int = 10
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    evaluation_interval: int = 3600  # seconds
    strategy_log_file: str = "strategy_evolutions.jsonl"
    tick_channel: str = "slow"

# ──────────────────────────────────────────────────────────────────────────
# 2. EvolutionManager
# ──────────────────────────────────────────────────────────────────────────
class EvolutionManager:
    MODULE_NAME = "evolution_manager"

    def __init__(self, nexus: NexusCore, cfg: Optional[EvolutionConfig] = None):
        self.logger = logging.getLogger("GrandNexus.EvolutionManager")
        self.nexus = nexus
        self.cfg = cfg or EvolutionConfig()
        self._running = False
        self._lock = threading.RLock()
        self._strategy_log = open(self.cfg.strategy_log_file, "a", encoding="utf-8")
        self.kb = nexus.get_module("kernel_bridge")
        if not self.kb:
            raise RuntimeError("KernelBridge is required for EvolutionManager")

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
            dependencies=["kernel_bridge", "executive_planner", "metrics_exporter"],
        )
        threading.Thread(target=self._evaluation_loop, daemon=True).start()
        self.logger.info("EvolutionManager started")

    def stop(self):
        self._running = False
        self._strategy_log.close()

    # ──────────────────────────────────────────────────────────────────────
    # Evolutionary optimization loop
    # ──────────────────────────────────────────────────────────────────────
    def _evaluation_loop(self):
        while self._running:
            self.logger.info("Starting evolutionary optimization round")
            best_strategy = self._run_evolution()
            self._apply_best_strategy(best_strategy)
            time.sleep(self.cfg.evaluation_interval)

    def _run_evolution(self):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # Ex. performance ↑, coût ↓
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        toolbox = base.Toolbox()
        toolbox.register("attr_float", np.random.uniform, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate_strategy)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selNSGA2)

        population = toolbox.population(n=self.cfg.population_size)
        algorithms.eaMuPlusLambda(population, toolbox,
                                  mu=self.cfg.population_size,
                                  lambda_=self.cfg.population_size,
                                  cxpb=self.cfg.crossover_rate,
                                  mutpb=self.cfg.mutation_rate,
                                  ngen=self.cfg.generations,
                                  verbose=False)

        best = tools.selBest(population, k=1)[0]
        strategy = {"parameters": best, "fitness": best.fitness.values}
        self._log_strategy(strategy)
        return strategy

    def _evaluate_strategy(self, individual):
        params = {"param_"+str(i): val for i, val in enumerate(individual)}
        child_id = self.kb.spawn_core()
        self.kb.forward_message(child_id, {
            "source": "evolution_manager",
            "target": "executive_planner",
            "type": "apply_parameters",
            "content": params
        })
        time.sleep(2)  # Simulated evaluation
        perf_metric = np.random.uniform(0.5, 1.0)  # Placeholder evaluation metric
        cost_metric = np.sum(individual)
        self.kb.terminate_core(child_id)
        return perf_metric, cost_metric

    # ──────────────────────────────────────────────────────────────────────
    # Applying & Logging strategies
    # ──────────────────────────────────────────────────────────────────────
    def _apply_best_strategy(self, strategy):
        params = {"best_strategy": strategy["parameters"]}
        self.nexus.send_message(
            source=self.MODULE_NAME,
            target="executive_planner",
            message_type="apply_strategy",
            content=params
        )
        self.logger.info(f"Applied best evolved strategy: {params}")

    def _log_strategy(self, strategy):
        record = {
            "timestamp": time.time(),
            "strategy": strategy,
            "id": str(uuid.uuid4())
        }
        with self._lock:
            self._strategy_log.write(json.dumps(record) + "\n")
            self._strategy_log.flush()

    # ──────────────────────────────────────────────────────────────────────
    # Health & Status
    # ──────────────────────────────────────────────────────────────────────
    def health_check(self):
        return {"running": self._running, "healthy": True}

    def get_status(self):
        return {"last_evolution": time.time()}
