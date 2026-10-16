# Logbook — GrandNexus 6.4 → 6.5 (repair)

Branch `main`, tag `v6.4-raw` = raw extraction (baseline). Each repair = 1 commit.
Replay the whole pipeline: `git checkout v6.4-raw -- grandnexus && bash tools/run_all.sh` (on a branch).

| # | Hypothesis | Result | Decision |
|---|-----------|----------|----------|
| 1 | The notebook is broken more by its structure than by its logic | imports: 8/41 code modules (raw baseline) → 41/41 after R1–R9 | package, then integrate |
| 2 | Some names come from Colab's shared namespace | pyflakes: 12 undefined names in 9 files → 0 (R10) | keep pyflakes in the loop |
| 3 | pyflakes also finds real bugs | 2 logical bugs (a clustering loop that wasn't making progress, ErrorRecovery stopped in dead code) → fixed (R11) | same |
| 4 | The system can boot from cold start | boot(): 16/16 steps OK (R12, boot.py) | test the wiring |
| 5 | Semantic memory and graph manager share one graph | NO: 2 separate graphs → 1 shared graph for 7 modules (R13) | ok |
| 6 | Perception recognizes the input type | 5/8: maths and `import` classified as TEXT (tie in scores, TEXT wins by order) | to calibrate (lab mode) |
| 7 | Priority routing beats chance | 1/5 vs 0.8/5 (random baseline): no real difference | the bottleneck is not routing but the perception → reasoning handoff (the reasoner waits for structured premises/rules) |

## What is NOT done yet
- Experimental .py layer from 15/05 (CognitiveOrchestrator, CognitiveClock, SafetyGuard, KernelBridge, EvolutionManager…): archived in `archive/`, not wired in. The orchestrator imports `ReasoningEngine`, `LearningManager`, `PerceptionModule` which do not exist.
- Only PerceptionCore and ReasoningCore declare capabilities (@cognitive_capability). The other 20 modules are only reachable by direct call or via the event bus.
- `CognitionManager.start()` (background threads) untested: boot(start_threads=True).
- Duplicates kept side by side, not merged: graph_engine_v1, llm_engine_lite (fallback + embeddings to be integrated into llm_engine), resource_manager_ml.
