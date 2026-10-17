# Logbook — GrandNexus 6.4 → 6.5 (repair)

Branch `main`, tag `v6.4-raw` = raw extraction (baseline). Each repair = 1 commit.
Replay: core = `tools/extract_notebook.py` then `tools/run_all.sh`; 15/05 layer = `tools/extract_layer_1505.py` then `tools/run_layer_1505.sh`.

| # | Hypothesis | Result | Decision |
|---|-----------|----------|----------|
| 1 | The notebook is broken more by its structure than by its logic | imports: 8/41 code modules (raw baseline) → 41/41 after R1–R9 | package, then integrate |
| 2 | Some names come from Colab's shared namespace | pyflakes: 12 undefined names in 9 files → 0 (R10) | keep pyflakes in the loop |
| 3 | pyflakes also finds real bugs | 2 logical bugs (clustering loop that never advanced, ErrorRecovery stopped in dead code) → fixed (R11) | same |
| 4 | The system can start from a block | boot(): 16/16 steps OK (R12, boot.py) | test the wiring |
| 5 | Semantic memory and graph manager share a graph | NO: 2 separate graphs → 1 shared graph for 7 modules (R13) | ok |
| 6 | Perception recognizes the input type | 5/8: maths and `import` classified as TEXT (ties in scores, TEXT wins by order) | to calibrate (lab mode) |
| 7 | Priority routing beats chance | 1/5 vs 0.8/5 (random baseline): no real difference | the bottleneck is not routing but the perception → reasoning handoff (the reasoner expects structured premises/rules) |

| 8 | The reasoner does not receive the premises | reason() stored the context under a different key than the one read by strategies → fixed (R14) | connect the extractor |
| 9 | Extracting premises is sufficient to reason | path A: 12/12 in grammar vs 6/12 (always-yes), 3/12 (random), 2/12 (old path); 0/5 out-of-grammar, 0 false "yes" | broaden the extractor (verbs, pronouns, relations) |
| 10 | The 15/05 layer is compatible with NexusCore | API: yes (register/get/send_message only). Names: 1 real conflict (working_memory v2) detected and blocked | layer extracted, collisions in _v2 |
| 11 | The orchestrator answers questions | after R15-R23, R27: 12/12, 0 answers to a different question, ~1 s/question (10 iterations × 0.1 s hard-coded) | ok |
| 12 | Everything runs together on a started NexusCore | BEFORE R24: the clock died after 1 tick (message queue: TypeError at equal priority). AFTER: ~8,000 messages in 3 s, 13 live modules, only 1 error (gymnasium missing) | fix addressing and slow shutdown |
| 13 | The CognitionManager talks to its modules | 12 called methods did not exist (quick_scan, evaluate, create_plan…) → delegation bridges (R25) | to be validated in threaded mode |
| 14 | The two WorkingMemory instances can coexist | v2 renamed 'working_memory_v2', AffectRegulator follows (R28): both run together, 0 errors | ok (Khaled's decision) |
| 15 | The extractor generalizes beyond its grammar | dev 0/5 → 5/5, regression 12/12 preserved; frozen reference BEFORE tuning (commit d5d2894): 6/7 vs 4/7 always-yes, 1.7/7 random; only failure = feminine 'végétarienne' | feminine morphology; next reference written by Khaled |
| 16 | Heavy dependencies are a wall | pip alone (level 1): 91 → 99 importable modules. But Redis/MongoDB are servers (connection refused / blocking); EvolutionManager requires KernelBridge | level 1 now, servers when a test requires them |

Bias to keep in mind (no. 15): the frozen reference was written by Claude, who knows his own grammar.
The 'Some birds sing' trap may have biased the choice to treat 'some' as non-universal. A reference
written by someone else remains the real measure.

## What is NOT done yet
- Layer 15/05: without requirements-extra.txt, 13 modules await an external dependency (torch, redis, neo4j, pymongo, pybullet, gymnasium, deap, opentelemetry, slack_sdk, sentence_transformers, sse_starlette).
- Dead addressings during runtime: self_model → 'monitor' (the core says 'cognitive_monitor'), messages sent before their recipient starts (startup order).
- Slow shutdown (~15 s, join with timeouts) and 2 threads that survive (including CognitiveMonitor's Report_Thread).
- Two concepts named CognitiveContext (orchestrator state / request context) to reconcile.
- Clock: 'fast' channel declared unhealthy despite ~120 ticks in 3 s (criteria to check).
- Only PerceptionCore and ReasoningCore declare capabilities (@cognitive_capability). The other 20 modules are only reachable by direct call or via the event bus.
- `CognitionManager.start()` (background threads) untested: boot(start_threads=True).
- Duplicates kept side by side, not merged: graph_engine_v1, llm_engine_lite (fallback + embeddings to graft into llm_engine), resource_manager_ml.
