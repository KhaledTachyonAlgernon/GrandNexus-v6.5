# Logbook — GrandNexus 6.4 → 6.5 (repair)

Branch `main`, tag `v6.4-raw` = raw extraction (baseline). Each fix = 1 commit.
To replay: core = `tools/extract_notebook.py` then `tools/run_all.sh`; 15/05 layer = `tools/extract_layer_1505.py` then `tools/run_layer_1505.sh`.

| # | Hypothesis | Result | Decision |
|---|-----------|----------|----------|
| 1 | The notebook is broken more by its structure than by its logic | imports: 8/41 code modules (raw baseline) → 41/41 after R1–R9 | package, then integrate |
| 2 | Names come from Colab's shared namespace | pyflakes: 12 undefined names in 9 files → 0 (R10) | keep pyflakes in the loop |
| 3 | pyflakes also finds real bugs | 2 logical bugs (clustering loop that wasn't progressing, ErrorRecovery stopping in dead code) → fixed (R11) | same |
| 4 | The system can start in one go | boot(): 16/16 steps OK (R12, boot.py) | test the wiring |
| 5 | Semantic memory and graph manager share a graph | NO: 2 separate graphs → 1 shared graph for 7 modules (R13) | ok |
| 6 | Perception recognizes the input type | 5/8: maths and `import` classified as TEXT (tie in scores, TEXT wins by order) | to calibrate (laboratory mode) |
| 7 | Priority routing beats randomness | 1/5 vs 0.8/5 (random baseline): no real difference | the bottleneck is not routing but the perception → reasoning handover (the reasoner expects structured premises/rules) |

| 8 | The reasoner does not receive the premises | reason() stored the context under a different key than the strategies read → fixed (R14) | connect the extractor |
| 9 | Extracting premises is sufficient for reasoning | path A: 12/12 in grammar vs 6/12 (always-yes), 3/12 (random), 2/12 (old path); 0/5 out-of-grammar, 0 false "yes" | expand the extractor (verbs, pronouns, relations) |
| 10 | The 15/05 layer is compatible with NexusCore | API: yes (register/get/send_message only). Names: 1 real conflict (working_memory v2) detected and blocked | layer extracted, _v2 collisions |
| 11 | The orchestrator answers questions | after R15–R23, R27: 12/12, 0 responses to other questions, ~1 s/question (10 iterations × 0.1 s hardcoded) | ok |
| 12 | Everything runs together on a started NexusCore | BEFORE R24: the clock died after 1 tick (message queue: TypeError at equal priority). AFTER: ~8,000 messages in 3 s, 13 modules alive, only 1 error (gymnasium missing) | fix addressings and slow shutdown |
| 13 | The CognitionManager talks to its modules | 12 called methods did not exist (quick_scan, evaluate, create_plan…) → delegation bridges (R25) | to be validated in threaded mode |
| 14 | The two WorkingMemory instances can coexist | v2 renamed "working_memory_v2", AffectRegulator follows (R28): both run together, 0 errors | ok (Khaled's decision) |
| 15 | The extractor generalizes beyond its grammar | dev 0/5 → 5/5, regression 12/12 preserved; fixed test case BEFORE tuning (commit d5d2894): 6/7 vs 4/7 always-yes, 1.7/7 chance; only failure = feminine "vegetarian" | feminine morphology; next test case written by Khaled |
| 16 | Heavy dependencies are a wall | pip only (level 1): 91 → 99 importable modules. But Redis/MongoDB are servers (connection refused / hang); EvolutionManager requires KernelBridge | level 1 for now, servers when a test requires them |
| 17 | The pathway reasons on sentences it has never seen (Khaled's test case, 10 items) | **4/10 = always-unknown (4/10)** → K1 FAILED; 0 false 'yes' → K2 PASSED. Of the 4 successes, 2 come from a failed analysis (no.1, no.6), 2 from a genuine mechanism (no.3 existential, no.9 no affirmation of the consequent) | missing capabilities identified, new test case required after fixes |
| 18 | Is the failure due to lack of practice, language, knowledge or the engine? (oracles per layer, tools/lab_niveaux.py) | practice: 3 runs = 4,4,4 (no learning possible); +language 6/10; +knowledge 5/10; +inference 7/10; +full chaining 9/10. The engine only chains 2 deductions out of 4 requested | priority 1 = engine (depth, contrapositive, global coherence); 2 = language (comparative test parser / LLM / grammar); 3 = knowledge; 4 = correction loop |

Raw result: tests_lab/resultats/temoin_khaled_2026-10-16_mesure1.txt. This test case is now "seen":
after the fixes, its score will no longer be valid as a measure of generalization.

Bias to keep in mind (no.15): the fixed test case was written by Claude, who knows his own grammar.
The 'Some birds sing' trap may have biased the choice to treat 'some' as non-universal. A test case
written by someone else remains the true measure.

## What is NOT done yet
- Layer 15/05: without requirements-extra.txt, 13 modules expect an external dependency (torch, redis, neo4j, pymongo, pybullet, gymnasium, deap, opentelemetry, slack_sdk, sentence_transformers, sse_starlette).
- Dead address mappings at runtime: self_model → "monitor" (the core calls it "cognitive_monitor"), messages sent before their recipient starts (startup order).
- Slow shutdown (~15 s, join with timeouts) and 2 threads that survive (including the CognitiveMonitor's Report_Thread).
- Two concepts named CognitiveContext (orchestrator state / request context) to reconcile.
- Clock: 'fast' channel declared unhealthy despite ~120 ticks in 3 s (criterion to check).
- Only PerceptionCore and ReasoningCore declare capabilities (@cognitive_capability). The other 20 modules are only reachable by direct call or via the event bus.
- `CognitionManager.start()` (background threads) untested: boot(start_threads=True).
- Duplicates kept side by side, not merged: graph_engine_v1, llm_engine_lite (fallback + embeddings to graft into llm_engine), resource_manager_ml.
