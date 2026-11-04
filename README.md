# GrandNexus v7.2.3-rc

**An anchored translator and a reference engine that decide without inventing, and a laboratory to measure whether a loop learns.**

Claude translates a text and a question into a small logic language (anchored to the text: every statement quotes
the span it comes from); the reference engine `nexus_logic` decides (yes / no / unknown / contradictory) with a printable
proof, checked against clingo; the answer is the strict verdict (text only), the background-knowledge verdict is reported apart.
Latest experiments are documented in LOGBOOK.md; generated benchmark outputs are intentionally excluded from version control.
Everything else in this repository (the 2025 cognitive-architecture scaffolding: memories, graph, orchestrator, 15/05 layer)
is an experimental ground around that core, kept and runnable but not demonstrated.

This archive is a complete, self-contained snapshot: no notebook chain, no `%%writefile`, no update to graft.

## Quick start (Colab or any Python 3.10+)
```bash
unzip GrandNexus_v7.2.3-rc_complete.zip -d /content      # -> /content/GrandNexus
pip install -q -r /content/GrandNexus/requirements.txt
```
```python
import sys; sys.path.insert(0, "/content/GrandNexus")
from grandnexus.boot import boot
gn = boot(llm="auto")          # nx pathway (Claude -> nexus_logic) if ANTHROPIC_API_KEY is set, else grammar only
gn.print_report()
tr = gn.modules()["text_reasoner"]
print(tr.answer("All men are mortal. Socrates is a man. Is Socrates mortal?")["answer"])
gn.shutdown()
```
Modes: laboratory (default, deterministic, nothing runs in the background) or live (`boot(start_core=True)` / `gn.start()`).

## What is where
| Path | Content |
|---|---|
| `grandnexus/` | the package (core, memories, cognition, graph, LLM interface, 15/05 layer) |
| `grandnexus/boot.py` | assembles and wires the whole system |
| `grandnexus/nexus_logic/` | the core: language spec, parser, reference engine (disjunctions, formulas, reductio), clingo export, bridge |
| `grandnexus/llm_interface/translator_v5.py` | Claude as anchored translator into the nexus_logic language (prompt v5.4), Claude-alone control |
| `grandnexus/cognition/reasoning/text_reasoner.py` | text -> program/premises -> reference engine -> verdict (grammar, JSON and nx pathways) |
| `grandnexus/cognition/deliberation.py` | the deliberation experiment (several readings, targeted re-reading; hypothesis rejected, logbook 44) |
| `grandnexus/llm_interface/llm_translator.py` | the earlier JSON translator (prompt versions v1-v4), kept for comparison |
| `tools/bench_folio.py`, `tools/bench_deliberation.py`, `tools/diff_engines.py` | FOLIO bench (content-sha freezing, trivial controls, --strict), deliberation bench, differential reference / clingo / home engine |
| `tools/test_bench.py` | bench for hand-written sets: freezes a set, measures 4 modes, never prints item texts |
| `tools/smoke_test.py`, `tools/lab_*.py` | import test, labs (controls, layer oracles) |
| `tests_lab/test_*.py` | offline tests (no API) |
| `LOGBOOK.md` | logbook: every hypothesis -> result -> decision (49 entries) |
| `tests_lab/resultats/` | every FOLIO run: results JSON + the translated programs |
| `archive/` | original 15/05 and 29/05 sources (untouched) |
| `.git` | full history: every repair is one commit (`git log --oneline`) |

## Checks
```bash
cd /content/GrandNexus
python tools/smoke_test.py
for t in tests_lab/test_*.py; do python $t; done
python tools/diff_engines.py tests_lab/programmes_nx      # reference engine vs clingo (pip install clingo)
python tools/bench_folio.py --split train --seed 2 --n 100 --modes translator,direct --strict   # needs ANTHROPIC_API_KEY
```

## Language
Code, tools, tests, logbook and README are in English. The legacy 2025 modules (outside the core) still carry French
comments and log messages; they will be translated in a later pass.
