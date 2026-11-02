# GrandNexus v6.10.1-rc

Modular cognitive architecture (Python). This archive is a complete, self-contained snapshot:
no notebook chain, no `%%writefile`, no update to graft.

## Quick start (Colab or any Python 3.10+)
```bash
unzip GrandNexus_v6.10.1-rc_disjonction_complet.zip -d /content      # -> /content/GrandNexus
pip install -q -r /content/GrandNexus/requirements.txt
```
```python
import sys; sys.path.insert(0, "/content/GrandNexus")
from grandnexus.boot import boot
gn = boot(llm="auto")          # Claude translator if ANTHROPIC_API_KEY is set (env var or Colab Secrets), else grammar only
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
| `grandnexus/cognition/reasoning/text_reasoner.py` | text -> premises -> deduction -> verdict |
| `grandnexus/llm_interface/llm_translator.py` | Claude as an anchored translator (prompt versions v1-v4) |
| `tools/banc_de_test_v2.py` | test bench: freezes a test set, measures 4 modes, never prints item texts |
| `tools/smoke_test.py`, `tools/lab_*.py` | import test, labs (controls, layer oracles) |
| `tests_lab/test_*.py` | offline tests (no API) |
| `CARNET_DE_BORD.md` | logbook: every hypothesis -> result -> decision |
| `archive/` | original 15/05 and 29/05 sources (untouched) |
| `.git` | full history: every repair is one commit (`git log --oneline`) |

## Checks
```bash
cd /content/GrandNexus
python tools/smoke_test.py
for t in tests_lab/test_*.py; do python $t; done
```
