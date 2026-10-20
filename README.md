# GrandNexus 6.5 (fix for 6.4 + 15/05 layer)

## Colab
```python
# optional (unlocks 8 modules from 15/05, see the 3 levels in the file):
# !pip install -q -r /content/GrandNexus/requirements-extra.txt
!unzip -q -o GrandNexus_v6.5.zip -d /content
%cd /content/GrandNexus
!python tools/smoke_test.py              # imports module by module (bugs vs missing dependencies)
!python tools/integration_test.py        # startup + capabilities
!python tools/lab_reasoning.py           # perception→reasoning path vs baselines + orchestrator
!python tools/layer_runtime_test.py 3    # the whole system running on a started NexusCore
```
```python
import sys; sys.path.insert(0, "/content/GrandNexus")
from grandnexus.boot import boot
gn = boot(); gn.print_report()
print(gn.modules()["text_reasoner"].answer("All men are mortal. Socrates is a man. Is Socrates mortal?")["answer"])
print(gn.orchestrator.process_input("No reptiles are mammals. A snake is a reptile. Is a snake a mammal?")["output"][0])
gn.shutdown()
```
History: `!git log --oneline` · Outcomes and decisions: `CARNET_DE_BORD.md`.
