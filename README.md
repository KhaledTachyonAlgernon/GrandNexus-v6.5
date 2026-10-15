# GrandNexus 6.5 (réparation de la 6.4 du 29/05/2025)

## Colab
```python
!unzip -q -o GrandNexus_v6.5.zip -d /content && cd /content/GrandNexus
%cd /content/GrandNexus
!python tools/smoke_test.py          # imports, module par module
!python tools/integration_test.py    # démarrage + perception + raisonnement vs témoin
```
```python
import sys; sys.path.insert(0, "/content/GrandNexus")
from grandnexus.boot import boot
from grandnexus.cognition.contracts import CognitiveProcessType as P
gn = boot(); gn.print_report()
print(gn.cognition.invoke(P.PERCEPTION, "2*x + 3 = 7").output["input_type"])
```
Historique : `!git log --oneline`. Diagnostic et résultats : `CARNET_DE_BORD.md`.
