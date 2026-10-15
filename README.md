# GrandNexus 6.5

GrandNexus est une architecture cognitive modulaire inspirée des systèmes biologiques et organiques. Elle combine orchestration de modules, mémoires hiérarchiques, graphe de connaissances, perception, raisonnement, apprentissage et métacognition.

Le projet se présente comme un **connexum entre IA** : un cadre expérimental pouvant servir de mémoire contextuelle pour un LLM, de cerveau numérique analytique ou de plateforme de coordination entre modules spécialisés.

> **État du projet :** GrandNexus 6.5 est une version de réparation et d’intégration de la base 6.4. Les tests d’importation et de démarrage passent, mais certaines capacités cognitives restent expérimentales. Consultez [`CARNET_DE_BORD.md`](CARNET_DE_BORD.md) pour le diagnostic détaillé.

## Sommaire

- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Vérifier l’installation](#vérifier-linstallation)
- [Utilisation rapide](#utilisation-rapide)
- [Démarrer le système](#démarrer-le-système)
- [Tests](#tests)
- [Utilisation dans Google Colab](#utilisation-dans-google-colab)
- [Configuration](#configuration)
- [Développement](#développement)
- [Limites connues](#limites-connues)
- [Historique Git](#historique-git)
- [Licence](#licence)

## Fonctionnalités

- **NexusCore** : orchestration centrale, registre de modules, cycle de vie et bus de messages.
- **Mémoire hiérarchique** : mémoire de travail, mémoire épisodique et mémoire sémantique.
- **Graphe de connaissances** : moteur de graphe en mémoire et gestionnaire de modules de graphe.
- **Cognition hybride** : perception, raisonnement symbolique, raisonnement neuronal/hybride, apprentissage et métacognition.
- **Routage par capacités** : les processus cognitifs sont dirigés vers des modules spécialisés selon leur capacité et leur priorité.
- **Récupération d’erreurs** : surveillance de l’état des modules et stratégies de récupération intégrées à l’orchestration.
- **Tests reproductibles** : tests d’importation module par module et test d’intégration du câblage complet.

## Architecture

```text
GrandNexus/
├── grandnexus/
│   ├── boot.py                  # Assemblage et démarrage du système
│   ├── core/                    # NexusCore, registre et bus d’événements
│   ├── memory/                  # Mémoires de travail, sémantique et épisodique
│   ├── graph/                   # Graphe de connaissances et gestionnaire de modules
│   ├── cognition/
│   │   ├── perception/          # Analyse et classification des entrées
│   │   ├── reasoning/           # Raisonnement symbolique et hybride
│   │   ├── learning/            # Apprentissage et motivation intrinsèque
│   │   ├── metacognition/       # Surveillance, confiance, adaptation et réflexion
│   │   └── cognition_manager.py # Orchestration des sous-modules cognitifs
│   ├── computation/             # Gestion des ressources
│   └── llm_interface/           # Interfaces et modèles de prompts pour LLM
├── tools/
│   ├── smoke_test.py            # Vérification des imports
│   ├── integration_test.py      # Test du démarrage et du routage
│   └── run_all.sh               # Script historique de réparation
├── archive/                     # Versions et expérimentations historiques
├── ARCHITECTURE.txt
├── CARNET_DE_BORD.md
├── LICENSE
└── README.md
```

Le point d’entrée recommandé est `grandnexus.boot.boot`. Cette fonction assemble les composants dans l’ordre suivant :

1. `NexusCore` ;
2. moteur de graphe ;
3. mémoires de travail, sémantique et épisodique ;
4. `GraphModuleManager` ;
5. `CognitionManager` et ses sous-modules ;
6. voies cognitives et capacités ;
7. partage du graphe entre les modules.

## Prérequis

- Python **3.9 ou supérieur** ;
- Git, si vous clonez le dépôt ;
- `psutil` pour l’importation de tous les modules et les tests complets ;
- SQLite, fourni avec la bibliothèque standard Python.

La base actuelle ne contient pas encore de `pyproject.toml` ni de `requirements.txt`. La majorité du code utilise la bibliothèque standard Python. Certains composants disposent de replis lorsque des bibliothèques scientifiques optionnelles ne sont pas disponibles.

## Installation

### Depuis GitHub

```bash
git clone https://github.com/KhaledTachyonAlgernon/GrandNexus-v6.5.git
cd GrandNexus-v6.5
```

Pour installer la dépendance nécessaire aux tests complets :

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows : .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install psutil
```

### Depuis une archive

Décompressez l’archive, puis placez-vous à la racine du projet :

```bash
unzip GrandNexus_v6.5.zip
cd GrandNexus
python3 -m pip install --user psutil
```

## Vérifier l’installation

Depuis la racine du dépôt, exécutez le smoke test :

```bash
python tools/smoke_test.py
```

Un environnement correctement préparé doit afficher :

```text
==> 55/55 modules importables
```

Si vous utilisez un environnement virtuel, vérifiez que la commande `python` correspond bien à celui-ci :

```bash
python --version
python -c "import psutil; print(psutil.__version__)"
```

## Utilisation rapide

Le projet n’est pas encore distribué comme paquet Python installé globalement. Depuis la racine du dépôt, Python peut importer directement le package :

```python
from grandnexus.boot import boot
from grandnexus.cognition.contracts import CognitiveProcessType

system = boot()
system.print_report()

result = system.cognition.invoke(
    CognitiveProcessType.PERCEPTION,
    "2*x + 3 = 7",
)
print(result.output)

system.shutdown()
```

Depuis un autre répertoire, ajoutez la racine du dépôt à `PYTHONPATH` :

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

ou dans Python :

```python
import sys
sys.path.insert(0, "/chemin/vers/GrandNexus")
```

### Types de processus cognitifs

Les types disponibles sont définis dans `grandnexus.cognition.contracts.CognitiveProcessType`. Par exemple :

```python
from grandnexus.cognition.contracts import CognitiveProcessType as P

system = boot()

perception = system.cognition.invoke(P.PERCEPTION, "Bonjour, GrandNexus")
reasoning = system.cognition.invoke(P.REASONING, "Si A implique B et A est vrai, que peut-on déduire ?")

print(perception.output)
print(reasoning.output)
system.shutdown()
```

Le résultat exact dépend du module sélectionné et de l’état expérimental des capacités. Utilisez `system.print_report()` pour inspecter chaque étape d’assemblage.

## Démarrer le système

La signature principale est :

```python
boot(
    db_file=None,
    start_threads=False,
    log_level=logging.WARNING,
    verbose=False,
    graph_config=None,
)
```

### Mode laboratoire — recommandé pour commencer

Par défaut, `start_threads=False`. Les modules sont initialisés sans lancer les boucles de fond, ce qui rend les essais plus déterministes :

```python
from grandnexus.boot import boot

system = boot(log_level=30)  # logging.WARNING
system.print_report()
# ... expérimentations ...
system.shutdown()
```

### Mode avec threads de fond

Pour démarrer les boucles de fond du gestionnaire cognitif :

```python
system = boot(start_threads=True, log_level=20)  # logging.INFO
try:
    # ... application utilisant GrandNexus ...
    pass
finally:
    system.shutdown()
```

### Persistance SQLite

Par défaut, `boot()` crée une base SQLite temporaire. Pour conserver la base entre deux exécutions :

```python
from grandnexus.boot import boot

system = boot(db_file="data/nexuscore.db")
# Les informations persistées par NexusCore seront enregistrées dans ce fichier.
system.shutdown()
```

Créez le dossier parent au préalable si nécessaire :

```bash
mkdir -p data
```

## Tests

### Smoke test

Vérifie que les modules principaux peuvent être importés :

```bash
python tools/smoke_test.py
```

### Test d’intégration

Vérifie notamment :

- l’initialisation de `NexusCore` ;
- l’enregistrement des mémoires et des modules ;
- le graphe partagé ;
- le démarrage de `CognitionManager` ;
- les voies de perception, raisonnement, apprentissage et métacognition ;
- le câblage et le routage des capacités.

```bash
python tools/integration_test.py
```

Le test affiche aussi des mesures expérimentales de classification et de routage. Ces mesures sont informatives et ne signifient pas que toutes les capacités cognitives sont déjà calibrées.

Pour exécuter les deux commandes :

```bash
python tools/smoke_test.py && python tools/integration_test.py
```

> `tools/run_all.sh` est un script historique qui rejoue les réparations de la version 6.4. Il modifie l’arbre Git et ne doit pas être exécuté sur `main` sans créer une branche de travail.

## Utilisation dans Google Colab

Dans une cellule Colab :

```python
!unzip -q -o GrandNexus_v6.5.zip -d /content
%cd /content/GrandNexus
!python tools/smoke_test.py
!python tools/integration_test.py
```

Puis, dans une autre cellule :

```python
import sys
sys.path.insert(0, "/content/GrandNexus")

from grandnexus.boot import boot
from grandnexus.cognition.contracts import CognitiveProcessType as P

gn = boot()
gn.print_report()
print(gn.cognition.invoke(P.PERCEPTION, "2*x + 3 = 7").output)
gn.shutdown()
```

## Configuration

### Niveau de logs

Les niveaux standards du module `logging` peuvent être utilisés :

```python
import logging
from grandnexus.boot import boot

system = boot(log_level=logging.INFO)
```

Pour afficher les traces complètes d’une étape de démarrage :

```python
system = boot(verbose=True)
```

### Configuration du graphe

Une configuration peut être transmise au gestionnaire de graphe :

```python
from grandnexus.boot import boot

system = boot(graph_config={
    # Paramètres spécifiques aux modules de graphe utilisés.
})
```

Les clés reconnues dépendent du moteur de graphe et du `GraphModuleManager` choisis.

## Développement

Avant toute modification, créez une branche de travail :

```bash
git switch -c feature/ma-modification
```

Après une modification :

```bash
python tools/smoke_test.py
python tools/integration_test.py
git diff --check
```

Les modules doivent préserver les contrats définis dans `grandnexus/cognition/contracts.py`. Pour comprendre l’état des réparations et les choix d’architecture, lisez :

- [`CARNET_DE_BORD.md`](CARNET_DE_BORD.md) ;
- [`ARCHITECTURE.txt`](ARCHITECTURE.txt) ;
- les docstrings de `grandnexus/boot.py` et `grandnexus/core/`.

## Limites connues

La version 6.5 est fonctionnelle comme base expérimentale, mais plusieurs éléments restent à développer :

- certaines entrées mathématiques et certains extraits de code sont encore classés `TEXT` par le module de perception ;
- le routage par priorité ne dépasse pas encore clairement le témoin aléatoire sur tous les exemples ;
- les threads de fond (`boot(start_threads=True)`) nécessitent davantage de tests ;
- plusieurs modules avancés restent accessibles par appel direct ou bus d’événements plutôt que par les capacités déclarées ;
- des doublons expérimentaux sont conservés côte à côte (`graph_engine_v1`, `llm_engine_lite`, `resource_manager_ml`) ;
- les composants avancés de l’archive ne sont pas branchés dans le démarrage principal ;
- il n’existe pas encore de packaging Python officiel ni de liste de dépendances complète.

Ces limites sont décrites plus précisément dans [`CARNET_DE_BORD.md`](CARNET_DE_BORD.md).

## Historique Git

La branche `main` contient la version réparée et intégrée. Les tags principaux sont :

```bash
git log --oneline
git tag --list
git show v6.4-raw
git show v6.5-repair
```

- `v6.4-raw` : extraction brute servant de témoin ;
- `v6.5-repair` : version réparée et documentée.

## Licence

Ce projet est distribué sous [Apache License 2.0](LICENSE).
