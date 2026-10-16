# GrandNexus 6.5

GrandNexus is a modular cognitive architecture inspired by biological and organic systems. It combines module orchestration, hierarchical memories, a knowledge graph, perception, reasoning, learning, and metacognition.

The project positions itself as a **nexus between AIs**: an experimental framework that can serve as contextual memory for an LLM, as an analytical digital brain, or as a coordination platform between specialized modules.

> **Project status:** GrandNexus 6.5 is a repair and integration release of the 6.4 base. Import and startup tests pass, but some cognitive capabilities remain experimental. See [`CARNET_DE_BORD.md`](CARNET_DE_BORD.md) for the detailed diagnosis.

## Contents

- [Features](#fonctionnalités)
- [Architecture](#architecture)
- [Prerequisites](#prérequis)
- [Installation](#installation)
- [Verify installation](#vérifier-linstallation)
- [Quick start](#utilisation-rapide)
- [Start the system](#démarrer-le-système)
- [Tests](#tests)
- [Using in Google Colab](#utilisation-dans-google-colab)
- [Configuration](#configuration)
- [Development](#développement)
- [Known limitations](#limites-connues)
- [Git history](#historique-git)
- [License](#licence)

## Features

- **NexusCore**: central orchestration, module registry, lifecycle and message bus.
- **Hierarchical memory**: working memory, episodic memory and semantic memory.
- **Knowledge graph**: in-memory graph engine and graph module manager.
- **Hybrid cognition**: perception, symbolic reasoning, neural/hybrid reasoning, learning and metacognition.
- **Capability-based routing**: cognitive processes are directed to specialized modules based on their capability and priority.
- **Error recovery**: monitoring module health and recovery strategies integrated into the orchestration.
- **Reproducible tests**: module-by-module import tests and a full wiring integration test.

## Architecture

```text
GrandNexus/
├── grandnexus/
│   ├── boot.py                  # Assembly and system startup
│   ├── core/                    # NexusCore, registry and event bus
│   ├── memory/                  # Working, semantic, and episodic memories
│   ├── graph/                   # Knowledge graph and module manager
│   ├── cognition/
│   │   ├── perception/          # Analysis and classification of inputs
│   │   ├── reasoning/           # Symbolic and hybrid reasoning
│   │   ├── learning/            # Learning and intrinsic motivation
│   │   ├── metacognition/       # Monitoring, confidence, adaptation, and reflection
│   │   └── cognition_manager.py # Orchestration of cognitive submodules
│   ├── computation/             # Resource management
│   └── llm_interface/           # Interfaces and prompt templates for LLMs
├── tools/
│   ├── smoke_test.py            # Import checks
│   ├── integration_test.py      # Startup and routing test
│   └── run_all.sh               # Legacy repair script
├── archive/                     # Historical versions and experiments
├── ARCHITECTURE.txt
├── CARNET_DE_BORD.md
├── LICENSE
└── README.md
```

The recommended entry point is `grandnexus.boot.boot`. This function assembles the components in the following order:

1. `NexusCore`;
2. graph engine;
3. working, semantic, and episodic memories;
4. `GraphModuleManager`;
5. `CognitionManager` and its submodules;
6. cognitive pathways and capabilities;
7. graph sharing between modules.

## Prerequisites

- Python **3.9 or higher**;
- Git, if you clone the repository;
- `psutil` for importing all modules and running full tests;
- SQLite, provided with the Python standard library.

The current base does not yet include a `pyproject.toml` or `requirements.txt`. Most of the code uses Python's standard library. Some components include fallbacks when optional scientific libraries are not available.

## Installation

### From GitHub

```bash
git clone https://github.com/KhaledTachyonAlgernon/GrandNexus-v6.5.git
cd GrandNexus-v6.5
```

To install the dependency required for full tests:

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install psutil
```

### From an archive

Extract the archive, then change to the project root:

```bash
unzip GrandNexus_v6.5.zip
cd GrandNexus
python3 -m pip install --user psutil
```

## Verify the installation

From the repository root, run the smoke test:

```bash
python tools/smoke_test.py
```

A correctly prepared environment should display:

```text
==> 55/55 importable modules
```

If you're using a virtual environment, make sure the `python` command points to it:

```bash
python --version
python -c "import psutil; print(psutil.__version__)"
```

## Quick start

The project is not yet distributed as a globally installed Python package. From the root of the repository, Python can import the package directly:

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

From another directory, add the repository root to `PYTHONPATH`:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

or in Python:

```python
import sys
sys.path.insert(0, "/chemin/vers/GrandNexus")
```

### Types of cognitive processes

The available types are defined in `grandnexus.cognition.contracts.CognitiveProcessType`. For example:

```python
from grandnexus.cognition.contracts import CognitiveProcessType as P

system = boot()

perception = system.cognition.invoke(P.PERCEPTION, "Bonjour, GrandNexus")
reasoning = system.cognition.invoke(P.REASONING, "Si A implique B et A est vrai, que peut-on déduire ?")

print(perception.output)
print(reasoning.output)
system.shutdown()
```

The exact result depends on the selected module and the experimental status of the capabilities. Use `system.print_report()` to inspect each assembly step.

## Starting the system

The main signature is:

```python
boot(
    db_file=None,
    start_threads=False,
    log_level=logging.WARNING,
    verbose=False,
    graph_config=None,
)
```

### Lab mode — recommended for getting started

By default, `start_threads=False`. Modules are initialized without starting background loops, which makes tests more deterministic:

```python
from grandnexus.boot import boot

system = boot(log_level=30)  # logging.WARNING
system.print_report()
# ... experiments ...
system.shutdown()
```

### Background threads mode

To start the cognitive manager's background loops:

```python
system = boot(start_threads=True, log_level=20)  # logging.INFO
try:
    # ... application using GrandNexus ...
    pass
finally:
    system.shutdown()
```

### SQLite Persistence

By default, `boot()` creates a temporary SQLite database. To keep the database between runs:

```python
from grandnexus.boot import boot

system = boot(db_file="data/nexuscore.db")
# The data persisted by NexusCore will be stored in this file.
system.shutdown()
```

Create the parent directory first if necessary:

```bash
mkdir -p data
```

## Tests

### Smoke test

Checks that the main modules can be imported:

```bash
python tools/smoke_test.py
```

### Integration test

Checks in particular:

- the initialization of `NexusCore`;
- the registration of memories and modules;
- the shared graph;
- the startup of `CognitionManager`;
- the perception, reasoning, learning, and metacognition pathways;
- the wiring and routing of capabilities.

```bash
python tools/integration_test.py
```

The test also displays experimental classification and routing metrics. These measurements are informational and do not imply that all cognitive capabilities are already calibrated.

To run both commands:

```bash
python tools/smoke_test.py && python tools/integration_test.py
```

> `tools/run_all.sh` is a legacy script that replays the fixes from version 6.4. It modifies the Git tree and should not be run on `main` without creating a working branch.

## Using Google Colab

In a Colab cell:

```python
!unzip -q -o GrandNexus_v6.5.zip -d /content
%cd /content/GrandNexus
!python tools/smoke_test.py
!python tools/integration_test.py
```

Then, in another cell:

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

### Logging level

The standard levels of the `logging` module can be used:

```python
import logging
from grandnexus.boot import boot

system = boot(log_level=logging.INFO)
```

To display full traces for a boot step:

```python
system = boot(verbose=True)
```

### Graph configuration

A configuration can be passed to the graph manager:

```python
from grandnexus.boot import boot

system = boot(graph_config={
    # Parameters specific to the graph modules used.
})
```

The recognized keys depend on the chosen graph engine and `GraphModuleManager`.

## Development

Before making any changes, create a working branch:

```bash
git switch -c feature/ma-modification
```

After making a change:

```bash
python tools/smoke_test.py
python tools/integration_test.py
git diff --check
```

Modules must preserve the contracts defined in `grandnexus/cognition/contracts.py`. To understand the state of fixes and the architectural choices, read:

- [`CARNET_DE_BORD.md`](CARNET_DE_BORD.md) ;
- [`ARCHITECTURE.txt`](ARCHITECTURE.txt) ;
- the docstrings of `grandnexus/boot.py` and `grandnexus/core/`.

## Known limitations

Version 6.5 is functional as an experimental base, but several elements remain to be developed:

- some mathematical entries and some code snippets are still classified as `TEXT` by the perception module;
- priority routing does not yet clearly outperform the random baseline on all examples;
- background threads (`boot(start_threads=True)`) require further testing;
- several advanced modules are still accessible via direct calls or the event bus rather than via declared capabilities;
- experimental duplicates are kept side-by-side (`graph_engine_v1`, `llm_engine_lite`, `resource_manager_ml`);
- advanced archive components are not integrated into the main startup;
- there is not yet official Python packaging nor a complete dependency list.

These limitations are described in more detail in [`CARNET_DE_BORD.md`](CARNET_DE_BORD.md).

## Git history

The `main` branch contains the repaired and integrated version. The main tags are:

```bash
git log --oneline
git tag --list
git show v6.4-raw
git show v6.5-repair
```

- `v6.4-raw`: raw extraction serving as a reference;
- `v6.5-repair`: repaired and documented version.

## License

This project is distributed under the [Apache License 2.0](LICENSE).
