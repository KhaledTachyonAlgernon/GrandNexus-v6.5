GrandNexus/
│
├── core/
│   ├── nexus_core.py
│   ├── errorRecoveryManager.py
│   └── registry.py
│
├── memory/
│   ├── memory_manager.py
│   ├── episodic.py
│   ├── semantic.py
│   └── working.py
│
├── cognition/
├── cognition_manager.py      # Orchestrateur central de la cognition
│
├── perception/
│   ├── perception_core.py    # Module de base pour l'interprétation des entrées
│   ├── symbolic_parser.py    # Parsing des expressions symboliques/mathématiques
│   ├── code_analyzer.py      # Analyse de code et structures formelles
│   ├── pattern_recognizer.py # Détection de motifs récurrents
│   └── filters/              # Filtres attentionnels et contextuels
│
├── reasoning/
│   ├── reasoning_core.py     # Core du système de raisonnement
│   ├── neural_reasoner.py    # Raisonnement par réseaux neuronaux (embeddings)
│   ├── hybrid_inference.py   # Module neuro-symbolique (LNN, DeepProbLog, etc.)
│   ├── planning.py           # Planification et décomposition d'objectifs
│   └── uncertainty.py        # Gestion de l'incertitude et raisonnement probabiliste
│
├── learning/
│   ├── learning_core.py      # Core du système d'apprentissage
│   ├── continuous_learner.py # Apprentissage continu (Avalanche)
│   ├── meta_learner.py       # Méta-apprentissage (learn2learn)
│   ├── reinforcement.py      # Apprentissage par renforcement
│   ├── intrinsic_motivation.py # Motivation et curiosité computationnelle
│   └── curriculum.py         # Gestion du curriculum d'apprentissage
│
└── metacognition/
    ├── monitor.py            # Surveillance des processus cognitifs
    ├── confidence.py         # Évaluation de la confiance/incertitude
    ├── adaptation.py         # Ajustement dynamique des paramètres
    └── reflection.py         # Auto-analyse des processus de raisonnement
│
├── graph/
│   ├── graph_engine.py
│   ├── pattern_mining.py
│   ├── graph_pattern_interface.py
│   └── graph_module_manager.py
│
├── computation/
│   └── resource_manager.py
│
└── llm_interface/
    ├── prompt_templates.py
    ├── response_parser.py
    └── knowledge_extractor.py
