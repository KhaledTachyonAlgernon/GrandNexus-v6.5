# Carnet de bord — GrandNexus 6.4 → 6.5 (réparation)

Branche `main`, tag `v6.4-raw` = extraction brute (témoin). Chaque réparation = 1 commit.
Rejouer toute la chaîne : `git checkout v6.4-raw -- grandnexus && bash tools/run_all.sh` (sur une branche).

| # | Hypothèse | Résultat | Décision |
|---|-----------|----------|----------|
| 1 | Le notebook est cassé par sa structure plus que par sa logique | imports : 8/41 modules de code (témoin brut) → 41/41 après R1–R9 | packager, puis intégrer |
| 2 | Des noms viennent de l'espace de noms partagé de Colab | pyflakes : 12 noms indéfinis dans 9 fichiers → 0 (R10) | garder pyflakes dans la boucle |
| 3 | pyflakes trouve aussi de vrais bugs | 2 bugs logiques (boucle de clustering qui ne progressait pas, arrêt d'ErrorRecovery en code mort) → corrigés (R11) | idem |
| 4 | Le système peut démarrer d'un bloc | boot() : 16/16 étapes OK (R12, boot.py) | tester le câblage |
| 5 | Mémoire sémantique et gestionnaire de graphe partagent un graphe | NON : 2 graphes séparés → 1 graphe partagé pour 7 modules (R13) | ok |
| 6 | La perception reconnaît le type d'entrée | 5/8 : maths et `import` classés TEXT (égalité de scores, TEXT gagne par ordre) | à calibrer (mode laboratoire) |
| 7 | Le routage par priorité bat le hasard | 1/5 vs 0.8/5 (témoin aléatoire) : pas de différence réelle | le goulot n'est pas le routage mais le passage perception → raisonnement (le raisonneur attend prémisses/règles structurées) |

## Ce qui n'est PAS encore fait
- Couche expérimentale du .py 15/05 (CognitiveOrchestrator, CognitiveClock, SafetyGuard, KernelBridge, EvolutionManager…) : archivée dans `archive/`, non branchée. L'orchestrateur importe `ReasoningEngine`, `LearningManager`, `PerceptionModule` qui n'existent pas.
- Seuls PerceptionCore et ReasoningCore déclarent des capacités (@cognitive_capability). Les 20 autres modules ne sont joignables que par appel direct ou par le bus d'événements.
- `CognitionManager.start()` (threads de fond) non testé : boot(start_threads=True).
- Doublons conservés côte à côte, non fusionnés : graph_engine_v1, llm_engine_lite (fallback + embeddings à greffer dans llm_engine), resource_manager_ml.
