#!/usr/bin/env bash
# Applique les réparations une par une, un commit chacune.
set -e
cd "$(dirname "$0")"
declare -A MSG=(
 [r2_syntax]="R2 syntaxe: attention_filter (méthode à 3 espaces) + reasoning_core (bloc sur-indenté L1686-1829)"
 [r3_dataclasses]="R3 dataclasses Task/TaskRequest: kw_only=True (champ obligatoire après défaut)"
 [r4_graph_factory]="R4 graph_engine V2: create_graph_engine remise au niveau module + alias GraphEngine"
 [r5_aliases]="R5 alias: PlanningEngine, ReinforcementLearner, AdaptationManager, Reflector, ConfidenceEstimator, MetaMonitor"
 [r6_missing_classes]="R6 LearningResult créée + create_pattern_mining sortie de la classe"
 [r7_event_bus]="R7 NexusCore: bus publication/abonnement (register_handler, publish_event, schedule_task)"
 [r8_cognitive_contract]="R8 contrat inter-modules: contracts.py + routage de capacités + constructeurs alignés"
 [r10_flat_namespace_imports]="R10 imports manquants hérités de l'espace de noms plat du notebook (pyflakes)"
 [r11_logic_bugs]="R11 bugs logiques: boucle de clustering (remaining coupé) + arrêt ErrorRecovery en code mort"
 [r9_robustness]="R9 robustesse: create_anomaly_filter hors classe, annotation ast différée, replis sans PyTorch"
)
for r in r2_syntax r3_dataclasses r4_graph_factory r5_aliases r6_missing_classes r7_event_bus r8_cognitive_contract r9_robustness r10_flat_namespace_imports r11_logic_bugs; do
  python3 repairs.py "$r"
  (cd .. && git add -A && git commit -qm "${MSG[$r]}")
done
