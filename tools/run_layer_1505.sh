#!/usr/bin/env bash
# Couche 15/05 : extraction puis réparations, un commit chacune.
set -e
cd "$(dirname "$0")"
declare -A MSG=(
 [r15_orchestrator_imports]="R15 orchestrateur: adaptateurs ReasoningEngine/LearningManager/PerceptionModule + fabrique sortie de la classe"
 [r16_layer_bugs]="R16 couche 15/05: dataclass Homeostasis, shell/JS collé (dialogue), alias grandnexus.dialogue, openai protégé"
 [r18_orchestrator_state]="R18 orchestrateur: état PERCEPTION référencé mais absent de l'énumération"
 [r19_working_memory_sort]="R19 WorkingMemory: tri sur le score seul (égalités -> TypeError)"
 [r20_working_memory_item]="R20 WorkingMemoryItem.access_count déclaré (utilisé partout, jamais défini)"
 [r21_orchestrator_session_output]="R21 orchestrateur: sortie limitée à la session courante (fuite de la réponse précédente)"
 [r22_orchestrator_cleanup]="R22 orchestrateur: actions finies libérées à chaque passage de file (famine après 1 question)"
 [r23_orchestrator_memory_queries]="R23 orchestrateur: requêtes mémoire typées (SemanticQuery/EpisodicQuery) + division par zéro en méta-analyse"
 [r26_attention_v2_default]="R26 AttentionFilter 15/05: max() sur config vide à la création"
 [r27_orchestrator_session_filter_first]="R27 orchestrateur: filtre de session avant la coupe (réponse courante évincée par les anciennes)"
 [r28_working_memory_v2_name]="R28 WorkingMemory 15/05 renommée working_memory_v2 (cohabite avec le cœur), AffectRegulator suit"
 [r29_layer_flat_imports]="R29 couche 15/05: import os (ExplainabilityEngine) et time (VirtualGripper) manquants"
 [r30_semantic_v2_dataclass]="R30 SemanticMemory 15/05: default_factory (Python 3.11+) + garde annoy absent (vu sur Colab)"
 [r17_resource_history]="R17 ResourceManager.get_usage_history (demandé par l'orchestrateur)"
)
for r in r15_orchestrator_imports r16_layer_bugs r17_resource_history r18_orchestrator_state r19_working_memory_sort r20_working_memory_item r21_orchestrator_session_output r22_orchestrator_cleanup r23_orchestrator_memory_queries r26_attention_v2_default r27_orchestrator_session_filter_first r28_working_memory_v2_name r29_layer_flat_imports r30_semantic_v2_dataclass; do
  python3 repairs.py "$r"
  (cd .. && git add -A && git commit -qm "${MSG[$r]}")
done
