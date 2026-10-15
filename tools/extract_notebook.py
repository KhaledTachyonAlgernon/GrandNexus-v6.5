"""
Étape 0 — Extraction brute du notebook GrandNexus V6.4 (29/05/2025) en paquet Python.
Aucune modification du code : chaque cellule est copiée telle quelle dans son fichier.
Les doublons sont conservés sous un autre nom (rien n'est jeté).
Usage : python tools/extract_notebook.py archive/GrandNexusV_6_4_29_05_2025.ipynb
"""
import json, os, sys

# cellule -> chemin dans le paquet (canonique d'après l'arbre d'architecture + imports réels)
CELL_MAP = {
    4:  "grandnexus/core/nexus_core.py",
    6:  "grandnexus/core/errorRecoveryManager.py",
    10: "grandnexus/core/registry.py",
    8:  "grandnexus/memory/memory_manager.py",
    12: "grandnexus/memory/episodic.py",
    14: "grandnexus/memory/semantic.py",
    17: "grandnexus/memory/working.py",
    19: "grandnexus/cognition/cognition_manager.py",
    21: "grandnexus/cognition/perception/perception_core.py",
    23: "grandnexus/cognition/perception/symbolic_parser.py",
    27: "grandnexus/cognition/perception/code_analyzer.py",
    29: "grandnexus/cognition/perception/pattern_recognizer.py",
    31: "grandnexus/cognition/perception/filters/attention_filter.py",
    25: "grandnexus/cognition/reasoning/symbolic_engine.py",
    33: "grandnexus/cognition/reasoning/reasoning_core.py",
    35: "grandnexus/cognition/reasoning/neural_reasoner.py",
    37: "grandnexus/cognition/reasoning/hybrid_inference.py",
    39: "grandnexus/cognition/reasoning/planning.py",
    41: "grandnexus/cognition/reasoning/uncertainty.py",
    43: "grandnexus/cognition/learning/learning_core.py",
    45: "grandnexus/cognition/learning/continuous_learner.py",
    47: "grandnexus/cognition/learning/meta_learner.py",
    49: "grandnexus/cognition/learning/reinforcement.py",
    51: "grandnexus/cognition/learning/intrinsic_motivation.py",
    53: "grandnexus/cognition/learning/curriculum.py",
    55: "grandnexus/cognition/metacognition/monitor.py",
    57: "grandnexus/cognition/metacognition/confidence.py",
    59: "grandnexus/cognition/metacognition/adaptation.py",
    61: "grandnexus/cognition/metacognition/reflection.py",
    63: "grandnexus/graph/graph_engine_v1.py",        # doublon : 1re version (conservée)
    65: "grandnexus/graph/graph_engine.py",           # canonique : V2, sur-ensemble de la V1
    67: "grandnexus/graph/pattern_mining.py",
    70: "grandnexus/graph/graph_pattern_interface.py",
    72: "grandnexus/graph/graph_module_manager.py",
    74: "grandnexus/computation/resource_manager_ml.py",  # doublon : variante ordonnanceur ML (conservée)
    76: "grandnexus/computation/resource_manager.py",     # canonique : quotas par module + NexusCore
    78: "grandnexus/llm_interface/llm_engine.py",         # canonique : intégration mémoire/graphe
    80: "grandnexus/llm_interface/llm_engine_lite.py",    # doublon : fallback + embeddings (conservée)
    82: "grandnexus/llm_interface/prompt_templates.py",
    84: "grandnexus/llm_interface/response_parser.py",
    86: "grandnexus/llm_interface/__init__.py",
}
ARCHIVE = {1: "ARCHITECTURE.txt", 15: "archive/cell15_semantic_methods_duplicate.py",
           68: "archive/cell68_cudf_demo_hors_sujet.py"}
for i in (90, 92, 95, 96, 97):
    ARCHIVE[i] = f"archive/cell{i}_gemini_helper.py"

def clean(src):
    # magies IPython commentées pour rester importable hors Colab
    return "\n".join(("# " + l) if l.lstrip().startswith(("%", "!")) else l for l in src.split("\n"))

def main(nb_path, out="."):
    nb = json.load(open(nb_path, encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        path = CELL_MAP.get(i) or ARCHIVE.get(i)
        if path is None:
            print(f"!! cellule {i} non mappée"); continue
        full = os.path.join(out, path)
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(clean(src) if path.endswith(".py") else src)
    # __init__.py manquants
    for root, dirs, files in os.walk(os.path.join(out, "grandnexus")):
        if "__init__.py" not in files:
            open(os.path.join(root, "__init__.py"), "w").write("")
    print("Extraction terminée.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "archive/GrandNexusV_6_4_29_05_2025.ipynb")
