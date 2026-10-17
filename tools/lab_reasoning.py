"""
Laboratoire — voie perception → prémisses → raisonnement.
Critères fixés AVANT la mesure :
  C1  jeu « dans la grammaire » : ≥ 10/12 réponses exactes
  C2  strictement mieux que les témoins (toujours-oui, hasard, ancienne voie)
  C3  jeu « généralisation » (paraphrases hors grammaire) : mesuré, sans objectif
      -> sert à voir l'illusion de compétence, pas à la cacher.
Réponses possibles : oui / non / inconnu / contradictoire.
"""
import logging, random, sys, pathlib, statistics
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from grandnexus.boot import boot
from grandnexus.cognition.contracts import CognitiveContext, CognitiveRequest

LABELS = ["oui", "non", "inconnu", "contradictoire"]
IN_GRAMMAR = [
    ("All men are mortal. Socrates is a man. Is Socrates mortal?", "oui"),
    ("If it rains the ground is wet. It rains. Is the ground wet?", "oui"),
    ("If it rains the ground is wet. Is the ground wet?", "inconnu"),
    ("All birds can fly. A penguin is a bird. Can a penguin fly?", "oui"),
    ("All birds can fly. A penguin is a bird. A penguin cannot fly. Can a penguin fly?", "contradictoire"),
    ("All cats are mammals. All mammals are animals. Tom is a cat. Is Tom an animal?", "oui"),
    ("No reptiles are mammals. A snake is a reptile. Is a snake a mammal?", "non"),
    ("Why does ice float on water?", "inconnu"),
    ("Tous les hommes sont mortels. Socrate est un homme. Socrate est-il mortel ?", "oui"),
    ("S'il pleut, le sol est mouillé. Il pleut. Le sol est-il mouillé ?", "oui"),
    ("Si le four est chaud, le gâteau cuit. Le gâteau cuit-il ?", "inconnu"),
    ("Aucun poisson n'est un oiseau. Le saumon est un poisson. Le saumon est-il un oiseau ?", "non"),
]
GENERALISATION = [
    ("Every human dies eventually. Socrates is human. Will Socrates die?", "oui"),
    ("Whales are mammals. Mammals breathe air. Do whales breathe air?", "oui"),
    ("Paris is in France. France is in Europe. Is Paris in Europe?", "oui"),
    ("Si Paul étudie, il réussit. Paul étudie. Paul réussit-il ?", "oui"),
    ("Aucun oiseau n'a de dents. Le moineau est un oiseau. Le moineau a-t-il des dents ?", "non"),
]

import json
HOLDOUT = [tuple(x) for x in json.load(open(pathlib.Path(__file__).resolve().parent.parent
                                            / "tests_lab" / "holdout_extracteur.json"))["items"]]

def old_route(gn, q):
    rc = gn.modules()["reasoning_core"]
    ctx = CognitiveContext()
    try:
        out = rc.solve_problem(CognitiveRequest(query=q, context=ctx), ctx, q)
    except Exception:
        return "inconnu"
    return "oui" if (out or {}).get("confidence", 0) > 0 else "inconnu"

def run(gn, dataset, title, seed=0):
    rnd = random.Random(seed)
    tr = gn.modules()["text_reasoner"]
    rows, score = [], {"voie": 0, "toujours-oui": 0, "ancienne voie": 0}
    rand_scores = []
    for q, gold in dataset:
        got = tr.answer(q, write_back=False)["answer"]
        score["voie"] += got == gold
        score["toujours-oui"] += gold == "oui"
        score["ancienne voie"] += old_route(gn, q) == gold
        rows.append((got == gold, q, gold, got))
    for _ in range(200):
        rand_scores.append(sum(rnd.choice(LABELS) == g for _, g in dataset))
    print(f"\n== {title} ({len(dataset)} questions)")
    for ok, q, gold, got in rows:
        print(f"  {'✓' if ok else '✗'} {q[:70]:72} attendu={gold:14} obtenu={got}")
    print(f"  voie perception→raisonnement : {score['voie']}/{len(dataset)}")
    print(f"  témoin toujours-oui          : {score['toujours-oui']}/{len(dataset)}")
    print(f"  témoin hasard (200 tirages)  : {statistics.mean(rand_scores):.1f}/{len(dataset)}")
    print(f"  ancienne voie (solve_problem): {score['ancienne voie']}/{len(dataset)}")
    return score, statistics.mean(rand_scores)

if __name__ == "__main__":
    gn = boot(log_level=logging.CRITICAL)
    s1, r1 = run(gn, IN_GRAMMAR, "Dans la grammaire")
    s2, _ = run(gn, GENERALISATION, "Généralisation (hors grammaire) — jeu de dev")
    s3, r3 = run(gn, HOLDOUT, "Témoin figé avant réglage (tests_lab/holdout_extracteur.json)")
    if gn.orchestrator is not None:
        import time
        ok = contaminated = 0; t0 = time.time()
        print("\n== Même jeu via le CognitiveOrchestrator (15/05), questions enchaînées")
        for q, gold in IN_GRAMMAR:
            r = gn.orchestrator.process_input(q)
            outs = [o for o in (r["output"] or []) if isinstance(o, dict) and "answer" in o]
            mine = [o for o in outs if o.get("question") == q]
            contaminated += len(outs) - len(mine)
            got = mine[0]["answer"] if mine else None
            ok += got == gold
            print(f"  {'✓' if got == gold else '✗'} {q[:60]:62} attendu={gold:14} obtenu={got}")
        print(f"  orchestrateur : {ok}/{len(IN_GRAMMAR)} | réponses d'une autre question : {contaminated} | "
              f"{(time.time() - t0) / len(IN_GRAMMAR):.2f}s par question")
    c1 = s1["voie"] >= 10
    c2 = s1["voie"] > max(s1["toujours-oui"], r1, s1["ancienne voie"])
    print(f"\nC1 (≥10/12) : {'PASSÉ' if c1 else 'ÉCHOUÉ'} | C2 (> témoins) : {'PASSÉ' if c2 else 'ÉCHOUÉ'} | "
          f"C3 dev : {s2['voie']}/{len(GENERALISATION)} | C4 témoin figé : {s3['voie']}/{len(HOLDOUT)} "
          f"(toujours-oui {s3['toujours-oui']}, hasard {r3:.1f})")
    gn.shutdown()
