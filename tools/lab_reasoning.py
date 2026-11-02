"""
Laboratory - perception -> premises -> reasoning pathway.
Criteria fixed BEFORE measuring:
  C1  'inside the grammar' set: >= 10/12 exact answers
  C2  strictly better than the controls (always-yes, random, old pathway)
  C3  'generalisation' set (paraphrases outside the grammar): measured, no target
      -> shows the illusion of competence instead of hiding it.
Possible answers: yes / no / unknown / contradictory.
"""
import logging, random, sys, pathlib, statistics
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from grandnexus.boot import boot
from grandnexus.nexus_logic import normalise_label
from grandnexus.cognition.contracts import CognitiveContext, CognitiveRequest

LABELS = ["yes", "no", "unknown", "contradictory"]
IN_GRAMMAR = [
    ("All men are mortal. Socrates is a man. Is Socrates mortal?", "yes"),
    ("If it rains the ground is wet. It rains. Is the ground wet?", "yes"),
    ("If it rains the ground is wet. Is the ground wet?", "unknown"),
    ("All birds can fly. A penguin is a bird. Can a penguin fly?", "yes"),
    ("All birds can fly. A penguin is a bird. A penguin cannot fly. Can a penguin fly?", "contradictory"),
    ("All cats are mammals. All mammals are animals. Tom is a cat. Is Tom an animal?", "yes"),
    ("No reptiles are mammals. A snake is a reptile. Is a snake a mammal?", "no"),
    ("Why does ice float on water?", "unknown"),
    ("Tous les hommes sont mortels. Socrate est un homme. Socrate est-il mortel ?", "yes"),
    ("S'il pleut, le sol est mouillé. Il pleut. Le sol est-il mouillé ?", "yes"),
    ("Si le four est chaud, le gâteau cuit. Le gâteau cuit-il ?", "unknown"),
    ("Aucun poisson n'est un oiseau. Le saumon est un poisson. Le saumon est-il un oiseau ?", "no"),
]
GENERALISATION = [
    ("Every human dies eventually. Socrates is human. Will Socrates die?", "yes"),
    ("Whales are mammals. Mammals breathe air. Do whales breathe air?", "yes"),
    ("Paris is in France. France is in Europe. Is Paris in Europe?", "yes"),
    ("Si Paul étudie, il réussit. Paul étudie. Paul réussit-il ?", "yes"),
    ("Aucun oiseau n'a de dents. Le moineau est un oiseau. Le moineau a-t-il des dents ?", "no"),
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
        return "unknown"
    return "yes" if (out or {}).get("confidence", 0) > 0 else "unknown"

def run(gn, dataset, title, seed=0):
    rnd = random.Random(seed)
    tr = gn.modules()["text_reasoner"]
    rows, score = [], {"pathway": 0, "always-yes": 0, "old pathway": 0}
    rand_scores = []
    dataset = [(q, normalise_label(g)) for q, g in dataset]
    for q, gold in dataset:
        got = tr.answer(q, mode="grammar", write_back=False)["answer"]   # grammar lab: never the LLM pathway
        score["pathway"] += got == gold
        score["always-yes"] += gold == "yes"
        score["old pathway"] += old_route(gn, q) == gold
        rows.append((got == gold, q, gold, got))
    for _ in range(200):
        rand_scores.append(sum(rnd.choice(LABELS) == g for _, g in dataset))
    print(f"\n== {title} ({len(dataset)} questions)")
    for ok, q, gold, got in rows:
        print(f"  {'✓' if ok else '✗'} {q[:70]:72} expected={gold:14} got={got}")
    print(f"  perception→reasoning pathway : {score['pathway']}/{len(dataset)}")
    print(f"  control always-yes          : {score['always-yes']}/{len(dataset)}")
    print(f"  random control (200 tirages)  : {statistics.mean(rand_scores):.1f}/{len(dataset)}")
    print(f"  old pathway (solve_problem): {score['old pathway']}/{len(dataset)}")
    return score, statistics.mean(rand_scores)

if __name__ == "__main__":
    gn = boot(log_level=logging.CRITICAL)
    s1, r1 = run(gn, IN_GRAMMAR, "Dans la grammaire")
    s2, _ = run(gn, GENERALISATION, "Generalisation (outside the grammar) - dev set")
    s3, r3 = run(gn, HOLDOUT, "Control set frozen before tuning (tests_lab/holdout_extracteur.json)")
    if gn.orchestrator is not None:
        import time
        ok = contaminated = 0; t0 = time.time()
        print("\n== Same set through the CognitiveOrchestrator (15/05), chained questions")
        for q, gold in IN_GRAMMAR:
            r = gn.orchestrator.process_input(q)
            outs = [o for o in (r["output"] or []) if isinstance(o, dict) and "answer" in o]
            mine = [o for o in outs if o.get("question") == q]
            contaminated += len(outs) - len(mine)
            got = mine[0]["answer"] if mine else None
            ok += got == gold
            print(f"  {'✓' if got == gold else '✗'} {q[:60]:62} expected={gold:14} got={got}")
        print(f"  orchestrator: {ok}/{len(IN_GRAMMAR)} | answers from another question : {contaminated} | "
              f"{(time.time() - t0) / len(IN_GRAMMAR):.2f}s per question")
    c1 = s1["pathway"] >= 10
    c2 = s1["pathway"] > max(s1["always-yes"], r1, s1["old pathway"])
    print(f"\nC1 (≥10/12) : {'PASS' if c1 else 'FAIL'} | C2 (> controls) : {'PASS' if c2 else 'FAIL'} | "
          f"C3 dev : {s2['pathway']}/{len(GENERALISATION)} | C4 frozen control set : {s3['pathway']}/{len(HOLDOUT)} "
          f"(always-yes {s3['always-yes']}, random {r3:.1f})")
    gn.shutdown()
