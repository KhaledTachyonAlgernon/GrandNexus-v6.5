"""
Mesure UNIQUE d'un frozen control set (file JSON de tests_lab/).
Criteria fixed before measuring:
  K1  score > best dumb baseline (always-yes, always-unknown, random)
  K2  zero false 'yes' (the pathway must not invent)
Also reports the orchestrator and, for every failure, the extraction trace (diagnosis, not tuning).
Usage : python tools/lab_temoin.py tests_lab/temoin_khaled_2026-10-16.json
"""
import json, logging, random, statistics, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from grandnexus.boot import boot
from grandnexus.nexus_logic import normalise_label

LABELS = ["yes", "no", "unknown", "contradictory"]

def main(path):
    items = [tuple(x) for x in json.load(open(path, encoding="utf-8"))["items"]]
    gn = boot(log_level=logging.CRITICAL)
    tr = gn.modules()["text_reasoner"]
    rows, ok, false_yes = [], 0, 0
    items = [(q, normalise_label(g)) for q, g in items]
    for q, gold in items:
        out = tr.answer(q, write_back=False)
        got = out["answer"]
        ok += got == gold
        false_yes += got == "yes" and gold != "yes"
        rows.append((got == gold, q, gold, got, out["extraction"]["trace"], out["extraction"]["unparsed"]))
    orch_ok = 0
    if gn.orchestrator is not None:
        for q, gold in items:
            r = gn.orchestrator.process_input(q)
            mine = [o for o in (r["output"] or []) if isinstance(o, dict) and o.get("question") == q]
            orch_ok += (mine[0]["answer"] if mine else None) == gold
    rnd = random.Random(0)
    rand = statistics.mean(sum(rnd.choice(LABELS) == g for _, g in items) for _ in range(500))
    always = {lab: sum(g == lab for _, g in items) for lab in LABELS}
    print(f"== {path} ({len(items)} items)\n")
    for i, (good, q, gold, got, trace, unparsed) in enumerate(rows, 1):
        print(f"{'✓' if good else '✗'} {i:2}. expected={gold:14} got={got:14} {q[:70]}")
        if not good:
            for t in trace:
                print(f"        {t[:120]}")
            for u in unparsed:
                print(f"        unparsed | {u[:110]}")
    best = max(max(always.values()), rand)
    print(f"\nperception→reasoning pathway : {ok}/{len(items)}   | orchestrateur : {orch_ok}/{len(items)}")
    print("controls : " + ", ".join(f"toujours-{k} {v}" for k, v in always.items()) + f", random {rand:.1f}")
    print(f"false yes: {false_yes}")
    print(f"K1 (> best control {best:.1f}) : {'PASS' if ok > best else 'FAIL'} | "
          f"K2 (0 false yes) : {'PASS' if false_yes == 0 else 'FAIL'}")
    gn.shutdown()

if __name__ == "__main__":
    main(sys.argv[1])
