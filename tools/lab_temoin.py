"""
Mesure UNIQUE d'un témoin figé (fichier JSON de tests_lab/).
Critères fixés avant la mesure :
  K1  score > meilleur témoin bête (toujours-oui, toujours-inconnu, hasard)
  K2  zéro « oui » à tort (la voie ne doit pas inventer)
Rapporte aussi l'orchestrateur et, pour chaque échec, la trace d'extraction (diagnostic, pas réglage).
Usage : python tools/lab_temoin.py tests_lab/temoin_khaled_2026-10-16.json
"""
import json, logging, random, statistics, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from grandnexus.boot import boot

LABELS = ["oui", "non", "inconnu", "contradictoire"]

def main(path):
    items = [tuple(x) for x in json.load(open(path, encoding="utf-8"))["items"]]
    gn = boot(log_level=logging.CRITICAL)
    tr = gn.modules()["text_reasoner"]
    rows, ok, false_yes = [], 0, 0
    for q, gold in items:
        out = tr.answer(q, write_back=False)
        got = out["answer"]
        ok += got == gold
        false_yes += got == "oui" and gold != "oui"
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
        print(f"{'✓' if good else '✗'} {i:2}. attendu={gold:14} obtenu={got:14} {q[:70]}")
        if not good:
            for t in trace:
                print(f"        {t[:120]}")
            for u in unparsed:
                print(f"        non analysé | {u[:110]}")
    best = max(max(always.values()), rand)
    print(f"\nvoie perception→raisonnement : {ok}/{len(items)}   | orchestrateur : {orch_ok}/{len(items)}")
    print("témoins : " + ", ".join(f"toujours-{k} {v}" for k, v in always.items()) + f", hasard {rand:.1f}")
    print(f"« oui » à tort : {false_yes}")
    print(f"K1 (> meilleur témoin {best:.1f}) : {'PASSÉ' if ok > best else 'ÉCHOUÉ'} | "
          f"K2 (0 oui à tort) : {'PASSÉ' if false_yes == 0 else 'ÉCHOUÉ'}")
    gn.shutdown()

if __name__ == "__main__":
    main(sys.argv[1])
