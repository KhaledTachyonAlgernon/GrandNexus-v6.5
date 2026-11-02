#!/usr/bin/env python
"""
Banc FOLIO (ou tout jeu JSONL {"id","text","query","expected"}) pour GrandNexus.
Jeu figé par SHA de CONTENU avant mesure (plus le piège fichier-réécrit du n°37), modes + 3 témoins triviaux,
stabilité sur N passages, ID du modèle noté, aucun texte d'item imprimé ni écrit
(sauf --save-programs, qui écrit les programmes traduits pour tools/diff_engines.py).

  pip install datasets                                   # une fois, pour télécharger FOLIO
  python tools/bench_folio.py --n 50 --modes translator,direct
  python tools/bench_folio.py --jsonl tests_lab/temoin_neuf_3.jsonl --modes translator,direct,grandnexus --runs 3

Written by the external reviewer (Fable 5.1, 2026-10-25); integrated unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grandnexus.nexus_logic import ENGINE_VERSION                         # noqa: E402
from grandnexus.llm_interface.translator_v5 import PROMPT_VERSION         # noqa: E402

LABELS = {"true": "oui", "false": "non", "uncertain": "inconnu", "unknown": "inconnu",
          "oui": "oui", "non": "non", "inconnu": "inconnu", "contradictoire": "contradictoire"}
API_MODES = ("translator", "direct", "grandnexus")
VERDICTS = ("contradictoire", "inconnu", "oui", "non")


def load_items(split: str, jsonl: str | None) -> list[dict]:
    if jsonl:
        rows = [json.loads(l) for l in open(jsonl, encoding="utf-8") if l.strip()]
    else:
        from datasets import load_dataset
        try:
            rows = list(load_dataset("yale-nlp/FOLIO", split=split))          # gated on the Hub (login + terms)
        except Exception:
            rows = list(load_dataset("tasksource/folio", split=split))         # public mirror, same columns
    items = []
    for i, r in enumerate(rows):
        text = r.get("text") or r.get("premises") or r.get("premise") or r.get("context") or ""
        text = "\n".join(text) if isinstance(text, list) else str(text)
        query = r.get("query") or r.get("conclusion") or r.get("question") or r.get("hypothesis") or ""
        exp = LABELS.get(str(r.get("expected", r.get("label", ""))).strip().lower())
        if text and query and exp:
            items.append({"id": str(r.get("id", r.get("example_id", i))), "text": text, "query": str(query), "expected": exp})
    return items


def freeze(items: list[dict]) -> str:
    return hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def as_question(q: str) -> str:
    return q if q.rstrip().endswith("?") else f"Is the following statement true? {q}"


def normalise(ans) -> str:
    a = str(ans).strip().lower()
    word = next((v for v in VERDICTS if v in a), None)
    return word or (a if a.replace(".", "", 1).lstrip("-").isdigit() else "inconnu")


def metrics(rows: list[dict], key: str = "pred") -> dict:
    n = len(rows) or 1
    g = lambda r: r.get(key, "erreur")          # a row that errored has no pred_strict (fix 6.9.2)
    acc = sum(g(r) == r["expected"] for r in rows)
    return {"acc": acc, "n": len(rows), "pct": round(100 * acc / n, 1),
            "false_oui": sum(g(r) == "oui" and r["expected"] != "oui" for r in rows),
            "false_non": sum(g(r) == "non" and r["expected"] != "non" for r in rows),
            "inventions": sum(g(r) in ("oui", "non") and r["expected"] == "inconnu" for r in rows),
            "inconnu": sum(g(r) == "inconnu" for r in rows),
            "contradictoire": sum(g(r) == "contradictoire" for r in rows),
            "erreurs": sum(g(r) == "erreur" for r in rows),
            "confusion": {f"{e}->{p}": c for (e, p), c in sorted(Counter((r["expected"], g(r)) for r in rows).items())}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--jsonl", help="jeu local : FOLIO jsonl ou {id,text,query,expected}")
    ap.add_argument("--n", type=int, default=0, help="sous-échantillon (0 = tout), tiré avec --seed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0, help="saute les N premiers items du tirage (ex. --skip 50 = les items jamais vus)")
    ap.add_argument("--runs", type=int, default=1, help="passages par mode API (stabilité)")
    ap.add_argument("--modes", default="translator,direct")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=str(ROOT / "tests_lab" / "resultats"))
    ap.add_argument("--save-programs", action="store_true", help="écrit les programmes traduits (ils citent le texte)")
    ap.add_argument("--strict", action="store_true", help="prédiction principale = verdict SANS connaissance de fond (FOLIO : prémisses seules)")
    a = ap.parse_args()

    items = load_items(a.split, a.jsonl)
    if a.n or a.skip:
        random.Random(a.seed).shuffle(items)
        items = items[a.skip:]
        if a.n:
            items = items[: a.n]
    sha = freeze(items)
    name = f"{'local' if a.jsonl else 'folio_' + a.split}_{len(items)}_{sha}_{time.strftime('%Y-%m-%d_%H%M%S')}"
    print(f"[{name}] {len(items)} items figés, sha {sha}, labels {dict(Counter(i['expected'] for i in items))}")

    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    tr = gn_reasoner = None
    if any(m in ("translator", "direct") for m in modes):
        from grandnexus.llm_interface.translator_v5 import Translator
        tr = Translator(model=a.model) if a.model else Translator()
    if "grandnexus" in modes:
        from grandnexus.boot import boot
        gn_reasoner = boot(llm="auto").modules()["text_reasoner"]

    majority = Counter(i["expected"] for i in items).most_common(1)[0][0]
    rows: dict[str, list[dict]] = {m: [] for m in modes + ["toujours_inconnu", "majorite", "hasard"]}
    outdir = pathlib.Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    progdir = outdir / "programmes" / name

    for m in rows:
        for run in range(a.runs if m in API_MODES else 1):
            rng = random.Random(a.seed + run)
            for it in items:
                row = {"id": it["id"], "expected": it["expected"], "run": run}
                try:
                    if m == "toujours_inconnu":
                        row["pred"] = "inconnu"
                    elif m == "majorite":
                        row["pred"] = majority
                    elif m == "hasard":
                        row["pred"] = rng.choice(["oui", "non", "inconnu"])
                    elif m == "translator":
                        r = tr.answer(it["text"], it["query"])
                        row.update(pred=r["verdict_strict"] if a.strict else r["verdict"], pred_full=r["verdict"],
                                   pred_strict=r["verdict_strict"], knowledge=r["depends_on_knowledge"],
                                   rejections=len(r["rejections"]), corrective=r["corrective_rounds"], calls=r["calls"],
                                   tokens=r["tokens_in"] + r["tokens_out"], model=r["model"], ids=r["message_ids"])
                        if a.save_programs:
                            progdir.mkdir(parents=True, exist_ok=True)
                            (progdir / f"{it['id']}_r{run}.nx").write_text(r["program"], encoding="utf-8")
                    elif m == "direct":
                        r = tr.direct(it["text"], as_question(it["query"]))
                        row.update(pred=normalise(r["answer"]), calls=r["calls"], tokens=r["tokens_in"] + r["tokens_out"],
                                   model=r["model"], ids=r["message_ids"])
                    elif m == "grandnexus":
                        r = gn_reasoner.answer(it["text"] + "\n" + as_question(it["query"]))
                        row["pred"] = normalise(r.get("answer", "") if isinstance(r, dict) else r)
                except Exception as e:  # noqa: BLE001 — on note et on continue
                    row.update(pred="erreur", error=f"{type(e).__name__}: {str(e)[:160]}")
                    n_err = sum(1 for r in rows[m] if r.get("pred") == "erreur") + 1
                    if n_err <= 3:
                        print(f"  !! {m} item {it['id']} : {row['error']}")
                    if n_err == 3 and len(rows[m]) == 2 and m in API_MODES:
                        print(f"  !! {m} : les 3 premiers appels ont échoué -> arrêt (clé API ? paquet anthropic ? réseau ?)")
                        rows[m].append(row)
                        break
                rows[m].append(row)
            print(f"  {m} run {run}: {metrics([r for r in rows[m] if r['run'] == run])['pct']}%")

    summary: dict = {"set": name, "sha": sha, "n": len(items), "runs": a.runs, "model": getattr(tr, "model", None),
                     "prompt_version": PROMPT_VERSION if tr else None, "engine_version": ENGINE_VERSION, "modes": {}}
    for m, rs in rows.items():
        per_run = {}
        for run in sorted({r["run"] for r in rs}):
            sub = [r for r in rs if r["run"] == run]
            per_run[run] = metrics(sub)
            if m == "translator":
                per_run[run]["strict"] = metrics(sub, "pred_strict")
                per_run[run]["depends_on_knowledge"] = sum(bool(r.get("knowledge")) for r in sub)
                per_run[run]["rejections"] = sum(r.get("rejections", 0) for r in sub)
        by_item: dict[str, set] = {}
        for r in rs:
            by_item.setdefault(r["id"], set()).add(r["pred"])
        summary["modes"][m] = {"runs": per_run, "stable_items": sum(len(s) == 1 for s in by_item.values()),
                               "calls": sum(r.get("calls", 0) for r in rs), "tokens": sum(r.get("tokens", 0) for r in rs)}

    for m in modes:
        errs = Counter(r.get("error", "") for r in rows[m] if r.get("pred") == "erreur")
        for msg, c in errs.most_common(3):
            print(f"erreurs {m} ({c}x) : {msg}")
    base = max(summary["modes"][b]["runs"][0]["pct"] for b in ("toujours_inconnu", "majorite", "hasard"))
    print(f"\nmeilleur témoin trivial : {base}%")
    for m in modes:
        for run, mt in summary["modes"][m]["runs"].items():
            print(f"{m} run {run}: {mt['acc']}/{mt['n']} = {mt['pct']}% | faux oui {mt['false_oui']} | faux non {mt['false_non']}"
                  f" | inventions {mt['inventions']} | inconnu {mt['inconnu']} | contradictoire {mt['contradictoire']} | erreurs {mt['erreurs']}")
            if "strict" in mt:
                s = mt["strict"]
                print(f"   strict (sans connaissance) : {s['acc']}/{s['n']} = {s['pct']}% | faux oui {s['false_oui']} | inventions {s['inventions']}"
                      f" | items dépendant d'une connaissance {mt['depends_on_knowledge']} | rejets {mt['rejections']}")
            print(f"   K1 {'PASS' if mt['pct'] > base else 'FAIL'} (> témoin {base}%) ; K2 faux oui = {mt['false_oui']}")
        if a.runs > 1 and m in API_MODES:
            print(f"   stabilité : {summary['modes'][m]['stable_items']}/{len(items)} items identiques sur {a.runs} passages")
    out = outdir / f"{name}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nécrit : {out}")


if __name__ == "__main__":
    main()
