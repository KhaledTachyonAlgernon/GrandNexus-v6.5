#!/usr/bin/env python
"""
Deliberation bench (FOLIO or any {id,text,query,expected} JSONL): k independent readings, engine
verdict for each, targeted re-reading when they diverge, and every decision policy computed from
the SAME readings — so the controls cost nothing extra:

  single      one reading (= the current system)          random   any one reading
  vote        plain majority of the k readings              adaptive read 2, a 3rd only if they disagree
  vote_confidence   majority weighted by each reading's self-estimate (metacognition role)
  reflective  vote + one targeted re-reading on divergence  unanimity  answer only if all readings agree
  direct      Claude alone (one call, one token)            + trivial baselines

Never prints item texts. --skip 50 --seed 0 = the 153 FOLIO validation items never looked at
(the first 50 of that draw drove the reductio / universal-question work and are "seen").

  python tools/bench_deliberation.py --skip 50 --n 60 --direct
  python tools/bench_deliberation.py --skip 50 --direct --readers diverse
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bench_folio import as_question, freeze, load_items, metrics, normalise      # noqa: E402
from grandnexus.nexus_logic import ENGINE_VERSION                                  # noqa: E402
from grandnexus.llm_interface.translator_v5 import PROMPT_VERSION                  # noqa: E402

POLICIES = ("single", "random", "vote", "vote_confidence", "adaptive", "reflective", "unanimity")


def build(readers: str, model: str, client=None):
    from grandnexus.llm_interface.translator_v5 import SYSTEM_TRANSLATE, Translator
    from grandnexus.cognition.deliberation import Deliberation, SYSTEM_TRANSLATE_ALT
    mk = (lambda m: Translator(client=client, model=m)) if client is not None else (lambda m: Translator(model=m))
    if readers == "diverse":
        trs = [mk(model), mk(model), mk("claude-haiku-5-5")]
        names, systems = ["sonnet", "sonnet-alt", "haiku"], [SYSTEM_TRANSLATE, SYSTEM_TRANSLATE_ALT, SYSTEM_TRANSLATE]
    else:
        k = int(readers) if readers.isdigit() else 3
        trs = [mk(model) for _ in range(k)]
        names, systems = [f"r{i + 1}" for i in range(k)], [SYSTEM_TRANSLATE] * k
    return Deliberation(trs, names, systems, reflector=trs[0]), trs


def main(argv=None, client=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", type=int, default=50)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--readers", default="3", help="'3' (same reader 3x) or 'diverse' (sonnet, sonnet-alt, haiku)")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--direct", action="store_true", help="also measure Claude alone")
    ap.add_argument("--out", default=str(ROOT / "tests_lab" / "resultats"))
    ap.add_argument("--save-programs", action="store_true")
    a = ap.parse_args(argv)

    items = load_items(a.split, a.jsonl)
    random.Random(a.seed).shuffle(items)
    items = items[a.skip:]
    if a.n:
        items = items[: a.n]
    sha = freeze(items)
    name = f"delib_{'local' if a.jsonl else 'folio_' + a.split}_skip{a.skip}_{len(items)}_{sha}_{time.strftime('%Y-%m-%d_%H%M%S')}"
    print(f"[{name}] {len(items)} items frozen (sha de contenu {sha}), labels {dict(Counter(i['expected'] for i in items))}")

    delib, trs = build(a.readers, a.model, client)
    rows = {p: [] for p in POLICIES}
    rows.update({"direct": [], "always_unknown": [], "majority": [], "random": []})
    stats = Counter()
    majority_label = Counter(i["expected"] for i in items).most_common(1)[0][0]
    rng = random.Random(a.seed)
    outdir = pathlib.Path(a.out)
    progdir = outdir / "programs" / name
    detail = []
    fails = 0
    for k, it in enumerate(items, 1):
        base = {"id": it["id"], "expected": it["expected"], "run": 0}
        try:
            res = delib.run(it["text"], it["query"])
            fails = 0
            for p, v in res["policies"].items():
                rows[p].append({**base, "pred": v})
            vs = [r.verdict for r in res["readings"]]
            stats["unanimous"] += len(set(vs)) == 1
            stats["first_two_agree"] += vs[0] == vs[1]
            stats["reflections"] += res["reflection"] is not None
            stats["rejections"] += sum(r.rejections for r in res["readings"])
            stats["cannot_express"] += sum(r.cannot_express for r in res["readings"])
            detail.append({"id": it["id"], "expected": it["expected"], "verdicts": vs,
                           "reflection": res["reflection"].verdict if res["reflection"] else None,
                           "confidences": [r.confidence for r in res["readings"]]})
            if a.save_programs:
                progdir.mkdir(parents=True, exist_ok=True)
                for r in res["readings"] + ([res["reflection"]] if res["reflection"] else []):
                    (progdir / f"{it['id']}_{r.reader}.nx").write_text(r.program, encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  !! item {it['id']} : {type(e).__name__}: {str(e)[:160]}")
            for p in POLICIES:
                rows[p].append({**base, "pred": "error"})
            if fails >= 3 and k == fails:
                print("  !! the first 3 items failed -> stop (key, credit, network?)")
                break
        if a.direct:
            try:
                d = trs[0].direct(it["text"], as_question(it["query"]))
                rows["direct"].append({**base, "pred": normalise(d["answer"])})
            except Exception as e:  # noqa: BLE001
                rows["direct"].append({**base, "pred": "error", "error": str(e)[:120]})
        rows["always_unknown"].append({**base, "pred": "unknown"})
        rows["majority"].append({**base, "pred": majority_label})
        rows["random"].append({**base, "pred": rng.choice(["yes", "no", "unknown"])})
        if k % 10 == 0:
            print(f"  {k}/{len(items)} items")

    n = len(detail) or 1
    calls = sum(len(t.receipts) for t in trs)
    tokens = sum(r.tokens_in + r.tokens_out for t in trs for r in t.receipts)
    summary = {"set": name, "sha": sha, "n": len(items), "readers": a.readers, "model": a.model,
               "prompt_version": PROMPT_VERSION, "engine_version": ENGINE_VERSION,
               "calls": calls, "tokens": tokens, "stats": dict(stats), "modes": {}}
    base_pct = max(metrics(rows[b])["pct"] for b in ("always_unknown", "majority", "random"))
    print(f"\nbest trivial control : {base_pct}%")
    order = list(POLICIES) + (["direct"] if a.direct else [])
    for m in order:
        mt = metrics(rows[m])
        summary["modes"][m] = mt
        print(f"{m:16} {mt['acc']:>3}/{mt['n']} = {mt['pct']:5}% | false yes {mt['false_yes']} | false no {mt['false_no']}"
              f" | inventions {mt['inventions']} | unknown {mt['unknown']} | errors {mt['errors']}")
    print(f"\nrumination : unanimous readings {stats['unanimous']}/{n} | first two agree {stats['first_two_agree']}/{n}"
          f" | targeted re-readings {stats['reflections']} | rejections {stats['rejections']} | cannot express {stats['cannot_express']}")
    print(f"cost : {calls} calls, {tokens} tokens | model {a.model} | readers {a.readers} | prompt {PROMPT_VERSION} | engine {ENGINE_VERSION}")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{name}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows, "detail": detail}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written: {out}")
    return summary


if __name__ == "__main__":
    main()
