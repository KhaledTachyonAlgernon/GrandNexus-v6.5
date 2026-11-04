#!/usr/bin/env python
"""
The loop that learns - does experience improve the reading on NEW items, without any hand-written rule?

Phase 1, experience (a pool the test never sees): read each item; when the strict verdict is wrong, the reader
is told only "wrong, expected X" and re-reads its own program; the correction is kept ONLY if the engine then
reaches the expected verdict (a verified experience). This is the loop: read -> verdict -> feedback ->
reflection -> memory.

Phase 2, test (a disjoint set, never seen): every item is read under three conditions, from the same memory:
  none      no memory                      (= the current system)
  static    the k most central experiences, the same for every item (consolidated experience, no retrieval)
  random    k experiences drawn at random  (control: is it the RELEVANCE of experience that helps?)
  learned   the k most similar experiences (the learning loop)

Two criteria, fixed before measuring (logbook 52-53):
  C-experience: static > none  (verified experience helps, even consolidated into a fixed set)
  C-relevance : learned > random AND learned > static (retrieving the RELEVANT experience adds something)
  both with asserted errors (false yes / no / contradictory) not above none. Trivial controls are printed as usual.
Run 1 (2026-11-02, memory of 18): none 77, random 83, learned 83 -> experience helps, relevance not shown.
No item text is printed; the memory is saved as JSON (it quotes the experience texts), the results as JSON.

  python tools/bench_learning.py --exp-n 150 --exp-seed 3 --test-n 100 --test-seed 4
  python tools/bench_learning.py --memory tests_lab/resultats/learning/<memory>.json --test-n 100 --test-seed 4   # reuse a memory
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import random
import sys
import time
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bench_folio import freeze, load_items, metrics                                   # noqa: E402
from grandnexus.cognition.experience import (Experience, ExperienceMemory,            # noqa: E402
                                             examples_block, statements_diff)
from grandnexus.nexus_logic import ENGINE_VERSION                                     # noqa: E402
from grandnexus.llm_interface.translator_v5 import PROMPT_VERSION, Translator          # noqa: E402

CONDITIONS = ("none", "static", "random", "learned")   # static = the k most central experiences, same for every item


def draw(split: str, seed: int, n: int, exclude: set) -> list:
    items = [it for it in load_items(split, None) if it["id"] not in exclude]
    random.Random(seed).shuffle(items)
    return items[:n] if n else items


def experience_phase(tr: Translator, items: list, memory: ExperienceMemory, store_correct: bool) -> dict:
    st = Counter()
    for k, it in enumerate(items, 1):
        try:
            r = tr.answer(it["text"], it["query"])
        except Exception as e:  # noqa: BLE001
            st["errors"] += 1
            print(f"  !! item {it['id']}: {type(e).__name__}: {str(e)[:120]}")
            continue
        st["read"] += 1
        if r["verdict_strict"] == it["expected"]:
            st["right_first"] += 1
            if store_correct:
                memory.add(Experience(it["text"], it["query"], r["program"], it["expected"], source_id=it["id"]))
            continue
        st["wrong_first"] += 1
        try:
            c = tr.reflect_correction(it["text"], it["query"], r["program"], r["verdict_strict"], it["expected"])
        except Exception as e:  # noqa: BLE001
            st["errors"] += 1
            print(f"  !! reflection {it['id']}: {type(e).__name__}: {str(e)[:120]}")
            continue
        if c["verdict_strict"] == it["expected"] and not c["rejections"]:
            st["verified"] += 1
            memory.add(Experience(it["text"], it["query"], c["program"], it["expected"], r["verdict_strict"],
                                  statements_diff(r["program"], c["program"]), it["id"]))
        else:
            st["unverified"] += 1
        if k % 10 == 0:
            print(f"  experience {k}/{len(items)}: wrong {st['wrong_first']}, verified corrections {st['verified']}")
    return dict(st)


def main(argv=None, client=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    ap.add_argument("--jsonl")
    ap.add_argument("--exp-n", type=int, default=150)
    ap.add_argument("--exp-seed", type=int, default=3)
    ap.add_argument("--test-n", type=int, default=100)
    ap.add_argument("--test-seed", type=int, default=4)
    ap.add_argument("--k", type=int, default=3, help="experiences shown per reading")
    ap.add_argument("--memory", help="reuse a saved memory; with --exp-n > 0 the experience phase EXTENDS it with new items")
    ap.add_argument("--store-correct", action="store_true", help="also store first-try correct readings (positive experiences)")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--exclude-from", nargs="*", default=[], help="result JSONs of earlier runs: their test items are excluded (independent replication)")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--out", default=str(ROOT / "tests_lab" / "resultats" / "learning"))
    a = ap.parse_args(argv)

    outdir = pathlib.Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tr = Translator(client=client, model=a.model) if client is not None else Translator(model=a.model)
    stamp = time.strftime("%Y-%m-%d_%H%M%S")

    if a.jsonl:
        pool = [json.loads(l) for l in open(a.jsonl, encoding="utf-8") if l.strip()]
        random.Random(a.exp_seed).shuffle(pool)
        exp_items, test_items = pool[: a.exp_n], pool[a.exp_n: a.exp_n + a.test_n]
    else:
        used = set()
        for f in a.exclude_from:
            for pattern in (glob.glob(f) or [f]):
                d = json.load(open(pattern, encoding="utf-8"))
                used |= {r["id"] for rows in d.get("rows", {}).values() for r in rows}
        exp_items = draw(a.split, a.exp_seed, a.exp_n, used) if (a.exp_n > 0 or not a.memory) else []   # --memory with --exp-n 0: no experience phase
        test_items = draw(a.split, a.test_seed, a.test_n, used | {it["id"] for it in exp_items} | (ExperienceMemory.load(a.memory).known_ids if a.memory else set()))
        if used:
            print(f"excluded {len(used)} items already used by earlier runs")
    assert not ({it["id"] for it in exp_items} & {it["id"] for it in test_items}), "experience and test sets overlap"
    exp_sha, test_sha = freeze(exp_items), freeze(test_items)
    print(f"experience pool: {len(exp_items)} items (sha {exp_sha}) | test set: {len(test_items)} items (sha {test_sha}), "
          f"labels {dict(Counter(i['expected'] for i in test_items))}")

    memory = ExperienceMemory.load(a.memory, seed=a.test_seed) if a.memory else ExperienceMemory(seed=a.test_seed)
    exp_stats = {"loaded": len(memory.entries)}
    if a.memory:
        print(f"memory loaded: {len(memory.entries)} experiences")
        exp_items = [it for it in exp_items if it["id"] not in memory.known_ids]
    if exp_items and (not a.memory or a.exp_n > 0):
        exp_stats.update(experience_phase(tr, exp_items, memory, a.store_correct))
        mem_path = outdir / f"memory_{a.split}_{len(memory.entries)}exp_{stamp}.json"
        memory.save(str(mem_path))
        print(f"experience: {exp_stats} | memory: {len(memory.entries)} experiences -> {mem_path}")
    static = memory.central(a.k)

    conditions = [c for c in a.conditions.split(",") if c]
    rows = {c: [] for c in conditions}
    rows.update({"always_unknown": [], "majority": [], "random_label": []})
    majority = Counter(i["expected"] for i in test_items).most_common(1)[0][0]
    rng = random.Random(a.test_seed)
    hits = Counter()
    for k, it in enumerate(test_items, 1):
        base = {"id": it["id"], "expected": it["expected"], "run": 0}
        for c in conditions:
            try:
                ex = ([] if c == "none" else static if c == "static"
                      else memory.retrieve(it["text"], it["query"], a.k, "random" if c == "random" else "similar"))
                hits[c] += len(ex)
                r = tr.answer(it["text"], it["query"], examples_block(ex))
                rows[c].append({**base, "pred": r["verdict_strict"], "pred_full": r["verdict"], "n_examples": len(ex),
                                "rejections": len(r["rejections"]), "calls": r["calls"], "tokens": r["tokens_in"] + r["tokens_out"]})
            except Exception as e:  # noqa: BLE001
                rows[c].append({**base, "pred": "erreur", "error": f"{type(e).__name__}: {str(e)[:120]}"})
        rows["always_unknown"].append({**base, "pred": "unknown"})
        rows["majority"].append({**base, "pred": majority})
        rows["random_label"].append({**base, "pred": rng.choice(["yes", "no", "unknown"])})
        if k % 10 == 0:
            print(f"  test {k}/{len(test_items)}")

    summary = {"experience_sha": exp_sha, "test_sha": test_sha, "n_exp": len(exp_items), "n_test": len(test_items), "k": a.k,
               "memory_size": len(memory.entries), "experience": exp_stats, "model": a.model, "prompt_version": PROMPT_VERSION,
               "engine_version": ENGINE_VERSION, "modes": {}}
    base_pct = max(metrics(rows[b])["pct"] for b in ("always_unknown", "majority", "random_label"))
    print(f"\nbest trivial control: {base_pct}%")
    for c in conditions:
        mt = metrics(rows[c])
        summary["modes"][c] = mt
        print(f"{c:8} {mt['acc']:>3}/{mt['n']} = {mt['pct']:5}% | false yes {mt['false_yes']} | false no {mt['false_no']} "
              f"| false contradictory {mt['false_contradictory']} | inventions {mt['inventions']} | unknown {mt['unknown']} "
              f"| examples shown {hits[c]} | calls {sum(r.get('calls', 0) for r in rows[c])} | tokens {sum(r.get('tokens', 0) for r in rows[c])}")
    m = summary["modes"]
    asserted = lambda x: m[x]["false_yes"] + m[x]["false_no"] + m[x]["false_contradictory"]
    if all(c in m for c in ("none", "static")):
        c_exp = m["static"]["acc"] > m["none"]["acc"] and asserted("static") <= asserted("none")
        print(f"\nC-experience (static > none, asserted errors not above none): {'PASS' if c_exp else 'FAIL'} "
              f"(static {m['static']['acc']}, none {m['none']['acc']}; asserted {asserted('static')} vs {asserted('none')})")
        summary["criteria_experience"] = c_exp
    if all(c in m for c in ("none", "random", "learned")) and "static" in m:
        c_rel = (m["learned"]["acc"] > m["random"]["acc"] and m["learned"]["acc"] > m["static"]["acc"] and asserted("learned") <= asserted("none"))
        print(f"C-relevance (learned > random and learned > static, asserted errors not above none): {'PASS' if c_rel else 'FAIL'} "
              f"(learned {m['learned']['acc']}, random {m['random']['acc']}, static {m['static']['acc']})")
        summary["criteria_relevance"] = c_rel
    elif all(c in m for c in ("none", "random", "learned")):
        ok = (m["learned"]["acc"] > m["none"]["acc"] and m["learned"]["acc"] > m["random"]["acc"] and asserted("learned") <= asserted("none"))
        print(f"\nCRITERIA (run 1 form): learned > none and learned > random, asserted errors not above none -> {'PASS' if ok else 'FAIL'}")
        summary["criteria_pass"] = ok
    # paired comparison per item (sign test): the honest statistic for small differences
    from math import comb

    def paired(x, y):
        X = {r["id"] for r in rows[x] if r["pred"] == r["expected"]}; Y = {r["id"] for r in rows[y] if r["pred"] == r["expected"]}
        w, l = len(X - Y), len(Y - X); n = w + l
        pv = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(0, min(w, l) + 1)) / 2 ** n)
        return w, l, pv
    summary["paired"] = {}
    for x, y in (("learned", "none"), ("learned", "random"), ("learned", "static"), ("static", "none"), ("random", "none")):
        if x in rows and y in rows:
            w, l, pv = paired(x, y)
            summary["paired"][f"{x}_vs_{y}"] = {"only_first_right": w, "only_second_right": l, "sign_test_p": round(pv, 3)}
            print(f"paired {x:7} vs {y:7}: only {x} right {w:>2} | only {y} right {l:>2} | sign test p = {pv:.3f}")
    out = outdir / f"learning_{a.split}_test{len(test_items)}_{test_sha}_{stamp}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written: {out}")
    return summary


if __name__ == "__main__":
    main()
