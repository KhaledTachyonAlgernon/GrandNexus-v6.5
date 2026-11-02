#!/usr/bin/env python
"""
Differential test on .nx programs (written by bench_folio.py --save-programs, or by hand):
  reference engine  vs  clingo (if `pip install clingo`)  vs  home engine (ReasoningCore through TextReasoner).
Every disagreement is a bug somewhere, or a documented limit (goal-directed step, absent from the ASP export;
universal, disjunctive and formula questions: reference engine only; the home engine has no disjunction).

  python tools/diff_engines.py tests_lab/resultats/programmes/<set>/ [-v]

Written by the external reviewer (Fable 5.1, 2026-10-25); home_verdict() (the adapter to the in-house engine) added here.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grandnexus.nexus_logic import Engine, decide, parse, Lit   # noqa: E402
from grandnexus.nexus_logic.asp import clingo_verdict            # noqa: E402

_VARS = ("X", "Y", "Z")


def _home_lit(l: Lit, varmap: dict, uf) -> str:
    """Lit -> in-house string atom: 'not pred(a,b)' with no spaces; variables mapped to X/Y/Z."""
    args = []
    for a in l.atom.args:
        if isinstance(a, str) and a[:1].isupper():
            args.append(varmap[a])
        elif isinstance(a, str):
            args.append(uf.find(a))
        else:
            args.append(str(a).replace(".", "_").replace("-", "m"))
    return ("not " if l.neg else "") + f"{l.atom.pred}({','.join(args)})"


def home_verdict(program_text: str) -> str | None:
    """Push a parsed program through the in-house pipeline (TextReasoner grounding + ReasoningCore
    deduction + verdict). None when the program uses what the home engine cannot express:
    more than 3 variables in a rule, more than 3 arguments, or a question with variables."""
    import logging
    logging.disable(logging.CRITICAL)
    from grandnexus.cognition.reasoning.reasoning_core import ReasoningCore
    from grandnexus.cognition.reasoning.text_reasoner import TextReasoner, _args
    prog = parse(program_text)
    if not prog.questions or prog.questions[0][0] in ("universal", "disjunctive", "formula") or any(l.vars() for l in prog.questions[0]) \
            or len(prog.questions[0]) != 1 or prog.disjunctions or prog.drules:
        return None                                   # the home engine has no disjunction
    ref = Engine(prog)                       # for same()/typical_/let canonicalisation only
    uf = ref.uf
    for lit, _, _ in prog.facts:
        if len(lit.atom.args) > 3:
            return None
    facts, rules, kfacts, krules = [], [], [], []
    for lit, sid, kn in prog.facts:
        (kfacts if kn else facts).append(_home_lit(ref.canon(lit), {}, uf))
    for r in prog.rules:
        vs = sorted({a for l in r.body + (r.head,) for a in l.vars()})
        if len(vs) > 3 or any(len(l.atom.args) > 3 for l in r.body + (r.head,)):
            return None
        vm = dict(zip(vs, _VARS))
        conds = [_home_lit(ref.canon(b), vm, uf) for b in r.body]
        rule = {"if": conds[0] if len(conds) == 1 else conds, "then": _home_lit(ref.canon(r.head), vm, uf),
                "confidence": 0.95}
        (krules if r.knowledge else rules).append(rule)
    goal = _home_lit(ref.canon(prog.questions[0][0]), {}, uf)
    bundle = {"text": "? " + goal, "parser": "nx", "facts": facts, "rules": rules, "goals": [goal],
              "knowledge_facts": kfacts, "knowledge_rules": krules, "trace": [], "unparsed": [], "llm": None}
    tr = TextReasoner(reasoning_core=ReasoningCore(), engine="home")      # the in-house engine itself (v7.0: default is the reference)
    out = tr._decide(bundle, 8, None, 0.0)
    return out.get("answer_with_knowledge") or out["answer"]              # compare the knowledge-aware verdict, like the reference


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    verbose = "-v" in sys.argv
    files = sorted(pathlib.Path(sys.argv[1]).glob("*.nx"))
    n = dis_asp = dis_home = 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        prog = parse(text)
        v = decide(prog)
        asp, home = clingo_verdict(prog), home_verdict(text)
        flags = []
        if asp is not None and asp != v.answer:
            dis_asp += 1
            flags.append("≠clingo" + (" (goal-directed step, expected)" if v.note.startswith("goal-directed") else ""))
        if home is not None and home != v.answer:
            dis_home += 1
            flags.append("≠home")
        n += 1
        if flags or verbose:
            print(f"{f.name:42s} ref={v.answer:14s} clingo={asp or '-':14s} home={home or '-':14s} {' '.join(flags)}")
    print(f"\n{n} programs ; clingo disagreements : {dis_asp} ; home-engine disagreements : {dis_home}")


if __name__ == "__main__":
    main()
