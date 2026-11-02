"""
bridge.py — From the in-house atom strings (TextReasoner / PremiseExtractor / JSON translator) to a
nexus_logic Program, so that every GrandNexus pathway is decided by the reference engine (v7.0).

In-house format:  "not pred(a,b)"  |  rules {"if": "atom" or ["atom", ...], "then": "atom"}
Variables are the uppercase names X, Y, Z (nexus_logic treats any Uppercase-initial term as a variable).
Constants may carry accents (the grammar produces "mouillé"): Lit objects are built directly, so the
ascii-only grammar of the text language does not apply here.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .core import Atom, Lit, Program, Rule, Verdict, decide

_ATOM = re.compile(r"^(not )?([^\W\d]\w*)\(([^()]*)\)$", re.UNICODE)


def lit(atom: str) -> Optional[Lit]:
    m = _ATOM.match(atom.strip())
    if not m:
        return None
    args = tuple(a.strip() for a in m.group(3).split(",") if a.strip())
    return Lit(Atom(m.group(2), args), bool(m.group(1)))


def program_from_atoms(facts: Iterable[str], rules: Iterable[Dict[str, Any]], goals: Iterable[str],
                       knowledge_facts: Iterable[str] = (), knowledge_rules: Iterable[Dict[str, Any]] = (),
                       same: Iterable[Tuple[str, str]] = ()) -> Program:
    prog = Program()
    n = 0

    def sid() -> str:
        nonlocal n
        n += 1
        return f"s{n}"

    for kn, fs in ((False, facts), (True, knowledge_facts)):
        for f in fs:
            l = lit(f)
            if l is not None and not l.vars():
                prog.facts.append((l, sid(), kn))
    for kn, rs in ((False, rules), (True, knowledge_rules)):
        for r in rs:
            conds = r["if"] if isinstance(r["if"], list) else [r["if"]]
            body = tuple(x for x in (lit(c) for c in conds) if x is not None)
            head = lit(r["then"])
            if head is None or len(body) != len(conds) or head in body:
                continue
            rule = Rule(body, head, sid(), kn)
            if rule.is_safe():
                prog.rules.append(rule)
    for a, b in same:
        prog.sames.append((a, b))
    gs = [g for g in (lit(x) for x in goals) if g is not None]
    if gs:
        prog.questions.append((gs[0],))                     # one question: the first goal
    return prog


def verdict_from_atoms(facts, rules, goals, knowledge_facts=(), knowledge_rules=(),
                       use_knowledge: bool = True) -> Verdict:
    return decide(program_from_atoms(facts, rules, goals, knowledge_facts, knowledge_rules),
                  use_knowledge=use_knowledge)
