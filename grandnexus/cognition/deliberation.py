"""
deliberation.py — Read several times, compare, re-read where the readings diverge.

Why (logbook 42): on FOLIO the engine is no longer the bottleneck (clingo agrees 100%); the
errors and the variance live in the single, one-shot reading of the text. The same sentence is
not always formalised the same way (items 905/1303/1412 were provable on one reading, not on the
next). Variance is turned into information: two readings that disagree reveal an ambiguity.

Roles (each one replaceable by an existing GrandNexus module later, then ablated):
  perception    several independent readings (translator calls, optionally diverse readers)
  working mem.  the readings kept side by side for the whole deliberation
  metacognition agreement between the engine verdicts; a per-reading confidence
  reflection    one targeted re-reading when the readings diverge: the reader sees its own
                previous programs and WHERE they differ, NOT the verdicts they led to (showing
                the verdicts would push towards whichever reading "decides")
  decision      policies computed from the same readings (single, vote, adaptive, reflective,
                unanimity), so every control comes from one run, at no extra cost

The engine is the reference engine nexus_logic (strict: premises only, FOLIO semantics).
"""
from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from grandnexus.llm_interface.translator_v5 import (SYSTEM_TRANSLATE, Translator, _user,
                                                    extract_program)
from grandnexus.nexus_logic import decide, parse

DECIDED = ("oui", "non", "contradictoire")

SYSTEM_TRANSLATE_ALT = SYSTEM_TRANSLATE + """

READING DISCIPLINE (alternative reader)
Before the program, inside the same fenced block, write one comment line per sentence of the text:
# s1: <how you formalise it, or why it cannot be expressed>
then the program. Comment lines are ignored by the engine."""

REFLECT_INSTRUCTIONS = """Several independent readings of the SAME text produced different programs.
They are shown below WITHOUT the answers they led to. Compare them statement by statement:
find the sentences of the text they formalise differently (or that one reading omitted), re-read
those sentences in the text, and decide what the text actually says.
Then output ONE complete program, the most faithful to the text, in the same format and rules
(anchors, no background knowledge unless obviously needed, # cannot express when needed)."""


@dataclass
class Reading:
    reader: str
    program: str
    verdict: str
    n_statements: int
    rejections: int
    cannot_express: int
    corrective_rounds: int
    proof: List[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Metacognitive self-estimate of a reading, from the reading itself (not from the truth)."""
        c = 1.0
        c -= 0.15 * min(self.rejections, 3)
        c -= 0.10 * min(self.cannot_express, 3)
        c -= 0.10 * self.corrective_rounds
        return max(0.05, round(c, 3))


def _verdict(program: str, text: str) -> Reading:
    prog = parse(program, source_text=text, require_anchor=True)
    v = decide(prog, use_knowledge=False)
    return Reading("", program, v.answer, prog.n_statements, len(prog.rejections),
                   program.count("# cannot express"), 0, v.proof)


def _statements(program: str) -> List[str]:
    out = []
    for line in program.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("```"):
            continue
        out.append(re.sub(r'\s*@\s*"[^"]*"\s*$', "", s))          # compare the logic, not the quotes
    return out


def divergences(programs: Sequence[str]) -> Dict[str, Any]:
    """Statements present in some readings and absent from others; differing questions."""
    sets = [set(_statements(p)) for p in programs]
    union = set().union(*sets)
    partial = sorted(s for s in union if sum(s in st for st in sets) < len(sets))
    questions = sorted({s for st in sets for s in st if s.startswith("?")})
    return {"partial_statements": partial, "questions": questions, "n_union": len(union)}


class Deliberation:
    """k readings -> engine verdicts -> agreement -> (if needed) one reflective re-reading."""

    def __init__(self, readers: Sequence[Translator], reader_names: Sequence[str],
                 systems: Optional[Sequence[str]] = None, reflector: Optional[Translator] = None,
                 seed: int = 0) -> None:
        assert len(readers) == len(reader_names)
        self.readers, self.names = list(readers), list(reader_names)
        self.systems = list(systems) if systems else [SYSTEM_TRANSLATE] * len(readers)
        self.reflector = reflector or readers[0]
        self.rng = random.Random(seed)

    # ---- perception: one reading
    def read(self, i: int, text: str, query: str) -> Reading:
        tr, system = self.readers[i], self.systems[i]
        raw = tr._call(system, _user(text, query), "translate")
        program = extract_program(raw)
        prog = parse(program, source_text=text, require_anchor=tr.require_anchor)
        rounds = 0
        while prog.rejections and rounds < tr.corrective_rounds:
            rounds += 1
            msg = (_user(text, query) + "\n\nYOUR PROGRAM:\n```nx\n" + program + "```\n\n"
                   "The checker rejected these lines:\n" + "\n".join(str(r) for r in prog.rejections)
                   + "\n\nReturn the complete corrected program (all lines) in the same format. "
                     "Keep the accepted lines; fix or remove only the rejected ones.")
            program = extract_program(tr._call(system, msg, "correct"))
            prog = parse(program, source_text=text, require_anchor=tr.require_anchor)
        r = _verdict(program, text)
        r.reader, r.corrective_rounds = self.names[i], rounds
        return r

    # ---- reflection: one targeted re-reading
    def reflect(self, text: str, query: str, readings: Sequence[Reading]) -> Reading:
        progs = [r.program for r in readings]
        div = divergences(progs)
        shown = "\n\n".join(f"READING {k + 1}:\n```nx\n{p.strip()}\n```" for k, p in enumerate(progs))
        msg = (_user(text, query) + "\n\n" + REFLECT_INSTRUCTIONS + "\n\n" + shown
               + "\n\nStatements that not all readings share:\n"
               + "\n".join(f"- {s}" for s in div["partial_statements"][:25]))
        program = extract_program(self.reflector._call(SYSTEM_TRANSLATE, msg, "reflect"))
        r = _verdict(program, text)
        r.reader = "reflection"
        return r

    # ---- the whole deliberation for one item (fixed design: every policy computable)
    def run(self, text: str, query: str) -> Dict[str, Any]:
        readings = [self.read(i, text, query) for i in range(len(self.readers))]
        verdicts = [r.verdict for r in readings]
        unanimous = len(set(verdicts)) == 1
        reflection = None if unanimous else self.reflect(text, query, readings)
        return {"readings": readings, "reflection": reflection, "policies": self.policies(readings, reflection)}

    # ---- decision policies, all from the same readings
    def policies(self, readings: Sequence[Reading], reflection: Optional[Reading]) -> Dict[str, str]:
        v = [r.verdict for r in readings]
        top, n_top = Counter(v).most_common(1)[0]
        majority = top if n_top > len(v) / 2 else "inconnu"
        weights: Dict[str, float] = {}
        for r in readings:
            weights[r.verdict] = weights.get(r.verdict, 0.0) + r.confidence
        weighted = max(weights.items(), key=lambda kv: kv[1])[0]
        adaptive = v[0] if len(v) >= 2 and v[0] == v[1] else majority
        if reflection is None:
            reflective = v[0]
        else:
            # the re-reading counts as one more voice; it decides only together with an earlier reading
            reflective = reflection.verdict if reflection.verdict in v else "inconnu"
        return {
            "single": v[0],                                  # = the current system (one reading)
            "random": self.rng.choice(v),                    # control: any one reading
            "vote": majority,                                # control: plain majority, no reflection
            "vote_confidence": weighted,                     # metacognition: confidence-weighted vote
            "adaptive": adaptive,                            # ruminate only when the first two disagree
            "reflective": reflective,                        # vote + targeted re-reading on divergence
            "unanimity": v[0] if len(set(v)) == 1 else "inconnu",   # caution: answer only if all agree
        }
