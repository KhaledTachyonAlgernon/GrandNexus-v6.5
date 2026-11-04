"""
Claude as an anchored translator - prompt v5.x, derived from LANGUAGE_SPEC (one source of truth shared with the parser).
Modes: answer() = translation + engine (verdict with and without background knowledge, proof, receipts);
direct() = Claude alone (control). FakeClient = fake SDK for offline tests.
Every result records the id of the model actually used.

Written by the external reviewer (Fable 5.1, 2026-10-25); DEFAULT_MODEL set to the measured model (claude-sonnet-5-5),
prompt evolved to v5.4 (universal questions, disjunctions, compound questions).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from grandnexus.nexus_logic import ENGINE_VERSION, LANGUAGE_SPEC, Program, decide, parse

PROMPT_VERSION = "v5.4"  # 5.3: disjunction example ; 5.4: compound questions as formulas, "doesn't B or C" convention
DEFAULT_MODEL = os.environ.get("GN_MODEL", "claude-sonnet-5-5")
VERDICTS = ("yes", "no", "unknown", "contradictory")

SYSTEM_TRANSLATE = """You translate a text and a question into a program for a small deductive engine.
The engine knows ONLY what you write. It cannot guess, has no common sense, and answers "unknown" for
anything it cannot derive. Your job is fidelity to the text, not helpfulness.

LANGUAGE
""" + LANGUAGE_SPEC + """
CONVENTIONS
1. Entities, categories and relations are lowercase ascii constants/predicates (no accents, no spaces).
   One constant per entity, reused on every line. Relations are predicates with arguments:
   taller(john, peter)  -- never a property like is(john, taller_than_peter).
   Category membership: man(socrates). Property: tall(john). Subjectless proposition: prop(rain) or rain.
2. Quantifiers: "all/every X are Y"  ->  x(V) => y(V).      "no X is Y"  ->  x(V) => not y(V).
   "some X are Y"  ->  a fresh constant: x(some_x_1). y(some_x_1).   (never a rule)
   "only X are Y"  ->  y(V) => x(V).     "X if and only if Y"  ->  two rules.
   "if A and B then C"  ->  a(..) & b(..) => c(..).  Up to 3 conditions; split longer rules.
   DISJUNCTION is supported: never write "# cannot express" for an "or" sentence.
   "A or B" / "either A or B (or both)"  ->  a(..) | b(..).      "all X are Y or Z"  ->  x(V) => y(V) | z(V).
   "either A or B but not both"  ->  a | b.  a => not b.  b => not a.
   "neither A nor B"  ->  not a.  not b.     "A unless B"  ->  not b => a.
   "A doesn't B or C"  ->  not b(a) | not c(a).   (FOLIO convention: at least one of the two is false)
   "neither B nor C"  ->  not b(a).  not c(a).   "if P then neither B nor C"  ->  p(a) => not b(a).  p(a) => not c(a).
   a compound question is written as a formula: "is A and B?"  ->  ? a(..) & b(..).
   "either (A and B) or (neither A nor B)"  ->  ? (a(..) & b(..)) | (not a(..) & not b(..)).
   Never define a helper predicate for a compound question: write the formula.
   Example — text: "Every pet is a cat or a dog. Rex is a pet. Rex is not a cat." question: "Is Rex a dog?"
   ```nx
   pet(V) => cat(V) | dog(V). @ "Every pet is a cat or a dog"
   pet(rex). @ "Rex is a pet"
   not cat(rex). @ "Rex is not a cat"
   ? dog(rex).
   ```
   Every variable of a conclusion must appear in a condition. Transitivity needs two conditions:
   in(X, Y) & in(Y, Z) => in(X, Z).
3. Negation: not tall(john).   "A does not B"  ->  not b(a).
4. Two names for one entity: same(a, b).   "the typical X" / "an X in general"  ->  the category constant x.
5. States that change over time: add a moment as last argument: full(reservoir, t1). empty(reservoir, t2).
   Order moments with before(t1, t2). Never assume a state persists; if the text implies persistence,
   write the rule explicitly.
6. Numbers: let n = 3 - 1 + 2.  then  has(john, apples, n).   "How many ..." questions use a variable:
   ? has(john, apples, N).
7. The question: exactly one line starting with "?". For "is it true that S" translate S itself.
   "A and B?" -> ? a(..) & b(..).
   A universal statement to verify keeps its quantifier: "All X are Y" -> ? x(V) => y(V).
   "No X is Y" / "X is never Y" / "X was not Y to any Z" -> ? x(V) => not y(V).
   Never turn a universal or negative statement into an existential positive question.
8. Anchoring: every statement taken from the text ends with @ "exact quote of the words that justify it"
   (copy the words exactly, no paraphrase). A statement you cannot anchor must not be written.
9. Background knowledge (not in the text) is allowed ONLY if needed to answer, ONLY as lines starting
   with "*", and must be obviously true (e.g. * man(X) => human(X).). Never add knowledge facts about
   the specific entities of the text.
10. If a sentence cannot be expressed, write a comment line: # cannot express: "exact quote"
    Do not approximate it with something the text does not say.

OUTPUT
One fenced block ```nx ... ``` with one statement per line, and nothing else."""

SYSTEM_DIRECT = """You answer a question from a text using only what the text says.
Reply with exactly one token: yes, no, unknown (not derivable from the text), contradictory (the text
contradicts itself), or a number for "how many" questions. No explanation."""

_FENCE = re.compile(r"```(?:nx)?[ \t]*\n(.*?)```", re.S)


def extract_program(reply: str) -> str:
    m = _FENCE.search(reply)
    return m.group(1) if m else reply


def _user(text: str, query: str) -> str:
    return f"TEXT:\n{text}\n\nQUESTION (or statement to verify):\n{query}"


def _default_client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        try:
            from google.colab import userdata  # type: ignore
            key = userdata.get("ANTHROPIC_API_KEY")
        except Exception:
            key = None
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set (env var or Colab secret)")
    import anthropic
    return anthropic.Anthropic(api_key=key)


class FakeClient:
    """Offline stand-in for the SDK: returns the canned replies in order (tests without API, zero cost)."""
    def __init__(self, replies: List[str]) -> None:
        self._replies = list(replies)
        self.messages = self

    def create(self, **kw) -> Any:
        if not self._replies:
            raise RuntimeError("FakeClient: no more canned replies")
        return SimpleNamespace(id=f"fake-{len(self._replies)}", model="fake",
                               content=[SimpleNamespace(type="text", text=self._replies.pop(0))],
                               usage=SimpleNamespace(input_tokens=0, output_tokens=0))


@dataclass
class Receipt:
    id: str
    model: str
    tokens_in: int
    tokens_out: int
    role: str


class Translator:
    def __init__(self, client=None, model: str = DEFAULT_MODEL, corrective_rounds: int = 1,
                 require_anchor: bool = True, max_tokens: int = 2000) -> None:
        self.client = client or _default_client()
        self.model, self.corrective_rounds, self.require_anchor, self.max_tokens = model, corrective_rounds, require_anchor, max_tokens
        self.receipts: List[Receipt] = []

    def _call(self, system: str, user: str, role: str) -> str:
        resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                           messages=[{"role": "user", "content": user}])
        u = getattr(resp, "usage", None)
        self.receipts.append(Receipt(getattr(resp, "id", "?"), getattr(resp, "model", self.model),
                                     getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0), role))
        return "".join(getattr(b, "text", "") for b in resp.content)

    def _tally(self, n0: int) -> Dict[str, Any]:
        rec = self.receipts[n0:]
        return {"calls": len(rec), "tokens_in": sum(r.tokens_in for r in rec), "tokens_out": sum(r.tokens_out for r in rec),
                "message_ids": [r.id for r in rec], "model": rec[0].model if rec else self.model}

    def translate(self, text: str, query: str, examples: str = "") -> Dict[str, Any]:
        """examples: an optional block of verified past readings (grandnexus.cognition.experience.examples_block),
        placed before the text; this is how the learning loop influences a reading, with no prompt change."""
        program = extract_program(self._call(SYSTEM_TRANSLATE, examples + _user(text, query), "translate"))
        prog = parse(program, source_text=text, require_anchor=self.require_anchor)
        rounds = 0
        while prog.rejections and rounds < self.corrective_rounds:      # tour correctif (v6.7.0), raisons incluses
            rounds += 1
            msg = (_user(text, query) + "\n\nYOUR PROGRAM:\n```nx\n" + program + "```\n\n"
                   "The checker rejected these lines:\n" + "\n".join(str(r) for r in prog.rejections)
                   + "\n\nReturn the complete corrected program (all lines) in the same format. "
                     "Keep the accepted lines; fix or remove only the rejected ones.")
            program = extract_program(self._call(SYSTEM_TRANSLATE, msg, "correct"))
            prog = parse(program, source_text=text, require_anchor=self.require_anchor)
        return {"program": program, "parsed": prog, "corrective_rounds": rounds}

    def answer(self, text: str, query: str, examples: str = "") -> Dict[str, Any]:
        n0 = len(self.receipts)
        t = self.translate(text, query, examples)
        prog: Program = t["parsed"]
        full = decide(prog, use_knowledge=True)
        strict = decide(prog, use_knowledge=False)
        return {"answer": full.short(), "answer_strict": strict.short(),
                "verdict": full.answer, "verdict_strict": strict.answer,
                "depends_on_knowledge": full.depends_on_knowledge, "bindings": full.bindings,
                "proof": full.proof, "note": full.note, "program": t["program"],
                "rejections": [str(r) for r in prog.rejections], "corrective_rounds": t["corrective_rounds"],
                "n_statements": prog.n_statements, "prompt_version": PROMPT_VERSION,
                "engine_version": ENGINE_VERSION, **self._tally(n0)}

    def reflect_correction(self, text: str, query: str, program: str, got: str, expected: str) -> Dict[str, Any]:
        """Learning loop: the reader is told only that its reading led to `got` while the expected verdict is
        `expected`; it re-reads its own program against the text and returns a corrected full program. The caller
        keeps it only if the engine then reaches the expected verdict (see tools/bench_learning.py)."""
        n0 = len(self.receipts)
        msg = (_user(text, query) + "\n\nYOUR PROGRAM:\n```nx\n" + program.strip() + "\n```\n\n"
               f"The engine answered '{got}' from this program, but the expected answer is '{expected}'.\n"
               "The engine is sound: the difference comes from the reading. Re-read the text sentence by sentence and find what the\n"
               "program misses or misstates (a dropped qualifier, a wrong quantifier, 'A or B' read as two negations, a missing\n"
               "statement, an unfaithful question). Do NOT add anything the text does not say and do not add background\n"
               "knowledge just to reach the expected answer. Output the complete corrected program in the same format.")
        corrected = extract_program(self._call(SYSTEM_TRANSLATE, msg, "reflect"))
        prog = parse(corrected, source_text=text, require_anchor=self.require_anchor)
        strict = decide(prog, use_knowledge=False)
        return {"program": corrected, "verdict_strict": strict.answer, "rejections": [str(r) for r in prog.rejections],
                **self._tally(n0)}

    def direct(self, text: str, query: str) -> Dict[str, Any]:
        """Control: Claude alone, no engine."""
        n0 = len(self.receipts)
        reply = self._call(SYSTEM_DIRECT, _user(text, query), "direct").strip().lower()
        word = next((v for v in VERDICTS if re.search(rf"\b{v}\b", reply)), None)
        num = re.search(r"-?\d+(?:\.\d+)?", reply)
        return {"answer": word or (num.group(0) if num else "unknown"), "raw": reply[:200], **self._tally(n0)}
