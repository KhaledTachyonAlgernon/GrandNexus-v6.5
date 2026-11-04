"""
experience.py - The loop that learns (logbook 18: three passes gave 4, 4, 4 because no channel turned an
error into an improvement; logbook 49-50: the remaining errors are translation errors, not engine errors).

Loop:  read -> program -> verdict -> feedback (the expected label only) -> reflection -> verified correction
       -> experience memory -> retrieved as examples for the next readings.

No rule is written by hand. The only feedback the system receives is "wrong" with the expected label; the
correction is produced by the reader itself, re-reading its own program, and is KEPT ONLY IF the engine then
reaches the expected verdict (a verified experience). At reading time, the k most similar past experiences
(TF-IDF over words and bigrams, no API call) are shown as examples. Controls in tools/bench_learning.py:
no memory (the current system) and the same memory retrieved at random (is it the relevance that helps?).
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    words = _WORD.findall(text.lower())
    return words + [a + "_" + b for a, b in zip(words, words[1:])]


@dataclass
class Experience:
    text: str                       # the premises that were read
    query: str                      # the question or statement to verify
    program: str                    # the verified (corrected) program
    expected: str                   # the verdict the engine reaches with it (= the expected label)
    wrong_verdict: str = ""         # the verdict the first reading led to
    changed: List[str] = field(default_factory=list)   # statements that differ from the first reading
    source_id: str = ""


class ExperienceMemory:
    """Verified corrections, retrieved by lexical similarity (TF-IDF cosine over words + bigrams)."""

    def __init__(self, entries: Optional[Sequence[Experience]] = None, seed: int = 0) -> None:
        self.entries: List[Experience] = list(entries or [])
        self.rng = random.Random(seed)
        self._df: Counter = Counter()
        self._vecs: List[Dict[str, float]] = []
        self._rebuild()

    # ---- storage
    def add(self, e: Experience) -> None:
        self.entries.append(e)
        self._rebuild()

    def _rebuild(self) -> None:
        docs = [Counter(_tokens(e.text + " " + e.query)) for e in self.entries]
        self._df = Counter()
        for d in docs:
            self._df.update(d.keys())
        n = max(1, len(docs))
        self._vecs = []
        for d in docs:
            v = {t: (1 + math.log(c)) * (1 + math.log((n + 1) / (self._df[t] + 1))) for t, c in d.items()}   # smoothed idf (sklearn style)
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            self._vecs.append({t: x / norm for t, x in v.items()})

    # ---- retrieval
    def retrieve(self, text: str, query: str, k: int = 3, mode: str = "similar") -> List[Experience]:
        if not self.entries or k <= 0:
            return []
        if mode == "random":
            return self.rng.sample(self.entries, min(k, len(self.entries)))
        d = Counter(_tokens(text + " " + query))
        n = len(self.entries)
        q = {t: (1 + math.log(c)) * (1 + math.log((n + 1) / (self._df.get(t, 0) + 1))) for t, c in d.items()}
        norm = math.sqrt(sum(x * x for x in q.values())) or 1.0
        scored = [(sum(q[t] / norm * v.get(t, 0.0) for t in q), i) for i, v in enumerate(self._vecs)]
        scored.sort(reverse=True)
        return [self.entries[i] for s, i in scored[:k] if s > 0]

    def central(self, k: int = 3) -> List[Experience]:
        """The k experiences most similar to all the others (medoids): a static, automatically chosen set,
        shown to every reading. Control: if static == learned, retrieval by relevance adds nothing."""
        if not self.entries:
            return []
        n = len(self._vecs)
        score = []
        for i, v in enumerate(self._vecs):
            tot = sum(sum(v.get(t, 0.0) * w.get(t, 0.0) for t in v) for j, w in enumerate(self._vecs) if j != i)
            score.append((tot / max(1, n - 1), i))
        score.sort(reverse=True)
        return [self.entries[i] for _, i in score[:k]]

    @property
    def known_ids(self) -> set:
        return {e.source_id for e in self.entries if e.source_id}

    # ---- persistence
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(e) for e in self.entries], f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str, seed: int = 0) -> "ExperienceMemory":
        with open(path, encoding="utf-8") as f:
            return cls([Experience(**d) for d in json.load(f)], seed=seed)


def examples_block(examples: Sequence[Experience]) -> str:
    """The examples as shown to the reader: past readings whose program was verified by the engine."""
    if not examples:
        return ""
    parts = ["EXAMPLES OF VERIFIED PAST READINGS (same language and conventions; adapt, do not copy):"]
    for i, e in enumerate(examples, 1):
        parts.append(f"\n--- example {i} ---\nTEXT:\n{e.text}\n\nQUESTION:\n{e.query}\n\nPROGRAM:\n```nx\n{e.program.strip()}\n```")
        if e.changed:
            parts.append("What the first reading got wrong: " + "; ".join(e.changed[:4]))
    return "\n".join(parts) + "\n\n"


def statements_diff(before: str, after: str) -> List[str]:
    """Statements of `after` absent from `before` (anchors removed), to tell the next reader what changed."""
    def stmts(p: str) -> List[str]:
        out = []
        for line in p.splitlines():
            s = line.strip()
            if s and not s.startswith("#") and not s.startswith("```"):
                out.append(re.sub(r'\s*@\s*"[^"]*"\s*$', "", s))
        return out
    b = set(stmts(before))
    return [s for s in stmts(after) if s not in b]
