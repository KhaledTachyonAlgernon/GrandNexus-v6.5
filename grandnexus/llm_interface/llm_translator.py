"""
llm_translator.py — LLM-backed language layer, used strictly as a TRANSLATOR.

Principle: "language translates, the engine decides".
  * text  -> facts / rules / goals  (then ReasoningCore deduces)
  * text  -> arithmetic expression  (then sympy computes)
The LLM never gives the final answer (except in `answer_directly`, which exists
only as a control baseline for experiments).

Anti-hallucination guard: every text-derived item must quote the span of the
text it comes from; unanchored items are rejected and reported. Background
knowledge (common sense, word meaning, persistence over time) is accepted only
in a separate, flagged list, so callers can tell when an answer depends on it.
"""

import hashlib
import json
import logging
import re
import time
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Tuple

from grandnexus.cognition.contracts import (
    CognitiveMode, CognitiveModuleBase, CognitiveProcessType, cognitive_capability,
)

log = logging.getLogger("GrandNexus.LLMTranslator")

DEFAULT_MODEL = "claude-sonnet-5-5"

PREMISES_SYSTEM = """You translate a short text (French or English) into logic for a symbolic reasoner.
You do NOT answer the question.

Output ONLY one JSON object, no prose, no code fence:
{"facts":[{"p":"<atom>","quote":"<exact span copied from the text>"}],
 "rules":[{"if":"<atom>","then":"<atom>","quote":"<exact span>"}],
 "goals":[{"p":"<atom>","quote":"<exact span of the question>"}],
 "knowledge":[{"p":"<atom>","why":"<short reason>"} or {"if":"<atom>","then":"<atom>","why":"<short reason>"}]}

Atoms (lowercase, no spaces):
- is(x,y)    x is a y / x is y (category or property)
- can(x,y)   x can y
- in(x,y)    containment of any kind: inside, in, part of, located in, at, left at home...
- verb(x,y) or verb(x) for any other verb, verb in base form: eat(luna,apple), come(bob,party), crash(server)
- negation: prefix "not " -> "not is(x,y)"
Constants: singular, snake_case, no articles. The SAME entity always gets the SAME constant in
facts, rules and goals ("le frère de Bob" -> brother_of_bob everywhere).
Universal statements use the variable X (and Y if needed):
  "All men are mortal"      -> {"if":"is(X,man)","then":"is(X,mortal)"}
  "No mammal has feathers"  -> {"if":"is(X,mammal)","then":"not have(X,feather)"}
  "Cats chase mice"          -> {"if":"is(X,cat)","then":"chase(X,mouse)"}
A conditional "if A then B" is one rule; a sentence with two consequences gives two rules.
"some / certains / quelques / many / most" are NOT universal: produce no rule for them.
Co-occurrence or time ("when the sun sets, the lights turn on") is a rule, never a cause:
never invent cause(...) atoms.
Every fact, rule and goal MUST quote the exact span of the text it comes from.
Anything not literally stated (common sense, word meanings like "a female cat is a cat",
"an apple is not meat", persistence over time) goes ONLY in "knowledge", with a short "why".
Never answer the question. Never add facts just to make the question answerable."""

EQUATION_SYSTEM = """You translate a word problem (French or English) into ONE arithmetic expression whose
value answers the question. Do NOT compute it.
Output ONLY one JSON object, no prose, no code fence:
{"expression":"<digits, + - * / ( ) . and spaces only>",
 "quantities":[{"value":<number>,"quote":"<exact span of the text containing this number>"}],
 "unit":"<unit or empty>"}
Every number used in the expression must appear in "quantities" with its quote.
If the text does not ask for a quantity, output {"expression":"","quantities":[],"unit":""}."""

DIRECT_SYSTEM = """Answer the question using only the text and ordinary common sense.
Reply with exactly ONE token: oui, non, inconnu, contradictoire — or a number if the question asks
for a quantity. No explanation."""

NUMBER_WORDS = {
    "zero": 0, "zéro": 0, "one": 1, "un": 1, "une": 1, "two": 2, "deux": 2, "three": 3, "trois": 3,
    "four": 4, "quatre": 4, "five": 5, "cinq": 5, "six": 6, "seven": 7, "sept": 7, "eight": 8, "huit": 8,
    "nine": 9, "neuf": 9, "ten": 10, "dix": 10, "eleven": 11, "onze": 11, "twelve": 12, "douze": 12,
    "a dozen": 12, "une douzaine": 12, "twenty": 20, "vingt": 20, "half": 0.5, "moitié": 0.5,
}

ATOM_RE = re.compile(r"^(not )?[^\W\d][\w]*\((?:[\w]+)(?:,[\w]+)?\)$", re.UNICODE)
EXPR_RE = re.compile(r"^[0-9+\-*/(). ]+$")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower().replace("’", "'").replace("`", "'")
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def anchored(quote: str, text: str, min_overlap: float = 0.8) -> bool:
    """True if `quote` is (almost) literally present in `text`."""
    q, t = _norm(quote), _norm(text)
    if not q:
        return False
    if q in t:
        return True
    qt, tt = q.split(), set(t.split())
    return len(qt) > 0 and sum(w in tt for w in qt) / len(qt) >= min_overlap


def extract_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None


class LLMTranslator(CognitiveModuleBase):
    """Translate text into premises or arithmetic for the symbolic engines."""

    def __init__(self, llm_engine: Any = None, call: Optional[Callable[[str, str], str]] = None,
                 model: str = DEFAULT_MODEL, max_tokens: int = 1500, use_knowledge: bool = True,
                 nexus_core: Any = None, **kwargs):
        if llm_engine is None and call is None:
            raise ValueError("LLMTranslator needs an llm_engine or a call(system, prompt) function")
        self.llm_engine = llm_engine
        self._call_fn = call
        self.model = model
        self.max_tokens = max_tokens
        self.use_knowledge = use_knowledge
        self.nexus_core = nexus_core
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.stats = {"calls": 0, "cache_hits": 0, "failures": 0, "rejected_items": 0, "seconds": 0.0}

    # -- transport ------------------------------------------------------------
    def _call(self, system: str, prompt: str) -> str:
        t0 = time.time()
        self.stats["calls"] += 1
        try:
            if self._call_fn is not None:
                return self._call_fn(system, prompt)
            from grandnexus.llm_interface.llm_engine import LLMRequest
            resp = self.llm_engine.complete(LLMRequest(
                prompt=prompt, temperature=0.0, max_tokens=self.max_tokens, request_type="chat",
                context={"model": self.model, "system": system}))
            return getattr(resp, "text", "") or ""
        finally:
            self.stats["seconds"] += time.time() - t0

    def _cached(self, task: str, text: str, system: str, prompt: str) -> Tuple[Optional[Dict[str, Any]], str]:
        key = hashlib.sha256(f"{task}|{self.model}|{text}".encode()).hexdigest()
        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key]["parsed"], self.cache[key]["raw"]
        try:
            raw = self._call(system, prompt)
        except Exception as e:
            self.stats["failures"] += 1
            log.warning(f"LLM call failed ({task}): {type(e).__name__}: {e}")
            return None, f"ERROR: {e}"
        parsed = extract_json(raw)
        if parsed is None and task != "direct":
            self.stats["failures"] += 1
        self.cache[key] = {"parsed": parsed, "raw": raw}
        return parsed, raw

    # -- text -> premises -----------------------------------------------------
    def to_premises(self, text: str) -> Dict[str, Any]:
        parsed, raw = self._cached("premises", text, PREMISES_SYSTEM, f"TEXT:\n{text}")
        out = {"facts": [], "rules": [], "goals": [], "knowledge_facts": [], "knowledge_rules": [],
               "rejected": [], "ok": parsed is not None, "raw": raw}
        if parsed is None:
            return out

        def keep(item: Dict[str, Any], atoms: List[str], kind: str, need_quote: bool) -> bool:
            bad_atoms = [a for a in atoms if not isinstance(a, str) or not ATOM_RE.match(a.strip())]
            if bad_atoms:
                out["rejected"].append({"kind": kind, "item": item, "reason": f"malformed atom {bad_atoms}"})
                return False
            if need_quote and not anchored(str(item.get("quote", "")), text):
                out["rejected"].append({"kind": kind, "item": item, "reason": "quote not found in text"})
                return False
            return True

        for f in parsed.get("facts", []) or []:
            if isinstance(f, dict) and keep(f, [f.get("p")], "fact", True):
                out["facts"].append(f["p"].strip())
        for r in parsed.get("rules", []) or []:
            if isinstance(r, dict) and keep(r, [r.get("if"), r.get("then")], "rule", True):
                out["rules"].append({"if": r["if"].strip(), "then": r["then"].strip(), "confidence": 0.95,
                                     "source": "llm:" + str(r.get("quote", ""))[:80]})
        for g in parsed.get("goals", []) or []:
            if isinstance(g, dict) and keep(g, [g.get("p")], "goal", True):
                out["goals"].append(g["p"].strip())
        for k in parsed.get("knowledge", []) or []:
            if not isinstance(k, dict):
                continue
            if "p" in k and keep(k, [k.get("p")], "knowledge", False):
                out["knowledge_facts"].append(k["p"].strip())
            elif "if" in k and keep(k, [k.get("if"), k.get("then")], "knowledge", False):
                out["knowledge_rules"].append({"if": k["if"].strip(), "then": k["then"].strip(),
                                               "confidence": 0.8, "source": "llm-knowledge:" + str(k.get("why", ""))[:80]})
        self.stats["rejected_items"] += len(out["rejected"])
        return out

    # -- text -> arithmetic ---------------------------------------------------
    def to_equation(self, text: str) -> Dict[str, Any]:
        parsed, raw = self._cached("equation", text, EQUATION_SYSTEM, f"TEXT:\n{text}")
        out = {"expression": "", "value": None, "unit": "", "rejected": [], "ok": False, "raw": raw}
        if not parsed:
            return out
        expr = str(parsed.get("expression", "")).strip()
        out["unit"] = str(parsed.get("unit", "") or "")
        if not expr:
            return out
        if not EXPR_RE.match(expr):
            out["rejected"].append({"reason": f"expression is not plain arithmetic: {expr!r}"})
            return out
        quoted = []
        for q in parsed.get("quantities", []) or []:
            try:
                v = float(q.get("value"))
            except (TypeError, ValueError):
                out["rejected"].append({"item": q, "reason": "non-numeric value"})
                continue
            quote = str(q.get("quote", ""))
            words = _norm(quote)
            present = (re.search(rf"(?<![\d.]){re.escape(str(q.get('value')))}(?![\d.])", quote) is not None
                       or any(n == v and re.search(rf"\b{re.escape(w)}\b", words) for w, n in NUMBER_WORDS.items())
                       or re.search(rf"(?<![\d.]){int(v) if v.is_integer() else v}(?![\d.])", quote) is not None)
            if not anchored(quote, text) or not present:
                out["rejected"].append({"item": q, "reason": "quantity not anchored in the text"})
                continue
            quoted.append(v)
        used = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", expr)]
        missing = [u for u in used if u not in quoted]
        if missing:
            out["rejected"].append({"reason": f"numbers {missing} used but not anchored"})
            self.stats["rejected_items"] += len(out["rejected"])
            return out
        try:
            import sympy
            val = sympy.nsimplify(sympy.sympify(expr, rational=True))
            value = int(val) if val.is_integer else float(val)
        except Exception as e:
            out["rejected"].append({"reason": f"evaluation failed: {e}"})
            return out
        out.update(expression=expr, value=value, ok=True)
        return out

    # -- control baseline only -----------------------------------------------
    def answer_directly(self, text: str) -> str:
        _, raw = self._cached("direct", text, DIRECT_SYSTEM, f"TEXT:\n{text}")
        token = _norm(raw).split()[:1]
        token = token[0] if token else ""
        mapping = {"yes": "oui", "no": "non", "unknown": "inconnu", "contradictory": "contradictoire"}
        return mapping.get(token, token)

    # -- capabilities ---------------------------------------------------------
    @cognitive_capability(capability_type=CognitiveProcessType.PERCEPTION,
                          description="Text -> anchored premises (LLM translator)", priority=53,
                          compatible_modes={CognitiveMode.DELIBERATIVE, CognitiveMode.REACTIVE})
    def translate_to_premises(self, request, context, query, **kwargs) -> Dict[str, Any]:
        out = self.to_premises(str(query))
        out["confidence"] = 0.0 if not out["ok"] else (0.9 if not out["rejected"] else 0.6)
        return out

    @cognitive_capability(capability_type=CognitiveProcessType.PERCEPTION,
                          description="Word problem -> anchored arithmetic expression (LLM translator)",
                          priority=58, compatible_modes={CognitiveMode.DELIBERATIVE, CognitiveMode.REACTIVE})
    def translate_to_equation(self, request, context, query, **kwargs) -> Dict[str, Any]:
        out = self.to_equation(str(query))
        out["confidence"] = 0.9 if out["ok"] else 0.0
        return out
