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

PREMISES_SYSTEM_V1 = """You translate a short text (French or English) into logic for a symbolic reasoner.
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

# v2 (6.6.3): examples are NEUTRAL. v1 contained hints taken from Khaled's 16/10 test set
# (brother_of_bob, female cat, apple/meat, sunset/lights, "left at home"), which inflated
# the bake-off on that set. v1 is kept above for reproducibility.
PREMISES_SYSTEM_V2 = """You translate a short text (French or English) into logic for a symbolic reasoner.
You do NOT answer the question.

Output ONLY one JSON object, no prose, no code fence:
{"facts":[{"p":"<atom>","quote":"<exact span copied from the text>"}],
 "rules":[{"if":"<atom>","then":"<atom>","quote":"<exact span>"}],
 "goals":[{"p":"<atom>","quote":"<exact span of the question>"}],
 "knowledge":[{"p":"<atom>","why":"<short reason>"} or {"if":"<atom>","then":"<atom>","why":"<short reason>"}]}

Atoms (lowercase, no spaces):
- is(x,y)    x is a y / x is y (category or property)
- can(x,y)   x can y
- in(x,y)    containment of any kind (inside, in, part of, located in, at...)
- verb(x,y) or verb(x) for any other verb, verb in base form: read(ana,book), leave(train), win(team,match)
- negation: prefix "not " -> "not is(x,y)"
Constants: singular, snake_case, no articles. The SAME entity always gets the SAME constant in
facts, rules and goals ("the manager of the shop" -> manager_of_shop everywhere).
Universal statements use the variable X (and Y if needed):
  "All squares are rectangles" -> {"if":"is(X,square)","then":"is(X,rectangle)"}
  "No fish can walk"           -> {"if":"is(X,fish)","then":"not can(X,walk)"}
  "Bees make honey"            -> {"if":"is(X,bee)","then":"make(X,honey)"}
A conditional "if A then B" is one rule; a sentence with two consequences gives two rules.
Rules have exactly ONE atom in "if" (no "and"): split or move the rest to "knowledge".
"some / certains / quelques / many / most" are NOT universal: produce no rule for them.
Co-occurrence or time ("when the bell rings, the students leave") is a rule, never a cause:
never invent cause(...) atoms.
Every fact, rule and goal MUST quote the exact span of the text it comes from.
Anything not literally stated (common sense, word meanings such as "a puppy is a dog",
"glass is fragile", persistence over time) goes ONLY in "knowledge", with a short "why".
Never answer the question. Never add facts just to make the question answerable."""

# v3 (6.7.0): relations between two entities as a predicate, impersonal propositions,
# rules with up to 3 conditions. Examples stay neutral (none taken from any test set).
PREMISES_SYSTEM_V3 = """You translate a short text (French or English) into logic for a symbolic reasoner.
You do NOT answer the question.

Output ONLY one JSON object, no prose, no code fence:
{"facts":[{"p":"<atom>","quote":"<exact span copied from the text>"}],
 "rules":[{"if":"<atom>" or ["<atom>","<atom>"],"then":"<atom>","quote":"<exact span>"}],
 "goals":[{"p":"<atom>","quote":"<exact span of the question>"}],
 "knowledge":[{"p":"<atom>","why":"<short reason>"} or {"if":"<atom>" or [...],"then":"<atom>","why":"<short reason>"}]}

Atoms (lowercase, no spaces):
- is(x,y)    x is a y / x is y (category or property of ONE entity)
- can(x,y)   x can y
- in(x,y)    containment of any kind (inside, in, part of, located in, at...)
- rel(x,y)   any relation between TWO entities, the relation as predicate:
             older(ana,ben), parent(ann,bob), before(breakfast,lunch), owns(lea,bike)
             (never encode a relation as a property such as is(ana,older_than_ben))
- verb(x,y) or verb(x) for any other verb, verb in base form: read(ana,book), leave(train)
- prop(x)    impersonal statement without a subject: "it is late" -> prop(late)
- negation: prefix "not " -> "not is(x,y)"
Constants: singular, snake_case, no articles. The SAME entity always gets the SAME constant in
facts, rules and goals ("the manager of the shop" -> manager_of_shop everywhere).
Every statement of the text that is a fact goes in "facts", with its quote.
Rules use the variables X, Y, Z. "if" is one atom or a list of up to 3 atoms (all required):
  "All squares are rectangles" -> {"if":"is(X,square)","then":"is(X,rectangle)"}
  "No fish can walk"           -> {"if":"is(X,fish)","then":"not can(X,walk)"}
  ancestry is transitive        -> {"if":["ancestor(X,Y)","ancestor(Y,Z)"],"then":"ancestor(X,Z)"}
Every variable of "then" must appear in "if".
A conditional "if A then B" is one rule; a sentence with two consequences gives two rules.
"some / certains / quelques / many / most" are NOT universal: produce no rule for them.
Co-occurrence or time ("when the bell rings, the students leave") is a rule, never a cause:
never invent cause(...) atoms.
Every fact, rule and goal MUST quote the exact span of the text it comes from.
Anything not literally stated (common sense, word meanings such as "a puppy is a dog",
properties of relations such as transitivity, persistence over time) goes ONLY in "knowledge",
with a short "why".
Never answer the question. Never add facts just to make the question answerable."""

PREMISES_SYSTEM_V4 = """You translate a short text (French or English) into logic for a symbolic reasoner.
You do NOT answer the question.

Output ONLY one JSON object, no prose, no code fence:
{"facts":[{"p":"<atom>","quote":"<exact span copied from the text>"}],
 "rules":[{"if":"<atom>" or ["<atom>","<atom>"],"then":"<atom>","quote":"<exact span>"}],
 "goals":[{"p":"<atom>","quote":"<exact span of the question>"}],
 "knowledge":[{"p":"<atom>","why":"<short reason>"} or {"if":"<atom>" or [...],"then":"<atom>","why":"<short reason>"}]}

Atoms (lowercase, no spaces):
- is(x,y)    x is a y / x is y (category or property of ONE entity)
- can(x,y)   x can y
- in(x,y)    containment of any kind (inside, in, part of, located in, at...)
- rel(x,y)   any relation between TWO entities, the relation as predicate:
             older(ana,ben), parent(ann,bob), before(breakfast,lunch), owns(lea,bike)
             (never encode a relation as a property such as is(ana,older_than_ben))
- verb(x,y) or verb(x) for any other verb, verb in base form: read(ana,book), leave(train)
- prop(x)    impersonal statement without a subject: "it is late" -> prop(late)
- negation: prefix "not " -> "not is(x,y)"
Constants: singular, snake_case, no articles. The SAME entity always gets the SAME constant in
facts, rules and goals ("the manager of the shop" -> manager_of_shop everywhere).
A typical or generic member of a category ("a typical nurse", "un infirmier typique", "nurses" in
general) is written with the category constant itself: nurse. If two constants still denote the
same entity, add the knowledge atom same(a,b).
Something that can change over time is written with the moment as last argument:
  "the shop was open on Monday"  -> open(shop,monday)
  "the lamp is off tonight"      -> not on(lamp,tonight)
  an event at a moment           -> repaired(bike,tuesday)
How states continue or change after an event is common sense: put it in "knowledge"
("repaired(bike,tuesday) => works(bike,wednesday)", with a short "why").
Every statement of the text that is a fact goes in "facts", with its quote.
Rules use the variables X, Y, Z. "if" is one atom or a list of up to 3 atoms (all required):
  "All squares are rectangles" -> {"if":"is(X,square)","then":"is(X,rectangle)"}
  "No fish can walk"           -> {"if":"is(X,fish)","then":"not can(X,walk)"}
  ancestry is transitive        -> {"if":["ancestor(X,Y)","ancestor(Y,Z)"],"then":"ancestor(X,Z)"}
Every variable of "then" must appear in "if".
A conditional "if A then B" is one rule; a sentence with two consequences gives two rules.
"some / certains / quelques / many / most" are NOT universal: produce no rule for them.
Co-occurrence or time ("when the bell rings, the students leave") is a rule, never a cause:
never invent cause(...) atoms.
Every fact, rule and goal MUST quote the exact span of the text it comes from.
Anything not literally stated (common sense, word meanings such as "a puppy is a dog",
properties of relations such as transitivity, persistence over time) goes ONLY in "knowledge",
with a short "why".
Never answer the question. Never add facts just to make the question answerable."""


# v4 (6.8.0): typical member = category constant, same(a,b) for co-reference, moments as last
# argument for states that change (neutral examples, none from any test set).
PREMISES_SYSTEM = PREMISES_SYSTEM_V4
PROMPT_VERSIONS = {"premises-v1": PREMISES_SYSTEM_V1, "premises-v2": PREMISES_SYSTEM_V2,
                   "premises-v3": PREMISES_SYSTEM_V3, "premises-v4": PREMISES_SYSTEM_V4}

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

ATOM_RE = re.compile(r"^(not )?[^\W\d][\w]*\((?:[\w]+)(?:,[\w]+){0,2}\)$", re.UNICODE)   # 1 to 3 arguments
EXPR_RE = re.compile(r"^[0-9+\-*/(). ]+$")
VAR_RE = re.compile(r"(?<=[(,])([XYZ])(?=[,)])")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower().replace("’", "'").replace("`", "'")
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


BARE_RE = re.compile(r"^(not )?([^\W\d]\w*)$", re.UNICODE)


def normalize_atom(atom: Any) -> Any:
    """'rain' -> 'prop(rain)', 'not rain' -> 'not prop(rain)'; anything else unchanged."""
    if not isinstance(atom, str):
        return atom
    a = re.sub(r"\s+", " ", atom.strip())
    m = BARE_RE.match(a)
    return f"{m.group(1) or ''}prop({m.group(2)})" if m else a


def condition_list(cond: Any) -> List[Any]:
    """A rule condition as a list of atoms (accepts a list, or 'A and B' / 'A ∧ B' / 'A & B')."""
    if isinstance(cond, list):
        return [normalize_atom(c) for c in cond]
    if isinstance(cond, str):
        return [normalize_atom(c) for c in re.split(r"\s+(?:and|et)\s+|\s*[∧&]\s*", cond.strip()) if c]
    return [cond]


def anchored(quote: str, text: str, min_overlap: float = 0.8) -> bool:
    """True if `quote` is (almost) literally present in `text`."""
    q, t = _norm(quote), _norm(text)
    if len(q.split()) < 2:                 # a single word proves nothing (review 1a)
        return False
    return q in t                          # exact (normalised) span only: no bag-of-words fallback (review 1a)


def extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """First complete JSON object in `raw` (code fences and trailing prose ignored)."""
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", raw):
        try:
            obj, _ = decoder.raw_decode(raw[m.start():])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


class LLMTranslator(CognitiveModuleBase):
    """Translate text into premises or arithmetic for the symbolic engines."""

    def __init__(self, llm_engine: Any = None, call: Optional[Callable[[str, str], str]] = None,
                 model: str = DEFAULT_MODEL, max_tokens: int = 1500, use_knowledge: bool = True,
                 nexus_core: Any = None, prompt_version: str = "premises-v4", repair_rounds: int = 1, **kwargs):
        if llm_engine is None and call is None:
            raise ValueError("LLMTranslator needs an llm_engine or a call(system, prompt) function")
        self.llm_engine = llm_engine
        self._call_fn = call
        self.model = model
        self.max_tokens = max_tokens
        self.use_knowledge = use_knowledge
        self.nexus_core = nexus_core
        self.prompt_version = prompt_version
        self.repair_rounds = repair_rounds          # one corrective round when items are rejected
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.stats = {"calls": 0, "cache_hits": 0, "failures": 0, "rejected_items": 0, "seconds": 0.0,
                      "input_tokens": 0, "output_tokens": 0,
                      "repairs": 0, "repairs_improved": 0}
        self.receipts: List[Dict[str, Any]] = []   # proof of real API calls (message id, model, tokens)

    # -- transport ------------------------------------------------------------
    def _call(self, system: str, prompt: str, task: str = "") -> str:
        t0 = time.time()
        self.stats["calls"] += 1
        try:
            if self._call_fn is not None:
                return self._call_fn(system, prompt)
            from grandnexus.llm_interface.llm_engine import LLMRequest
            # no temperature: recent Claude models reject non-default sampling parameters
            resp = self.llm_engine.complete(LLMRequest(
                prompt=prompt, temperature=None, max_tokens=self.max_tokens, request_type="chat",
                context={"model": self.model, "system": system}))
            receipt = getattr(resp, "receipt", None)
            if receipt:
                self.receipts.append({"task": task, "seconds": round(time.time() - t0, 2), **receipt})
                self.stats["input_tokens"] += receipt.get("input_tokens") or 0
                self.stats["output_tokens"] += receipt.get("output_tokens") or 0
            return getattr(resp, "text", "") or ""
        finally:
            self.stats["seconds"] += time.time() - t0

    def _cached(self, task: str, text: str, system: str, prompt: str) -> Tuple[Optional[Dict[str, Any]], str]:
        key = hashlib.sha256(f"{task}|{self.model}|{self.prompt_version}|{text}".encode()).hexdigest()
        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key]["parsed"], self.cache[key]["raw"]
        try:
            raw = self._call(system, prompt, task)
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
        system = PROMPT_VERSIONS.get(self.prompt_version, PREMISES_SYSTEM)
        parsed, raw = self._cached("premises", text, system, f"TEXT:\n{text}")
        out = self._validate(parsed, raw, text)
        # one corrective round: the engine tells the translator what it refused, and why
        if self.repair_rounds > 0 and parsed is not None and out["rejected"]:
            problems = "\n".join(f"- {r['reason']}: {json.dumps(r.get('item'), ensure_ascii=False)[:200]}"
                                  for r in out["rejected"][:8])
            prompt = (f"TEXT:\n{text}\n\nYour previous JSON:\n{raw[:3000]}\n\n"
                      f"The reasoner REJECTED these items:\n{problems}\n\n"
                      "Output the full corrected JSON object, following the format rules exactly.")
            self.stats["repairs"] += 1
            parsed2, raw2 = self._cached("premises-repair", text, system, prompt)
            out2 = self._validate(parsed2, raw2, text)
            if parsed2 is not None and (len(out2["rejected"]), -len(out2["facts"])) < (len(out["rejected"]), -len(out["facts"])):
                self.stats["repairs_improved"] += 1
                out2["repaired"] = True
                out = out2
        self.stats["rejected_items"] += len(out["rejected"])
        return out

    def _validate(self, parsed: Optional[Dict[str, Any]], raw: str, text: str) -> Dict[str, Any]:
        out = {"facts": [], "rules": [], "goals": [], "knowledge_facts": [], "knowledge_rules": [],
               "rejected": [], "ok": parsed is not None, "raw": raw}
        if parsed is None:
            return out

        def bad(item, kind, reason):
            out["rejected"].append({"kind": kind, "item": item, "reason": reason})
            return False

        def atom_ok(item, atoms, kind):
            wrong = [a for a in atoms if not isinstance(a, str) or not ATOM_RE.match(a)]
            return bad(item, kind, f"malformed atom {wrong}") if wrong else True

        def rule_ok(item, conds, then, kind, need_quote):
            if not conds or len(conds) > 3:
                return bad(item, kind, "a rule needs 1 to 3 conditions")
            if not atom_ok(item, conds + [then], kind):
                return False
            v_if = set().union(*(VAR_RE.findall(c) for c in conds))
            v_then = set(VAR_RE.findall(then))
            if v_then - v_if:   # range restriction: in(X,Y) => in(X,Z) would put anything anywhere
                return bad(item, kind, f"unsafe rule: variable(s) {sorted(v_then - v_if)} only in the conclusion")
            if then in conds:
                return bad(item, kind, "rule concludes its own condition")
            if need_quote and not anchored(str(item.get("quote", "")), text):
                return bad(item, kind, "quote not found in text")
            return True

        def rule(item, conds, then, conf, src):
            return {"if": conds[0] if len(conds) == 1 else conds, "then": then, "confidence": conf, "source": src}

        def ground(item, a, kind):
            return not VAR_RE.search(a) or bad(item, kind, "variable in a fact or goal (review 2.5): use a constant or a rule")

        for f in parsed.get("facts", []) or []:
            if not isinstance(f, dict):
                continue
            a = normalize_atom(f.get("p"))
            if atom_ok(f, [a], "fact") and ground(f, a, "fact") and (anchored(str(f.get("quote", "")), text) or bad(f, "fact", "quote not found in text")):
                out["facts"].append(a)
        for r in parsed.get("rules", []) or []:
            if not isinstance(r, dict):
                continue
            conds, then = condition_list(r.get("if")), normalize_atom(r.get("then"))
            if rule_ok(r, conds, then, "rule", True):
                out["rules"].append(rule(r, conds, then, 0.95, "llm:" + str(r.get("quote", ""))[:80]))
        for g in parsed.get("goals", []) or []:
            if not isinstance(g, dict):
                continue
            a = normalize_atom(g.get("p"))
            if atom_ok(g, [a], "goal") and ground(g, a, "goal") and (anchored(str(g.get("quote", "")), text) or bad(g, "goal", "quote not found in text")):
                out["goals"].append(a)
        for k in parsed.get("knowledge", []) or []:
            if not isinstance(k, dict):
                continue
            if "p" in k:
                a = normalize_atom(k.get("p"))
                if atom_ok(k, [a], "knowledge") and ground(k, a, "knowledge"):
                    out["knowledge_facts"].append(a)
            elif "if" in k:
                conds, then = condition_list(k.get("if")), normalize_atom(k.get("then"))
                if rule_ok(k, conds, then, "knowledge", False):
                    out["knowledge_rules"].append(rule(k, conds, then, 0.8, "llm-knowledge:" + str(k.get("why", ""))[:80]))
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
        mapping = {"yes": "yes", "no": "no", "unknown": "unknown", "contradictory": "contradictory"}
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
