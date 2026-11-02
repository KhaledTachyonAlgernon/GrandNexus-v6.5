"""
text_reasoner.py — Perception -> premises -> ReasoningCore pathway.

    text ─► premises   grammar (PremiseExtractor) and/or LLM translator (anchored quotes)
         ─► grounding  universal rules (variables X, Y) instantiated on known constants,
                       containment ("in") made transitive
         ─► ReasoningCore deduction (existing strategies, chained to a fixpoint)
         ─► verdict    oui / non / inconnu / contradictoire (global consistency check),
                       or a number for quantity questions (expression computed by sympy)
         ─► write-back shared graph, working memory, event bus

Modes:  "grammar"    rule-based extractor only (deterministic, no API)
        "llm"        LLM translator only
        "hybrid"     grammar first; LLM only when the grammar cannot decide
        "llm_direct" CONTROL BASELINE: the LLM answers by itself (no engine)
"""

import itertools
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from grandnexus.cognition.contracts import (
    CognitiveContext, CognitiveMode, CognitiveModuleBase, CognitiveProcessType,
    CognitiveRequest, cognitive_capability,
)
from grandnexus.cognition.perception.premise_extractor import PremiseExtractor

ATOM = re.compile(r"^(not )?([^\W\d]\w*)\((\w+(?:,\w+){0,2})\)$", re.UNICODE)   # 1 to 3 arguments
VARIABLE = re.compile(r"^[XYZ]$")
QUANTITY_QUESTION = re.compile(r"\b(combien|how many|how much|quel(?:le)? est le nombre|what is the number)\b", re.I)
DECIDED = ("oui", "non", "contradictoire")
TYPICAL = re.compile(r"^(?:typical|generic|average|typique|generique|générique)_(\w+)$|^(\w+)_(?:typique|typical|generique|générique)$")


def _negate(p: str) -> str:
    return p[4:] if p.startswith("not ") else "not " + p


def _args(atom: str) -> List[str]:
    m = ATOM.match(atom.strip())
    return m.group(3).split(",") if m else []



def _rename(atom: str, mapping: Dict[str, str]) -> str:
    m = ATOM.match(atom.strip()) if isinstance(atom, str) else None
    if not m:
        return atom
    args = [mapping.get(a, a) for a in m.group(3).split(",")]
    return f"{m.group(1) or ''}{m.group(2)}({','.join(args)})"


def _alias_map(pair_atoms: List[str], constants: List[str], goals: List[str]) -> Dict[str, str]:
    """Co-reference: same(a,b) atoms + the typical-member convention (typical_x == x).
    Canonical name: the constant used in a goal if any, else the shortest."""
    pairs = []
    for a in pair_atoms:
        m = ATOM.match(a.strip())
        if m and not m.group(1) and m.group(2) == "same" and len(m.group(3).split(",")) == 2:
            pairs.append(tuple(m.group(3).split(",")))
    cs = set(constants)
    for c in constants:
        t = TYPICAL.match(c)
        base = t and (t.group(1) or t.group(2))
        if base and base in cs:
            pairs.append((c, base))
    parent: Dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        parent[find(a)] = find(b)
    in_goal = {x for g in goals for x in _args(g)}
    groups: Dict[str, List[str]] = {}
    for x in list(parent):
        groups.setdefault(find(x), []).append(x)
    mapping = {}
    for members in groups.values():
        canon = sorted(members, key=lambda x: (x not in in_goal, len(x), x))[0]
        mapping.update({x: canon for x in members if x != canon})
    return mapping


class TextReasoner(CognitiveModuleBase):

    def __init__(self, reasoning_core: Any = None, perception_core: Any = None,
                 extractor: Optional[PremiseExtractor] = None, graph_engine: Any = None,
                 working_memory: Any = None, nexus_core: Any = None, translator: Any = None,
                 mode: Optional[str] = None, nx_translator: Any = None, engine: str = "reference", **kwargs):
        """engine: "reference" (nexus_logic, v7.0 default) or "home" (ReasoningCore, kept for comparison).
        nx_translator: a translator_v5.Translator (text -> nexus_logic program); enables mode "nx"."""
        self.engine = engine
        self.nx_translator = nx_translator
        self.reasoning_core = reasoning_core
        self.perception_core = perception_core
        self.extractor = extractor or PremiseExtractor(nexus_core=nexus_core)
        self.graph_engine = graph_engine
        self.working_memory = working_memory
        self.nexus_core = nexus_core
        self.translator = translator
        self.mode = mode or ("nx" if nx_translator is not None else "hybrid" if translator is not None else "grammar")
        self.logger = logging.getLogger("GrandNexus.TextReasoner")
        self._node_ids: Dict[str, str] = {}

    # =========================================================================
    def answer(self, text: str, max_steps: int = 8, write_back: bool = True,
               mode: Optional[str] = None) -> Dict[str, Any]:
        mode = mode or self.mode
        if mode == "nx" and self.nx_translator is None:
            mode = "hybrid" if self.translator is not None else "grammar"
        if self.translator is None and mode not in ("grammar", "nx"):
            mode = "grammar"
        t0 = time.time()
        perception = self._perceive(text)

        if mode == "nx":
            out = self._answer_nx(text, perception, t0)
            if write_back:
                self._write_back(text, out["extraction"]["facts"], out)
            return out

        if mode == "llm_direct":
            ans = self.translator.answer_directly(text)
            return self._result(ans, [], 0.5 if ans else 0.0,
                                {"facts": [], "rules": [], "goals": [], "unparsed": [],
                                 "trace": ["control baseline: the LLM answered directly"]},
                                [], [], "llm_direct", perception, t0)

        if QUANTITY_QUESTION.search(text):
            if mode in ("llm", "hybrid"):
                out = self._answer_quantity(text, perception, t0)
                if out is not None:
                    if write_back:
                        self._write_back(text, [], out)
                    return out
            else:
                # the grammar has no arithmetic: say so instead of forcing a meaningless goal
                return self._result("inconnu", [], 0.0,
                                    {"facts": [], "rules": [], "goals": ["quantity"],
                                     "unparsed": ["quantity question: needs the LLM translator (mode 'llm' or 'hybrid')"],
                                     "trace": ["quantity | detected, not handled by the grammar"]},
                                    [], [], mode, perception, t0)

        if mode in ("grammar", "hybrid"):
            out = self._decide(self._from_grammar(text), max_steps, perception, t0)
            if mode == "grammar" or out["answer"] in DECIDED:
                if write_back:
                    self._write_back(text, out["extraction"]["facts"], out)
                return out
        out = self._decide(self._from_llm(text), max_steps, perception, t0)
        if write_back:
            self._write_back(text, out["extraction"]["facts"], out)
        return out

    # =========================================================================
    @staticmethod
    def _split_question(text: str):
        """Last sentence ending with '?' is the question; the rest is the text."""
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        qs = [p for p in parts if p.rstrip().endswith("?")]
        if not qs:
            return text.strip(), ""
        q = qs[-1]
        body = " ".join(p for p in parts if p is not q)
        return body.strip(), q.strip()

    def _answer_nx(self, text: str, perception: Any, t0: float) -> Dict[str, Any]:
        """v7.0 core pathway: Claude translates into the nexus_logic language, the reference engine decides.
        Policy: the answer is the STRICT verdict (text only); a verdict that needs background knowledge is
        reported apart, starred, never counted as the answer."""
        body, query = self._split_question(text)
        r = self.nx_translator.answer(body, query)
        strict, full = r["verdict_strict"], r["verdict"]
        answer = r["answer_strict"] if strict == "oui" else strict      # numbers for "how many" questions
        out = self._result(answer, [{"goal": query, "answer": answer, "confidence": 0.95 if strict in DECIDED else 0.0,
                                     "support": [{"statement": l, "from": "proof"} for l in r["proof"][:12]]}],
                           0.95 if strict in DECIDED else 0.0,
                           {"facts": [], "rules": [], "goals": [query] if query else [],
                            "unparsed": r["rejections"], "trace": ["nx | " + l for l in r["program"].splitlines() if l.strip()]},
                           [], ["nexus_logic:" + r["engine_version"]], "nx", perception, t0)
        out["strict_answer"] = strict
        out["answer_with_knowledge"] = r["answer"] if full == "oui" else full
        out["depends_on_knowledge"] = bool(r["depends_on_knowledge"]) and full != strict
        out["program"] = r["program"]
        out["receipts"] = {"calls": r["calls"], "message_ids": r["message_ids"], "model": r["model"]}
        return out

    def _deduce_reference(self, facts: List[str], rules: List[Dict[str, Any]], goals: List[str],
                          knowledge_facts: List[str] = (), knowledge_rules: List[Dict[str, Any]] = ()):
        """Deduction by the reference engine (v7.0): returns (verdicts, derived, reasoning_info)."""
        from grandnexus.nexus_logic import program_from_atoms, decide
        prog = program_from_atoms(facts, rules, goals, knowledge_facts, knowledge_rules)
        v = decide(prog, use_knowledge=bool(knowledge_facts or knowledge_rules))
        conf = 0.95 if v.answer in DECIDED else 0.0
        verdicts = [{"goal": g, "answer": v.answer, "confidence": conf,
                     "support": [{"statement": l.strip(), "from": "proof"} for l in v.proof[:12]],
                     "contradictions": v.contradictions} for g in goals[:1]]
        derived = [re.sub(r"\s+\[.*$", "", l).strip().replace(", ", ",") for l in v.proof if "[s" in l and "=>" in l]
        return verdicts, derived, {"strategies_used": ["nexus_logic"], "note": v.note}

    def _perceive(self, text: str) -> Optional[Dict[str, Any]]:
        if self.perception_core is None:
            return None
        try:
            ctx = CognitiveContext()
            return self.perception_core.perceive_input(CognitiveRequest(query=text, context=ctx), ctx, text)
        except Exception as e:
            return {"error": str(e)}

    def _from_grammar(self, text: str) -> Dict[str, Any]:
        ex = self.extractor.extract(text)
        return {"text": text, "parser": "grammar", "facts": list(ex.facts), "rules": list(ex.rules), "goals": list(ex.goals),
                "knowledge_facts": [], "knowledge_rules": [], "trace": list(ex.trace),
                "unparsed": list(ex.unparsed), "llm": None}

    def _from_llm(self, text: str) -> Dict[str, Any]:
        p = self.translator.to_premises(text)
        def cond(r):
            return " ∧ ".join(r["if"]) if isinstance(r["if"], list) else r["if"]
        trace = ([f"fact     | {f}" for f in p["facts"]] + [f"rule     | {cond(r)} => {r['then']}" for r in p["rules"]]
                 + [f"goal     | {g}" for g in p["goals"]] + [f"knowledge| {k}" for k in p["knowledge_facts"]]
                 + [f"knowledge| {cond(r)} => {r['then']}" for r in p["knowledge_rules"]]
                 + (["repaired | one corrective round was used"] if p.get("repaired") else []))
        use_k = getattr(self.translator, "use_knowledge", True)
        return {"text": text, "parser": "llm", "facts": p["facts"], "rules": p["rules"], "goals": p["goals"],
                "knowledge_facts": p["knowledge_facts"] if use_k else [],
                "knowledge_rules": p["knowledge_rules"] if use_k else [], "trace": trace,
                "unparsed": [f"rejected: {r['reason']}" for r in p["rejected"]]
                            + ([] if p["ok"] else ["LLM output unusable"]),
                "llm": {"rejected": p["rejected"], "ok": p["ok"], "raw": (p.get("raw") or "")[:3000]}}

    # -- grounding ------------------------------------------------------------
    @staticmethod
    def _ground(rules: List[Dict[str, Any]], constants: List[str]) -> List[Dict[str, Any]]:
        """Instantiate the variables X, Y, Z of every rule on the known constants.
        A rule condition may be one atom or a list of atoms (all required)."""
        out, seen = [], set()

        def key(r):
            return (tuple(r["if"]) if isinstance(r["if"], list) else r["if"], r["then"])
        for r in rules:
            conds = r["if"] if isinstance(r["if"], list) else [r["if"]]
            variables = sorted({a for atom in conds + [r["then"]] for a in _args(atom) if VARIABLE.match(a)})
            if not variables:
                if key(r) not in seen:
                    seen.add(key(r))
                    out.append(r)
                continue
            pool = constants[:40] if len(variables) <= 2 else constants[:15]
            for combo in itertools.product(pool, repeat=len(variables)):
                sub = dict(zip(variables, combo))

                def apply(atom: str) -> str:
                    m = ATOM.match(atom.strip())
                    args = [sub.get(a, a) for a in m.group(3).split(",")]
                    return f"{m.group(1) or ''}{m.group(2)}({','.join(args)})"
                g_if = [apply(c) for c in conds]
                if len(g_if) > 1 and len(set(g_if)) < len(g_if):
                    continue                     # degenerate grounding (same condition twice)
                g = {**r, "if": g_if[0] if len(g_if) == 1 else g_if, "then": apply(r["then"])}
                if key(g) not in seen:
                    seen.add(key(g))
                    out.append(g)
        return out

    @staticmethod
    def _containment_axioms(facts: List[str]) -> List[Dict[str, Any]]:
        pairs = {tuple(_args(f)) for f in facts if f.startswith("in(") and len(_args(f)) == 2}
        closure, rules, changed = set(pairs), [], True
        while changed:
            changed = False
            for (a, b) in list(closure):
                for (b2, c) in list(closure):
                    if b == b2 and a != c and (a, c) not in closure:
                        closure.add((a, c))
                        changed = True
                        rules.append({"if": f"in({b},{c})", "then": f"in({a},{c})", "confidence": 0.95,
                                      "source": "axiom: containment is transitive"})
        return rules

    # -- deduction + verdict --------------------------------------------------
    def _deduce(self, facts: List[str], rules: List[Dict[str, Any]], max_steps: int):
        known = {f: {"statement": f, "confidence": 1.0, "from": "premise"} for f in facts}
        reasoning: Dict[str, Any] = {}
        if self.reasoning_core is not None and facts and rules:
            ctx = CognitiveContext(parameters={"premises": list(facts), "rules": list(rules),
                                               "max_steps": max_steps, "exploration_factor": 0.0})
            reasoning = self.reasoning_core.reason(CognitiveRequest(query="deduce", context=ctx), ctx, "deduce")
            for c in reasoning.get("conclusions", []) or []:
                st = c.get("statement")
                if st and st not in known:
                    known[st] = {"statement": st, "confidence": c.get("confidence", 0.95),
                                 "from": c.get("derived_from")}
        return known, reasoning

    @staticmethod
    def _verdicts(known: Dict[str, Any], goals: List[str]) -> List[Dict[str, Any]]:
        contradictions = sorted(p for p in known if not p.startswith("not ") and ("not " + p) in known)
        verdicts = []
        for g in goals:
            if contradictions:          # inconsistent premises: no answer can be trusted
                verdicts.append({"goal": g, "answer": "contradictoire", "confidence": 0.0,
                                 "support": [known[c] for c in contradictions[:3]]
                                            + [known["not " + c] for c in contradictions[:3]],
                                 "contradictions": contradictions})
                continue
            # convention (aligned with the reference engine nexus_logic): "oui" means the asked
            # literal holds, even when it is negative ("is it true that the lamp is not on?" -> oui)
            pos, neg = g in known, _negate(g) in known
            if pos:
                v, conf = "oui", known[g]["confidence"]
            elif neg:
                v, conf = "non", known[_negate(g)]["confidence"]
            else:
                v, conf = "inconnu", 0.0
            verdicts.append({"goal": g, "answer": v, "confidence": conf,
                             "support": [known[x] for x in (g, _negate(g)) if x in known]})
        return verdicts

    @staticmethod
    def _apply_aliases(b: Dict[str, Any], use_knowledge: bool) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Merge constants that denote the same entity (same(a,b) and typical_x == x)."""
        def conds(r):
            return r["if"] if isinstance(r["if"], list) else [r["if"]]
        atoms = b["facts"] + b["goals"] + [a for r in b["rules"] for a in conds(r) + [r["then"]]]
        if use_knowledge:
            atoms += b["knowledge_facts"] + [a for r in b["knowledge_rules"] for a in conds(r) + [r["then"]]]
        constants = list(dict.fromkeys(x for a in atoms for x in _args(a) if not VARIABLE.match(x)))
        pair_atoms = b["facts"] + (b["knowledge_facts"] if use_knowledge else [])
        mapping = _alias_map(pair_atoms, constants, b["goals"])

        def rule(r):
            c = [_rename(a, mapping) for a in conds(r)]
            return {**r, "if": c[0] if len(c) == 1 else c, "then": _rename(r["then"], mapping)}
        not_same = lambda a: not a.startswith("same(")
        nb = {**b, "facts": [_rename(a, mapping) for a in b["facts"] if not_same(a)],
              "goals": [_rename(a, mapping) for a in b["goals"]],
              "rules": [rule(r) for r in b["rules"]],
              "knowledge_facts": [_rename(a, mapping) for a in b["knowledge_facts"] if not_same(a)],
              "knowledge_rules": [rule(r) for r in b["knowledge_rules"]]}
        return nb, mapping

    def _decide(self, b: Dict[str, Any], max_steps: int, perception: Any, t0: float) -> Dict[str, Any]:
        b_strict, map_s = self._apply_aliases(b, use_knowledge=False)
        b_full, map_k = self._apply_aliases(b, use_knowledge=True)
        b = {**b_full, "trace": list(b["trace"]) + [f"alias    | {k} -> {v}" for k, v in map_k.items()]}
        facts = list(dict.fromkeys(b_strict["facts"]))
        goals = list(dict.fromkeys(b_strict["goals"]))
        # a question about a category reasons on a typical member ("Do whales breathe air?")
        categories = {_args(r["if"])[1] for r in b["rules"] + b["knowledge_rules"]
                      if isinstance(r["if"], str) and r["if"].startswith("is(") and len(_args(r["if"])) == 2
                      and VARIABLE.match(_args(r["if"])[0])}
        for g in goals:
            a = _args(g)
            if a and a[0] in categories and f"is({a[0]},{a[0]})" not in facts:     # no guard (review 2.4)
                facts.append(f"is({a[0]},{a[0]})")
        constants = list(dict.fromkeys(a for atom in facts + goals + b["knowledge_facts"] for a in _args(atom)
                                       if not VARIABLE.match(a)))
        has_k = bool(b["knowledge_facts"] or b["knowledge_rules"] or map_k != map_s)
        if self.engine == "reference":
            # v7.0: variables are native, so no grounding; containment transitivity stays a ground axiom set
            strict_rules = list(b_strict["rules"]) + self._containment_axioms(facts)
            strict, derived_s, reasoning = self._deduce_reference(facts, strict_rules, goals)
            if has_k:
                goals_k = list(dict.fromkeys(b["goals"]))
                facts_k = list(dict.fromkeys(b["facts"] + [f for f in b["knowledge_facts"] if f not in b["facts"]]))
                verdicts, derived, reasoning = self._deduce_reference(
                    facts_k, list(b["rules"]) + self._containment_axioms(facts_k), goals_k, [], b["knowledge_rules"])
            else:
                verdicts, derived = strict, derived_s
            # policy (v7.0): the answer is the strict verdict; the knowledge-based one is reported apart
            answer = strict[0]["answer"] if strict else ("inconnu" if "?" in str(b.get("text", "")) else ("faits enregistrés" if facts else "inconnu"))
            out = self._result(answer, strict, strict[0]["confidence"] if strict else 0.0,
                               {"facts": facts, "rules": [(r["if"], r["then"]) for r in strict_rules],
                                "goals": goals, "unparsed": b["unparsed"], "trace": b["trace"]},
                               derived_s, reasoning.get("strategies_used", []), b["parser"], perception, t0)
            out["strict_answer"] = strict[0]["answer"] if strict else None
            out["answer_with_knowledge"] = verdicts[0]["answer"] if (has_k and verdicts) else out["strict_answer"]
            out["depends_on_knowledge"] = bool(has_k and strict and verdicts and strict[0]["answer"] != verdicts[0]["answer"])
            if b.get("llm") is not None:
                out["llm"] = b["llm"]
            return out
        strict_rules = self._ground(b_strict["rules"], constants) + self._containment_axioms(facts)
        known_s, reasoning = self._deduce(facts, strict_rules, max_steps)
        strict = self._verdicts(known_s, goals)
        if has_k:
            goals = list(dict.fromkeys(b["goals"]))
            facts_k = list(dict.fromkeys(b["facts"] + [f for f in b["knowledge_facts"] if f not in b["facts"]]))
            for g in goals:                                   # typical member, after knowledge aliases
                a = _args(g)
                if a and a[0] in categories and f"is({a[0]},{a[0]})" not in facts_k:
                    facts_k.append(f"is({a[0]},{a[0]})")
            constants = list(dict.fromkeys(constants + [x for atom in facts_k + goals for x in _args(atom)
                                                        if not VARIABLE.match(x)]))
            rules_k = self._ground(b["rules"] + b["knowledge_rules"], constants) + self._containment_axioms(facts_k)
            known, reasoning = self._deduce(facts_k, rules_k, max_steps)
            verdicts = self._verdicts(known, goals)
        else:
            known, verdicts = known_s, strict
        derived = [k for k, v in known.items() if v.get("from") != "premise"]
        asked = "?" in str(b.get("text", ""))
        if not verdicts and asked:
            b["unparsed"] = list(b["unparsed"]) + ["question present but no goal could be formalised"]
        answer = verdicts[0]["answer"] if verdicts else ("faits enregistrés" if facts and not asked else "inconnu")
        out = self._result(answer, verdicts, verdicts[0]["confidence"] if verdicts else 0.0,
                           {"facts": facts, "rules": [(r["if"], r["then"]) for r in strict_rules],
                            "goals": goals, "unparsed": b["unparsed"], "trace": b["trace"]},
                           derived, reasoning.get("strategies_used", []), b["parser"], perception, t0)
        out["strict_answer"] = strict[0]["answer"] if strict else None
        out["depends_on_knowledge"] = bool(has_k and strict and verdicts and strict[0]["answer"] != verdicts[0]["answer"])
        if b.get("llm") is not None:
            out["llm"] = b["llm"]
        return out

    def _answer_quantity(self, text: str, perception: Any, t0: float) -> Optional[Dict[str, Any]]:
        eq = self.translator.to_equation(text)
        if not eq["expression"] and not eq["rejected"]:
            return None                         # not a numeric question after all
        value = eq["value"] if eq["ok"] else None
        answer = str(value) if value is not None else "inconnu"
        trace = [f"equation | {eq['expression']} = {value} {eq['unit']}".strip()] + \
                [f"rejected | {r.get('reason')}" for r in eq["rejected"]]
        out = self._result(answer, [{"goal": "quantity", "answer": answer,
                                     "confidence": 0.9 if value is not None else 0.0,
                                     "support": [{"statement": eq["expression"], "from": "sympy"}]}],
                           0.9 if value is not None else 0.0,
                           {"facts": [], "rules": [], "goals": ["quantity"], "unparsed": [], "trace": trace},
                           [], ["llm_translation", "sympy_evaluation"], "llm-equation", perception, t0)
        out["equation"] = {k: eq[k] for k in ("expression", "value", "unit", "rejected")}
        return out

    @staticmethod
    def _result(answer, verdicts, confidence, extraction, derived, strategies, parser, perception, t0):
        return {"answer": answer, "verdicts": verdicts, "confidence": confidence, "extraction": extraction,
                "derived": derived, "strategies_used": strategies, "parser": parser,
                "input_type": (perception or {}).get("input_type") if isinstance(perception, dict) else None,
                "processing_time": time.time() - t0}

    # =========================================================================
    def _node(self, name: str) -> Optional[str]:
        if self.graph_engine is None:
            return None
        if name not in self._node_ids:
            self._node_ids[name] = self.graph_engine.add_node(label="Concept",
                                                               properties={"name": name, "source": "text_reasoner"})
        return self._node_ids[name]

    def _write_back(self, text: str, facts: List[str], out: Dict[str, Any]) -> None:
        for st in list(facts) + [d for d in out.get("derived", []) if d]:
            m = ATOM.match(st.strip())
            if not m or len(m.group(3).split(",")) != 2:
                continue                                   # the graph stores binary relations only
            neg, pred = m.group(1), m.group(2)
            a, b = m.group(3).split(",")
            try:
                na, nb = self._node(a), self._node(b)
                if na and nb:
                    label = {"is": "is_a"}.get(pred, pred)
                    self.graph_engine.add_edge(na, nb, ("not_" if neg else "") + label,
                                               properties={"source": "text_reasoner", "derived": st not in facts})
            except Exception as e:
                self.logger.debug(f"graph write: {e}")
        if self.working_memory is not None:
            try:
                self.working_memory.store_item(content={"question": text, "answer": out["answer"],
                                                        "verdicts": out["verdicts"], "kind": "reasoning"},
                                               source="reasoning", importance=max(0.3, out["confidence"]))
            except Exception as e:
                self.logger.debug(f"working memory write: {e}")
        if self.nexus_core is not None and hasattr(self.nexus_core, "publish_event"):
            self.nexus_core.publish_event("reasoning.conclusion", {"question": text, "answer": out["answer"],
                                                                    "derived": out.get("derived", [])},
                                          source="text_reasoner")

    @cognitive_capability(
        capability_type=CognitiveProcessType.REASONING,
        description="Reason over text: premises (grammar and/or LLM) then deduction",
        priority=30,
        compatible_modes={CognitiveMode.DELIBERATIVE, CognitiveMode.REACTIVE},
    )
    def reason_from_text(self, request, context, query, **kwargs) -> Dict[str, Any]:
        return self.answer(str(query), max_steps=int(context.get("max_steps", 8) or 8),
                           mode=context.get("mode"))
