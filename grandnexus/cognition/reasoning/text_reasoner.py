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
from typing import Any, Dict, List, Optional

from grandnexus.cognition.contracts import (
    CognitiveContext, CognitiveMode, CognitiveModuleBase, CognitiveProcessType,
    CognitiveRequest, cognitive_capability,
)
from grandnexus.cognition.perception.premise_extractor import PremiseExtractor

ATOM = re.compile(r"^(not )?([^\W\d]\w*)\((\w+)(?:,(\w+))?\)$", re.UNICODE)
VARIABLE = re.compile(r"^[XYZ]$")
QUANTITY_QUESTION = re.compile(r"\b(combien|how many|how much|quel(?:le)? est le nombre|what is the number)\b", re.I)
DECIDED = ("oui", "non", "contradictoire")


def _negate(p: str) -> str:
    return p[4:] if p.startswith("not ") else "not " + p


def _args(atom: str) -> List[str]:
    m = ATOM.match(atom.strip())
    return [a for a in (m.group(3), m.group(4)) if a] if m else []


class TextReasoner(CognitiveModuleBase):

    def __init__(self, reasoning_core: Any = None, perception_core: Any = None,
                 extractor: Optional[PremiseExtractor] = None, graph_engine: Any = None,
                 working_memory: Any = None, nexus_core: Any = None, translator: Any = None,
                 mode: Optional[str] = None, **kwargs):
        self.reasoning_core = reasoning_core
        self.perception_core = perception_core
        self.extractor = extractor or PremiseExtractor(nexus_core=nexus_core)
        self.graph_engine = graph_engine
        self.working_memory = working_memory
        self.nexus_core = nexus_core
        self.translator = translator
        self.mode = mode or ("hybrid" if translator is not None else "grammar")
        self.logger = logging.getLogger("GrandNexus.TextReasoner")
        self._node_ids: Dict[str, str] = {}

    # =========================================================================
    def answer(self, text: str, max_steps: int = 8, write_back: bool = True,
               mode: Optional[str] = None) -> Dict[str, Any]:
        mode = mode or self.mode
        if self.translator is None and mode != "grammar":
            mode = "grammar"
        t0 = time.time()
        perception = self._perceive(text)

        if mode == "llm_direct":
            ans = self.translator.answer_directly(text)
            return self._result(ans, [], 0.5 if ans else 0.0,
                                {"facts": [], "rules": [], "goals": [], "unparsed": [],
                                 "trace": ["control baseline: the LLM answered directly"]},
                                [], [], "llm_direct", perception, t0)

        if mode in ("llm", "hybrid") and QUANTITY_QUESTION.search(text):
            out = self._answer_quantity(text, perception, t0)
            if out is not None:
                if write_back:
                    self._write_back(text, [], out)
                return out

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
        return {"parser": "grammar", "facts": list(ex.facts), "rules": list(ex.rules), "goals": list(ex.goals),
                "knowledge_facts": [], "knowledge_rules": [], "trace": list(ex.trace),
                "unparsed": list(ex.unparsed), "llm": None}

    def _from_llm(self, text: str) -> Dict[str, Any]:
        p = self.translator.to_premises(text)
        trace = ([f"fact     | {f}" for f in p["facts"]] + [f"rule     | {r['if']} => {r['then']}" for r in p["rules"]]
                 + [f"goal     | {g}" for g in p["goals"]] + [f"knowledge| {k}" for k in p["knowledge_facts"]]
                 + [f"knowledge| {r['if']} => {r['then']}" for r in p["knowledge_rules"]])
        use_k = getattr(self.translator, "use_knowledge", True)
        return {"parser": "llm", "facts": p["facts"], "rules": p["rules"], "goals": p["goals"],
                "knowledge_facts": p["knowledge_facts"] if use_k else [],
                "knowledge_rules": p["knowledge_rules"] if use_k else [], "trace": trace,
                "unparsed": [f"rejected: {r['reason']}" for r in p["rejected"]]
                            + ([] if p["ok"] else ["LLM output unusable"]),
                "llm": {"rejected": p["rejected"], "ok": p["ok"]}}

    # -- grounding ------------------------------------------------------------
    @staticmethod
    def _ground(rules: List[Dict[str, Any]], constants: List[str]) -> List[Dict[str, Any]]:
        out, seen = [], set()
        for r in rules:
            variables = sorted({a for atom in (r["if"], r["then"]) for a in _args(atom) if VARIABLE.match(a)})
            if not variables:
                if (r["if"], r["then"]) not in seen:
                    seen.add((r["if"], r["then"]))
                    out.append(r)
                continue
            for combo in itertools.product(constants[:40], repeat=len(variables)):
                sub = dict(zip(variables, combo))

                def apply(atom: str) -> str:
                    m = ATOM.match(atom.strip())
                    args = [sub.get(a, a) for a in (m.group(3), m.group(4)) if a]
                    return f"{m.group(1) or ''}{m.group(2)}({','.join(args)})"
                g = {**r, "if": apply(r["if"]), "then": apply(r["then"])}
                if (g["if"], g["then"]) not in seen:
                    seen.add((g["if"], g["then"]))
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
            pos, neg = g in known, _negate(g) in known
            if pos:
                v, conf = ("oui" if not g.startswith("not ") else "non"), known[g]["confidence"]
            elif neg:
                v, conf = ("non" if not g.startswith("not ") else "oui"), known[_negate(g)]["confidence"]
            else:
                v, conf = "inconnu", 0.0
            verdicts.append({"goal": g, "answer": v, "confidence": conf,
                             "support": [known[x] for x in (g, _negate(g)) if x in known]})
        return verdicts

    def _decide(self, b: Dict[str, Any], max_steps: int, perception: Any, t0: float) -> Dict[str, Any]:
        facts = list(dict.fromkeys(b["facts"]))
        goals = list(dict.fromkeys(b["goals"]))
        # a question about a category reasons on a typical member ("Do whales breathe air?")
        categories = {_args(r["if"])[1] for r in b["rules"] + b["knowledge_rules"]
                      if r["if"].startswith("is(") and len(_args(r["if"])) == 2 and VARIABLE.match(_args(r["if"])[0])}
        for g in goals:
            a = _args(g)
            if a and a[0] in categories and not any(_args(f)[:1] == [a[0]] for f in facts):
                facts.append(f"is({a[0]},{a[0]})")
        constants = list(dict.fromkeys(a for atom in facts + goals + b["knowledge_facts"] for a in _args(atom)
                                       if not VARIABLE.match(a)))
        strict_rules = self._ground(b["rules"], constants) + self._containment_axioms(facts)
        known_s, reasoning = self._deduce(facts, strict_rules, max_steps)
        strict = self._verdicts(known_s, goals)
        has_k = bool(b["knowledge_facts"] or b["knowledge_rules"])
        if has_k:
            facts_k = facts + [f for f in b["knowledge_facts"] if f not in facts]
            rules_k = self._ground(b["rules"] + b["knowledge_rules"], constants) + self._containment_axioms(facts_k)
            known, reasoning = self._deduce(facts_k, rules_k, max_steps)
            verdicts = self._verdicts(known, goals)
        else:
            known, verdicts = known_s, strict
        derived = [k for k, v in known.items() if v.get("from") != "premise"]
        answer = verdicts[0]["answer"] if verdicts else ("faits enregistrés" if facts else "inconnu")
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
            if not m or not m.group(4):
                continue
            neg, pred, a, b = m.group(1), m.group(2), m.group(3), m.group(4)
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
