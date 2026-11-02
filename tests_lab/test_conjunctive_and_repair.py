"""Offline tests (no API): conjunctive rules, bare propositions, corrective round, unsafe rule."""
import json, logging, sys
logging.disable(logging.CRITICAL)
sys.path.insert(0, ".")
from grandnexus.cognition.reasoning.reasoning_core import ReasoningCore
from grandnexus.cognition.reasoning.text_reasoner import TextReasoner
from grandnexus.llm_interface.llm_translator import LLMTranslator
rc = ReasoningCore()
def run(text, replies, label):
    it = iter(replies)
    tl = LLMTranslator(call=lambda s, p: json.dumps(next(it)))
    o = TextReasoner(reasoning_core=rc, translator=tl, mode="llm").answer(text, write_back=False)
    got = o["answer"] if o["answer"] != "unknown" else o.get("answer_with_knowledge", o["answer"])  # v7.0: starred answer apart
    print(f"{label:46} -> {got:14} star={o.get('depends_on_knowledge')} calls={tl.stats['calls']} repaired={tl.stats['repairs_improved']} rejected={[r['reason'][:40] for r in o['llm']['rejected']]}")
    return o
# 1. comparative chain with a conjunctive transitivity rule in knowledge (neutral names)
T1 = "Ana is older than Ben. Ben is older than Carl. Is Ana older than Carl?"
run(T1, [{"facts": [{"p": "older(ana,ben)", "quote": "Ana is older than Ben"}, {"p": "older(ben,carl)", "quote": "Ben is older than Carl"}],
          "rules": [], "goals": [{"p": "older(ana,carl)", "quote": "Is Ana older than Carl"}],
          "knowledge": [{"if": ["older(X,Y)", "older(Y,Z)"], "then": "older(X,Z)", "why": "older is transitive"}]}], "1 transitivity (conjunctive knowledge)")
# 2. same, but "Is Carl older than Ana?" must NOT be yes
T2 = "Ana is older than Ben. Ben is older than Carl. Is Carl older than Ana?"
run(T2, [{"facts": [{"p": "older(ana,ben)", "quote": "Ana is older than Ben"}, {"p": "older(ben,carl)", "quote": "Ben is older than Carl"}],
          "rules": [], "goals": [{"p": "older(carl,ana)", "quote": "Is Carl older than Ana"}],
          "knowledge": [{"if": ["older(X,Y)", "older(Y,Z)"], "then": "older(X,Z)", "why": "older is transitive"}]}], "2 reverse question (must not be oui)")
# 3. impersonal proposition as a bare word
T3 = "When it is late, the shop closes. It is late. Is the shop closed?"
run(T3, [{"facts": [{"p": "late", "quote": "It is late"}], "rules": [{"if": "late", "then": "is(shop,closed)", "quote": "When it is late, the shop closes"}],
          "goals": [{"p": "is(shop,closed)", "quote": "Is the shop closed"}], "knowledge": []}], "3 bare proposition 'late' -> prop(late)")
# 4. corrective round: first reply malformed, second fixed
run(T3, [{"facts": [{"p": "late()", "quote": "It is late"}], "rules": [{"if": "late()", "then": "is(shop,closed)", "quote": "When it is late, the shop closes"}],
          "goals": [{"p": "is(shop,closed)", "quote": "Is the shop closed"}], "knowledge": []},
         {"facts": [{"p": "prop(late)", "quote": "It is late"}], "rules": [{"if": "prop(late)", "then": "is(shop,closed)", "quote": "When it is late, the shop closes"}],
          "goals": [{"p": "is(shop,closed)", "quote": "Is the shop closed"}], "knowledge": []}], "4 corrective round fixes 'late()'")
# 5. unsafe conjunctive rule rejected
run(T1, [{"facts": [{"p": "older(ana,ben)", "quote": "Ana is older than Ben"}], "rules": [], "goals": [{"p": "older(ana,carl)", "quote": "Is Ana older than Carl"}],
          "knowledge": [{"if": ["older(X,Y)", "older(Y,W)"], "then": "older(X,Z)", "why": "bad"}]},
         {"facts": [{"p": "older(ana,ben)", "quote": "Ana is older than Ben"}], "rules": [], "goals": [{"p": "older(ana,carl)", "quote": "Is Ana older than Carl"}], "knowledge": []}],
    "5 unsafe conjunctive rule (W/Z) rejected")
