"""Offline tests (no API): typical member == category, same(a,b) co-reference, time-indexed states."""
import json, logging, sys
logging.disable(logging.CRITICAL)
sys.path.insert(0, ".")
from grandnexus.cognition.reasoning.reasoning_core import ReasoningCore
from grandnexus.cognition.reasoning.text_reasoner import TextReasoner
from grandnexus.llm_interface.llm_translator import LLMTranslator
rc = ReasoningCore()
ok = True
def run(text, reply, label, expected):
    global ok
    tl = LLMTranslator(call=lambda s, p: json.dumps(reply), repair_rounds=0)
    o = TextReasoner(reasoning_core=rc, translator=tl, mode="llm").answer(text, write_back=False)
    got = o["answer"] if o["answer"] != "unknown" else o.get("answer_with_knowledge", o["answer"])  # v7.0: starred answer apart
    good = got == expected
    ok &= good
    alias = [t for t in o["extraction"]["trace"] if t.startswith("alias")]
    print(("✓ " if good else "✗ ") + f"{label:50} -> {got:10} (text only : {o['strict_answer']}) {alias}")
T = "A typical nurse works at night. Working at night requires coffee. Do nurses need coffee?"
F = [{"p": "work(typical_nurse,night)", "quote": "A typical nurse works at night"}]
R = [{"if": "work(X,night)", "then": "need(X,coffee)", "quote": "Working at night requires coffee"}]
G = [{"p": "need(nurse,coffee)", "quote": "Do nurses need coffee"}]
run(T, {"facts": F, "rules": R, "goals": G, "knowledge": []}, "typical member (typical_nurse == nurse)", "yes")
F2 = [{"p": "work(average_carer,night)", "quote": "A typical nurse works at night"}]
run(T, {"facts": F2, "rules": R, "goals": G, "knowledge": [{"p": "same(average_carer,nurse)", "why": "same entity"}]},
    "co-reference same(a,b) declared by Claude", "yes")
run(T, {"facts": F2, "rules": R, "goals": G, "knowledge": []}, "sans same() : no invented link", "unknown")
L = "The lamp was on this morning. Paul switched it off at noon. Is the lamp on this afternoon?"
run(L, {"facts": [{"p": "on(lamp,morning)", "quote": "The lamp was on this morning"},
                  {"p": "switched_off(lamp,noon)", "quote": "Paul switched it off at noon"}],
        "rules": [], "goals": [{"p": "on(lamp,afternoon)", "quote": "Is the lamp on this afternoon"}],
        "knowledge": [{"if": "switched_off(lamp,noon)", "then": "not on(lamp,afternoon)", "why": "stays off until switched on"}]},
    "dated states: no false contradiction", "no")
L2 = "The lamp was on this morning. Is the lamp on this afternoon?"
run(L2, {"facts": [{"p": "on(lamp,morning)", "quote": "The lamp was on this morning"}], "rules": [],
         "goals": [{"p": "on(lamp,afternoon)", "quote": "Is the lamp on this afternoon"}], "knowledge": []},
    "no persistence assumed without a rule", "unknown")
print("ALL OK" if ok else "FAIL")
