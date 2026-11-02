"""Offline tests (no API): 3-argument atoms (event with object and moment), and a question that
cannot be formalised must give "inconnu" (never "faits enregistrés")."""
import json, logging, sys
logging.disable(logging.CRITICAL)
sys.path.insert(0, ".")
from grandnexus.cognition.reasoning.reasoning_core import ReasoningCore
from grandnexus.cognition.reasoning.text_reasoner import TextReasoner
from grandnexus.llm_interface.llm_translator import LLMTranslator
rc = ReasoningCore(); ok = True
def run(text, reply, label, expected):
    global ok
    tl = LLMTranslator(call=lambda s, p: json.dumps(reply), repair_rounds=0)
    o = TextReasoner(reasoning_core=rc, translator=tl, mode="llm").answer(text, write_back=False)
    good = o["answer"] == expected; ok &= good
    print(("✓ " if good else "✗ ") + f"{label:52} -> {o['answer']:10} rejets={[r['reason'][:30] for r in o['llm']['rejected']]}")
T = "Lea bought books on Monday. Did Lea buy books on Monday?"
F = [{"p": "buy(lea,book,monday)", "quote": "Lea bought books on Monday"}]
run(T, {"facts": F, "rules": [], "goals": [{"p": "buy(lea,book,monday)", "quote": "Did Lea buy books on Monday"}], "knowledge": []},
    "atome à 3 arguments accepté", "oui")
T2 = "Lea bought books on Monday. Did Lea buy books on Tuesday?"
run(T2, {"facts": F, "rules": [], "goals": [{"p": "buy(lea,book,tuesday)", "quote": "Did Lea buy books on Tuesday"}], "knowledge": []},
    "autre moment : pas de report implicite", "inconnu")
T3 = "Lea bought books on Monday. Is Lea richer than Tom?"
run(T3, {"facts": F, "rules": [], "goals": [], "knowledge": []}, "question non formalisée -> inconnu", "inconnu")
run(T3, {"facts": F, "rules": [], "goals": [{"p": "richer(lea,tom,now,extra)", "quote": "Is Lea richer than Tom"}], "knowledge": []},
    "atome à 4 arguments rejeté -> inconnu", "inconnu")
run("Lea bought books on Monday.", {"facts": F, "rules": [], "goals": [], "knowledge": []}, "sans question : faits enregistrés", "faits enregistrés")
print("TOUS OK" if ok else "ÉCHEC")
