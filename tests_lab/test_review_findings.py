"""Findings of the external code review (Fable 5.1, 2026-10-25), reproduced offline.
On v6.8.1 every line showed the FLAW; on v7.0 each must read: 1a unknown+rejection, 1b unknown, 2.4 yes, 2.5 rejectioné, 2.3 oui, 2.2 oui."""
import json, logging, sys
logging.disable(logging.CRITICAL)
sys.path.insert(0, ".")
from grandnexus.cognition.reasoning.reasoning_core import ReasoningCore
from grandnexus.cognition.reasoning.text_reasoner import TextReasoner
from grandnexus.llm_interface.llm_translator import LLMTranslator, anchored
rc = ReasoningCore()
def run(text, reply, mode="llm"):
    tl = LLMTranslator(call=lambda s, p: json.dumps(reply), repair_rounds=0)
    return TextReasoner(reasoning_core=rc, translator=tl, mode=mode).answer(text, write_back=False)
T = "Ana est plus âgée que Ben. Ben n'est pas médecin."
print("1a bag-of-words anchoring  :", anchored("Ana est médecin", T), "| single word :", anchored("Ben", T))
o = run(T + " Ana est-elle médecin ?", {"facts": [{"p": "is(ana,medecin)", "quote": "Ana est médecin"}], "rules": [],
        "goals": [{"p": "is(ana,medecin)", "quote": "Ana est-elle médecin"}], "knowledge": []})
print("1a invented fact accepted  :", o["answer"], "| rejections :", o["llm"]["rejected"])
o = run("Socrates is a man. Is Socrates mortal?", {"facts": [{"p": "is(socrates,man)", "quote": "Socrates is a man"}], "rules": [],
        "goals": [{"p": "is(socrates,mortal)", "quote": "Is Socrates mortal"}],
        "knowledge": [{"p": "is(socrates,mortal)", "why": "men die"}]})
print("1b knowledge = but          :", o["answer"], "| star :", o["depends_on_knowledge"], "| text only :", o["strict_answer"])
o = run("A typical nurse is kind. Nurses work at the hospital. Does a typical nurse work at the hospital?",
        {"facts": [{"p": "is(nurse,kind)", "quote": "A typical nurse is kind"}],
         "rules": [{"if": "is(X,nurse)", "then": "work(X,hospital)", "quote": "Nurses work at the hospital"}],
         "goals": [{"p": "work(nurse,hospital)", "quote": "Does a typical nurse work at the hospital"}], "knowledge": []})
print("2.4 garde typical member :", o["answer"])
o = run("All men are mortal. Is Socrates mortal?", {"facts": [{"p": "is(X,man)", "quote": "All men are mortal"}], "rules": [],
        "goals": [{"p": "is(socrates,mortal)", "quote": "Is Socrates mortal"}], "knowledge": []})
print("2.5 fact with variable   :", "accepted" if "is(X,man)" in o["extraction"]["facts"] else "rejected")
o = run("The lamp is not on. Is it true that the lamp is not on?", {"facts": [{"p": "not on(lamp)", "quote": "The lamp is not on"}], "rules": [],
        "goals": [{"p": "not on(lamp)", "quote": "Is it true that the lamp is not on"}], "knowledge": []})
print("2.3 negative goal          :", o["answer"], '(a human reads "yes")')
tr = TextReasoner(reasoning_core=None, translator=LLMTranslator(call=lambda s, p: json.dumps({"facts": [{"p": "is(socrates,man)", "quote": "Socrates is a man"}],
     "rules": [{"if": "is(X,man)", "then": "is(X,mortal)", "quote": "All men are mortal"}], "goals": [{"p": "is(socrates,mortal)", "quote": "Is Socrates mortal"}], "knowledge": []}), repair_rounds=0), mode="llm")
print("2.2 engine absent        :", tr.answer("All men are mortal. Socrates is a man. Is Socrates mortal?", write_back=False)["answer"], "(no error)")
