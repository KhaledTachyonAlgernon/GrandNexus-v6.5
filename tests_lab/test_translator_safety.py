"""Offline safety test (no API): an unsafe LLM rule must not create a false 'yes'."""
import json, logging, sys
logging.disable(logging.CRITICAL)
sys.path.insert(0, ".")
from grandnexus.cognition.reasoning.reasoning_core import ReasoningCore
from grandnexus.cognition.reasoning.text_reasoner import TextReasoner
from grandnexus.llm_interface.llm_translator import LLMTranslator
TEXT = "The key is inside the drawer. The drawer is part of the desk. The desk is located in the office. Is the key in the kitchen?"
REPLY = {"facts": [{"p": "in(key,drawer)", "quote": "The key is inside the drawer"},
                   {"p": "in(drawer,desk)", "quote": "The drawer is part of the desk"},
                   {"p": "in(desk,office)", "quote": "The desk is located in the office"}],
         "rules": [], "goals": [{"p": "in(key,kitchen)", "quote": "Is the key in the kitchen"}],
         "knowledge": [{"if": "in(X,Y)", "then": "in(X,Z)", "why": "containment is transitive"}]}
tl = LLMTranslator(call=lambda system, prompt: json.dumps(REPLY))
tr = TextReasoner(reasoning_core=ReasoningCore(), translator=tl, mode="llm")
out = tr.answer(TEXT, write_back=False)
print("Is the key in the kitchen? ->", out["answer"], "| text only :", out["strict_answer"],
      "| rejections :", [r["reason"] for r in out["llm"]["rejected"]])
assert out["answer"] == "unknown", "unsafe rule produced an answer"
print("OK: unsafe rule rejected")
