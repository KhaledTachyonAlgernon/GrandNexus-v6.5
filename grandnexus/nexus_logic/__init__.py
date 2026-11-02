"""Formal language + parser + reference deduction engine. Semantics: see core.py."""
from .core import (ENGINE_VERSION, LABEL_ALIASES, LANGUAGE_SPEC, MAX_ARITY, normalise_label, Atom, DRule, Engine, Lit, Program, Rejection,
                   Rule, Verdict, decide, leaves, parse, parse_literal, run, safe_eval)
from .bridge import program_from_atoms, verdict_from_atoms

__all__ = ["ENGINE_VERSION", "LABEL_ALIASES", "normalise_label", "LANGUAGE_SPEC", "MAX_ARITY", "Atom", "DRule", "Engine", "Lit", "Program",
           "Rejection", "Rule", "Verdict", "decide", "leaves", "parse", "parse_literal", "run", "safe_eval",
           "program_from_atoms", "verdict_from_atoms"]
