"""Langage formel + analyseur + moteur de déduction de référence. Sémantique : voir core.py."""
from .core import (ENGINE_VERSION, LANGUAGE_SPEC, MAX_ARITY, Atom, DRule, Engine, Lit, Program, Rejection,
                   Rule, Verdict, decide, leaves, parse, parse_literal, run, safe_eval)

__all__ = ["ENGINE_VERSION", "LANGUAGE_SPEC", "MAX_ARITY", "Atom", "DRule", "Engine", "Lit", "Program",
           "Rejection", "Rule", "Verdict", "decide", "leaves", "parse", "parse_literal", "run", "safe_eval"]
