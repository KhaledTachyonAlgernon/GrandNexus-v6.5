"""Langage formel + analyseur + moteur de déduction de référence. Sémantique : voir core.py."""
from .core import (ENGINE_VERSION, LANGUAGE_SPEC, MAX_ARITY, Atom, DRule, Engine, Lit, Program, Rejection,
                   Rule, Verdict, decide, leaves, parse, parse_literal, run, safe_eval)
from .bridge import program_from_atoms, verdict_from_atoms

__all__ = ["ENGINE_VERSION", "LANGUAGE_SPEC", "MAX_ARITY", "Atom", "DRule", "Engine", "Lit", "Program",
           "Rejection", "Rule", "Verdict", "decide", "leaves", "parse", "parse_literal", "run", "safe_eval",
           "program_from_atoms", "verdict_from_atoms"]
