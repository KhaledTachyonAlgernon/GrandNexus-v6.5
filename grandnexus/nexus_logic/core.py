"""
nexus_logic — langage formel, analyseur et moteur de déduction de référence (candidat pour remplacer le
moteur de text_reasoner, ou lui servir d'oracle différentiel).

Ce que le carnet a montré : la plupart des échecs venaient (a) d'un écart entre ce que le prompt demande et
ce que l'analyseur accepte (arité, relations, propositions, conjonctions), (b) de propriétés du moteur
redécouvertes au cas par cas (R32 profondeur/contraposée, R37 sûreté, R38 conjonctions, same(), temps).
Ici le langage est défini UNE fois (LANGUAGE_SPEC, montré tel quel au traducteur, appliqué tel quel par
l'analyseur) et la sémantique du moteur est explicite, donc testable et comparable à clingo (asp.py).

Sémantique
- Faits signés clos ; règles à corps conjonctif et conclusion unique ; aucun symbole de fonction
  => base de Herbrand finie => le chaînage avant naïf termine toujours.
- Négation classique et EXPLICITE : `not p(a)` est un littéral stocké. Un littéral négatif en condition
  ne s'apparie qu'à un littéral négatif stocké (pas de négation par échec). Absence == inconnu.
- Sûreté (R37) : chaque variable de la conclusion d'une règle apparaît dans une condition.
- Contraposées (R32) ajoutées automatiquement quand elles sont sûres ; sinon réservées à un pas dirigé
  par le but sur la question posée (likes(X,Y) => happy(X). not happy(john). ? likes(john, mary) -> non).
- same(a, b) fusionne deux constantes (union-find) ; typical_x == x en option (convention du prompt v4).
- let c = expr lie une constante numérique (arithmétique évaluée sans eval()).
- Verdict : contradictoire si la clôture contient p et not p ; oui si le but est dérivé ; non si sa
  négation l'est ; inconnu sinon. Une question avec variables renvoie les liaisons (« combien », « qui »).
- Chaque littéral dérivé garde sa justification : preuve imprimable, énoncés utilisés, dépendance à la
  connaissance de fond (énoncés marqués *), sans second passage.
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

ENGINE_VERSION = "ref-2.0"  # 1.1: reductio ; 1.2: universal questions ; 2.0: disjunctions (case split)
MAX_ARITY = 4  # relation ternaire + moment ; le carnet a perdu un item sur une limite à 2 puis 3

LANGUAGE_SPEC = """\
program   := { line }                         one statement per line; a line ends with "."
line      := [ "*" ] statement [ @ "span" ]   "*" = background knowledge (not in the text)
                                              span = exact quote of the text words justifying the statement
statement := fact | rule | same | let | question
fact      := literal "."
           | literal "|" literal { "|" literal } "."   disjunction: at least one of them holds
rule      := literal { "&" literal } "=>" literal "."
             every variable of the conclusion must appear in a condition
           | literal { "&" literal } "=>" literal "|" literal { "|" literal } "."   disjunctive conclusion
same      := "same(" const ", " const ")."    two names for one entity
let       := "let " const " = " expr "."      expr: numbers, + - * / ( ), earlier let-constants
question  := "? " literal { "&" literal } "."  exactly one per program; may contain a variable
           | "? " literal { "&" literal } "=>" literal "."   universal question ("all/no X are Y")
literal   := [ "not " ] atom
atom      := pred [ "(" term { ", " term } ")" ]   0 to 4 terms; a bare word w means prop(w)
term      := const | var | number
pred      := [a-z][a-z0-9_]*                  lowercase ascii, no accents
const     := [a-z][a-z0-9_]*                  lowercase ascii, no accents, no spaces
var       := [A-Z][A-Za-z0-9_]*               starts with an uppercase letter
number    := integer or decimal
comment   := "#" to end of line (own line)
"""

Term = Union[str, int, float]


def is_var(t: Term) -> bool:
    return isinstance(t, str) and t[:1].isupper()


# ----------------------------------------------------------------------------- termes et littéraux
@dataclass(frozen=True)
class Atom:
    pred: str
    args: Tuple[Term, ...] = ()

    def __str__(self) -> str:
        return f"{self.pred}({', '.join(str(a) for a in self.args)})" if self.args else self.pred


@dataclass(frozen=True)
class Lit:
    atom: Atom
    neg: bool = False

    def flip(self) -> "Lit":
        return Lit(self.atom, not self.neg)

    @property
    def key(self) -> Tuple[bool, str, int]:
        return (self.neg, self.atom.pred, len(self.atom.args))

    def vars(self) -> Set[str]:
        return {a for a in self.atom.args if is_var(a)}

    def __str__(self) -> str:
        return ("not " if self.neg else "") + str(self.atom)


@dataclass(frozen=True)
class Rule:
    body: Tuple[Lit, ...]
    head: Lit
    sid: str
    knowledge: bool = False
    derived: bool = False  # contraposée

    def is_safe(self) -> bool:
        return self.head.vars() <= set().union(*(b.vars() for b in self.body))

    def __str__(self) -> str:
        return " & ".join(str(b) for b in self.body) + " => " + str(self.head)


@dataclass(frozen=True)
class DRule:
    """Rule with a disjunctive conclusion: body => h1 | h2 | ... (case split at solving time)."""
    body: Tuple[Lit, ...]
    heads: Tuple[Lit, ...]
    sid: str
    knowledge: bool = False

    def is_safe(self) -> bool:
        bv = set().union(*(b.vars() for b in self.body))
        return all(h.vars() <= bv for h in self.heads)

    def __str__(self) -> str:
        return " & ".join(str(b) for b in self.body) + " => " + " | ".join(str(h) for h in self.heads)


def subst_lit(lit: Lit, s: Dict[str, Term]) -> Lit:
    return Lit(Atom(lit.atom.pred, tuple(s.get(a, a) if is_var(a) else a for a in lit.atom.args)), lit.neg)


def match(pat: Lit, ground: Lit, subst: Dict[str, Term]) -> Optional[Dict[str, Term]]:
    """Appariement d'un motif sur un littéral clos ; renvoie la substitution étendue ou None."""
    if pat.key != ground.key:
        return None
    s = dict(subst)
    for p, g in zip(pat.atom.args, ground.atom.args):
        if is_var(p):
            if s.setdefault(p, g) != g:
                return None
        elif p != g:
            return None
    return s


def contrapositives(r: Rule) -> List[Rule]:
    """c1 & ... & cn => h  donne, pour chaque i :  (autres cj) & not h => not ci."""
    return [Rule(r.body[:i] + r.body[i + 1:] + (r.head.flip(),), c.flip(), r.sid, r.knowledge, derived=True)
            for i, c in enumerate(r.body)]


# ----------------------------------------------------------------------------- arithmétique sûre
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}


def _norm_num(v: Union[int, float]) -> Union[int, float]:
    return int(v) if isinstance(v, float) and v.is_integer() else v


def safe_eval(expr: str, env: Dict[str, Union[int, float]]) -> Union[int, float]:
    """Évalue + - * / // % ** et parenthèses sur des nombres et des constantes `let` ; jamais eval()."""
    def ev(n: ast.AST) -> Union[int, float]:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in env:
                return env[n.id]
            raise ValueError(f"unknown constant in expression: {n.id}")
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 64:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](a, b)
        raise ValueError("unsupported expression")
    try:
        return _norm_num(ev(ast.parse(expr.strip(), mode="eval").body))
    except (SyntaxError, ZeroDivisionError, OverflowError) as e:
        raise ValueError(f"bad expression: {e}") from None


# ----------------------------------------------------------------------------- analyseur
_IDENT = r"[a-z][a-z0-9_]*"
_VAR = r"[A-Z][A-Za-z0-9_]*"
_NUM = r"-?\d+(?:\.\d+)?"
_ATOM_RE = re.compile(rf"^({_IDENT})\s*(?:\((.*)\))?$")
_TERM_RE = re.compile(rf"^(?:{_IDENT}|{_VAR}|{_NUM})$")
_ANCHOR_RE = re.compile(r'\s*@\s*"([^"]*)"\s*$')
_LET_RE = re.compile(rf"^let\s+({_IDENT})\s*=\s*(.+)$")
_SAME_RE = re.compile(rf"^same\s*\(\s*({_IDENT})\s*,\s*({_IDENT})\s*\)$")
_RESERVED = {"not", "let", "same"}


@dataclass
class Rejection:
    line: int
    text: str
    reason: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.reason} :: {self.text.strip()}"


@dataclass
class Program:
    facts: List[Tuple[Lit, str, bool]] = field(default_factory=list)      # (littéral, id, connaissance ?)
    rules: List[Rule] = field(default_factory=list)
    sames: List[Tuple[str, str]] = field(default_factory=list)
    lets: Dict[str, Union[int, float]] = field(default_factory=dict)
    questions: List[Tuple] = field(default_factory=list)                 # conjonction de littéraux, ou ("universal", i)
    universal_questions: List["Rule"] = field(default_factory=list)       # ? body => head
    disjunctions: List[Tuple[Tuple[Lit, ...], str, bool]] = field(default_factory=list)   # a | b.  (ground)
    drules: List["DRule"] = field(default_factory=list)                   # body => a | b
    rejections: List[Rejection] = field(default_factory=list)
    anchors: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)                       # lignes "# cannot express"

    @property
    def n_statements(self) -> int:
        return len(self.facts) + len(self.rules) + len(self.sames) + len(self.lets) + len(self.disjunctions) + len(self.drules)


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", s.lower()).split())


def parse_term(s: str) -> Term:
    s = s.strip()
    if not _TERM_RE.match(s):
        raise ValueError(f"bad term '{s}'")
    if re.fullmatch(_NUM, s):
        return _norm_num(float(s) if "." in s else int(s))
    return s


def parse_literal(s: str) -> Lit:
    s = s.strip()
    neg = False
    if s.startswith("not "):
        neg, s = True, s[4:].strip()
    elif s.startswith("-"):
        neg, s = True, s[1:].strip()
    m = _ATOM_RE.match(s)
    if not m:
        raise ValueError(f"bad atom '{s}'")
    pred, inner = m.group(1), m.group(2)
    if pred in _RESERVED:
        raise ValueError(f"'{pred}' is reserved")
    args = tuple(parse_term(a) for a in inner.split(",")) if inner and inner.strip() else ()
    if len(args) > MAX_ARITY:
        raise ValueError(f"arity {len(args)} > {MAX_ARITY}")
    if not args:                      # mot nu -> prop(mot)  (v6.7.0 : « bare atoms normalised »)
        pred, args = "prop", (pred,)
    return Lit(Atom(pred, args), neg)


def _split_line(line: str) -> List[str]:
    """Découpe une ligne en énoncés : '.' hors guillemets suivi d'un blanc/fin, ancre @ "..." rattachée."""
    out, buf, inq, i, n = [], [], False, 0, len(line)
    while i < n:
        ch = line[i]
        if ch == '"':
            inq = not inq
        buf.append(ch)
        if ch == "." and not inq and (i + 1 == n or line[i + 1].isspace()):
            j = i + 1
            while j < n and line[j].isspace():
                j += 1
            if j < n and line[j] == "@":
                k = line.find('"', j)
                k2 = line.find('"', k + 1) if k != -1 else -1
                if k2 != -1:
                    buf.append(line[i + 1:k2 + 1])
                    i = k2
            out.append("".join(buf).strip())
            buf = []
        i += 1
    rest = "".join(buf).strip()
    if rest:
        out.append(rest)
    return out


def _parse_statement(prog: Program, seg: str, lineno: int, sid: str, src: Optional[str], require_anchor: bool) -> None:
    knowledge = seg.startswith("*")
    line = seg[1:].strip() if knowledge else seg
    anchor = None
    m = _ANCHOR_RE.search(line)
    if m:
        anchor, line = (m.group(1).strip() or None), line[: m.start()].rstrip()
    if not line.endswith("."):
        prog.rejections.append(Rejection(lineno, seg, "missing final '.'"))
        return
    body = line[:-1].strip()
    try:
        if body.startswith("?"):
            q = body[1:]
            if "=>" in q:                                     # universal question: ? body => head
                lhs, rhs = q.split("=>", 1)
                conds = tuple(parse_literal(c) for c in lhs.split("&"))
                head = parse_literal(rhs)
                if not head.vars() <= set().union(*(c.vars() for c in conds)):
                    raise ValueError("universal question: every variable of the conclusion must appear in a condition")
                prog.universal_questions.append(Rule(conds, head, sid))
                prog.questions.append(("universal", len(prog.universal_questions) - 1))
            else:
                prog.questions.append(tuple(parse_literal(c) for c in q.split("&")))
            return
        if body.startswith("let "):
            mm = _LET_RE.match(body)
            if not mm:
                raise ValueError("bad let (expected: let name = expression)")
            prog.lets[mm.group(1)] = safe_eval(mm.group(2), prog.lets)
            return
        if body.startswith("same"):
            mm = _SAME_RE.match(body)
            if not mm:
                raise ValueError("bad same (expected: same(a, b))")
            prog.sames.append((mm.group(1), mm.group(2)))
            return
        if src is not None and not knowledge:            # ancrage : l'énoncé doit citer le texte
            if anchor is None:
                if require_anchor:
                    raise ValueError('unanchored statement: quote the text span with @ "..."')
            elif _norm(anchor) not in src:
                raise ValueError("anchor not found in the text")
        if anchor:
            prog.anchors[sid] = anchor
        if "=>" in body:
            lhs, rhs = body.split("=>", 1)
            conds = tuple(parse_literal(c) for c in lhs.split("&"))
            if "|" in rhs:                                    # disjunctive conclusion
                heads = tuple(parse_literal(h) for h in rhs.split("|"))
                drule = DRule(conds, heads, sid, knowledge)
                if not drule.is_safe():
                    raise ValueError("unsafe rule: every variable of the conclusion must appear in a condition (R37)")
                prog.drules.append(drule)
                return
            head = parse_literal(rhs)
            if head in conds:
                raise ValueError("rule concludes one of its own conditions")
            rule = Rule(conds, head, sid, knowledge)
            if not rule.is_safe():
                raise ValueError("unsafe rule: every variable of the conclusion must appear in a condition (R37)")
            prog.rules.append(rule)
        elif "|" in body:                                     # ground disjunction
            lits = tuple(parse_literal(x) for x in body.split("|"))
            if any(l.vars() for l in lits):
                raise ValueError("disjunction with a variable: use constants, or a rule with a disjunctive conclusion")
            prog.disjunctions.append((lits, sid, knowledge))
        else:
            lit = parse_literal(body)
            if lit.vars():
                raise ValueError("fact with a variable: use a constant, or a rule")
            prog.facts.append((lit, sid, knowledge))
    except ValueError as e:
        prog.rejections.append(Rejection(lineno, seg, str(e)))


def parse(text: str, source_text: Optional[str] = None, require_anchor: bool = False) -> Program:
    """Analyse un programme ; les énoncés refusés vont dans prog.rejections avec la raison (pour le tour correctif)."""
    prog = Program()
    src = _norm(source_text) if source_text else None
    n = 0
    for lineno, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("```"):
            continue
        if stripped.startswith("#"):
            prog.notes.append(stripped)
            continue
        for seg in _split_line(stripped):
            n += 1
            _parse_statement(prog, seg, lineno, f"s{n}", src, require_anchor)
    return prog


# ----------------------------------------------------------------------------- moteur
class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if (len(rb), rb) < (len(ra), ra):   # représentant = nom le plus court (lisible dans les preuves)
                ra, rb = rb, ra
            self.parent[rb] = ra


@dataclass
class Verdict:
    answer: str                                   # oui | non | inconnu | contradictoire
    bindings: List[Dict[str, Term]] = field(default_factory=list)
    proof: List[str] = field(default_factory=list)
    used: List[str] = field(default_factory=list)  # ids des énoncés utilisés
    depends_on_knowledge: bool = False
    contradictions: List[str] = field(default_factory=list)
    note: str = ""

    def short(self) -> str:
        """Réponse courte : la valeur liée pour « combien / qui », sinon le verdict."""
        if self.answer == "oui" and self.bindings and all(len(b) == 1 for b in self.bindings):
            return ", ".join(sorted({str(next(iter(b.values()))) for b in self.bindings}))
        return self.answer


class Engine:
    def __init__(self, prog: Program, use_knowledge: bool = True, contrapositives: bool = True,
                 merge_typical: bool = True, max_facts: int = 50_000, reductio: bool = True,
                 extra_facts: Optional[Sequence[Lit]] = None) -> None:
        self.prog, self.use_knowledge, self.merge_typical, self.max_facts = prog, use_knowledge, merge_typical, max_facts
        self.reductio, self.extra_facts = reductio, list(extra_facts or [])
        self.uf = _UnionFind()
        self.kb: Set[Lit] = set()
        self.index: Dict[Tuple[bool, str, int], List[Lit]] = {}
        self.why: Dict[Lit, Tuple[Optional[Rule], Tuple[Lit, ...], str]] = {}
        self.knowledge_sids: Set[str] = set()
        self.rules: List[Rule] = []
        self.goal_rules: List[Rule] = []      # contraposées non sûres : un pas dirigé par le but seulement
        self.facts0: List[Lit] = []
        self.overflow = False
        self.rounds = 0
        self._build(contrapositives)
        self._chase()
        self.contradictions: List[Lit] = sorted((l for l in self.kb if not l.neg and l.flip() in self.kb), key=str)
        self.pending: List[Tuple[Tuple[Lit, ...], str]] = []       # ground disjunctions not yet decided
        self.refuted_disjunction: Optional[Tuple[Lit, ...]] = None  # a disjunction with every disjunct refuted
        self._collect_disjunctions()

    @property
    def inconsistent(self) -> bool:
        return bool(self.contradictions) or self.refuted_disjunction is not None

    def _collect_disjunctions(self) -> None:
        cands: List[Tuple[Tuple[Lit, ...], str]] = []
        for lits, sid, kn in self.prog.disjunctions:
            if kn and not self.use_knowledge:
                continue
            cands.append((tuple(self.canon(l) for l in lits), sid))
        for r in self.prog.drules:
            if r.knowledge and not self.use_knowledge:
                continue
            body = tuple(self.canon(b) for b in r.body)
            for sub in self._matches(body, {}):
                cands.append((tuple(subst_lit(self.canon(h), sub) for h in r.heads), r.sid))
        seen = set()
        for lits, sid in cands:
            if lits in seen:
                continue
            seen.add(lits)
            if any(l in self.kb for l in lits):
                continue                                           # already satisfied
            if all(l.flip() in self.kb for l in lits):
                self.refuted_disjunction = lits                    # branch inconsistent
                return
            self.pending.append((lits, sid))

    # --- construction
    def _all_terms(self) -> Iterator[Term]:
        for lit, _, _ in self.prog.facts:
            yield from lit.atom.args
        for r in self.prog.rules:
            for l in r.body + (r.head,):
                yield from l.atom.args
        for q in self.prog.questions:
            if q and q[0] == "universal":
                continue
            for l in q:
                yield from l.atom.args

    def canon(self, lit: Lit) -> Lit:
        """Substitue les constantes `let` et les représentants same()/typical_."""
        def ct(t: Term) -> Term:
            if isinstance(t, str) and not is_var(t):
                return self.prog.lets[t] if t in self.prog.lets else self.uf.find(t)
            return t
        return Lit(Atom(lit.atom.pred, tuple(ct(a) for a in lit.atom.args)), lit.neg)

    def _build(self, with_contrapositives: bool) -> None:
        for a, b in self.prog.sames:
            self.uf.union(a, b)
        if self.merge_typical:
            for t in list(self._all_terms()):
                if isinstance(t, str) and not is_var(t) and t.startswith("typical_") and len(t) > 8:
                    self.uf.union(t, t[8:])
        for lit, sid, kn in self.prog.facts:
            if kn and not self.use_knowledge:
                continue
            if kn:
                self.knowledge_sids.add(sid)
            c = self.canon(lit)
            self.facts0.append(c)
            self._add(c, None, (), sid)
        for h in self.extra_facts:                       # hypotheses of a reductio trial
            self._add(self.canon(h), None, (), "hypothesis")
        for r in self.prog.rules:
            if r.knowledge and not self.use_knowledge:
                continue
            if r.knowledge:
                self.knowledge_sids.add(r.sid)
            cr = Rule(tuple(self.canon(b) for b in r.body), self.canon(r.head), r.sid, r.knowledge)
            self.rules.append(cr)
            if with_contrapositives:
                for c in contrapositives(cr):
                    (self.rules if c.is_safe() else self.goal_rules).append(c)

    def _add(self, lit: Lit, rule: Optional[Rule], prem: Tuple[Lit, ...], sid: str) -> bool:
        if lit in self.kb:
            return False
        if len(self.kb) >= self.max_facts:
            self.overflow = True
            return False
        self.kb.add(lit)
        self.index.setdefault(lit.key, []).append(lit)
        self.why[lit] = (rule, prem, sid)
        return True

    # --- chaînage
    def _matches(self, body: Sequence[Lit], subst: Dict[str, Term], i: int = 0) -> Iterator[Dict[str, Term]]:
        if i == len(body):
            yield subst
            return
        for g in self.index.get(body[i].key, ()):
            s = match(body[i], g, subst)
            if s is not None:
                yield from self._matches(body, s, i + 1)

    def _chase(self) -> None:
        changed = True
        while changed and not self.overflow:
            changed = False
            self.rounds += 1
            new: List[Tuple[Lit, Rule, Tuple[Lit, ...]]] = []
            for r in self.rules:
                for s in self._matches(r.body, {}):
                    h = subst_lit(r.head, s)
                    if h not in self.kb:
                        new.append((h, r, tuple(subst_lit(b, s) for b in r.body)))
            for h, r, prem in new:
                if self._add(h, r, prem, r.sid):
                    changed = True

    def _goal_directed(self, goal: Lit) -> bool:
        for r in self.goal_rules:
            s = match(r.head, goal, {})
            if s is None:
                continue
            for s2 in self._matches(r.body, s):
                return self._add(goal, r, tuple(subst_lit(b, s2) for b in r.body), r.sid)
        return False

    # --- preuves
    def _proof(self, lit: Lit, depth: int = 0, seen: frozenset = frozenset()) -> List[str]:
        rule, prem, sid = self.why[lit]
        pad = "  " * depth
        if rule is None:
            return [f"{pad}{lit}   [{'knowledge' if sid in self.knowledge_sids else 'text'} {sid}]"]
        head = f"{pad}{lit}   [{sid}{' contrapositive' if rule.derived else ''}: {rule}]"
        if lit in seen:
            return [head + " ..."]
        out = [head]
        for p in prem:
            out += self._proof(p, depth + 1, seen | {lit})
        return out

    def _used(self, lit: Lit) -> Set[str]:
        out: Set[str] = set()
        seen: Set[Lit] = set()
        stack = [lit]
        while stack:
            l = stack.pop()
            if l in seen:
                continue
            seen.add(l)
            rule, prem, sid = self.why[l]
            out.add(sid)
            stack.extend(prem)
        return out

    def _verdict(self, ans: str, lits: List[Lit], bindings: Optional[List[Dict[str, Term]]] = None) -> Verdict:
        used: Set[str] = set().union(*(self._used(l) for l in lits))
        via_goal = any(self.why[l][0] is not None and self.why[l][0] in self.goal_rules for l in lits)
        return Verdict(ans, bindings=bindings or [], proof=sum((self._proof(l) for l in lits), []),
                       used=sorted(used), depends_on_knowledge=bool(used & self.knowledge_sids),
                       note="goal-directed contrapositive (not exported to ASP)" if via_goal else "")

    # --- réponse
    def answer(self, goal: Optional[Sequence[Lit]] = None) -> Verdict:
        if goal is None:
            if not self.prog.questions:
                return Verdict("inconnu", note="no question formalised")   # jamais « faits enregistrés »
            goal = self.prog.questions[0]
        if goal and goal[0] == "universal":
            return self._answer_universal(self.prog.universal_questions[goal[1]])
        gs = tuple(self.canon(l) for l in goal)
        if self.overflow:
            return Verdict("inconnu", note=f"fact limit {self.max_facts} reached")
        if self.refuted_disjunction is not None and not self.contradictions:
            d = self.refuted_disjunction
            return Verdict("contradictoire", contradictions=[" | ".join(str(l) for l in d)],
                           proof=[f"CONTRADICTION: every disjunct of {' | '.join(str(l) for l in d)} is refuted"]
                           + sum((self._proof(l.flip()) for l in d), []))
        if self.contradictions:
            c = self.contradictions[0]
            used = self._used(c) | self._used(c.flip())
            return Verdict("contradictoire", contradictions=[f"{l} / {l.flip()}" for l in self.contradictions],
                           proof=[f"CONTRADICTION {c} / {c.flip()}"] + self._proof(c) + self._proof(c.flip()),
                           used=sorted(used), depends_on_knowledge=bool(used & self.knowledge_sids))
        if any(l.vars() for l in gs):
            binds = list(self._matches(gs, {}))
            if not binds:
                return Verdict("inconnu", note="no instance derivable")
            return self._verdict("oui", [subst_lit(l, binds[0]) for l in gs], bindings=binds)
        if all(l in self.kb or self._goal_directed(l) for l in gs):
            return self._verdict("oui", list(gs))
        for l in gs:
            if l.flip() in self.kb or self._goal_directed(l.flip()):
                return self._verdict("non", [l.flip()])
        # Reductio (classical refutation, FOLIO run of 2026-10-25: 10 of 16 "non" items were missed).
        # The knowledge base is consistent here. If assuming the goal makes it inconsistent, the
        # goal is false; if assuming its negation does, the goal is true. Sound: the rules only
        # derive classical consequences, so KB + g inconsistent means KB entails not g.
        if self.reductio and len(gs) == 1:
            g = gs[0]
            for hyp, ans in ((g, "non"), (g.flip(), "oui")):
                trial = Engine(self.prog, self.use_knowledge, True, self.merge_typical, self.max_facts,
                               reductio=False, extra_facts=self.extra_facts + [hyp])
                if trial.inconsistent and not trial.overflow:
                    c = trial.contradictions[0] if trial.contradictions else None
                    if c is None:                               # a disjunction with every disjunct refuted
                        return Verdict(ans, proof=[f"REDUCTIO: assume {hyp}: a disjunction has all its disjuncts refuted"],
                                       note="by refutation (reductio ad absurdum)")
                    used = (trial._used(c) | trial._used(c.flip())) - {"hypothesis"}
                    return Verdict(ans, proof=[f"REDUCTIO: assume {hyp}"] + trial._proof(c) + trial._proof(c.flip()),
                                   used=sorted(used), depends_on_knowledge=bool(used & trial.knowledge_sids),
                                   note="by refutation (reductio ad absurdum)")
        return Verdict("inconnu")

    def _answer_universal(self, q: Rule) -> Verdict:
        """? body => head  ("all X are Y", "no X is Y" with a negated head).
        non : a counterexample among the known constants (body derived, head refuted);
        oui : universal introduction — fresh constants for the variables, body assumed, head derived;
        inconnu otherwise. (FOLIO 2026-10-25: items 306 and 325 were universal negatives that the
        translator could only approximate by an existential question, which produced two false "oui".)"""
        if self.overflow:
            return Verdict("inconnu", note=f"fact limit {self.max_facts} reached")
        if self.contradictions:
            return self.answer(())                      # reports "contradictoire" as usual
        body = tuple(self.canon(b) for b in q.body)
        head = self.canon(q.head)
        for s in self._matches(body, {}):               # counterexample search
            h = subst_lit(head, s)
            if h.flip() in self.kb:
                return self._verdict("non", [h.flip()] + [subst_lit(b, s) for b in body])
            if self.reductio:
                trial = Engine(self.prog, self.use_knowledge, True, self.merge_typical, self.max_facts,
                               reductio=False, extra_facts=self.extra_facts + [h])
                if trial.inconsistent and not trial.overflow:
                    return Verdict("non", proof=[f"COUNTEREXAMPLE {s}: assuming {h} is inconsistent"],
                                   note="universal refuted by a counterexample (reductio)")
        fresh = {v: f"fresh_{v.lower()}" for b in body for v in b.vars()}
        hyp = [subst_lit(b, fresh) for b in body]
        trial = Engine(self.prog, self.use_knowledge, True, self.merge_typical, self.max_facts,
                       reductio=False, extra_facts=self.extra_facts + hyp)
        target = subst_lit(head, fresh)
        if target in trial.kb and not trial.contradictions:
            used = trial._used(target) - {"hypothesis"}
            return Verdict("oui", proof=[f"UNIVERSAL: for a fresh {fresh}, assume {' & '.join(str(h) for h in hyp)}"]
                           + trial._proof(target), used=sorted(used),
                           depends_on_knowledge=bool(used & trial.knowledge_sids), note="universal introduction")
        if self.reductio:                                # head's negation inconsistent under the assumption
            trial2 = Engine(self.prog, self.use_knowledge, True, self.merge_typical, self.max_facts,
                            reductio=False, extra_facts=self.extra_facts + hyp + [target.flip()])
            if trial2.inconsistent and not trial2.overflow and not trial.inconsistent:
                return Verdict("oui", proof=[f"UNIVERSAL by refutation: assume {' & '.join(str(h) for h in hyp)} and {target.flip()}"],
                               note="universal introduction (reductio)")
        return Verdict("inconnu", note="universal question: no counterexample, no derivation")

    def answers(self) -> List[Verdict]:
        return [self.answer(q) for q in self.prog.questions] or [self.answer()]


MAX_BRANCHES = 256


def leaves(prog: Program, use_knowledge: bool = True, extra: Optional[List[Lit]] = None,
           budget: Optional[List[int]] = None, **kw) -> List[Engine]:
    """Case split on the pending disjunctions: every consistent complete branch (a 'leaf')."""
    budget = budget if budget is not None else [MAX_BRANCHES]
    eng = Engine(prog, use_knowledge=use_knowledge, extra_facts=extra or [], **kw)
    if eng.inconsistent or eng.overflow:
        return []
    if not eng.pending:
        return [eng]
    lits, _ = eng.pending[0]
    out: List[Engine] = []
    for l in lits:
        budget[0] -= 1
        if budget[0] < 0:
            raise RuntimeError("too many disjunction branches")
        out += leaves(prog, use_knowledge, (extra or []) + [l], budget, **kw)
    return out


def decide(prog: Program, use_knowledge: bool = True, **kw) -> Verdict:
    """Entailment under all case splits: a verdict holds only if every consistent branch gives it.
    Without disjunctions this is exactly Engine(prog).answer()."""
    if not prog.disjunctions and not prog.drules:
        return Engine(prog, use_knowledge=use_knowledge, **kw).answer()
    try:
        ls = leaves(prog, use_knowledge, **kw)
    except RuntimeError as e:
        return Verdict("inconnu", note=str(e))
    if not ls:
        root = Engine(prog, use_knowledge=use_knowledge, **kw)
        v = root.answer()
        return v if v.answer == "contradictoire" else Verdict("contradictoire", note="every disjunction branch is inconsistent")
    vs = [leaf.answer() for leaf in ls]
    answers = {v.answer for v in vs}
    if len(answers) == 1:
        v = vs[0]
        v.note = (v.note + " | " if v.note else "") + f"holds in all {len(ls)} disjunction branches"
        return v
    return Verdict("inconnu", note=f"disjunction: {len(ls)} branches disagree ({', '.join(sorted(answers))})")


def run(program_text: str, source_text: Optional[str] = None, **engine_kw) -> Tuple[Program, Verdict]:
    prog = parse(program_text, source_text=source_text)
    return prog, decide(prog, **engine_kw)
