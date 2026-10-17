"""Outils de réparation : remplacements textuels ASSERTÉS (échec bruyant si l'ancre a bougé)."""
import pathlib, re
ROOT = pathlib.Path(__file__).resolve().parent.parent

def read(p): return (ROOT / p).read_text(encoding="utf-8")
def write(p, s): (ROOT / p).write_text(s, encoding="utf-8")

def replace_once(p, old, new):
    s = read(p); n = s.count(old)
    assert n == 1, f"{p}: ancre trouvée {n} fois -> {old[:80]!r}"
    write(p, s.replace(old, new))

def append(p, text):
    s = read(p)
    if text.strip() in s: return
    write(p, s.rstrip("\n") + "\n\n" + text.strip("\n") + "\n")

def reindent(p, start, end, delta):
    """Décale de `delta` espaces les lignes [start, end] (numérotation 1)."""
    lines = read(p).split("\n")
    for k in range(start - 1, end):
        l = lines[k]
        if not l.strip(): continue
        if delta < 0:
            assert l.startswith(" " * -delta), f"{p}:{k+1} pas assez indentée"
            lines[k] = l[-delta:]
        else:
            lines[k] = " " * delta + l
    write(p, "\n".join(lines))

def copy_new(name, dest):
    """Installe un fichier neuf depuis tools/new_files/."""
    src = ROOT / "tools" / "new_files" / name
    (ROOT / dest).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

def add_import(p, line):
    """Ajoute une ligne d'import avant le premier import de niveau module."""
    s = read(p)
    if line in s.split("\n"): return
    lines = s.split("\n")
    i = next((k for k, l in enumerate(lines) if l.startswith(("import ", "from ")) and "__future__" not in l), None)
    if i is None:  # pas d'import au niveau module : juste après la docstring d'en-tête
        q = [k for k, l in enumerate(lines) if l.strip().startswith('"""')]
        i = (q[1] + 1) if len(q) >= 2 else 0
    lines.insert(i, line + "  # réparation 6.5 : fourni avant par l'espace de noms partagé du notebook")
    write(p, "\n".join(lines))

def dedent_from(p, prefix):
    """Ramène au niveau module une fonction indentée par erreur dans une classe
    (et tout ce qui la suit jusqu'à la fin du fichier)."""
    lines = read(p).split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("    " + prefix))
    for k in range(i, len(lines)):
        if lines[k].startswith("    "): lines[k] = lines[k][4:]
    write(p, "\n".join(lines))
