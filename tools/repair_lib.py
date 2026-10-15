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
