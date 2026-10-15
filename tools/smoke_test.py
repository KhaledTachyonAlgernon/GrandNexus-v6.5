"""
Test de fumée GrandNexus : chaque module est importé dans un processus isolé.
Sortie : tableau OK/ÉCHEC + première cause. Sert de témoin avant/après chaque réparation.
Usage : python tools/smoke_test.py [--json resultats.json]
"""
import os, sys, json, subprocess, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

def modules():
    for p in sorted((ROOT / "grandnexus").rglob("*.py")):
        rel = p.relative_to(ROOT).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        yield ".".join(parts)

def try_import(mod):
    code = f"import logging; logging.disable(logging.CRITICAL); import importlib; importlib.import_module({mod!r})"
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    if r.returncode == 0:
        return True, ""
    lines = [l for l in r.stderr.strip().split("\n") if l.strip()]
    return False, lines[-1][:160] if lines else "erreur inconnue"

def main():
    results = {}
    for m in modules():
        ok, err = try_import(m)
        results[m] = (ok, err)
        print(f"{'OK    ' if ok else 'ÉCHEC '} {m}" + ("" if ok else f"\n        └─ {err}"))
    n_ok = sum(1 for ok, _ in results.values() if ok)
    print(f"\n==> {n_ok}/{len(results)} modules importables")
    if "--json" in sys.argv:
        json.dump(results, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1, ensure_ascii=False)
    return n_ok, len(results)

if __name__ == "__main__":
    main()
