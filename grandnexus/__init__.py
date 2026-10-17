"""
GrandNexus — paquet racine.

Le code importe les sous-systèmes cognitifs sous deux chemins :
  grandnexus.cognition.reasoning.X   et   grandnexus.reasoning.X
(idem perception / learning / metacognition). Plutôt que de dupliquer des fichiers,
un alias d'import fait pointer grandnexus.<sous-système> vers
grandnexus.cognition.<sous-système> : même module, mêmes classes, un seul état.
"""
import importlib
import importlib.abc
import importlib.util
import sys

__version__ = "6.5.0-repair"

_ALIASES = {
    "grandnexus.perception": "grandnexus.cognition.perception",
    "grandnexus.reasoning": "grandnexus.cognition.reasoning",
    "grandnexus.learning": "grandnexus.cognition.learning",
    "grandnexus.metacognition": "grandnexus.cognition.metacognition",
    "grandnexus.dialogue": "grandnexus.io",  # couche 15/05 : en-tête io/, import dialogue/
}


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, real_name):
        self.real_name = real_name

    def create_module(self, spec):
        return importlib.import_module(self.real_name)

    def exec_module(self, module):
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for alias, real in _ALIASES.items():
            if fullname == alias or fullname.startswith(alias + "."):
                real_name = real + fullname[len(alias):]
                if importlib.util.find_spec(real_name) is None:
                    return None
                spec = importlib.util.spec_from_loader(fullname, _AliasLoader(real_name),
                                                       is_package=fullname == alias)
                return spec
        return None


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())
