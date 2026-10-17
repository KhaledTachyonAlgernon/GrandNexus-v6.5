"""Paquet cognition.perception (réparation 6.5 : export paresseux de PerceptionModule)."""

def __getattr__(attr):
    if attr == "PerceptionModule":
        from grandnexus.cognition.adapters import PerceptionModule
        return PerceptionModule
    raise AttributeError(attr)
