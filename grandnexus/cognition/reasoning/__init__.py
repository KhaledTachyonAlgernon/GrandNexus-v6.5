"""Paquet cognition.reasoning (réparation 6.5 : export paresseux de ReasoningEngine)."""

def __getattr__(attr):
    if attr == "ReasoningEngine":
        from grandnexus.cognition.adapters import ReasoningEngine
        return ReasoningEngine
    raise AttributeError(attr)
