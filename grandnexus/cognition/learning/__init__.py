"""Paquet cognition.learning (réparation 6.5 : export paresseux de LearningManager)."""

def __getattr__(attr):
    if attr == "LearningManager":
        from grandnexus.cognition.adapters import LearningManager
        return LearningManager
    raise AttributeError(attr)
