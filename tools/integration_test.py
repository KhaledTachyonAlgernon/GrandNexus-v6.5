"""
Test d'intégration GrandNexus + témoin bête.
Question : le routage par priorité (invoke) donne-t-il plus de réponses utiles
qu'un choix de capacité au hasard ?  Réponse « utile » = succès, confiance > 0.
Usage : python tools/integration_test.py
"""
import logging, random, sys, pathlib, statistics
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from grandnexus.boot import boot
from grandnexus.cognition.contracts import (CognitiveProcessType as P, CognitiveContext,
                                            CognitiveRequest)

PERCEPTION_SET = [  # (entrée, type attendu)
    ("2*x + 3 = 7", "MATHEMATICAL"), ("x^2 - 4 = 0", "MATHEMATICAL"),
    ("def f(x):\n    return x*2", "CODE"), ("import os\nprint(os.getcwd())", "CODE"),
    ('{"a": 1, "b": [1, 2]}', "STRUCTURED_DATA"), ("[1, 2, 3]", "STRUCTURED_DATA"),
    ("Bonjour, comment vas-tu ?", "TEXT"), ("Les oiseaux migrent en automne.", "TEXT"),
]
REASONING_SET = [
    "If all birds can fly and a penguin is a bird, can a penguin fly?",
    "All men are mortal. Socrates is a man. Is Socrates mortal?",
    "If it rains the ground is wet. It rains. Is the ground wet?",
    "What is the relation between a cat and a mammal?",
    "Why does ice float on water?",
]

def useful(r):
    out = r.output if isinstance(r.output, dict) else {}
    return bool(r.success and (r.confidence or out.get("confidence", 0)) > 0)

def call(cap, q):
    ctx = CognitiveContext()
    req = CognitiveRequest(query=q, context=ctx)
    try:
        out = cap.handler(req, ctx, q)
        conf = out.get("confidence", 0.0) if isinstance(out, dict) else 0.0
        return bool(out) and not (isinstance(out, dict) and out.get("error")) and conf > 0
    except Exception:
        return False

def main(seed=0):
    random.seed(seed)
    gn = boot(log_level=logging.CRITICAL)
    gn.print_report()
    cm = gn.cognition
    print("\nCapacités câblées :")
    for t, caps in cm.capabilities.summary().items():
        print(f"  {t}: {caps}")

    print("\n[1] Perception : type détecté vs attendu")
    hits = 0
    for q, expected in PERCEPTION_SET:
        r = cm.invoke(P.PERCEPTION, q)
        got = (r.output or {}).get("input_type") if isinstance(r.output, dict) else None
        hits += got == expected
        print(f"  {'✓' if got == expected else '✗'} {q[:28]!r:32} attendu={expected:16} obtenu={got}")
    print(f"  ==> {hits}/{len(PERCEPTION_SET)} types corrects")

    print("\n[2] Raisonnement : routage par priorité vs témoin aléatoire")
    caps = cm.capabilities.find(P.REASONING)
    routed = [useful(cm.invoke(P.REASONING, q)) for q in REASONING_SET]
    rand = [statistics.mean(call(random.choice(caps), q) for _ in range(10)) for q in REASONING_SET]
    per_cap = {c.name: [call(c, q) for q in REASONING_SET] for c in caps}
    print(f"  routage priorité : {sum(routed)}/{len(REASONING_SET)} réponses utiles")
    print(f"  témoin aléatoire : {sum(rand):.1f}/{len(REASONING_SET)} (moyenne sur 10 tirages)")
    for name, v in per_cap.items():
        print(f"  capacité seule {name:15}: {sum(v)}/{len(REASONING_SET)}")
    print("\nBus :", gn.core.event_bus_stats()["published"])
    gn.shutdown()

if __name__ == "__main__":
    main()
