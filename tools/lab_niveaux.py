"""
Localiser le « manque de QI » : à quel niveau chaque échec se joue-t-il ?

Méthode des oracles (ablation par couches), sur le témoin de Khaled (déjà « vu »,
utilisé ici pour DIAGNOSTIQUER, pas pour scorer une généralisation) :

  E0  système actuel (extracteur + moteur)                     -> référence
  E1  pratique : 3 passages successifs, même session            -> s'améliore-t-il ?
  N1  oracle LANGUE : prémisses écrites à la main, fidèles au
      texte seul (aucune connaissance ajoutée)                  -> le moteur suffit-il ?
  N2  + oracle CONNAISSANCES : axiomes et faits du monde
      (contenance transitive, pomme ≠ viande, remplir ⇒ plein…) -> manque-t-il du savoir ?
  N3  + oracle MOTEUR : pas d'inférence fournis à la main
      (contraposée, contrôle global de cohérence)               -> manque-t-il des règles d'inférence ?

Le gain de chaque niveau dit OÙ corriger. Le moteur utilisé reste ReasoningCore
(stratégies existantes) ; seul le verdict est calculé ici.
"""
import json, logging, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

ITEMS = json.load(open(pathlib.Path(__file__).resolve().parent.parent /
                       "tests_lab" / "temoin_khaled_2026-10-16.json", encoding="utf-8"))["items"]

def R(a, b):
    return {"if": a, "then": b, "confidence": 0.95}

# Formalisation de référence, couche par couche.
# texte : ce que la phrase dit, et rien d'autre ; savoir : ce qu'un humain ajoute ;
# moteur : ce qu'un moteur complet déduirait par lui-même (fourni ici à la main).
GOLD = {
 1: dict(goal="in(eau,maison)",
         texte=dict(facts=["in(eau,bouteille)", "in(bouteille,sac_paul)", "in(sac_paul,maison)"], rules=[]),
         savoir=dict(facts=[], rules=[R("in(sac_paul,maison)", "in(bouteille,maison)"),
                                      R("in(bouteille,maison)", "in(eau,maison)")]),  # contenance transitive
         moteur=dict(facts=[], rules=[])),
 2: dict(goal="respecte(luna,regime_carnivore)",
         texte=dict(facts=["is(luna,chatte)", "mange(luna,pomme)"],
                    rules=[R("is(luna,chat)", "is(luna,carnivore)"),
                           R("is(luna,carnivore)", "not mange(luna,non_viande)")]),
         savoir=dict(facts=["is(pomme,non_viande)"],
                     rules=[R("is(luna,chatte)", "is(luna,chat)"),                  # chatte = chat femelle
                            R("mange(luna,pomme)", "mange(luna,non_viande)"),        # pomme n'est pas de la viande
                            R("mange(luna,non_viande)", "not respecte(luna,regime_carnivore)")]),
         moteur=dict(facts=[], rules=[], global_consistency=True)),
 3: dict(goal="is(python,easy_to_learn)",
         texte=dict(facts=["is(python,programming_language)"], rules=[]),
         savoir=dict(facts=[], rules=[]), moteur=dict(facts=[], rules=[])),
 4: dict(goal="vient(frere_de_bob,fete)",
         texte=dict(facts=["invite(alice,bob)"],
                    rules=[R("invite(alice,bob)", "vient(bob,fete)"),
                           R("invite(alice,bob)", "vient(frere_de_bob,fete)")]),   # « avec son frère »
         savoir=dict(facts=[], rules=[]), moteur=dict(facts=[], rules=[])),
 5: dict(goal="write(software_developer,code)",
         texte=dict(facts=["is(software_developer,software_developer)"],
                    rules=[R("is(software_developer,software_developer)", "build(software_developer,application)"),
                           R("build(software_developer,application)", "write(software_developer,code)")]),
         savoir=dict(facts=[], rules=[]), moteur=dict(facts=[], rules=[])),
 6: dict(goal="cause(soleil,allumage_lumieres)",
         texte=dict(facts=["couche(soleil)"], rules=[R("couche(soleil)", "allume(lumieres)")]),
         savoir=dict(facts=[], rules=[]), moteur=dict(facts=[], rules=[])),
 7: dict(goal="in(key,office)",
         texte=dict(facts=["in(key,drawer)", "part_of(drawer,desk)", "located_in(desk,office)"], rules=[]),
         savoir=dict(facts=[], rules=[R("part_of(drawer,desk)", "in(drawer,desk)"),
                                      R("located_in(desk,office)", "in(desk,office)"),
                                      R("in(drawer,desk)", "in(key,desk)"),
                                      R("in(desk,office)", "in(drawer,office)"),
                                      R("in(drawer,office)", "in(key,office)")]),
         moteur=dict(facts=[], rules=[])),
 8: dict(goal="is(pingouin,mammifere)",
         texte=dict(facts=["a(pingouin,plume)"],
                    rules=[R("is(pingouin,mammifere)", "not a(pingouin,plume)")]),
         savoir=dict(facts=[], rules=[]),
         moteur=dict(facts=[], rules=[R("a(pingouin,plume)", "not is(pingouin,mammifere)")])),  # contraposée
 9: dict(goal="crash(server)",
         texte=dict(facts=["offline(website)"], rules=[R("crash(server)", "offline(website)")]),
         savoir=dict(facts=[], rules=[]), moteur=dict(facts=[], rules=[])),
 10: dict(goal="vide(reservoir,apres_midi)",
          texte=dict(facts=["vide(reservoir,matin)", "remplit(mecanicien,reservoir,midi)"], rules=[]),
          savoir=dict(facts=[], rules=[R("remplit(mecanicien,reservoir,midi)", "plein(reservoir,midi)")]),
          moteur=dict(facts=[], rules=[R("plein(reservoir,midi)", "not vide(reservoir,apres_midi)")])),  # persistance dans le temps
}

def neg(p):
    return p[4:] if p.startswith("not ") else "not " + p

def verdict(rc, facts, rules, goal, global_consistency=False, fixpoint=False):
    """fixpoint=True : on relance ReasoningCore avec ses propres conclusions jusqu'à
    stabilité, ce que ferait un moteur à chaînage avant complet."""
    from grandnexus.cognition.contracts import CognitiveContext, CognitiveRequest
    known = set(facts)
    while facts and rules:
        ctx = CognitiveContext(parameters={"premises": sorted(known), "rules": list(rules),
                                           "max_steps": 8, "exploration_factor": 0.0})
        out = rc.reason(CognitiveRequest(query=goal, context=ctx), ctx, goal)
        new = {c.get("statement") for c in out.get("conclusions", []) if c.get("statement")} - known
        known |= new
        if not fixpoint or not new:
            break
    if global_consistency and any(neg(k) in known for k in known):
        return "contradictoire", known
    pos, ng = goal in known, neg(goal) in known
    return ("contradictoire" if pos and ng else "oui" if pos else "non" if ng else "inconnu"), known

def chain_depth_probe(rc):
    """Le moteur enchaîne-t-il plus de 2 déductions ? (a⇒b⇒c⇒d⇒e)"""
    rules = [R("p(a)", "p(b)"), R("p(b)", "p(c)"), R("p(c)", "p(d)"), R("p(d)", "p(e)")]
    _, known = verdict(rc, ["p(a)"], rules, "p(e)")
    return max(i for i, x in enumerate("abcde") if f"p({x})" in known)

def main():
    logging.disable(logging.CRITICAL)
    from grandnexus.boot import boot
    gn = boot(log_level=logging.CRITICAL)
    rc, tr = gn.modules()["reasoning_core"], gn.modules()["text_reasoner"]
    labels = {i + 1: ITEMS[i][1] for i in range(len(ITEMS))}

    # E0 / E1 : système actuel, trois passages (pratique)
    runs = []
    for _ in range(3):
        runs.append({i + 1: tr.answer(q)["answer"] for i, (q, _) in enumerate(ITEMS)})
    e = [sum(r[i] == labels[i] for i in labels) for r in runs]

    # N1..N3 : oracles cumulés
    table = {}
    for i, g in GOLD.items():
        f1, r1 = g["texte"]["facts"], g["texte"]["rules"]
        f2, r2 = f1 + g["savoir"]["facts"], r1 + g["savoir"]["rules"]
        f3, r3 = f2 + g["moteur"]["facts"], r2 + g["moteur"]["rules"]
        n1, _ = verdict(rc, f1, r1, g["goal"])
        n2, _ = verdict(rc, f2, r2, g["goal"])
        n3, _ = verdict(rc, f3, r3, g["goal"], g["moteur"].get("global_consistency", False))
        n4, _ = verdict(rc, f3, r3, g["goal"], True, fixpoint=True)
        table[i] = (runs[0][i], n1, n2, n3, n4)

    print("item attendu        E0 actuel      N1 +langue     N2 +savoir     N3 +inférences N4 +chaînage    niveau du manque")
    for i, (e0, n1, n2, n3, n4) in table.items():
        lab = labels[i]
        lvl = ("aucun" if e0 == lab else "LANGUE" if n1 == lab else "LANGUE+SAVOIR" if n2 == lab else
               "MOTEUR (inférence)" if n3 == lab else "MOTEUR (profondeur)" if n4 == lab else
               "étiquette ou formalisation")
        mark = lambda v: ("✓ " if v == lab else "✗ ") + v
        print(f"{i:>3}  {lab:14} {mark(e0):14} {mark(n1):14} {mark(n2):14} {mark(n3):14} {mark(n4):14} {lvl}")
    tot = lambda k: sum(table[i][k] == labels[i] for i in table)
    print(f"\nscores : E0 {tot(0)}/10 → N1 +langue {tot(1)}/10 → N2 +savoir {tot(2)}/10 → "
          f"N3 +inférences {tot(3)}/10 → N4 +chaînage complet {tot(4)}/10")
    print(f"pratique (E1) : 3 passages successifs = {e} (identiques : {len(set(map(str, runs))) == 1})")
    print(f"profondeur d'enchaînement du moteur : {chain_depth_probe(rc)} déduction(s) sur 4 demandées")
    gn.shutdown()

if __name__ == "__main__":
    main()
