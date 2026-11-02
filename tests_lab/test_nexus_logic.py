"""Offline tests of the reference engine nexus_logic (no API): the logbook's cases + reductio + universal questions."""
import sys
sys.path.insert(0, ".")
from grandnexus.nexus_logic import run
SRC = "Ana est plus âgée que Ben. Ben n'est pas médecin."
CASES = [
 ("depth 4", "p(a). p(a) => q(a). q(a) => r(a). r(a) => s(a). s(a) => t(a). ? t(a).", "yes"),
 ("modus tollens", "mammal(X) => not feathers(X). feathers(penguin). ? mammal(penguin).", "no"),
 ("R37 unsafe rule rejected", "in(key,drawer). in(X,Y) => in(X,Z). ? in(key,kitchen).", "unknown"),
 ("R38 transitivity", "older(a,b). older(b,c). older(X,Y) & older(Y,Z) => older(X,Z). ? older(a,c).", "yes"),
 ("reverse question", "older(a,b). older(b,c). older(X,Y) & older(Y,Z) => older(X,Z). ? older(c,a).", "unknown"),
 ("same()", "tall(ana). same(ana, anna). ? tall(anna).", "yes"),
 ("dated states", "on(lamp,t1). off(lamp,t2). * off(X,T) => not on(X,T). ? on(lamp,t2).", "no"),
 ("no persistence assumed", "on(lamp,t1). ? on(lamp,t2).", "unknown"),
 ("let / apples", "let n = 3 - 1 + 2. has(john,apple,n). ? has(john,apple,N).", "4"),
 ("negative goal", "not on(lamp). ? not on(lamp).", "yes"),
 ("contradiction", "fly(X) => wing(X). fly(bird). not wing(bird). ? fly(bird).", "contradictory"),
 ("fact with variable rejected", "man(X). ? man(socrates).", "unknown"),
 ("conjunctive contrapositive", "show(bm). not dl(bm). show(X) & binge(X) => dl(X). show(X) & pop(X) => binge(X). ? pop(bm).", "no"),
 ("reductio", "digital(V) => not analog(V). printed(V) => analog(V). streaming(V) => digital(V). hardcover(V) => printed(V). streaming(n) => hardcover(n). ? streaming(n).", "no"),
 ("universal negative refuted", "politician(hs). member_of(hs, nazi). ? politician(X) => not member_of(X, nazi).", "no"),
 ("universal proved by rules", "growth(V) => profit(V). profit(V) => not suitable(V). ? growth(V) => not suitable(V).", "yes"),
 ("universal unknown", "growth(g1). profit(g1). ? growth(V) => profit(V).", "unknown"),
]
ok = True
for name, prog, exp in CASES:
    p, v = run(prog)
    got = v.short()
    good = got == exp
    ok &= good
    print(("✓ " if good else "✗ ") + f"{name:30} -> {got:14} expected {exp}")
p, v = run('is_doctor(ana). @ "Ana est médecin"\n? is_doctor(ana).', source_text=SRC)
good = v.answer == "unknown" and p.rejections and "anchor" in p.rejections[0].reason
ok &= good
print(("✓ " if good else "✗ ") + f"{'strict anchoring':30} -> {v.answer:14} rejection={p.rejections[0].reason if p.rejections else None}")
print("ALL OK" if ok else "FAIL")
