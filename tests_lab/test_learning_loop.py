"""Offline test of the learning loop (no API): a scripted reader reads carelessly, is corrected on the experience
pool (verified by the engine), and reads correctly on the test set only when a RELEVANT experience is shown."""
import json, pathlib, sys, tempfile
from types import SimpleNamespace
sys.path.insert(0, "."); sys.path.insert(0, "tools")

CARELESS = '```nx\nbird(X) => fly(X). @ "Every bird in the zoo can fly"\npenguin({n}). @ "{N} is a penguin"\n? fly({n}).\n```'
CAREFUL = ('```nx\nbird(X) => fly(X). @ "Every bird in the zoo can fly"\npenguin({n}). @ "{N} is a penguin"\n'
           'penguin(X) => bird(X). @ "Penguins are birds in the zoo"\npenguin(X) => not fly(X). @ "Penguins cannot fly"\n? fly({n}).\n```')
NAMES = ["Tweety", "Pingu", "Skipper", "Kowalski", "Rico", "Private"]
def item(i):
    n = NAMES[i]
    return {"id": str(i), "text": f"Every bird in the zoo can fly. {n} is a penguin. Penguins are birds in the zoo. Penguins cannot fly.",
            "query": f"Can {n} fly?", "expected": "contradictory"}
DOGS = ["Rex", "Fido", "Max"]
def dog(i):
    n = DOGS[i]
    return {"id": f"d{i}", "text": f"All dogs bark. {n} is a dog.", "query": f"Does {n} bark?", "expected": "yes"}
DOGPROG = '```nx\ndog(X) => bark(X). @ "All dogs bark"\ndog({n}). @ "{N} is a dog"\n? bark({n}).\n```'

class Scripted:
    def __init__(self): self.messages, self.calls = self, []
    def create(self, model, max_tokens, system, messages):
        u = messages[0]["content"]; self.calls.append(u)
        tail = u.split("TEXT:")[-1]
        dn = next((x for x in DOGS if x in tail), None)
        if dn:
            body = DOGPROG.format(n=dn.lower(), N=dn)         # the dog pattern is always read right
        else:
            n = next(x for x in NAMES if x in tail)
            if "The engine answered" in u:                    # reflection -> the careful program
                body = CAREFUL.format(n=n.lower(), N=n)
            elif "EXAMPLES OF VERIFIED PAST READINGS" in u and "penguin(X) => not fly(X)" in u:
                body = CAREFUL.format(n=n.lower(), N=n)       # a relevant example shown -> careful reading
            else:
                body = CARELESS.format(n=n.lower(), N=n)      # careless first reading
        return SimpleNamespace(id="fake", model=model, content=[SimpleNamespace(text=body)],
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1))

d = tempfile.mkdtemp()
jl = pathlib.Path(d) / "pool.jsonl"
# pool order is shuffled by the bench with --exp-seed: experience = 6 items (3 penguin, 3 dog), test = the 3 remaining penguins
pool = [item(0), item(1), item(2), dog(0), dog(1), dog(2), item(3), item(4), item(5)]
jl.write_text("\n".join(json.dumps(x) for x in pool), encoding="utf-8")
import bench_learning as B
s = B.main(["--jsonl", str(jl), "--exp-n", "6", "--test-n", "3", "--k", "1", "--exp-seed", "0", "--out", d], client=Scripted())
m = s["modes"]
ok = (s["experience"].get("verified", 0) >= 1 and m["none"]["acc"] == 0 and m["learned"]["acc"] == 3 and m["random"]["acc"] <= 3)
print("ALL OK" if ok else f"FAIL {json.dumps({k: v['acc'] for k, v in m.items()})} {s['experience']}")
