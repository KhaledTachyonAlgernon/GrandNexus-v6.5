"""Offline test of the deliberation bench (no API): scripted readings that agree, diverge, need a
corrective round, and a reflective re-reading that resolves a divergence."""
import json, pathlib, sys, tempfile
from types import SimpleNamespace
sys.path.insert(0, "."); sys.path.insert(0, "tools")

T1 = "All men are mortal. Socrates is a man."
T2 = "Every bird in the zoo can fly. Tweety is a penguin. Penguins are birds in the zoo. Penguins cannot fly."
T3 = "Ana is older than Ben."
P = {
 "t1": '```nx\nman(X) => mortal(X). @ "All men are mortal"\nman(socrates). @ "Socrates is a man"\n? mortal(socrates).\n```',
 # a careless reading forgets one premise -> inconnu ; a careful one -> contradictoire
 "t2_careless": '```nx\nbird(X) => fly(X). @ "Every bird in the zoo can fly"\npenguin(tweety). @ "Tweety is a penguin"\n? fly(tweety).\n```',
 "t2_careful": '```nx\nbird(X) => fly(X). @ "Every bird in the zoo can fly"\npenguin(tweety). @ "Tweety is a penguin"\npenguin(X) => bird(X). @ "Penguins are birds in the zoo"\npenguin(X) => not fly(X). @ "Penguins cannot fly"\n? fly(tweety).\n```',
 "t3_bad": '```nx\nolder(ana, ben). @ "Ana is older than Bob"\n? older(ana, ben).\n```',
 "t3_good": '```nx\nolder(ana, ben). @ "Ana is older than Ben"\n? older(ana, ben).\n```',
}
SCRIPT = {T1: ["t1", "t1", "t1"], T2: ["t2_careless", "t2_careful", "t2_careless"], T3: ["t3_bad", "t3_good", "t3_good"]}

class Scripted:
    def __init__(self):
        self.messages, self.count = self, {}
    def create(self, model, max_tokens, system, messages):
        u = messages[0]["content"]
        text = next(t for t in SCRIPT if t in u)
        if "one token" in system:
            body = "oui"
        elif "Several independent readings" in u:
            body = P["t2_careful"]
        elif "rejected these lines" in u:
            body = P["t3_good"]
        else:
            i = self.count.get(text, 0); self.count[text] = i + 1
            body = P[SCRIPT[text][i % 3]]
        return SimpleNamespace(id="fake", model=model, content=[SimpleNamespace(text=body)],
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1))

d = tempfile.mkdtemp()
items = [{"id": "1", "text": T1, "query": "Is Socrates mortal?", "expected": "oui"},
         {"id": "2", "text": T2, "query": "Can Tweety fly?", "expected": "contradictoire"},
         {"id": "3", "text": T3, "query": "Is Ana older than Ben?", "expected": "oui"}]
jl = pathlib.Path(d) / "set.jsonl"; jl.write_text("\n".join(json.dumps(i) for i in items), encoding="utf-8")
import bench_deliberation as B
s = B.main(["--jsonl", str(jl), "--skip", "0", "--direct", "--out", d], client=Scripted())
m = s["modes"]
ok = (m["single"]["acc"] == 2 and m["vote"]["acc"] == 2 and m["reflective"]["acc"] == 3
      and s["stats"]["reflections"] == 1 and s["stats"]["unanimous"] == 2)
print("TOUS OK" if ok else f"ÉCHEC {json.dumps({k: v['acc'] for k, v in m.items()})} {s['stats']}")
