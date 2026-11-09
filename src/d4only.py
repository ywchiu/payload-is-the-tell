import json, sys
from pathlib import Path
import numpy as np, collections
sys.path.insert(0, str(Path(__file__).parent)); import detect as D
o = json.load(open(D.DATA / "organic_msgs.json")); cl = json.load(open(D.DATA / "claims.json"))
agg = collections.defaultdict(list)
for c in ["S0", "S1", "S2", "S3", "S4"]:
    for s in range(3):
        sw = json.load(open(D.DATA / f"swarm_{c}_s{s}.json")); msgs = o + sw; acc = D.accounts_of(msgs)
        scored = sorted(a for a, v in acc.items() if len(v) >= 2); lab = {a: int(a.startswith("swarm:")) for a in scored}
        ours, naive, info = D.d4_claim(msgs, acc, scored, cl)
        r = D.evaluate({"ours": ours, "naive": naive, "uns": info["out_uns"]}, scored, lab)
        for k, v in r.items(): agg[(c, k)].append(v)
for (c, k), L in agg.items():
    print(c, k, {m: round(np.mean([x[m] for x in L]), 3) for m in L[0]})
