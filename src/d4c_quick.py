import json, sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); import detect as D
o = json.load(open(D.DATA / "organic_msgs.json")); cl = json.load(open(D.DATA / "claims.json"))
for c in ["S0", "S1", "S2", "S3", "S4", "S5"]:
    rs = []
    for s in range(3):
        sw = json.load(open(D.DATA / f"swarm_{c}_s{s}.json")); msgs = o + sw; acc = D.accounts_of(msgs)
        scored = sorted(a for a, v in acc.items() if len(v) >= 2); lab = {a: int(a.startswith("swarm:")) for a in scored}
        a, _, _ = D.d4_claim(msgs, acc, scored, cl); b, _, _ = D.d4_claim(msgs, acc, scored, cl, inherit=True)
        rs.append(D.evaluate({"D4": a, "D4c": b}, scored, lab))
    print(c, {k: [round(np.mean([r[k][m] for r in rs]), 3) for m in ("auroc", "ap", "r@k")] for k in rs[0]})
