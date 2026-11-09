"""Frozen 0.5% alert threshold (data/ops.json, fitted on dev organic-only data) applied to the Gemini-generated swarms -> data/ops_gemini.json"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import detect as D

ROOT = Path(__file__).resolve().parent.parent; DATA = ROOT / "data"
thr = json.load(open(DATA / "ops.json"))["threshold"]
org = json.load(open(DATA / "organic_msgs.json")); cl = json.load(open(DATA / "claims.json"))
rows = []
for c in ["S1", "S2", "S3", "S4", "S5"]:
    for s in range(3):
        msgs = org + json.load(open(DATA / f"swarm_{c}G_s{s}.json")); acc = D.accounts_of(msgs)
        scored = sorted(a for a, v in acc.items() if len(v) >= D.MIN_MSGS)
        for k, inh in (("D4", False), ("D4c", True)):
            sc, _, _ = D.d4_claim(msgs, acc, scored, cl, inherit=inh)
            sw = [a for a in scored if a.startswith("swarm:")]; og = [a for a in scored if not a.startswith("swarm:")]
            tp = sum(sc[a] > thr[k] for a in sw); fp = sum(sc[a] > thr[k] for a in og)
            rows.append(dict(cond=c + "G", seed=s, det=k, tp=tp, fp=fp, n_sw=len(sw), prec=tp / (tp + fp) if tp + fp else None,
                             rec=tp / len(sw), org_rate=fp / len(og)))
json.dump(rows, open(DATA / "ops_gemini.json", "w"), indent=0)
for k in ("D4", "D4c"):
    for c in ["S1", "S2", "S3", "S4", "S5"]:
        r = [x for x in rows if x["det"] == k and x["cond"] == c + "G"]; p = [x["prec"] for x in r if x["prec"] is not None]
        print(k, c, "fp", [x["fp"] for x in r], "prec", [round(x, 2) for x in p], "mean", round(np.mean(p), 2) if p else None,
              "rec", [round(x["rec"], 2) for x in r], "mean", round(np.mean([x["rec"] for x in r]), 2))
