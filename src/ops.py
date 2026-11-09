"""Operating point: fix an alert threshold on ORGANIC-ONLY development data (<= 0.5% of organic accounts alerted),
freeze it, and report realised precision / recall / organic alert rate on every injected pool (dev and held-out).
No LLM calls; D4 scores are recomputed for each pool because they are population-level."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import detect as D

ROOT = Path(__file__).resolve().parent.parent
ALERT = 0.005
CONDS = ["S0", "S1", "S2", "S3", "S4", "S5", "S4k", "S4m"]


def pool(data, sw):
    org = json.load(open(data / "organic_msgs.json")); cl = json.load(open(data / "claims.json"))
    msgs = org + sw; acc = D.accounts_of(msgs)
    scored = sorted(a for a, v in acc.items() if len(v) >= D.MIN_MSGS)
    d4, _, _ = D.d4_claim(msgs, acc, scored, cl)
    d4c, _, _ = D.d4_claim(msgs, acc, scored, cl, inherit=True)
    return scored, {"D4": d4, "D4c": d4c}


dev = ROOT / "data"; ho = ROOT / "data" / "heldout"
scored, sc = pool(dev, [])
thr = {k: float(np.quantile([v[a] for a in scored], 1 - ALERT)) for k, v in sc.items()}
print("thresholds (dev organic-only, 99.5th pct):", thr)
rows = []
for win, data in (("dev", dev), ("heldout", ho)):
    scored0, sc0 = pool(data, [])  # no-injection control
    for k in thr:
        rows.append(dict(win=win, cond="none", seed=-1, det=k, alerts=sum(sc0[k][a] > thr[k] for a in scored0),
                         org_rate=float(np.mean([sc0[k][a] > thr[k] for a in scored0])), prec=None, rec=None))
    for c in CONDS:
        for s in (0, 1, 2):
            f = data / f"swarm_{c}_s{s}.json"
            if not f.exists(): continue
            scored1, sc1 = pool(data, json.load(open(f)))
            sw = [a for a in scored1 if a.startswith("swarm:")]; og = [a for a in scored1 if not a.startswith("swarm:")]
            for k in thr:
                tp = sum(sc1[k][a] > thr[k] for a in sw); fp = sum(sc1[k][a] > thr[k] for a in og)
                rows.append(dict(win=win, cond=c, seed=s, det=k, alerts=tp + fp, org_rate=fp / len(og),
                                 prec=tp / (tp + fp) if tp + fp else None, rec=tp / len(sw)))
json.dump({"threshold": thr, "alert_budget": ALERT, "rows": rows}, open(dev / "ops.json", "w"), indent=0)
for win in ("dev", "heldout"):
    for k in thr:
        print(win, k)
        for c in ["none"] + CONDS:
            r = [x for x in rows if x["win"] == win and x["det"] == k and x["cond"] == c]
            if not r: continue
            p = [x["prec"] for x in r if x["prec"] is not None]
            print(f"  {c:5s} alerts {np.mean([x['alerts'] for x in r]):6.1f}  organic-alert-rate {np.mean([x['org_rate'] for x in r]):.4f}"
                  f"  precision {np.mean(p) if p else float('nan'):.2f}  recall {np.mean([x['rec'] for x in r if x['rec'] is not None] or [float('nan')]):.2f}")
