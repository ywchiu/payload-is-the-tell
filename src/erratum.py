"""Erratum check (2026-11-11): the unprompted gate matched brand names in root posts by substring, so 「可不可以」
('can you') counted as a mention of the brand 可不可. Recompute every population-level D4 variant with the submitted
gate (COLLIDE=()) and the fixed gate (detect.COLLIDE), in both windows, plus the frozen 0.5% alarm.
No LLM calls. Writes data/erratum.json and prints the comparison."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import detect as D

ROOT = Path(__file__).resolve().parent.parent
FIX = D.COLLIDE
WINS = {"dev": ROOT / "data", "heldout": ROOT / "data" / "heldout"}
CONDS = ["S0", "S1", "S2", "S3", "S4", "S5", "S4k", "S4m", "S1G", "S2G", "S3G", "S4G", "S5G"]
DETS = ("D4 claim (ours)", "D4a claim-naive", "D4b claim-unsolicited", "D4c claim+provenance")


def scores(msgs, acc, scored, cl):
    d4, d4a, info = D.d4_claim(msgs, acc, scored, cl)
    d4c, _, _ = D.d4_claim(msgs, acc, scored, cl, inherit=True)
    return {"D4 claim (ours)": d4, "D4a claim-naive": d4a, "D4b claim-unsolicited": info["out_uns"], "D4c claim+provenance": d4c}


def pool(data, sw, cl, org):
    msgs = org + sw; acc = D.accounts_of(msgs)
    scored = sorted(a for a, v in acc.items() if len(v) >= D.MIN_MSGS)
    return msgs, acc, scored


out = {"eval": {}, "ops": {}}
dev_org = json.load(open(WINS["dev"] / "organic_msgs.json")); dev_cl = json.load(open(WINS["dev"] / "claims.json"))
thr = {}
for tag, col in (("old", ()), ("new", FIX)):
    D.COLLIDE = col
    m, a, sc = pool(WINS["dev"], [], dev_cl, dev_org)
    s = scores(m, a, sc, dev_cl)
    thr[tag] = {k: float(np.quantile([s[k][x] for x in sc], 1 - 0.005)) for k in ("D4 claim (ours)", "D4c claim+provenance")}
out["thresholds"] = thr
for win, data in WINS.items():
    org = json.load(open(data / "organic_msgs.json")); cl = json.load(open(data / "claims.json"))
    for c in CONDS:
        for sd in range(3):
            f = data / f"swarm_{c}_s{sd}.json"
            if not f.exists(): continue
            sw = json.load(open(f)); msgs, acc, scored = pool(data, sw, cl, org)
            lab = {x: int(x.startswith("swarm:")) for x in scored}
            for tag, col in (("old", ()), ("new", FIX)):
                D.COLLIDE = col
                s = scores(msgs, acc, scored, cl)
                r = D.evaluate(s, scored, lab)
                out["eval"][f"{win}|{c}_s{sd}|{tag}"] = {k: r[k] for k in DETS}
                for k in ("D4 claim (ours)", "D4c claim+provenance"):
                    al = [x for x in scored if s[k][x] > thr[tag][k]]
                    tp = sum(lab[x] for x in al); ns = sum(lab.values())
                    out["ops"][f"{win}|{c}_s{sd}|{tag}|{k}"] = {"alerts": len(al), "prec": tp / len(al) if al else None,
                                                               "rec": tp / ns if ns else None}
            print(win, c, sd, "done", flush=True)
json.dump(out, open(ROOT / "data" / "erratum.json", "w"), indent=1)

# summary: condition means, old -> new
for win in WINS:
    for c in CONDS:
        for k in DETS:
            row = []
            for tag in ("old", "new"):
                v = [out["eval"][key][k] for key in out["eval"] if key.startswith(f"{win}|{c}_s") and key.endswith("|" + tag)]
                if v: row.append((np.mean([x["auroc"] for x in v]), np.mean([x["ap"] for x in v]), np.mean([x["r@k"] for x in v])))
            if len(row) == 2 and any(abs(row[0][i] - row[1][i]) > 5e-4 for i in range(3)):
                print(f"{win:8s}{c:5s}{k:24s} AUROC/AP/R@30 {row[0][0]:.3f}/{row[0][1]:.3f}/{row[0][2]:.3f} -> {row[1][0]:.3f}/{row[1][1]:.3f}/{row[1][2]:.3f}")
print("thresholds", thr)
