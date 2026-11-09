"""Sensitivity: swarm size (repeated random account subsamples) and D4 entity-merge ratio. No LLM detectors."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import detect as D
DATA = D.DATA
N_REP = 5  # random subsamples per (cond, seed, N<30)
organic = json.load(open(DATA / "organic_msgs.json")); claims = json.load(open(DATA / "claims.json"))
rows = []
for cond in ["S1", "S2", "S3", "S4"]:
    for seed in [0, 1, 2]:
        sw_all = json.load(open(DATA / f"swarm_{cond}_s{seed}.json"))
        ids = sorted({m["account"] for m in sw_all})
        for N in [5, 10, 20, 30]:
            for rep in range(1 if N == len(ids) else N_REP):
                keep = set(random.Random(f"{cond}-{seed}-{N}-{rep}").sample(ids, N))
                sw = [m for m in sw_all if m["account"] in keep]
                msgs = organic + sw; acc = D.accounts_of(msgs)
                scored = sorted(a for a, v in acc.items() if len(v) >= D.MIN_MSGS)
                labels = {a: int(a.startswith("swarm:")) for a in scored}
                sc = {"D1 lexical": D.d1_lexical(msgs, acc, scored), "D2 temporal": D.d2_temporal(msgs, acc, scored),
                      "D3 semantic": D.d3_semantic(msgs, acc, scored)}
                sc["D4 claim (ours)"], sc["D4a claim-naive"], _i = D.d4_claim(msgs, acc, scored, claims); sc["D4b claim-unsolicited"] = _i["out_uns"]
                if N == len(ids):
                    for thr in (0.34, 0.67, 1.0):
                        sc[f"D4 ratio={thr}"], _, _ = D.d4_claim(msgs, acc, scored, claims, thr=thr)
                r = D.evaluate(sc, scored, labels)
                base = sum(labels.values()) / len(scored)  # AP of a random ranking
                for k, v in r.items():
                    rows.append(dict(cond=cond, seed=seed, N=N, rep=rep, det=k, base=base, **v))
                print(cond, seed, N, rep, {k: round(v["ap"], 3) for k, v in r.items() if k.startswith("D4 claim")}, flush=True)
json.dump(rows, open(DATA / "sens.json", "w"), indent=0)
