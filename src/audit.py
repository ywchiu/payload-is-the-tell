"""Independent-model audit of the Gemma claim extractor and of entity merging (not a human audit).

  python src/audit.py sample   -> data/audit/items.json  (stratified, shuffled, opaque ids; no extractor output)
  python src/audit.py run      -> data/audit/gemini_*.json  (Gemini 3.8 Flash via the `agy` CLI, blind to Gemma labels)
  python src/audit.py score    -> data/audit/summary.json + printed table

Strata per window (dev, held-out), 25 messages each: random organic; organic messages from the top-ranked organic
accounts under PayloadConv; S4 payload; S5 payload; established-brand (S4k/S4m) payload; swarm cover messages.
"""
import json, random, re, subprocess, sys, concurrent.futures as cf
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import detect as D

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "audit"; OUT.mkdir(exist_ok=True)
WINS = {"dev": ROOT / "data", "heldout": ROOT / "data" / "heldout"}
N = 25
MODEL = "gemini-3.8-flash-medium"
BRANDS = ("霧嶼", "可不可", "迷客夏")

PROMPT = """You are annotating Taiwanese social-media messages for a research audit. For EACH item below, read the MESSAGE (and, if given, the ROOT post it replies to, which is context only).
List every specific brand / store / product / person that the MESSAGE ITSELF recommends, promotes, praises or criticises (resolve references like "這家" using the root post). For each give:
  name: the entity's name as commonly written (full name if possible)
  stance: 1 = positive / recommending, 0 = neutral mention, -1 = negative
  root_mentions: true if the ROOT post explicitly names this entity (false if there is no root post)
If the message endorses or discusses no specific entity, return an empty list for it.
Return ONLY a JSON array, one object per item, in the same order: [{{"id": "...", "targets": [{{"name": "...", "stance": 1, "root_mentions": false}}]}}]

ITEMS:
{items}"""

ENT_PROMPT = """Below are clusters of entity names that an automatic system merged as aliases of ONE real-world entity (Taiwanese social media, Oct-Nov 2026). For each cluster, say whether ALL names refer to the same real-world entity.
Return ONLY a JSON array: [{{"cid": "...", "same_entity": true/false, "wrong_names": ["names that do not belong"]}}]

CLUSTERS:
{clusters}"""


def load(win):
    d = WINS[win]
    org = json.load(open(d / "organic_msgs.json")); cl = json.load(open(d / "claims.json"))
    sw = {c: [m for s in range(3) for m in json.load(open(d / f"swarm_{c}_s{s}.json"))] for c in ("S2", "S3", "S4", "S5", "S4k", "S4m")}
    return org, cl, sw


def sample():
    items = []
    for win in WINS:
        rng = random.Random(f"audit-{win}")
        org, cl, sw = load(win)
        roots = {m["msg_id"]: m["text"] for m in org + [x for v in sw.values() for x in v] if m["kind"] == "post"}
        # top organic accounts under PayloadConv in the S4 seed-0 pool
        msgs = org + json.load(open(WINS[win] / "swarm_S4_s0.json")); acc = D.accounts_of(msgs)
        scored = sorted(a for a, v in acc.items() if len(v) >= 2)
        out, _, _ = D.d4_claim(msgs, acc, scored, cl)
        top = [a for a in sorted(scored, key=lambda a: -out[a]) if not a.startswith("swarm:")][:40]
        high = [msgs[i] for a in top for i in acc[a] if cl.get(msgs[i]["msg_id"], {}).get("stance", 0) > 0 and cl[msgs[i]["msg_id"]]["target"]]
        strata = {
            "organic_random": rng.sample([m for m in org if len(m["text"]) > 4], N),
            "organic_top": rng.sample(high, min(N, len(high))),
            "S4_payload": rng.sample([m for m in sw["S4"] if m["payload"] and m["kind"] == "comment"], N),
            "S5_payload": rng.sample([m for m in sw["S5"] if m["payload"] and m["kind"] == "comment"], N),
            "brand_payload": rng.sample([m for m in sw["S4k"] + sw["S4m"] if m["payload"]], N),
            "swarm_cover": rng.sample([m for c in ("S2", "S3", "S4", "S5") for m in sw[c] if not m["payload"]], N),
        }
        for st, ms in strata.items():
            for m in ms:
                items.append(dict(win=win, stratum=st, msg_id=m["msg_id"], text=m["text"][:400],
                                  root=roots.get(m["post_id"], "")[:300] if m["kind"] == "comment" else ""))
    random.Random(7).shuffle(items)
    for k, it in enumerate(items): it["id"] = f"q{k:03d}"
    json.dump(items, open(OUT / "items.json", "w"), ensure_ascii=False, indent=0)
    print(len(items), "items")


def agy(prompt):
    r = subprocess.run(["agy", "-p", prompt, "--model", MODEL, "--print-timeout", "600s"], capture_output=True, text=True, timeout=700)
    return r.stdout


def parse(t):
    m = re.search(r"\[.*\]", t, re.S)
    return json.loads(m.group(0))


def run():
    items = json.load(open(OUT / "items.json"))
    B = 25
    batches = [items[i:i + B] for i in range(0, len(items), B)]

    def go(k):
        f = OUT / f"gemini_{k:02d}.json"
        if f.exists(): return
        body = "\n\n".join(f"[{it['id']}]\nMESSAGE: {it['text']}" + (f"\nROOT: {it['root']}" if it["root"] else "") for it in batches[k])
        for attempt in range(3):
            try:
                res = parse(agy(PROMPT.format(items=body)))
                if {r["id"] for r in res} >= {it["id"] for it in batches[k]}:
                    json.dump(res, open(f, "w"), ensure_ascii=False); print("batch", k, "ok", flush=True); return
            except Exception as e:
                print("batch", k, "retry", attempt, e, flush=True)
    with cf.ThreadPoolExecutor(6) as ex:
        list(ex.map(go, range(len(batches))))
    # entity-merge audit: top-15 entities per window (S4 seed-0 pool) with >= 2 alias strings
    f = OUT / "gemini_entities.json"
    if not f.exists():
        clusters = []
        for win in WINS:
            org, cl, _ = load(win)
            msgs = org + json.load(open(WINS[win] / "swarm_S4_s0.json")); acc = D.accounts_of(msgs)
            scored = sorted(a for a, v in acc.items() if len(v) >= 2)
            _, _, info = D.d4_claim(msgs, acc, scored, cl)
            names = defaultdict(list)
            for t, c in info["tmap"].items(): names[c].append(t)
            for c, _ in sorted(info["naive"].items(), key=lambda x: -x[1])[:25]:
                if len(names[c]) >= 2: clusters.append({"cid": f"{win}-{c}", "names": sorted(names[c])[:30]})
        body = "\n".join(f"[{x['cid']}] " + " | ".join(x["names"]) for x in clusters)
        res = parse(agy(ENT_PROMPT.format(clusters=body)))
        json.dump({"clusters": clusters, "gemini": res}, open(f, "w"), ensure_ascii=False, indent=0)


def _n(t): return D._norm(t)


def match(a, b):
    a, b = _n(a), _n(b)
    return len(a) >= 2 and len(b) >= 2 and (a in b or b in a)


def score():
    items = {it["id"]: it for it in json.load(open(OUT / "items.json"))}
    gem = {r["id"]: r.get("targets") or [] for f in sorted(OUT.glob("gemini_[0-9]*.json")) for r in json.load(open(f))}
    cls = {w: json.load(open(WINS[w] / "claims.json")) for w in WINS}
    rows = []
    for i, it in items.items():
        if i not in gem: continue
        g = gem[i]; c = cls[it["win"]][it["msg_id"]]
        g_pos = [t["name"] for t in g if int(t.get("stance") or 0) > 0]
        m_pos = D.split_targets(c["target"]) if c["target"] and c["stance"] > 0 else []
        brand = next((b for b in BRANDS if b in it["text"] or b in it["root"]), None)
        rows.append(dict(id=i, win=it["win"], stratum=it["stratum"], g_has=bool(g_pos), m_has=bool(m_pos),
                         tgt_agree=bool(g_pos and m_pos and any(match(x, y) for x in g_pos for y in m_pos)),
                         g_brand=any(any(b in t for b in BRANDS) for t in g_pos), m_brand=any(any(b in t for b in BRANDS) for t in m_pos),
                         g_root=[bool(t.get("root_mentions")) for t in g if int(t.get("stance") or 0) > 0 and any(match(t["name"], y) for y in m_pos)],
                         rule_root=[any(len(n) >= 2 and n in it["root"] for n in [y]) for y in m_pos] if it["root"] else [False] * len(m_pos)))
    def kappa(a, b):
        a, b = np.array(a, float), np.array(b, float); po = np.mean(a == b); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        return (po - pe) / (1 - pe) if pe < 1 else float("nan")
    summ = {}
    for st in ["organic_random", "organic_top", "S4_payload", "S5_payload", "brand_payload", "swarm_cover", "ALL"]:
        R = [r for r in rows if st == "ALL" or r["stratum"] == st]
        both = [r for r in R if r["g_has"] and r["m_has"]]
        summ[st] = dict(n=len(R), pos_agree=float(np.mean([r["g_has"] == r["m_has"] for r in R])), kappa=kappa([r["g_has"] for r in R], [r["m_has"] for r in R]),
                        gemma_pos=float(np.mean([r["m_has"] for r in R])), gemini_pos=float(np.mean([r["g_has"] for r in R])),
                        target_agree_given_both=float(np.mean([r["tgt_agree"] for r in both])) if both else None,
                        gemma_brand=float(np.mean([r["m_brand"] for r in R])), gemini_brand=float(np.mean([r["g_brand"] for r in R])))
    gr = [(a, b) for r in rows for a, b in zip(r["g_root"][:1], r["rule_root"][:1])] if rows else []
    summ["gate_agreement"] = dict(n=len(gr), agree=float(np.mean([a == b for a, b in gr])) if gr else None)
    ef = OUT / "gemini_entities.json"
    if ef.exists():
        E = json.load(open(ef)); summ["entity_clusters"] = dict(n=len(E["gemini"]), same=sum(bool(x.get("same_entity")) for x in E["gemini"]),
                                                                 wrong=[x for x in E["gemini"] if not x.get("same_entity")])
    json.dump(summ, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1)
    for k, v in summ.items(): print(k, v)


if __name__ == "__main__":
    {"sample": sample, "run": run, "score": score}[sys.argv[1]]()
