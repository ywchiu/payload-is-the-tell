"""Normalize raw InfoMiner rows into a message table (organic, label=0)."""
import json, collections, random, os
D = os.environ.get("DATA_DIR", "data")
import glob
rows = [json.loads(l) for f in sorted(glob.glob(f"{D}/organic_raw*.jsonl")) for l in open(f)]
posts = {r["id"]: r for r in rows if r["kind"] == "post"}
msgs, seen = [], set()
for r in rows:
    if r["id"] in seen: continue
    seen.add(r["id"])
    text = ((r.get("title") or "") + "\n" + (r.get("content") or "")).strip() if r["kind"] == "post" else (r.get("content") or "").strip()
    if not text or text == "None": continue
    src = r.get("source") if r["kind"] == "post" else r.get("post_source")
    aid = r.get("author_id") or r.get("author")
    if not aid or aid == "None": continue
    try: tm = int(float(r.get("tm")))
    except Exception: continue
    msgs.append(dict(msg_id=r["id"], account=f"{src}:{aid}", text=text[:600], tm=tm, post_id=r["id"] if r["kind"] == "post" else r["post_id"],
                     kind=r["kind"], source=src, label=0, cond="organic"))
json.dump(msgs, open(f"{D}/organic_msgs.json", "w"), ensure_ascii=False)
acc = collections.Counter(m["account"] for m in msgs)
print("msgs", len(msgs), "accounts", len(acc), "dist", sorted(collections.Counter(min(v, 10) for v in acc.values()).items()))
# delays of comments relative to post
d = [m["tm"] - posts[m["post_id"]]["tm"] and (m["tm"] - int(float(posts[m["post_id"]]["tm"]))) for m in msgs if m["kind"] == "comment" and m["post_id"] in posts]
d = [x for x in d if 0 < x < 30 * 86400]
d.sort(); print("delay pct (h)", [round(d[int(len(d) * q)] / 3600, 2) for q in (.1, .25, .5, .75, .9)] if d else None)
json.dump(d, open(f"{D}/organic_delays.json", "w"))
