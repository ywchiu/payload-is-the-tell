"""Fetch organic (real) Taiwanese discussion on a consumer topic from InfoMiner -> data/organic_raw.jsonl"""
import sys, time, json
sys.path.insert(0, "src"); import im
KW = "(手搖飲 OR 飲料店 OR 珍奶 OR 珍珠奶茶 OR 無糖茶 OR 茶飲 OR 五十嵐 OR 50嵐 OR 迷客夏 OR 可不可 OR 清心福全)"
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 21
OUT = sys.argv[3] if len(sys.argv) > 3 else "data/organic_raw.jsonl"
KEEP = {"ptt", "threads", "dcard", "mobile01", "fans", "fb_group", "youtube"}
now = int(time.time()); start = now - DAYS * 86400
if len(sys.argv) > 2 and sys.argv[2] != "-":
    start = int(time.mktime(time.strptime(sys.argv[2], "%Y-%m-%d"))); now = start + DAYS * 86400
out = open(OUT, "w")
seen = set(); n_posts = n_push = 0
# slice by 2-day windows to avoid caps
t = start
while t < now:
    t2 = min(t + 2 * 86400, now); page = 1
    while True:
        d = im.qsearch(f"{KW} && tm: (>={t} && <={t2})", page=page, size=100)
        rows = d.get("data", [])
        for x in rows:
            if x["id"] in seen: continue
            seen.add(x["id"])
            src = x.get("source", "")
            if not any(src == k or src.startswith(k) for k in KEEP): continue
            rec = {k: x.get(k) for k in ("id", "source", "source_name", "author", "author_id", "datetime", "tm", "title", "content", "url", "comment_count", "like_count")}
            rec["kind"] = "post"; out.write(json.dumps(rec, ensure_ascii=False) + "\n"); n_posts += 1
            if src == "ptt" or int(x.get("comment_count") or 0) > 0:
                for p in range(1, 6):
                    try: pd = im.pushs(x["id"], p).get("data", [])
                    except Exception as e: print("push err", e); break
                    for c in pd:
                        c2 = {k: c.get(k) for k in ("id", "author", "author_id", "content", "push_type", "datetime", "tm", "source")}
                        c2.update(kind="comment", post_id=x["id"], post_source=src)
                        out.write(json.dumps(c2, ensure_ascii=False) + "\n"); n_push += 1
                    if len(pd) < 20: break
        if len(rows) < 100 or page >= 20: break
        page += 1
    print(time.strftime("%m-%d", time.localtime(t)), "posts", n_posts, "comments", n_push, flush=True)
    t = t2
print("DONE posts", n_posts, "comments", n_push)
