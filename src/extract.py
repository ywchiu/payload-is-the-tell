"""LLM claim extraction for every message (organic + swarm). Comments are shown with their root post so that
anaphora (\"this shop\") resolves. Cached in data/claims.json keyed by msg_id."""
import os
import asyncio, json, sys, glob, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import llm
DATA = Path(os.environ.get("DATA_DIR") or Path(__file__).resolve().parent.parent / "data").resolve()

PROMPT = """你是社群內容分析器。讀下面這則台灣社群留言，判斷它是否在「正面推薦或宣傳某個具體品牌/店家/商品」。
輸出 JSON：{{"target": "被推薦的品牌/店家/商品名稱（盡量寫成正式全名；沒有就空字串）", "stance": 1 或 0 或 -1（對該 target 的態度；沒有 target 則 0）, "claim": "對該 target 的核心主張，15字內；沒有就空字串"}}
只輸出 JSON。
{context}
留言：{text}"""
CTX = """
這則留言回覆的原貼文如下（只用來理解留言裡「這家」「他們」「這杯」等指代的是誰；只判斷留言本身的態度，不要判斷原貼文）：
原貼文：{root}
"""


async def extract_all(msgs):
    roots = {m["msg_id"]: m["text"] for m in msgs if m["kind"] == "post"}
    out_f = DATA / "claims.json"
    done = json.load(open(out_f)) if out_f.exists() else {}
    def prompt(m):
        ctx = CTX.format(root=roots[m["post_id"]][:300]) if m["kind"] == "comment" and m["post_id"] in roots else ""
        return PROMPT.format(text=m["text"][:400], context=ctx)
    h = {m["msg_id"]: hashlib.md5(prompt(m).encode()).hexdigest()[:12] for m in msgs}
    todo = [m for m in msgs if done.get(m["msg_id"], {}).get("h") != h[m["msg_id"]]]  # re-extract stale entries
    print("extract todo", len(todo), flush=True)

    async def one(client, m):
        try:
            t = await llm.chat(client, [{"role": "user", "content": prompt(m)}], temp=0.0, max_tokens=120)
            j = llm.parse_json(t)
            done[m["msg_id"]] = {"target": str(j.get("target") or "").strip(), "stance": int(j.get("stance") or 0), "claim": str(j.get("claim") or "").strip(), "h": h[m["msg_id"]]}
        except Exception:
            done[m["msg_id"]] = {"target": "", "stance": 0, "claim": "", "h": h[m["msg_id"]]}

    async def go(client):
        B = 500
        for i in range(0, len(todo), B):
            await asyncio.gather(*[one(client, m) for m in todo[i:i + B]])
            json.dump(done, open(out_f, "w"), ensure_ascii=False)
            print("extracted", min(i + B, len(todo)), "/", len(todo), flush=True)
    await llm.run_all(go)
    return done


if __name__ == "__main__":
    msgs = json.load(open(DATA / "organic_msgs.json"))
    for f in sorted(glob.glob(str(DATA / "swarm_S*_s[0-9].json"))):
        msgs += json.load(open(f))
    asyncio.run(extract_all(msgs))
