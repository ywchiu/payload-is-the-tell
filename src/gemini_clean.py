"""Cross-family swarms (Gemini via agy, files swarm_S*G_s*.json): drop refusals in place and report counts.
A refusal is kept out of the pool (it would not be posted); counts are written to data/gemini_refusals.json."""
import json, re, glob
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
# full-text match (an earlier first-80-characters version missed client-side blocks and long refusal essays)
REF = re.compile(r"blocked by Gemini's filters|我無法|無法協助|無法為您|不能協助|無法配合|抱歉，我|I can(?:'|no)t|I'm unable|安全準則|安全規範|安全政策|^\s*(?:本|此|這則|該)請求")
stats = {}
for f in sorted(glob.glob(str(DATA / "swarm_S*G_s[0-9].json"))):
    raw = Path(f[:-5] + ".raw.json")  # idempotent: always clean from the untouched generation
    if not raw.exists(): raw.write_text(open(f).read())
    ms = json.load(open(raw))
    bad = [m for m in ms if REF.search(m["text"])]
    keep = [m for m in ms if m not in bad]
    # a dropped S5 seed post would orphan its replies: re-point nothing, just report it
    stats[Path(f).name] = dict(n=len(ms), refused=len(bad), refused_payload=sum(m["payload"] for m in bad),
                               refused_posts=sum(m["kind"] == "post" for m in bad))
    json.dump(keep, open(f, "w"), ensure_ascii=False)
json.dump(stats, open(DATA / "gemini_refusals.json", "w"), indent=1)
for k, v in stats.items(): print(k, v)
