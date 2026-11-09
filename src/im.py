"""Thin InfoMiner client."""
import os, time, requests
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
for line in (ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"'))
H = {"auth-header": os.environ["INFOMINER_KEY"], "content-type": "application/json"}
BASE = "https://api.infominer.io"
S = requests.Session()

def post(path, body, tries=3):
    for i in range(tries):
        r = S.post(BASE + path, json=body, headers=H, timeout=60)
        if r.status_code in (502, 503, 504):
            time.sleep(2 + 2 * i); continue
        r.raise_for_status(); return r.json()
    r.raise_for_status()

def qsearch(q, page=1, size=100):
    return post("/querystring_search", {"querystring": q, "sorted": "tm", "page": page, "size": size, "withPushs": False})

def pushs(post_id, page=1):
    return post("/pushs", {"post_id": post_id, "page": page})
