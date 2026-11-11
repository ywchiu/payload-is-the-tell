"""Account-level coordination detectors + evaluation.

For each (condition, seed): pool = organic messages + one injected swarm.
Scored set = accounts with >= MIN_MSGS messages (swarm activity is sampled from the same distribution).

D1 lexical    : char n-gram TF-IDF near-duplicate similarity to other accounts
D2 temporal   : repeated co-activity (same thread within DT seconds) with the same partner accounts
D3 semantic   : sentence-embedding density (mean cos-sim to top-k nearest other accounts)
D4 claim      : (ours) payload-convergence: target-level clusters of positive promotional claims;
                score = #distinct accounts endorsing the target x stance purity
D5 llm-solo   : per-account LLM monitor that sees only that account's messages (12 msgs x 200 chars, evenly over time)
ENS           : rank-average of D1-D4
"""
import os
import asyncio, json, sys, glob, hashlib, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, str(Path(__file__).parent))
import llm

DATA = Path(os.environ.get("DATA_DIR") or Path(__file__).resolve().parent.parent / "data").resolve()
MIN_MSGS = 2
# Erratum (2026-11-11): everyday words that contain a short brand alias as a substring are masked before the
# root-text match of the unprompted gate (e.g. 「可不可以」 'can you' contains the brand 可不可). Set to () for the
# behaviour of the submitted version. See ERRATUM.md.
COLLIDE = ("可不可以", "都可以")
DT = 3600
_EMB = None


def embedder():
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return _EMB


def embed(texts, tag):
    """Cached embeddings keyed by text hash."""
    cf = DATA / f"emb_{tag}.npz"
    cache = dict(np.load(cf, allow_pickle=True)["d"].item()) if cf.exists() else {}
    keys = [hashlib.md5(t.encode()).hexdigest() for t in texts]
    miss = [(k, t) for k, t in zip(keys, texts) if k not in cache]
    if miss:
        uniq = dict(miss)
        vecs = embedder().encode(list(uniq.values()), batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        cache.update(zip(uniq.keys(), vecs))
        np.savez(cf, d=np.array(cache, dtype=object))
    return np.stack([cache[k] for k in keys]) if keys else np.zeros((0, 384))


def accounts_of(msgs):
    acc = defaultdict(list)
    for i, m in enumerate(msgs):
        acc[m["account"]].append(i)
    return acc


# ---------------- D1 lexical
def d1_lexical(msgs, acc, scored):
    texts = [m["text"] for m in msgs]
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True).fit_transform(texts)
    owner = np.array([m["account"] for m in msgs])
    idx = np.array(sorted({i for a in scored for i in acc[a]}))
    S = (X[idx] @ X.T).toarray()
    for r, i in enumerate(idx):
        S[r, owner == owner[i]] = 0
    best = dict(zip(idx, S.max(1)))
    out = {}
    for a in scored:
        v = sorted((best[i] for i in acc[a]), reverse=True)
        out[a] = float(np.mean(v[:2]))
    return out


# ---------------- D2 temporal co-activity
def d2_temporal(msgs, acc, scored):
    by_post = defaultdict(list)
    for m in msgs:
        by_post[m["post_id"]].append((m["tm"], m["account"]))
    pair = defaultdict(int)
    for p, L in by_post.items():
        L.sort()
        for x in range(len(L)):
            for y in range(x + 1, len(L)):
                if L[y][0] - L[x][0] > DT: break
                a, b = L[x][1], L[y][1]
                if a != b:
                    pair[(a, b)] += 1; pair[(b, a)] += 1
    rep = defaultdict(int); tot = defaultdict(int)
    for (a, b), w in pair.items():
        tot[a] += w
        if w >= 2: rep[a] += 1
    return {a: rep[a] + 0.001 * tot[a] for a in scored}


# ---------------- D3 semantic density
def d3_semantic(msgs, acc, scored, k=3):
    E = embed([m["text"] for m in msgs], "msg")
    names = list(acc)
    V = np.stack([E[acc[a]].mean(0) for a in names]); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    pos = {a: i for i, a in enumerate(names)}
    si = np.array([pos[a] for a in scored])
    S = V[si] @ V.T
    S[np.arange(len(si)), si] = -1
    top = -np.sort(-S, axis=1)[:, :k]
    return {a: float(top[r].mean()) for r, a in enumerate(scored)}


# ---------------- D4 claim / payload convergence (ours)
def _norm(t):
    return "".join(ch for ch in t.lower() if ch.isalnum())


def canon_targets(names, ratio=0.5):
    """Merge entity strings that are variants of each other: after normalization, the shorter (>=2 chars)
    is a substring of the longer and covers >= `ratio` of it (e.g. 霧嶼 ~ 霧嶼茶研, 萬安 ~ 蔣萬安).
    Sentence embeddings were too coarse for short CJK names (they merged unrelated brands)."""
    par = list(range(len(names)))
    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    nn = [_norm(t) for t in names]
    big = defaultdict(list)
    for j, t in enumerate(nn):
        for b in {t[k:k + 2] for k in range(len(t) - 1)}:
            big[b].append(j)
    for j, t in enumerate(nn):
        if len(t) < 2: continue
        cands = set(big.get(t[:2], []))
        for k in cands:
            u = nn[k]
            if k == j or len(u) < len(t): continue
            if t in u and len(t) / len(u) >= ratio:
                par[f(j)] = f(k)
    return {t: f(j) for j, t in enumerate(names)}


_SPLIT = re.compile(r"\s*[,，、/／;；]\s*")


def split_targets(t):
    """The extractor sometimes lists several endorsed entities in one string ("A, B"). Split them so that a
    co-mention cannot bridge two different entities during alias merging. Names that merely contain a symbol
    (e.g. "D&E") are kept whole; we split only on list separators and only if every part is a plausible name."""
    parts = [x for x in _SPLIT.split(t.strip()) if x]
    return parts if len(parts) > 1 and all(len(_norm(x)) >= 2 for x in parts) else [t]


def d4_claim(msgs, acc, scored, claims, thr=0.5, inherit=False):
    ann = [claims.get(m["msg_id"], {"target": "", "stance": 0, "claim": ""}) for m in msgs]
    tgt_idx = [i for i, c in enumerate(ann) if c["target"]]
    if not tgt_idx:
        z = {a: 0.0 for a in scored}
        return z, dict(z), {"out_uns": dict(z), "out_np": dict(z)}
    parts = {i: split_targets(ann[i]["target"]) for i in tgt_idx}
    tmap = canon_targets(sorted({x for ps in parts.values() for x in ps}), ratio=thr)
    ents = {i: {tmap[x] for x in parts[i]} for i in tgt_idx}  # canonical entities endorsed/mentioned by message i
    cnames = defaultdict(list)
    for t, c in tmap.items():
        cnames[c].append(t)
    # unsolicited = endorsement placed in a thread whose root post does not already talk about that entity
    root = {m["msg_id"]: i for i, m in enumerate(msgs) if m["kind"] == "post"}

    def unsolicited(i, c):
        m = msgs[i]
        if m["kind"] == "post":
            return inherit  # (inherit) a thread-starting endorsement is itself unprompted
        r = root.get(m["post_id"])
        if r is None: return False  # root not collected: give the benefit of the doubt (treat as solicited)
        if c in ents.get(r, ()):
            # (inherit) replies echoing a *promotional* root inherit its provenance instead of being excused
            return inherit and ann[r]["stance"] > 0
        rt = msgs[r]["text"]
        for w in COLLIDE:
            rt = rt.replace(w, " ")
        return not any(len(n) >= 2 and n in rt for n in cnames[c])

    pos_acc = defaultdict(set); uns_acc = defaultdict(set); n_pos = defaultdict(int); n_all = defaultdict(int)
    uns = {}; uns_threads = defaultdict(lambda: defaultdict(int))
    for i in tgt_idx:
        for c in ents[i]:
            n_all[c] += 1
            if ann[i]["stance"] > 0:
                n_pos[c] += 1; pos_acc[c].add(msgs[i]["account"])
                uns[(i, c)] = unsolicited(i, c)
                if uns[(i, c)]:
                    uns_acc[c].add(msgs[i]["account"]); uns_threads[c][msgs[i]["post_id"]] += 1
    pur = {c: (n_pos[c] / n_all[c]) ** 2 for c in n_all}

    def eff_threads(c):  # exp(entropy) of where unsolicited endorsements land: rallies concentrate, swarms spread
        v = np.array(list(uns_threads[c].values()), dtype=float)
        if not len(v): return 0.0
        p = v / v.sum(); return float(np.exp(-(p * np.log(p)).sum()))
    naive = {c: len(pos_acc[c]) * pur[c] for c in n_all}
    uns_only = {c: len(uns_acc[c]) * pur[c] for c in n_all}
    ours = {c: np.sqrt(len(uns_acc[c]) * eff_threads(c)) * pur[c] for c in n_all}
    nopur = {c: np.sqrt(len(uns_acc[c]) * eff_threads(c)) for c in n_all}
    out_naive, out, out_uns, out_np = {}, {}, {}, {}
    for a in scored:
        pe = [(i, c) for i in acc[a] if i in ents and ann[i]["stance"] > 0 for c in ents[i]]
        out_naive[a] = float(max((naive[c] for i, c in pe), default=0.0))
        out[a] = float(max((ours[c] for i, c in pe if uns[(i, c)]), default=0.0))
        out_uns[a] = float(max((uns_only[c] for i, c in pe if uns[(i, c)]), default=0.0))
        out_np[a] = float(max((nopur[c] for i, c in pe if uns[(i, c)]), default=0.0))
    return out, out_naive, {"tmap": tmap, "ours": ours, "naive": naive, "uns_only": uns_only, "out_uns": out_uns, "out_np": out_np}


# ---------------- D5 single-account LLM monitor
D5_PROMPT = """以下是某個社群帳號在一段時間內的所有留言。請評估這個帳號是「受操控的業配/口碑行銷假帳號（可能由 AI 產生）」的機率。
只輸出一個 0 到 100 的整數。

留言：
{msgs}"""


D5_CTX_PROMPT = """以下是某個社群帳號在一段時間內的留言；每則留言前面附上它所回覆的原貼文開頭（如果有）。請評估這個帳號是「受操控的業配/口碑行銷假帳號（可能由 AI 產生）」的機率。
只輸出一個 0 到 100 的整數。

{msgs}"""


def sample12(msgs, acc, a):
    """D5's budget: 12 messages, sampled evenly over the account's timeline (same policy for every account)."""
    idx = sorted(acc[a], key=lambda i: msgs[i]["tm"])
    if len(idx) > 12: idx = [idx[int(round(x))] for x in np.linspace(0, len(idx) - 1, 12)]
    return idx


async def d5_llm(msgs, acc, scored, ctx=False):
    """ctx=False: 12 msgs x 200 chars, no thread context (the original monitor).
    ctx=True : matched-information control -- 12 msgs x 400 chars, each with 300 chars of its root post,
               i.e. the same per-message view that D4's extractor gets."""
    out = {}
    root = {m["msg_id"]: m["text"] for m in msgs if m["kind"] == "post"}
    async def one(client, a):
        idx = sample12(msgs, acc, a)
        if ctx:
            body = "\n".join((f"[原貼文] {root[msgs[i]['post_id']][:300]}\n" if msgs[i]["kind"] == "comment" and msgs[i]["post_id"] in root else "")
                             + f"[留言] {msgs[i]['text'][:400]}" for i in idx)
            prompt = D5_CTX_PROMPT.format(msgs=body)
        else:
            prompt = D5_PROMPT.format(msgs="\n".join(f"- {msgs[i]['text'][:200]}" for i in idx))
        t = await llm.chat(client, [{"role": "user", "content": prompt}], temp=0.0, max_tokens=8)
        try: out[a] = float("".join(ch for ch in t if ch.isdigit())[:3] or 0)
        except Exception: out[a] = 0.0
    async def go(client):
        await asyncio.gather(*[one(client, a) for a in scored])
    await llm.run_all(go)
    return out


def a1_account_only(msgs, acc, scored, claims):
    """Zero-call per-account control: the largest fraction of an account's messages that positively endorse one
    (split, alias-merged) entity. Uses the same extractions as D4 but no cross-account evidence."""
    ann = [claims.get(m["msg_id"], {"target": "", "stance": 0}) for m in msgs]
    names = sorted({x for c in ann if c["target"] for x in split_targets(c["target"])})
    tmap = canon_targets(names) if names else {}
    out = {}
    for a in scored:
        cnt = defaultdict(int)
        for i in acc[a]:
            if ann[i]["target"] and ann[i]["stance"] > 0:
                for c in {tmap[x] for x in split_targets(ann[i]["target"])}: cnt[c] += 1
        out[a] = max(cnt.values(), default=0) / len(acc[a])
    return out


def rank01(d, keys):
    v = np.array([d[k] for k in keys], dtype=float)
    from scipy.stats import rankdata
    r = rankdata(v)
    return dict(zip(keys, r / len(v)))


def evaluate(scores, scored, labels):
    y = np.array([labels[a] for a in scored]); k = int(y.sum())
    res = {}
    for name, d in scores.items():
        s = np.array([d[a] for a in scored])
        # recall@k with expected value under random tie-breaking at the k-th score
        th = np.sort(s)[::-1][k - 1]
        above = s > th; tie = s == th; slots = k - above.sum()
        rk = (y[above].sum() + slots * y[tie].mean()) / k
        res[name] = {"auroc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s)), "r@k": float(rk)}
    return res


def run(cond, seed, organic, claims, use_llm=True):
    """use_llm: run D5 (self-hosted Gemma on H200)."""
    sw = json.load(open(DATA / f"swarm_{cond}_s{seed}.json"))
    msgs = organic + sw
    acc = accounts_of(msgs)
    scored = sorted(a for a, v in acc.items() if len(v) >= MIN_MSGS)
    labels = {a: int(a.startswith("swarm:")) for a in scored}
    scores = {"D1 lexical": d1_lexical(msgs, acc, scored), "D2 temporal": d2_temporal(msgs, acc, scored),
              "D3 semantic": d3_semantic(msgs, acc, scored)}
    scores["D4 claim (ours)"], scores["D4a claim-naive"], info = d4_claim(msgs, acc, scored, claims)
    scores["D4b claim-unsolicited"] = info.pop("out_uns")
    scores["D4d claim-no-purity"] = info.pop("out_np")
    for n in ("D1 lexical", "D2 temporal", "D3 semantic"):  # direction-inverted surface baselines
        scores[n + " (inv)"] = {a: -v for a, v in scores[n].items()}
    scores["X platforms"] = {a: float(len({msgs[i]["source"] for i in acc[a]})) for a in scored}  # shortcut check
    scores["D4c claim+provenance"], _, _ = d4_claim(msgs, acc, scored, claims, inherit=True)
    scores["A1 account-only"] = a1_account_only(msgs, acc, scored, claims)
    # restricted D4: population score computed only from the 12 messages per account that D5 sees
    keep = sorted({i for a in acc for i in sample12(msgs, acc, a)} | {i for i, m in enumerate(msgs) if m["kind"] == "post"})  # roots stay visible
    rmsgs = [msgs[i] for i in keep]; racc = accounts_of(rmsgs)
    r4, _, _ = d4_claim(rmsgs, racc, [a for a in scored if a in racc], claims)
    scores["D4 on D5 sample"] = {a: r4.get(a, 0.0) for a in scored}
    if use_llm:
        scores["D5 LLM-solo"] = asyncio.run(d5_llm(msgs, acc, scored))
        scores["D5 +context"] = asyncio.run(d5_llm(msgs, acc, scored, ctx=True))
    R = [rank01(scores[n], scored) for n in ("D1 lexical", "D2 temporal", "D3 semantic", "D4 claim (ours)")]
    scores["ENS D1-4"] = {a: float(np.mean([r[a] for r in R])) for a in scored}
    res = evaluate(scores, scored, labels)
    meta = {"n_scored": len(scored), "n_swarm": sum(labels.values()), "n_swarm_msgs": len(sw)}
    return res, meta, scores, info, scored, labels


if __name__ == "__main__":
    argv = [x for x in sys.argv if not x.startswith("--")]
    sys.argv = argv + [x for x in sys.argv if x.startswith("--")]
    conds = argv[1].split(",") if len(argv) > 1 else ["S0", "S1", "S2", "S3", "S4"]
    seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
    USE_LLM = "--no-llm" not in sys.argv
    SALT = (os.environ.get("SCORE_SALT") or os.urandom(16).hex()).encode()[:64]
    organic = json.load(open(DATA / "organic_msgs.json"))
    claims = json.load(open(DATA / "claims.json"))
    allres = {}
    for c in conds:
        for s in seeds:
            res, meta, scores, info, scored, labels = run(c, s, organic, claims, use_llm=USE_LLM)
            allres[f"{c}_s{s}"] = {"res": res, "meta": meta}
            print(c, s, meta, {k: round(v["auroc"], 3) for k, v in res.items()}, flush=True)
            # organic IDs -> keyed hash; the key (SCORE_SALT, random if unset) is never written to disk
            pid = lambda a: a if a.startswith("swarm:") else "org:" + hashlib.blake2b(a.encode(), key=SALT, digest_size=8).hexdigest()
            json.dump({"scores": {n: {pid(a): v for a, v in d.items()} for n, d in scores.items()},
                       "labels": {pid(a): v for a, v in labels.items()}}, open(DATA / f"scores_{c}_s{s}.json", "w"), ensure_ascii=False)
    json.dump(allres, open(DATA / os.environ.get("RESULTS", "results.json"), "w"), indent=1)
