"""Figures + tables for the report."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; OUT = ROOT / "report" / "figs"; OUT.mkdir(parents=True, exist_ok=True)
tag = sys.argv[1] if len(sys.argv) > 1 else ""
R = json.load(open(DATA / f"results{tag}.json"))
CONDS = ["S0", "S1", "S2", "S3", "S4", "S5"]
CLAB = ["S0\ntemplate", "S1\nnaive LLM", "S2\npersona", "S3\norchestr.", "S4\nevasive", "S5\nadaptive"]
DETS = [("D1 lexical", "D1 lexical", "#8c8c8c", "o"), ("D2 temporal", "D2 temporal", "#b07aa1", "s"),
        ("D3 semantic", "D3 semantic", "#76b7b2", "^"), ("D5 LLM-solo", "D5 per-account LLM", "#edc948", "D"),
        ("ENS D1-4", "Ensemble D1-4", "#59a14f", "v"), ("D4a claim-naive", "D4a claim (naive)", "#f28e2b", "x"),
        ("D4c claim+provenance", "D4c +provenance", "#9c755f", "P"), ("D4 claim (ours)", "D4 payload convergence (ours)", "#e15759", "*")]


def agg(det, met):
    m, s = [], []
    for c in CONDS:
        v = [R[k]["res"][det][met] for k in R if k.startswith(c + "_") and det in R[k]["res"]]
        m.append(np.mean(v) if v else np.nan); s.append(np.std(v) if v else np.nan)
    return np.array(m), np.array(s)


fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
for ax, met, title in [(axes[0], "auroc", "AUROC"), (axes[1], "ap", "Average precision")]:
    for key, lab, col, mk in DETS:
        m, s = agg(key, met)
        if np.isnan(m).all(): continue
        ax.errorbar(range(len(CONDS)), m, yerr=s, label=lab, color=col, marker=mk, lw=2 if "ours" in key else 1.2, ms=8 if "ours" in key else 5, capsize=2)
    ax.set_xticks(range(len(CONDS))); ax.set_xticklabels(CLAB, fontsize=8); ax.set_title(title); ax.grid(alpha=.3)
    if met == "auroc": ax.axhline(.5, color="k", ls=":", lw=.8); ax.set_ylim(0, 1.02)
    else:
        base = np.mean([R[k]["meta"]["n_swarm"] / R[k]["meta"]["n_scored"] for k in R]); ax.axhline(base, color="k", ls=":", lw=.8); ax.set_yscale("log")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
plt.tight_layout(rect=(0, 0.11, 1, 1)); plt.savefig(OUT / f"main{tag}.pdf"); plt.savefig(OUT / f"main{tag}.png", dpi=150)

# inverted surface baseline: direction AND detector chosen on the development window only (mean AUROC over S2-S4)
RD = json.load(open(DATA / "results.json"))
INV = max(("D1 lexical (inv)", "D2 temporal (inv)", "D3 semantic (inv)"),
          key=lambda d: np.mean([RD[k]["res"][d]["auroc"] for k in RD if k[:2] in ("S2", "S3", "S4") and d in RD[k]["res"]] or [0]))
print("inverted baseline chosen on dev:", INV)

# table
rows = []
_d = {d[0]: d for d in DETS}
TDETS = [_d[k] for k in ("D1 lexical", "D2 temporal", "D3 semantic")] + [(INV, INV.split()[0] + " inverted (dir.\\ chosen on dev)", None, None)] + \
        [_d[k] for k in ("D5 LLM-solo", "ENS D1-4", "D4a claim-naive")] + [("D4b claim-unsolicited", "D4b claim (+unprompted gate)", None, None),
        ("D4d claim-no-purity", "D4 without purity $\\pi_c$", None, None), ("D4c claim+provenance", "D4c (+provenance; vs.\\ S5)", None, None), ("D4 claim (ours)", "\\textbf{D4 \\ours{} (ours)}", None, None)]
for key, lab, *_ in TDETS:
    cells = []
    for c in CONDS:
        v = [R[k]["res"][key] for k in R if k.startswith(c + "_") and key in R[k]["res"]]
        if not v: cells.append("--"); continue
        cells.append(f"{np.mean([x['auroc'] for x in v]):.2f} / {np.mean([x['ap'] for x in v]):.2f} / {np.mean([x['r@k'] for x in v]):.2f}")
    rows.append((lab, cells))
with open(OUT / f"table{tag}.tex", "w") as f:
    for lab, cells in rows:
        f.write(lab + " & " + " & ".join(cells) + " \\\\\n")
print(open(OUT / f"table{tag}.tex").read())

# sensitivity
sf = DATA / f"sens{tag}.json"
if sf.exists():
    S = json.load(open(sf))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, cond in zip(axes, ["S2", "S4"]):
        for key, lab, col, mk in [d for d in DETS if d[0] not in ("D5 LLM-solo", "ENS D1-4", "D4c claim+provenance")]:
            pts = sorted({r["N"] for r in S})
            m = [np.mean([r["ap"] for r in S if r["cond"] == cond and r["det"] == key and r["N"] == n] or [np.nan]) for n in pts]
            if np.isnan(m).all(): continue
            ax.plot(pts, m, marker=mk, color=col, label=lab, lw=2 if "ours" in key else 1.2)
        base = [np.mean([r["base"] for r in S if r["cond"] == cond and r["N"] == n]) for n in pts]
        ax.plot(pts, base, color="k", ls=":", lw=1, label="chance (prevalence)")
        ax.set_yscale("log"); ax.set_xlabel("swarm size (accounts)"); ax.set_ylabel("average precision"); ax.set_title(f"{cond}: AP vs swarm size"); ax.grid(alpha=.3)
    axes[0].legend(fontsize=7, loc="lower right")
    plt.tight_layout(); plt.savefig(OUT / f"size{tag}.pdf"); plt.savefig(OUT / f"size{tag}.png", dpi=150)
    for r in S:
        if "ratio" in r["det"]: print(r["cond"], r["seed"], r["det"], round(r["auroc"], 3), round(r["ap"], 3))
    for cond in ("S2", "S4"):
        for n in sorted({r["N"] for r in S}):
            v = [r for r in S if r["cond"] == cond and r["N"] == n and r["det"] == "D4 claim (ours)"]
            print("size", cond, n, "AUROC %.3f AP %.3f R@k %.3f base %.4f" % tuple(np.mean([x[k] for x in v]) for k in ("auroc", "ap", "r@k", "base")))
