"""Controls table (S4 variants): established brands and matched-information controls -> report/figs/controls.tex"""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
R = {"dev": json.load(open(ROOT / "data/results.json")), "ho": json.load(open(ROOT / "data/heldout/results.json"))}
ROWS = [("S4", "D4 claim (ours)", "\\ours{}, fictional brand (S4)"),
        ("S4k", "D4 claim (ours)", "\\ours{}, established brand: 可不可 (common)"),
        ("S4m", "D4 claim (ours)", "\\ours{}, established brand: 迷客夏 (moderate)"),
        ("S4k", "D5 +context", "D5 + context, 可不可"),
        ("S4", "A1 account-only", "Account-only score on D4's extractions"),
        ("S4", "D4 on D5 sample", "\\ours{} using only D5's 12-message sample"),
        ("S4", "D5 LLM-solo", "D5 (12 msgs $\\times$ 200 chars, no context)"),
        ("S4", "D5 +context", "D5 + context (12 $\\times$ 400 chars + root posts)")]
def cell(win, c, d):
    v = [R[win][k]["res"][d] for k in R[win] if k.startswith(c + "_")]
    return " / ".join("%.2f" % np.mean([x[m] for x in v]) for m in ("auroc", "ap", "r@k"))
with open(ROOT / "report/figs/controls.tex", "w") as f:
    for i, (c, d, lab) in enumerate(ROWS):
        if i == 4: f.write("\\midrule\n")
        f.write(f"{lab} & {cell('dev', c, d)} & {cell('ho', c, d)} \\\\\n")
print(open(ROOT / "report/figs/controls.tex").read())
