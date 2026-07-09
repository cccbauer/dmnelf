#!/usr/bin/env python3
"""
plot_frontal_comparison.py
--------------------------
Grouped bar chart comparing full-montage vs frontal EEG decoding accuracy.

Panel A: Within-cohort LOSO (n=17)
  - Full montage group model
  - Frontal single best electrode
  - Frontal multivariate, no calibration
  - Frontal multivariate + pseudo-calibration (1 run)

Panel B: Cross-cohort nf1 (n=19)
  - Full montage group model
  - Frontal multivariate, no calibration (cross-cohort generalizes without cal)

Output: results/figures/frontal_comparison.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
RES  = PROJ / "results"
OUT  = RES / "figures"
OUT.mkdir(exist_ok=True)

TARGETS = ["PDA", "GSR_CEN", "CEN", "DMN", "GSR_PDA"]
LABELS  = ["PDA", "GSR-CEN", "CEN", "DMN", "GSR-PDA"]

C_FULL   = "#2166ac"   # dark blue  — full montage
C_FS     = "#74add1"   # light blue — frontal single
C_FM_NO  = "#f4a582"   # light orange — frontal multi, no cal
C_FM_CAL = "#d6604d"   # dark red-orange — frontal multi + pseudo-cal


def get_r(csv_path, target, method, n_cal):
    df = pd.read_csv(csv_path)
    d  = df[(df.target == target) & (df.method == method) & (df.n_cal_runs == n_cal)]
    return d.r.dropna().values


def bar_group(ax, x, w, configs, targets):
    n = len(configs)
    offsets = np.linspace(-(n - 1) / 2 * w, (n - 1) / 2 * w, n)
    for (csv, method, n_cal, color, label), off in zip(configs, offsets):
        means, sems = [], []
        for t in targets:
            rs = get_r(csv, t, method, n_cal)
            means.append(np.nanmean(rs))
            sems.append(np.nanstd(rs, ddof=1) / np.sqrt(len(rs)) if len(rs) > 1 else 0)
        ax.bar(x + off, means, w, yerr=sems, color=color, label=label,
               capsize=3, error_kw=dict(linewidth=0.8),
               edgecolor="white", linewidth=0.4)


loso_full  = RES / "efp_calibrated_loso.csv"
loso_fs    = RES / "efp_calibrated_loso_frontal.csv"
loso_fm    = RES / "efp_calibrated_loso_frontal_multi.csv"
cross_full = RES / "efp_calibrated_crosscohort_tr.csv"
cross_fm   = RES / "efp_calibrated_crosscohort_tr_frontal_multi.csv"

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
x = np.arange(len(TARGETS))
w = 0.17

# ── Panel A: LOSO ────────────────────────────────────────────────────────────
bar_group(axes[0], x, w, [
    (loso_full, "group_only", 1, C_FULL,   "Full montage"),
    (loso_fs,   "group_only", 1, C_FS,     "Frontal single electrode"),
    (loso_fm,   "group_only", 1, C_FM_NO,  "Frontal multi, no cal"),
    (loso_fm,   "pseudo_cal", 1, C_FM_CAL, "Frontal multi + pseudo-cal"),
], TARGETS)

axes[0].axhline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.5)
axes[0].set_xticks(x); axes[0].set_xticklabels(LABELS, fontsize=10)
axes[0].set_ylabel("Mean Pearson r  (±SEM)", fontsize=10)
axes[0].set_title("A   Within-cohort LOSO  (n = 17)", fontsize=11,
                   fontweight="bold", loc="left")
axes[0].legend(fontsize=8.5, frameon=False, loc="upper right")
axes[0].spines[["top", "right"]].set_visible(False)
axes[0].set_ylim(-0.06, 0.22)

# ── Panel B: Cross-cohort nf1 ─────────────────────────────────────────────
bar_group(axes[1], x, w * 1.4, [
    (cross_full, "group_only", 1, C_FULL,   "Full montage"),
    (cross_fm,   "group_only", 1, C_FM_CAL, "Frontal multi (no cal needed)"),
], TARGETS)

axes[1].axhline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.5)
axes[1].set_xticks(x); axes[1].set_xticklabels(LABELS, fontsize=10)
axes[1].set_ylabel("Mean Pearson r  (±SEM)", fontsize=10)
axes[1].set_title("B   Cross-cohort nf1  (n = 19)", fontsize=11,
                   fontweight="bold", loc="left")
axes[1].legend(fontsize=8.5, frameon=False, loc="upper right")
axes[1].spines[["top", "right"]].set_visible(False)
axes[1].set_ylim(-0.06, 0.22)

fig.tight_layout(w_pad=3)
out = OUT / "frontal_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
