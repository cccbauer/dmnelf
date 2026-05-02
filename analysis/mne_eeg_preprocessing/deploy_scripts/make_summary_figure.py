#!/usr/bin/env python3
"""
make_summary_figure.py
Regenerate results/stats/summary_figure_250Hz.png from local CSVs.

Panel A: Two-stage group t-test bars (all tasks vs feedback)
Panel B: LMM fixed-effect forest plot with 95% CI (dagger = task interaction p<0.05)
Panel C: Logistic LORO AUC per subject, group mean +/- SE vs chance (0.5)
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as spstats

HERE   = Path(__file__).parent
STATS  = HERE / "results" / "stats"
TAG    = "250Hz"
OUT    = STATS / f"summary_figure_{TAG}.png"

MS_LABELS = ["DMN", "AUD", "VIS", "CEN", "SAL", "DANT", "SOM"]
N_MS = len(MS_LABELS)
x    = np.arange(N_MS)

C_ALL = "#5080D0"
C_FB  = "#E07020"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def sig_stars(p):
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def get_arr(rows, analysis, task_set, key):
    subset = sorted(
        [r for r in rows if r["analysis"] == analysis and r["task_set"] == task_set],
        key=lambda r: int(r["ms_idx"])
    )
    out = []
    for r in subset:
        v = r.get(key, "")
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return np.array(out)


grp = read_csv(STATS / f"group_stats_{TAG}.csv")
log = read_csv(STATS / f"combination_logistic_{TAG}.csv")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(
    "Microstate selectivity for PDA+ vs PDA− epochs\n"
    "(pda_direct, Benjamini–Hochberg FDR corrected)",
    fontsize=12, fontweight="bold"
)

# ── Panel A: Two-stage t-test ──────────────────────────────────
ax = axes[0]
w  = 0.38

m_all  = get_arr(grp, "ttest", "all",      "beta")
se_all = get_arr(grp, "ttest", "all",      "se")
p_all  = get_arr(grp, "ttest", "all",      "p_fdr")
m_fb   = get_arr(grp, "ttest", "feedback", "beta")
se_fb  = get_arr(grp, "ttest", "feedback", "se")
p_fb   = get_arr(grp, "ttest", "feedback", "p_fdr")

ax.bar(x - w/2, m_all, w, color=C_ALL, alpha=0.80, label="all tasks",
       yerr=se_all, capsize=4, error_kw={"elinewidth": 1.2})
ax.bar(x + w/2, m_fb,  w, color=C_FB,  alpha=0.80, label="feedback",
       yerr=se_fb,  capsize=4, error_kw={"elinewidth": 1.2})
ax.axhline(0, color="black", lw=0.8, ls="--")

_ymax = np.nanmax(np.abs(np.concatenate([m_all + se_all, m_fb + se_fb]))) * 1.3
for i in range(N_MS):
    s = sig_stars(p_all[i])
    if s:
        ax.text(x[i] - w/2, m_all[i] + se_all[i] + _ymax * 0.04, s,
                ha="center", va="bottom", fontsize=13, color=C_ALL, fontweight="bold")
    s = sig_stars(p_fb[i])
    if s:
        ax.text(x[i] + w/2, m_fb[i] + se_fb[i] + _ymax * 0.04, s,
                ha="center", va="bottom", fontsize=13, color=C_FB, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(MS_LABELS, fontsize=10)
ax.set_ylabel("Group mean ΔT-hat (PDA+ − PDA−)", fontsize=10)
ax.set_title("A   Two-stage t-test (primary)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.tick_params(axis="y", labelsize=9)

# ── Panel B: LMM fixed effects ─────────────────────────────────
ax = axes[1]

b_all  = get_arr(grp, "lmm", "all",      "beta")
b_fb   = get_arr(grp, "lmm", "feedback", "beta")
cl_all = get_arr(grp, "lmm", "all",      "ci_lo")
ch_all = get_arr(grp, "lmm", "all",      "ci_hi")
cl_fb  = get_arr(grp, "lmm", "feedback", "ci_lo")
ch_fb  = get_arr(grp, "lmm", "feedback", "ci_hi")
p_lm_all = get_arr(grp, "lmm", "all",      "p_fdr")
p_lm_fb  = get_arr(grp, "lmm", "feedback", "p_fdr")
ip_all   = get_arr(grp, "lmm", "all",      "interaction_p")

if not np.all(np.isnan(b_all)):
    e_all = [b_all - cl_all, ch_all - b_all]
    e_fb  = [b_fb  - cl_fb,  ch_fb  - b_fb ]
    ax.errorbar(x - 0.12, b_all, yerr=e_all, fmt="o", color=C_ALL,
                capsize=5, markersize=8, label="all tasks", lw=1.8)
    ax.errorbar(x + 0.12, b_fb,  yerr=e_fb,  fmt="s", color=C_FB,
                capsize=5, markersize=8, label="feedback",  lw=1.8)
    ax.axhline(0, color="black", lw=0.8, ls="--")

    _ymax2 = np.nanmax(np.abs(np.concatenate([ch_all, ch_fb]))) * 1.35
    for i in range(N_MS):
        # significance stars (FDR)
        s = sig_stars(p_lm_all[i])
        if s:
            ax.text(x[i] - 0.12, b_all[i] + e_all[1][i] + _ymax2 * 0.04, s,
                    ha="center", va="bottom", fontsize=13, color=C_ALL, fontweight="bold")
        s = sig_stars(p_lm_fb[i])
        if s:
            ax.text(x[i] + 0.12, b_fb[i] + e_fb[1][i] + _ymax2 * 0.04, s,
                    ha="center", va="bottom", fontsize=13, color=C_FB, fontweight="bold")
        # task interaction dagger
        try:
            if not np.isnan(ip_all[i]) and ip_all[i] < 0.05:
                ax.text(x[i], max(ch_all[i], ch_fb[i]) + _ymax2 * 0.10, "†",
                        ha="center", va="bottom", fontsize=13, color="black")
        except (TypeError, ValueError):
            pass
else:
    ax.text(0.5, 0.5, "LMM not available\n(statsmodels not installed)",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=11, color="gray")

ax.set_xticks(x); ax.set_xticklabels(MS_LABELS, fontsize=10)
ax.set_ylabel("LMM beta (pda_sign, ±95% CI)", fontsize=10)
ax.set_title("B   LMM confirmatory (pda_sign | subject random slope)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.tick_params(axis="y", labelsize=9)
ax.text(0.02, 0.98, "† task interaction p<0.05", transform=ax.transAxes,
        fontsize=8, va="top", color="black", style="italic")

# ── Panel C: Logistic LORO AUC ─────────────────────────────────
ax = axes[2]

aucs = np.array([float(r["mean_auc"]) for r in log])
subjects = [r["subject"] for r in log]
n = len(aucs)

jitter = np.random.default_rng(42).uniform(-0.15, 0.15, n)
ax.scatter(np.zeros(n) + jitter, aucs, color=C_ALL, alpha=0.75, s=60, zorder=3)
for subj, xi, yi in zip(subjects, jitter, aucs):
    ax.text(xi + 0.02, yi, subj.replace("sub-dmnelf", "s"), fontsize=7,
            va="center", color="gray")

mean_auc = float(np.mean(aucs))
se_auc   = float(np.std(aucs, ddof=1) / np.sqrt(n))
t_stat, p_vs_chance = spstats.ttest_1samp(aucs, 0.5)

ax.errorbar([0], [mean_auc], yerr=[se_auc], fmt="D", color=C_FB,
            markersize=10, capsize=6, lw=2.2, zorder=5, label="group mean ±SE")
ax.axhline(0.5, color="black", lw=1.0, ls="--", label="chance (0.5)")

ax.set_xlim(-0.6, 0.6)
ax.set_xticks([])
ax.set_ylabel("LORO CV AUC", fontsize=10)
ax.set_title("C   Logistic combination (feedback runs, LORO CV)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.tick_params(axis="y", labelsize=9)

txt = (f"Group AUC = {mean_auc:.3f} ± {se_auc:.3f}\n"
       f"p = {p_vs_chance:.4f} vs chance (0.5)")
ax.text(0.98, 0.02, txt, ha="right", va="bottom", transform=ax.transAxes,
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

# ── Save ───────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig(str(OUT), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT}")
