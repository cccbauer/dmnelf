#!/usr/bin/env python3
"""
02_stats_plv_cluster.py
Paired pre-vs-post PLV connectivity stats (t-test + Cohen's d +
BH-FDR, 5 bands) on plv_connectivity.csv, plus a pre/post bar chart.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as spstats
import pandas as pd
from statsmodels.stats.multitest import multipletests

# -- Paths ----------------------------------------------------
CLUSTER_BASE = Path("/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/connectivity_prepost")
RESULTS_DIR  = CLUSTER_BASE / "results"
BAND_ORDER   = ['delta', 'theta', 'alpha', 'beta', 'gamma']
ALPHA = 0.05

# -- Helpers ----------------------------------------------------
def cohens_d_paired(pre, post):
    diff = np.asarray(post) - np.asarray(pre)
    if len(diff) < 2 or diff.std(ddof=1) == 0:
        return float("nan")
    return float(diff.mean() / diff.std(ddof=1))

def sig_stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

# ================================================================
print()
print("=" * 60)
print("Loading plv_connectivity.csv")
print("=" * 60)

df = pd.read_csv(str(RESULTS_DIR / "plv_connectivity.csv"))
df["plv"] = pd.to_numeric(df["plv"], errors="coerce")
print("Rows loaded: " + str(len(df)))
print("Subjects:    " + str(df["subject"].nunique()))

# Average the 2 runs within each condition per subject per band
cond_means = (df.groupby(["subject", "condition", "band"])["plv"]
                .mean().reset_index())

# ================================================================
print()
print("=" * 60)
print("Paired t-test (ttest_rel) + paired Cohen's d  [5 bands, one FDR family]")
print("=" * 60)

stat_rows = []
raw_pvals = []

for band in BAND_ORDER:
    wide = cond_means[cond_means["band"] == band].pivot(
        index="subject", columns="condition", values="plv")
    wide = wide.dropna(subset=["pre", "post"])
    n = len(wide)
    if n < 3:
        stat_rows.append(dict(
            band=band, mean_pre=None, mean_post=None, mean_diff=None,
            t=None, p=None, p_fdr=None, cohens_d=None, n=n,
        ))
        raw_pvals.append(1.0)
        continue
    pre, post = wide["pre"].values, wide["post"].values
    t, p = spstats.ttest_rel(post, pre)
    d = cohens_d_paired(pre, post)
    stat_rows.append(dict(
        band=band,
        mean_pre=round(float(pre.mean()), 5),
        mean_post=round(float(post.mean()), 5),
        mean_diff=round(float(post.mean() - pre.mean()), 5),
        t=round(float(t), 4), p=round(float(p), 6),
        p_fdr=None, cohens_d=round(d, 4), n=n,
    ))
    raw_pvals.append(float(p))

_, p_fdr, _, _ = multipletests(raw_pvals, alpha=ALPHA, method="fdr_bh")
for row, pf in zip(stat_rows, p_fdr):
    row["p_fdr"] = round(float(pf), 6)
    row["sig_fdr"] = bool(pf < ALPHA)
    if row["p"] is not None:
        stars = sig_stars(pf)
        print("  [ttest] " + row["band"]
              + "  diff=" + str(row["mean_diff"])
              + "  t=" + str(row["t"])
              + "  p_fdr=" + str(row["p_fdr"]) + "  " + stars)

stats_df = pd.DataFrame(stat_rows)
stats_csv = RESULTS_DIR / "stats_pre_vs_post_plv.csv"
stats_df.to_csv(str(stats_csv), index=False)
print()
print("Saved: " + str(stats_csv))

# ================================================================
print()
print("Generating pre-vs-post PLV bar chart...")

x = np.arange(len(BAND_ORDER))
w_bar = 0.38
C_PRE  = "#5080D0"
C_POST = "#E07020"

means_pre, sems_pre, means_post, sems_post, pfs = [], [], [], [], []
for band in BAND_ORDER:
    wide = cond_means[cond_means["band"] == band].pivot(
        index="subject", columns="condition", values="plv")
    wide = wide.dropna(subset=["pre", "post"])
    pre, post = wide["pre"].values, wide["post"].values
    n = len(pre)
    means_pre.append(pre.mean() if n else np.nan)
    means_post.append(post.mean() if n else np.nan)
    sems_pre.append(pre.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0)
    sems_post.append(post.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0)
    row = [r for r in stat_rows if r["band"] == band][0]
    pfs.append(row["p_fdr"] if row["p_fdr"] is not None else 1.0)
means_pre  = np.array(means_pre);  sems_pre  = np.array(sems_pre)
means_post = np.array(means_post); sems_post = np.array(sems_post)

fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle("rtBPD nf1 rest PLV connectivity (DMN-relevant pairs), pre vs post\n"
              "(whole-run PLV, paired t-test, BH-FDR, n="
              + str(cond_means["subject"].nunique()) + " subjects)",
              fontsize=12, fontweight="bold")

ax.bar(x - w_bar / 2, means_pre,  w_bar, color=C_PRE,  alpha=0.85,
       label="pre",  yerr=sems_pre,  capsize=4)
ax.bar(x + w_bar / 2, means_post, w_bar, color=C_POST, alpha=0.85,
       label="post", yerr=sems_post, capsize=4)

ymax = np.nanmax(np.concatenate([means_pre + sems_pre,
                                   means_post + sems_post])) * 1.15
ymax = ymax if ymax > 0 else 1.0
for i in range(len(BAND_ORDER)):
    s = sig_stars(pfs[i])
    if s:
        yb = max(means_pre[i] + sems_pre[i], means_post[i] + sems_post[i])
        ax.text(x[i], yb + ymax * 0.03, s, ha="center", va="bottom",
                 fontsize=13, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(BAND_ORDER, fontsize=11)
ax.set_ylabel("PLV", fontsize=11)
ax.legend(fontsize=10)
ax.tick_params(axis="y", labelsize=10)

plt.tight_layout()
bar_path = RESULTS_DIR / "plv_connectivity_prepost.png"
plt.savefig(str(bar_path), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  " + str(bar_path))

print()
print("=" * 60)
print("DONE")
print("=" * 60)