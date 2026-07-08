#!/usr/bin/env python3
"""
compare_efp_vs_bandpower.py
---------------------------
Head-to-head: the single-electrode EFP fingerprint (efp_meirhasson, nested-CV v3)
vs the multivariate band-power decoder (eeg_bold_coupling, ridge/elasticnet).

Two clean, apples-to-apples tiers:
  1. Within-subject  (both: per-subject CV; caveat below)
  2. Cross-cohort double replication (both: train ALL DMNELF -> predict rtBPD nf1 & nf2;
     identical protocol, no selection bias) <- the decisive generalization test.

Caveat on tier 1: EFP within is nested-CV de-biased (v3); band-power within is the
older non-nested run with per-target subject-count variation. Cross-cohort is fully
matched. Band-power numbers use elasticnet (its stronger model).

Writes: comparison_table.md, fig_crosscohort_headtohead.png, fig_within_headtohead.png
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FP = Path(__file__).resolve().parent.parent
EFP = FP / "efp_meirhasson" / "results"
BP = FP / "eeg_bold_coupling" / "results"
OUT = Path(__file__).resolve().parent

# common targets (band-power decoded only PDA/GSR_DMN/GSR_CEN/RAW_DMN within-subject)
# EFP name <-> band-power name
PAIRS = [("DMN", "RAW_DMN"), ("PDA", "PDA"), ("GSR_CEN", "GSR_CEN"),
         ("GSR_DMN", "GSR_DMN"), ("GSR_PDA", "GSR_PDA")]
BP_MODEL = "elasticnet"


def efp_within(t):
    d = pd.read_csv(EFP / "full" / "efp_persubject_all.csv")
    v = d[(d.target == t) & (d.method == "EFP") & (d.resolution == "tr")].mean_r
    return float(v.mean()) if len(v) else np.nan


def bp_within(bt):
    f = BP / "multivariate" / f"group_{bt}_{BP_MODEL}.csv"
    if not f.exists():
        return np.nan
    return float(pd.read_csv(f).mean_subject_r.iloc[0])


def efp_cc(t, tag=""):
    f = EFP / f"cross_cohort_efp_summary_tr{tag}.csv"
    d = pd.read_csv(f); d = d[(d.target == t) & (d.method == "EFP")]
    if not len(d):
        return np.nan, np.nan
    return float(d.mean_r.iloc[0]), float(d.sign_flip_p.iloc[0])


def bp_cc(bt, tag=""):
    f = BP / f"cross_cohort_coupling_summary{tag}.csv"
    d = pd.read_csv(f); d = d[(d.target == bt) & (d.method == BP_MODEL)]
    if not len(d):
        return np.nan, np.nan
    return float(d.mean_r.iloc[0]), float(d.sign_flip_p.iloc[0])


def star(p):
    return "*" if (p is not None and np.isfinite(p) and p < 0.05) else " "


# ---- assemble table ----
rows = []
for et, bt in PAIRS:
    e_w, b_w = efp_within(et), bp_within(bt)
    e1, e1p = efp_cc(et, "");     e2, e2p = efp_cc(et, "_nf2")
    b1, b1p = bp_cc(bt, "");      b2, b2p = bp_cc(bt, "_nf2")
    rows.append(dict(target=et, efp_within=e_w, bp_within=b_w,
                     efp_nf1=e1, efp_nf1_p=e1p, efp_nf2=e2, efp_nf2_p=e2p,
                     bp_nf1=b1, bp_nf1_p=b1p, bp_nf2=b2, bp_nf2_p=b2p))
df = pd.DataFrame(rows)
df.to_csv(OUT / "comparison_data.csv", index=False)

# ---- markdown table ----
L = ["# EFP vs multivariate band-power — head-to-head", "",
     f"Band-power model = {BP_MODEL} (its stronger model). EFP = nested-CV v3, single electrode.",
     "`*` = sign-flip p < 0.05. **Cross-cohort replicates** = significant in BOTH rtBPD sessions.", "",
     "## Within-subject (caveat: EFP nested/de-biased; band-power non-nested)", "",
     "| Target | EFP (1 electrode) | Band-power (155 feat) |", "|---|---|---|"]
for _, r in df.iterrows():
    L.append(f"| {r.target} | {r.efp_within:.3f} | {r.bp_within:.3f} |")
L += ["", "## Cross-cohort double replication (train DMNELF → predict rtBPD)", "",
      "| Target | EFP nf1 | EFP nf2 | Band-power nf1 | Band-power nf2 | Replicates (both sess.) |",
      "|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    efp_rep = "EFP ✓" if (r.efp_nf1_p < 0.05 and r.efp_nf2_p < 0.05) else ""
    bp_rep = "BP ✓" if (r.bp_nf1_p < 0.05 and r.bp_nf2_p < 0.05) else ""
    rep = " / ".join([x for x in (efp_rep, bp_rep) if x]) or "neither"
    L.append(f"| {r.target} | {r.efp_nf1:+.3f}{star(r.efp_nf1_p)} | {r.efp_nf2:+.3f}{star(r.efp_nf2_p)} "
             f"| {r.bp_nf1:+.3f}{star(r.bp_nf1_p)} | {r.bp_nf2:+.3f}{star(r.bp_nf2_p)} | {rep} |")
efp_n = int(((df.efp_nf1_p < 0.05) & (df.efp_nf2_p < 0.05)).sum())
bp_n = int(((df.bp_nf1_p < 0.05) & (df.bp_nf2_p < 0.05)).sum())
L += ["", f"**Double-replication scorecard (of {len(df)} targets): EFP {efp_n}/{len(df)}, "
      f"band-power {bp_n}/{len(df)}.**",
      "", "Band-power transfers the RAW/arousal-loaded networks well but collapses on the "
      "GSR'd (arousal-removed) and differential (PDA) targets across cohorts; the single-electrode "
      "EFP fingerprint holds. Consistent with eeg_bold_coupling's own finding that its cross-cohort "
      "signal is largely global arousal."]
(OUT / "comparison_table.md").write_text("\n".join(L))
print("\n".join(L))

# ---- figure 1: cross-cohort head-to-head ----
tg = df.target.tolist()
x = np.arange(len(tg)); w = 0.2
fig, ax = plt.subplots(figsize=(10, 4.8))
bars = [("EFP nf1", df.efp_nf1, df.efp_nf1_p, "#1f77b4"),
        ("EFP nf2", df.efp_nf2, df.efp_nf2_p, "#4c9be8"),
        ("Band-power nf1", df.bp_nf1, df.bp_nf1_p, "#c0504d"),
        ("Band-power nf2", df.bp_nf2, df.bp_nf2_p, "#e08e8b")]
for j, (lab, vals, ps, col) in enumerate(bars):
    xs = x + (j - 1.5) * w
    ax.bar(xs, vals, w, label=lab, color=col, edgecolor="k", linewidth=0.5)
    for xi, v, p in zip(xs, vals, ps):
        if np.isfinite(p) and p < 0.05:
            ax.text(xi, v + (0.004 if v >= 0 else -0.012), "*", ha="center", fontsize=11, fontweight="bold")
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(tg)
ax.set_ylabel("cross-cohort transfer r");
ax.set_title("Cross-cohort double replication — EFP generalizes on network-specific targets, band-power does not",
             fontsize=11, fontweight="bold")
ax.legend(ncol=2, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "fig_crosscohort_headtohead.png", dpi=150); plt.close(fig)

# ---- figure 2: within-subject ----
fig, ax = plt.subplots(figsize=(8.5, 4.4))
w = 0.38
ax.bar(x - w/2, df.efp_within, w, label="EFP (1 electrode, nested)", color="#1f77b4", edgecolor="k", linewidth=0.5)
ax.bar(x + w/2, df.bp_within, w, label="Band-power (155 feat, non-nested)", color="#c0504d", edgecolor="k", linewidth=0.5)
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(tg)
ax.set_ylabel("within-subject CV r")
ax.set_title("Within-subject — band-power leads (multivariate; note non-nested caveat)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "fig_within_headtohead.png", dpi=150); plt.close(fig)
print("\nsaved figures to", OUT)
