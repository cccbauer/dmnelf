#!/usr/bin/env python3
"""
06_cen_pda_proxy_cluster.py
Evaluate CEN T-hat as a real-time PDA proxy.
Computes r(T-hat_m, PDA_direct) per microstate and compares to ElasticNet decoder.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import json
import csv
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────
CLUSTER_BASE    = Path("/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3")
MICROSTATES_DIR = CLUSTER_BASE / "microstates"
FEATURES_DIR    = CLUSTER_BASE / "features"
TARGETS_DIR     = CLUSTER_BASE / "targets"
MODELS_DIR      = CLUSTER_BASE / "models"
OUT_DIR         = CLUSTER_BASE / "results" / "cen_proxy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────
SUBJECTS      = ['sub-dmnelf001', 'sub-dmnelf004', 'sub-dmnelf005', 'sub-dmnelf006', 'sub-dmnelf007', 'sub-dmnelf008', 'sub-dmnelf009', 'sub-dmnelf010', 'sub-dmnelf011', 'sub-dmnelf1001', 'sub-dmnelf1002', 'sub-dmnelf1003']
MISSING_RUNS  = {'sub-dmnelf1002': [('rest', '02')], 'sub-dmnelf1003': [('feedback', '04')]}
SFREQ_TAG     = "250Hz"
TASKS = {
    "rest":      ["01", "02"],
    "shortrest": ["01"],
    "feedback":  ["01", "02", "03", "04"],
}

# ── Load microstate labels ────────────────────────────
assignments_path = MICROSTATES_DIR / "assignments_250Hz.json"
with open(str(assignments_path)) as _f:
    _asgn = json.load(_f)
labels    = [_asgn.get(str(i), "MS" + str(i)) for i in range(7)]
cen_idx   = next((i for i, l in enumerate(labels) if "CEN" in l.upper()), None)
dmn_idx   = next((i for i, l in enumerate(labels) if "DMN" in l.upper()), None)
print("Labels:  " + str(labels))
print("CEN idx: " + str(cen_idx) + "  DMN idx: " + str(dmn_idx))

# ── Main loop ─────────────────────────────────────────
print()
print("=" * 55)
print("Computing correlations")
print("=" * 55)

rows = []

for subject in SUBJECTS:
    print()
    print("--- " + subject + " ---")
    for task, runs in TASKS.items():
        for run in runs:
            if subject in MISSING_RUNS:
                if (task, run) in MISSING_RUNS[subject]:
                    continue

            feat_path = (FEATURES_DIR / (subject + "_task-" + task
                          + "_run-" + run + "_250Hz_features.npy"))
            pda_path  = (TARGETS_DIR  / (subject + "_task-" + task
                          + "_run-" + run + "_pda_direct.npy"))
            if not feat_path.exists() or not pda_path.exists():
                continue

            feats = np.load(str(feat_path))  # (n_vols, 9)
            pda   = np.load(str(pda_path))   # (n_vols,)
            n     = min(len(feats), len(pda))
            feats, pda = feats[:n], pda[:n]

            row = {"subject": subject, "task": task, "run": run}

            # r per microstate
            for m in range(7):
                r_val = float(pearsonr(feats[:, m], pda)[0])
                row["r_" + labels[m]] = round(r_val, 4)

            # CEN−DMN proxy
            if cen_idx is not None and dmn_idx is not None:
                proxy = feats[:, cen_idx] - feats[:, dmn_idx]
                row["r_CEN_minus_DMN"] = round(float(pearsonr(proxy, pda)[0]), 4)
            else:
                row["r_CEN_minus_DMN"] = float("nan")

            # Decoder LORO r (personal, feedback only)
            cv_path = MODELS_DIR / (subject + "_250Hz_cv_personal.json")
            if task == "feedback" and cv_path.exists():
                with open(str(cv_path)) as _f:
                    _cv = json.load(_f)
                row["r_decoder_personal"] = round(float(_cv.get("loro_r", float("nan"))), 4)
            else:
                row["r_decoder_personal"] = float("nan")

            rows.append(row)
            if cen_idx is not None:
                print("  " + task + " run-" + run
                      + "  r_CEN=" + str(row["r_" + labels[cen_idx]]))

# ── Save CSV ──────────────────────────────────────────
csv_path = OUT_DIR / "cen_proxy_correlations_250Hz.csv"
if rows:
    fieldnames = list(rows[0].keys())
    with open(str(csv_path), "w", newline="") as _f:
        writer = csv.DictWriter(_f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print()
    print("Saved: " + str(csv_path))

# ── Group figure ─────────────────────────────────────
print()
print("=" * 55)
print("Generating group figure")
print("=" * 55)

import re
_r_cols = ["r_" + l for l in labels] + ["r_CEN_minus_DMN"]
_decoder_col = "r_decoder_personal"

# Mean across ALL tasks/runs per subject, then group mean
sub_means = {}
for row in rows:
    s = row["subject"]
    if s not in sub_means:
        sub_means[s] = {c: [] for c in _r_cols + [_decoder_col]}
    for c in _r_cols:
        v = row.get(c, float("nan"))
        if not (v != v):  # not NaN
            sub_means[s][c].append(v)
    v = row.get(_decoder_col, float("nan"))
    if not (v != v):
        sub_means[s][_decoder_col].append(v)

all_cols  = _r_cols + [_decoder_col]
bar_means = []
bar_sems  = []
for c in all_cols:
    vals = [np.mean(sub_means[s][c]) for s in sub_means
            if sub_means[s][c]]
    bar_means.append(float(np.mean(vals)) if vals else float("nan"))
    bar_sems.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)

bar_labels = labels + ["CEN−DMN", "Decoder\n(personal)"]
colors = ["#888888"] * 7 + ["#FF1A00", "#0077FF"]
if cen_idx is not None:
    colors[cen_idx] = "#FF6600"
if dmn_idx is not None:
    colors[dmn_idx] = "#0055AA"

fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(all_cols))
bars = ax.bar(x, bar_means, color=colors, alpha=0.85,
              yerr=bar_sems, capsize=4, error_kw={"elinewidth": 1.2})
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_xticks(x)
ax.set_xticklabels(bar_labels, fontsize=9)
ax.set_ylabel("Pearson r  (T-hat vs PDA_direct)", fontsize=9)
ax.set_title("CEN T-hat as PDA proxy  —  group mean ± SEM  (n="
             + str(len(sub_means)) + ")", fontsize=10, fontweight="bold")
ax.tick_params(axis="y", labelsize=8)
# Highlight zero line and add value labels
for bar, m in zip(bars, bar_means):
    if m != m: continue
    ax.text(bar.get_x() + bar.get_width() / 2,
            m + (0.005 if m >= 0 else -0.012),
            str(round(m, 3)), ha="center", va="bottom" if m >= 0 else "top",
            fontsize=7, color="black")
plt.tight_layout()
fig_path = OUT_DIR / "cen_proxy_group_250Hz.png"
plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: " + str(fig_path))

print()
print("=" * 55)
print("DONE")
print("=" * 55)