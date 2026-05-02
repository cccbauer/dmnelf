#!/usr/bin/env python3
"""
03_compute_pda_cluster.py
Compute PDA_group and PDA_personal targets from DiFuMo-68 TSVs.
Baseline z-score matches MURFI real-time computation (Hinds 2011).
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────
CLUSTER_BASE  = Path("/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3")
TSV_DIR       = CLUSTER_BASE / "difumo_timeseries"
OUT_DIR       = CLUSTER_BASE / "targets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS     = ['sub-dmnelf001', 'sub-dmnelf004', 'sub-dmnelf005', 'sub-dmnelf006', 'sub-dmnelf007', 'sub-dmnelf008', 'sub-dmnelf009', 'sub-dmnelf010', 'sub-dmnelf011', 'sub-dmnelf1001', 'sub-dmnelf1002', 'sub-dmnelf1003']
MISSING_RUNS = {'sub-dmnelf1002': [('rest', '02')], 'sub-dmnelf1003': [('feedback', '04')]}
OVERWRITE    = True

# DiFuMo-64 group indices (0-based)
# Verified against labels_64_dictionary.csv (Dadi et al. 2020)
DMN_IDX = [3, 6, 29, 35, 38, 58, 60, 61]
CEN_IDX = [4, 31, 47, 48, 50, 51]

# MURFI baseline: first 25 volumes (30s at TR=1.2s)
BASELINE_VOLS = 25

TASK_RUNS = {
    "rest":      ["01", "02"],
    "shortrest": ["01"],
    "feedback":  ["01", "02", "03", "04"],
}

# ── Helper ────────────────────────────────────────────
def baseline_zscore(x, n=BASELINE_VOLS):
    """Z-score relative to first n volumes (MURFI convention)."""
    mu  = x[:n].mean(axis=0)
    sig = x[:n].std(axis=0)
    sig[sig < 1e-10] = 1e-10
    return (x - mu) / sig

# ── Main loop ─────────────────────────────────────────
print("=" * 55)
print("Computing PDA targets")
print("  DMN_IDX: " + str(DMN_IDX))
print("  CEN_IDX: " + str(CEN_IDX))
print("  Overwrite: " + str(OVERWRITE))
print("=" * 55)

n_done    = 0
n_skipped = 0
n_missing = 0

for subject in SUBJECTS:
    for task, runs in TASK_RUNS.items():
        for run in runs:

            # Skip known missing runs
            if subject in MISSING_RUNS:
                if (task, run) in MISSING_RUNS[subject]:
                    continue

            tsv_name = (subject + "_ses-dmnelf_task-" + task
                        + "_run-" + run
                        + "_desc-difumo64_timeseries.tsv")
            tsv_path = TSV_DIR / tsv_name

            out_group    = OUT_DIR / (subject + "_task-" + task
                          + "_run-" + run + "_pda_group.npy")
            out_personal = OUT_DIR / (subject + "_task-" + task
                          + "_run-" + run + "_pda_personal.npy")

            if out_group.exists() and out_personal.exists():
                if not OVERWRITE:
                    print("  EXISTS (skip): " + subject
                          + " " + task + " run-" + run)
                    n_skipped += 1
                    continue

            if not tsv_path.exists():
                print("  MISSING TSV: " + tsv_name)
                n_missing += 1
                continue

            # Load TSV
            df = pd.read_csv(str(tsv_path), sep="\t")
            n_vols = len(df)

            # Extract ROI columns (64 parcels)
            roi_cols = ["ROI_" + str(i+1).zfill(2)
                        for i in range(64)]
            roi_data = df[roi_cols].values.astype(np.float32)

            # Baseline z-score all parcels
            roi_z = baseline_zscore(roi_data)

            # PDA_group
            cen_z     = roi_z[:, CEN_IDX].mean(axis=1)
            dmn_z     = roi_z[:, DMN_IDX].mean(axis=1)
            pda_group = (cen_z - dmn_z).astype(np.float32)

            # PDA_personal
            if "DMN_personal" in df.columns and "CEN_personal" in df.columns:
                dmn_p = df["DMN_personal"].values.reshape(-1, 1).astype(np.float64)
                cen_p = df["CEN_personal"].values.reshape(-1, 1).astype(np.float64)
                dmn_p_z = baseline_zscore(dmn_p).ravel()
                cen_p_z = baseline_zscore(cen_p).ravel()
                pda_personal = (cen_p_z - dmn_p_z).astype(np.float32)
            else:
                print("  WARNING: no personal columns — using group for " + subject)
                pda_personal = pda_group.copy()

            np.save(str(out_group),    pda_group)
            np.save(str(out_personal), pda_personal)

            print("  " + subject + " " + task + " run-" + run
                  + "  n=" + str(n_vols)
                  + "  pda_g_std=" + "{:.3f}".format(pda_group.std())
                  + "  pda_p_std=" + "{:.3f}".format(pda_personal.std()))
            n_done += 1

print()
print("=" * 55)
print("DONE")
print("  Computed: " + str(n_done))
print("  Skipped:  " + str(n_skipped))
print("  Missing:  " + str(n_missing))
print("=" * 55)