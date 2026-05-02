#!/usr/bin/env python3
"""
00c_add_personal_parcels_cluster.py
Add DMN_personal and CEN_personal weighted composite columns
to existing DiFuMo-64 timeseries TSVs.
Weights come from parcel_weights.json produced by 00b.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
import json
from pathlib import Path

CLUSTER_BASE = Path("/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3")
MASKS_ROOT   = Path("/projects/swglab/data/DMNELF/derivatives/network_masks")
TSV_DIR      = CLUSTER_BASE / "difumo_timeseries"

SUBJECTS = ['sub-dmnelf001', 'sub-dmnelf002', 'sub-dmnelf003', 'sub-dmnelf004', 'sub-dmnelf005', 'sub-dmnelf006', 'sub-dmnelf007', 'sub-dmnelf008', 'sub-dmnelf009', 'sub-dmnelf010', 'sub-dmnelf011', 'sub-dmnelf1001', 'sub-dmnelf1002', 'sub-dmnelf1003', 'sub-dmnelf999']
TASKS    = {'rest': ['01', '02'], 'shortrest': ['01'], 'feedback': ['01', '02', '03', '04']}

print("=" * 55)
print("Adding personalized parcel composites")
print("=" * 55)

n_updated  = 0
n_skipped  = 0
n_missing  = 0

for subject in SUBJECTS:
    # Load parcel weights for this subject
    weights_name = (subject
                    + "_space-MNI152NLin6Asym_res-2"
                    + "_parcel_weights.json")
    weights_path = MASKS_ROOT / subject / weights_name

    if not weights_path.exists():
        print("  MISSING weights: " + subject)
        n_missing += 1
        continue

    with open(str(weights_path)) as f:
        w = json.load(f)

    dmn_w = np.array(w["dmn_weights"])  # (64,)
    cen_w = np.array(w["cen_weights"])  # (64,)

    print()
    print("--- " + subject + " ---")

    for task, runs in TASKS.items():
        for run in runs:

            tsv_name = (subject
                        + "_ses-dmnelf"
                        + "_task-" + task
                        + "_run-" + run
                        + "_desc-difumo64_timeseries.tsv")
            tsv_path = TSV_DIR / tsv_name

            if not tsv_path.exists():
                print("  MISSING TSV: " + tsv_name)
                n_missing += 1
                continue

            # Load TSV
            df = pd.read_csv(str(tsv_path), sep="\t")

            # Skip if already has personal columns
            if "DMN_personal" in df.columns:
                print("  EXISTS (skip): " + tsv_name)
                n_skipped += 1
                continue

            # Extract ROI columns (64 columns)
            roi_cols = ["ROI_" + str(i+1).zfill(2) for i in range(64)]
            missing_cols = [c for c in roi_cols if c not in df.columns]
            if missing_cols:
                print("  ERROR missing ROI cols: " + str(missing_cols[:3]))
                continue

            roi_data = df[roi_cols].values  # (n_vols, 64)

            # Compute weighted composites
            dmn_personal = roi_data @ dmn_w  # (n_vols,)
            cen_personal = roi_data @ cen_w  # (n_vols,)

            # Append columns
            df["DMN_personal"] = dmn_personal
            df["CEN_personal"] = cen_personal

            # Save in place
            df.to_csv(str(tsv_path), sep="\t", index=False)
            print("  Updated: " + tsv_name
                  + "  shape=" + str(df.shape))
            n_updated += 1

print()
print("=" * 55)
print("DONE")
print("  Updated: " + str(n_updated))
print("  Skipped: " + str(n_skipped))
print("  Missing: " + str(n_missing))
print("=" * 55)