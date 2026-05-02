# 00c_add_personal_parcels.py
# Run locally: python 00c_add_personal_parcels.py
# Deploys cluster script, submits SLURM job.
#
# What it does on the cluster:
#   For each subject × task × run:
#   1. Load DiFuMo-64 TSV (64 ROI columns)
#   2. Load subject parcel_weights.json from Step 0b
#   3. Compute weighted composites:
#        DMN_personal(t) = sum(w_dmn_p * ROI_p(t))
#        CEN_personal(t) = sum(w_cen_p * ROI_p(t))
#   4. Append two columns → 66-column TSV (in place)
#
# Output: same TSV filenames, now with 66 columns:
#   volume | time | ROI_01..ROI_64 | DMN_personal | CEN_personal

import py_compile
import time
from pathlib import Path
from utils import run_ssh, scp_to
from config import (
    CLUSTER_BASE, SLURM_ACCOUNT, PYTHON,
    SUBJECTS, LOCAL_BASE
)

MASKS_ROOT = "/projects/swglab/data/DMNELF/derivatives/network_masks"

TASKS = {
    "rest":      ["01", "02"],
    "shortrest": ["01"],
    "feedback":  ["01", "02", "03", "04"],
}

# ── 1. Build cluster-side script ───────────────────────────
lines = [
    '#!/usr/bin/env python3',
    '"""',
    '00c_add_personal_parcels_cluster.py',
    'Add DMN_personal and CEN_personal weighted composite columns',
    'to existing DiFuMo-64 timeseries TSVs.',
    'Weights come from parcel_weights.json produced by 00b.',
    '"""',
    'import sys',
    'sys.stdout.reconfigure(line_buffering=True)',
    'import numpy as np',
    'import pandas as pd',
    'import json',
    'from pathlib import Path',
    '',
    'CLUSTER_BASE = Path("' + CLUSTER_BASE + '")',
    'MASKS_ROOT   = Path("' + MASKS_ROOT + '")',
    'TSV_DIR      = CLUSTER_BASE / "difumo_timeseries"',
    '',
    'SUBJECTS = ' + str(SUBJECTS),
    'TASKS    = ' + str(TASKS),
    '',
    'print("=" * 55)',
    'print("Adding personalized parcel composites")',
    'print("=" * 55)',
    '',
    'n_updated  = 0',
    'n_skipped  = 0',
    'n_missing  = 0',
    '',
    'for subject in SUBJECTS:',
    '    # Load parcel weights for this subject',
    '    weights_name = (subject',
    '                    + "_space-MNI152NLin6Asym_res-2"',
    '                    + "_parcel_weights.json")',
    '    weights_path = MASKS_ROOT / subject / weights_name',
    '',
    '    if not weights_path.exists():',
    '        print("  MISSING weights: " + subject)',
    '        n_missing += 1',
    '        continue',
    '',
    '    with open(str(weights_path)) as f:',
    '        w = json.load(f)',
    '',
    '    dmn_w = np.array(w["dmn_weights"])  # (64,)',
    '    cen_w = np.array(w["cen_weights"])  # (64,)',
    '',
    '    print()',
    '    print("--- " + subject + " ---")',
    '',
    '    for task, runs in TASKS.items():',
    '        for run in runs:',
    '',
    '            tsv_name = (subject',
    '                        + "_ses-dmnelf"',
    '                        + "_task-" + task',
    '                        + "_run-" + run',
    '                        + "_desc-difumo64_timeseries.tsv")',
    '            tsv_path = TSV_DIR / tsv_name',
    '',
    '            if not tsv_path.exists():',
    '                print("  MISSING TSV: " + tsv_name)',
    '                n_missing += 1',
    '                continue',
    '',
    '            # Load TSV',
    '            df = pd.read_csv(str(tsv_path), sep="\\t")',
    '',
    '            # Skip if already has personal columns',
    '            if "DMN_personal" in df.columns:',
    '                print("  EXISTS (skip): " + tsv_name)',
    '                n_skipped += 1',
    '                continue',
    '',
    '            # Extract ROI columns (64 columns)',
    '            roi_cols = ["ROI_" + str(i+1).zfill(2) for i in range(64)]',
    '            missing_cols = [c for c in roi_cols if c not in df.columns]',
    '            if missing_cols:',
    '                print("  ERROR missing ROI cols: " + str(missing_cols[:3]))',
    '                continue',
    '',
    '            roi_data = df[roi_cols].values  # (n_vols, 64)',
    '',
    '            # Compute weighted composites',
    '            dmn_personal = roi_data @ dmn_w  # (n_vols,)',
    '            cen_personal = roi_data @ cen_w  # (n_vols,)',
    '',
    '            # Append columns',
    '            df["DMN_personal"] = dmn_personal',
    '            df["CEN_personal"] = cen_personal',
    '',
    '            # Save in place',
    '            df.to_csv(str(tsv_path), sep="\\t", index=False)',
    '            print("  Updated: " + tsv_name',
    '                  + "  shape=" + str(df.shape))',
    '            n_updated += 1',
    '',
    'print()',
    'print("=" * 55)',
    'print("DONE")',
    'print("  Updated: " + str(n_updated))',
    'print("  Skipped: " + str(n_skipped))',
    'print("  Missing: " + str(n_missing))',
    'print("=" * 55)',
]

# ── 2. Save cluster script locally ─────────────────────────
script_name = "00c_add_personal_parcels_cluster.py"
script_path = LOCAL_BASE / "scripts" / script_name
script_path.parent.mkdir(parents=True, exist_ok=True)

with open(script_path, "w") as f:
    f.write("\n".join(lines))

# ── 3. Syntax check ────────────────────────────────────────
print("Checking syntax...")
try:
    py_compile.compile(str(script_path), doraise=True)
    print("Syntax OK: " + script_name)
except py_compile.PyCompileError as e:
    print("SYNTAX ERROR: " + str(e))
    raise

# ── 4. Build SLURM script ──────────────────────────────────
sbatch_lines = [
    "#!/bin/bash",
    "#SBATCH --job-name=add_personal_parcels",
    "#SBATCH --output=" + CLUSTER_BASE + "/logs/add_personal_parcels_%j.out",
    "#SBATCH --error="  + CLUSTER_BASE + "/logs/add_personal_parcels_%j.err",
    "#SBATCH --partition=short",
    "#SBATCH --time=01:00:00",
    "#SBATCH --cpus-per-task=2",
    "#SBATCH --mem=16G",
    "#SBATCH --account=" + SLURM_ACCOUNT,
    "",
    PYTHON + " " + CLUSTER_BASE + "/scripts/" + script_name,
]

sbatch_name = "00c_add_personal_parcels.sh"
sbatch_path = LOCAL_BASE / "scripts" / sbatch_name
with open(sbatch_path, "w") as f:
    f.write("\n".join(sbatch_lines))

# ── 5. Deploy ──────────────────────────────────────────────
print("\nDeploying scripts...")
scp_to(script_path,
       CLUSTER_BASE + "/scripts/" + script_name,
       verbose=False)
scp_to(sbatch_path,
       CLUSTER_BASE + "/scripts/" + sbatch_name,
       verbose=False)
print("Deployed: " + script_name)
print("Deployed: " + sbatch_name)

# ── 6. Submit ──────────────────────────────────────────────
print("\nSubmitting SLURM job...")
result = run_ssh("sbatch " + CLUSTER_BASE + "/scripts/" + sbatch_name)
job_id = ""
for line in result.stdout.strip().split("\n"):
    if "Submitted" in line:
        job_id = line.strip().split()[-1]
        print("Job ID: " + job_id)

# ── 7. Monitor ─────────────────────────────────────────────
if job_id:
    print("\nMonitoring job " + job_id + "  (Ctrl+C to stop watching)")
    print("-" * 55)
    try:
        while True:
            r = run_ssh(
                "squeue -j " + job_id
                + " --format=%.8i_%.8T_%.10M 2>/dev/null",
                verbose=False
            )
            status = r.stdout.strip()
            if status and "JOBID" not in status.split("\n")[-1]:
                print(status)
            else:
                print("Job finished — checking log...")
                log = run_ssh(
                    "tail -30 " + CLUSTER_BASE
                    + "/logs/add_personal_parcels_" + job_id + ".out 2>/dev/null",
                    verbose=False
                )
                print(log.stdout)
                break
            time.sleep(15)
    except KeyboardInterrupt:
        print("\nStopped watching. Check manually:")
        print("  squeue -j " + job_id)
        print("  tail -f " + CLUSTER_BASE
              + "/logs/add_personal_parcels_" + job_id + ".out")