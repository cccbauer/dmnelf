# 03_compute_pda.py
# Run locally: python 03_compute_pda.py [--overwrite]
# Deploys cluster script, submits SLURM job.
#
# What it does on the cluster:
#   For each subject x task x run:
#   1. Load DiFuMo-64 TSV (68 columns including DMN_personal, CEN_personal)
#   2. Baseline z-score each parcel (first 25 volumes, matching MURFI)
#   3. Compute two PDA variants:
#      - PDA_group:    mean(CEN_group_z) - mean(DMN_group_z)
#      - PDA_personal: CEN_personal_z - DMN_personal_z
#   4. Save (n_vols,) float32 arrays
#
# DiFuMo-64 group indices (0-based, verified vs labels_64_dictionary.csv):
#   DMN: [3,6,29,35,38,58,60,61]
#     PCC(3), STS/angular(6), sup rostral(29), angular sup(35),
#     dmPFC(38), angular inf(58), MTG(60), sup frontal(61)
#   CEN: [4,31,47,48,50,51]
#     parieto-occipital(4), IPS(31), IPS-RH(47), IFsulcus(48),
#     MFG(50), IFG(51)
#
# Output:
#   targets/{subject}_task-{task}_run-{run}_pda_group.npy
#   targets/{subject}_task-{task}_run-{run}_pda_personal.npy

import argparse
import py_compile
import time
from pathlib import Path
from utils import run_ssh, scp_to
from config import (
    CLUSTER_BASE, SLURM_ACCOUNT, PYTHON,
    SUBJECTS_EEG_FMRI_ALL, LOCAL_BASE,
    MISSING_RUNS, DMN_IDX, CEN_IDX
)

parser = argparse.ArgumentParser()
parser.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing PDA files")
args = parser.parse_args()

# ── 1. Build cluster-side script ───────────────────────────
lines = [
    '#!/usr/bin/env python3',
    '"""',
    '03_compute_pda_cluster.py',
    'Compute PDA_group and PDA_personal targets from DiFuMo-68 TSVs.',
    'Baseline z-score matches MURFI real-time computation (Hinds 2011).',
    '"""',
    'import sys',
    'sys.stdout.reconfigure(line_buffering=True)',
    'import numpy as np',
    'import pandas as pd',
    'from pathlib import Path',
    '',
    '# ── Paths ─────────────────────────────────────────────',
    'CLUSTER_BASE  = Path("' + CLUSTER_BASE + '")',
    'TSV_DIR       = CLUSTER_BASE / "difumo_timeseries"',
    'OUT_DIR       = CLUSTER_BASE / "targets"',
    'OUT_DIR.mkdir(parents=True, exist_ok=True)',
    '',
    'SUBJECTS     = ' + str(SUBJECTS_EEG_FMRI_ALL),
    'MISSING_RUNS = ' + str(MISSING_RUNS),
    'OVERWRITE    = ' + str(args.overwrite),
    '',
    '# DiFuMo-64 group indices (0-based)',
    '# Verified against labels_64_dictionary.csv (Dadi et al. 2020)',
    'DMN_IDX = ' + str(DMN_IDX),
    'CEN_IDX = ' + str(CEN_IDX),
    '',
    '# MURFI baseline: first 25 volumes (30s at TR=1.2s)',
    'BASELINE_VOLS = 25',
    '',
    'TASK_RUNS = {',
    '    "rest":      ["01", "02"],',
    '    "shortrest": ["01"],',
    '    "feedback":  ["01", "02", "03", "04"],',
    '}',
    '',
    '# ── Helper ────────────────────────────────────────────',
    'def baseline_zscore(x, n=BASELINE_VOLS):',
    '    """Z-score relative to first n volumes (MURFI convention)."""',
    '    mu  = x[:n].mean(axis=0)',
    '    sig = x[:n].std(axis=0)',
    '    sig[sig < 1e-10] = 1e-10',
    '    return (x - mu) / sig',
    '',
    '# ── Main loop ─────────────────────────────────────────',
    'print("=" * 55)',
    'print("Computing PDA targets")',
    'print("  DMN_IDX: " + str(DMN_IDX))',
    'print("  CEN_IDX: " + str(CEN_IDX))',
    'print("  Overwrite: " + str(OVERWRITE))',
    'print("=" * 55)',
    '',
    'n_done    = 0',
    'n_skipped = 0',
    'n_missing = 0',
    '',
    'for subject in SUBJECTS:',
    '    for task, runs in TASK_RUNS.items():',
    '        for run in runs:',
    '',
    '            # Skip known missing runs',
    '            if subject in MISSING_RUNS:',
    '                if (task, run) in MISSING_RUNS[subject]:',
    '                    continue',
    '',
    '            tsv_name = (subject + "_ses-dmnelf_task-" + task',
    '                        + "_run-" + run',
    '                        + "_desc-difumo64_timeseries.tsv")',
    '            tsv_path = TSV_DIR / tsv_name',
    '',
    '            out_group    = OUT_DIR / (subject + "_task-" + task',
    '                          + "_run-" + run + "_pda_group.npy")',
    '            out_personal = OUT_DIR / (subject + "_task-" + task',
    '                          + "_run-" + run + "_pda_personal.npy")',
    '',
    '            if out_group.exists() and out_personal.exists():',
    '                if not OVERWRITE:',
    '                    print("  EXISTS (skip): " + subject',
    '                          + " " + task + " run-" + run)',
    '                    n_skipped += 1',
    '                    continue',
    '',
    '            if not tsv_path.exists():',
    '                print("  MISSING TSV: " + tsv_name)',
    '                n_missing += 1',
    '                continue',
    '',
    '            # Load TSV',
    '            df = pd.read_csv(str(tsv_path), sep="\\t")',
    '            n_vols = len(df)',
    '',
    '            # Extract ROI columns (64 parcels)',
    '            roi_cols = ["ROI_" + str(i+1).zfill(2)',
    '                        for i in range(64)]',
    '            roi_data = df[roi_cols].values.astype(np.float32)',
    '',
    '            # Baseline z-score all parcels',
    '            roi_z = baseline_zscore(roi_data)',
    '',
    '            # PDA_group',
    '            cen_z     = roi_z[:, CEN_IDX].mean(axis=1)',
    '            dmn_z     = roi_z[:, DMN_IDX].mean(axis=1)',
    '            pda_group = (cen_z - dmn_z).astype(np.float32)',
    '',
    '            # PDA_personal',
    '            if "DMN_personal" in df.columns and "CEN_personal" in df.columns:',
    '                dmn_p = df["DMN_personal"].values.reshape(-1, 1).astype(np.float64)',
    '                cen_p = df["CEN_personal"].values.reshape(-1, 1).astype(np.float64)',
    '                dmn_p_z = baseline_zscore(dmn_p).ravel()',
    '                cen_p_z = baseline_zscore(cen_p).ravel()',
    '                pda_personal = (cen_p_z - dmn_p_z).astype(np.float32)',
    '            else:',
    '                print("  WARNING: no personal columns — using group for " + subject)',
    '                pda_personal = pda_group.copy()',
    '',
    '            np.save(str(out_group),    pda_group)',
    '            np.save(str(out_personal), pda_personal)',
    '',
    '            print("  " + subject + " " + task + " run-" + run',
    '                  + "  n=" + str(n_vols)',
    '                  + "  pda_g_std=" + "{:.3f}".format(pda_group.std())',
    '                  + "  pda_p_std=" + "{:.3f}".format(pda_personal.std()))',
    '            n_done += 1',
    '',
    'print()',
    'print("=" * 55)',
    'print("DONE")',
    'print("  Computed: " + str(n_done))',
    'print("  Skipped:  " + str(n_skipped))',
    'print("  Missing:  " + str(n_missing))',
    'print("=" * 55)',
]

# ── 2. Save cluster script locally ─────────────────────────
script_name = "03_compute_pda_cluster.py"
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
job_name = "compute_pda"
run_cmd  = (PYTHON + " " + CLUSTER_BASE + "/scripts/" + script_name
            + (" --overwrite" if args.overwrite else ""))

sbatch_lines = [
    "#!/bin/bash",
    "#SBATCH --job-name=" + job_name,
    "#SBATCH --output=" + CLUSTER_BASE + "/logs/" + job_name + "_%j.out",
    "#SBATCH --error="  + CLUSTER_BASE + "/logs/" + job_name + "_%j.err",
    "#SBATCH --partition=short",
    "#SBATCH --time=01:00:00",
    "#SBATCH --cpus-per-task=2",
    "#SBATCH --mem=16G",
    "#SBATCH --account=" + SLURM_ACCOUNT,
    "",
    run_cmd,
]

sbatch_name = "03_compute_pda.sh"
sbatch_path = LOCAL_BASE / "scripts" / sbatch_name
with open(sbatch_path, "w") as f:
    f.write("\n".join(sbatch_lines))

# ── 5. Deploy ──────────────────────────────────────────────
print("\nDeploying...")
scp_to(script_path,
       CLUSTER_BASE + "/scripts/" + script_name,
       verbose=False)
scp_to(sbatch_path,
       CLUSTER_BASE + "/scripts/" + sbatch_name,
       verbose=False)
print("Deployed: " + script_name)
print("Command:  " + run_cmd)

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
    print("\nMonitoring job " + job_id + "  (Ctrl+C to stop)")
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
                    "tail -20 " + CLUSTER_BASE
                    + "/logs/" + job_name + "_" + job_id
                    + ".out 2>/dev/null",
                    verbose=False
                )
                print(log.stdout)
                break
            time.sleep(15)
    except KeyboardInterrupt:
        print("\nStopped watching.")
        print("  tail -f " + CLUSTER_BASE
              + "/logs/" + job_name + "_" + job_id + ".out")