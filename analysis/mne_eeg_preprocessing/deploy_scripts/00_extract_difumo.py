# 00_extract_difumo.py
# Run locally: python 00_extract_difumo.py
# Deploys cluster script, submits SLURM job.
#
# What it does on the cluster:
#   For each subject x task x run:
#   1. Load fMRIPrep preprocessed BOLD (MNI152NLin6Asym res-2)
#   2. Extract timeseries from DiFuMo-64 probabilistic parcels
#      using NiftiMapsMasker (weighted average per parcel)
#      standardize="psc" — percent signal change normalization
#   3. Save as TSV: volume, time, ROI_01..ROI_64
#
# Output:
#   difumo_timeseries/
#     {subject}_ses-dmnelf_task-{task}_run-{run}_desc-difumo64_timeseries.tsv

import py_compile
import time
from pathlib import Path
from utils import run_ssh, scp_to, make_cluster_dirs
from config import (
    CLUSTER_BASE, SLURM_ACCOUNT, PYTHON,
    SUBJECTS, LOCAL_BASE
)

FMRIPREP_ROOT = "/projects/swglab/data/DMNELF/derivatives/fmriprep_25.2.5_fmap"
DIFUMO_CACHE  = "/projects/swglab/software/nilearn_data"

TASKS = {
    "rest":      ["01", "02"],
    "shortrest": ["01"],
    "feedback":  ["01", "02", "03", "04"],
}

# ── 1. Build cluster-side script ───────────────────────────
lines = [
    '#!/usr/bin/env python3',
    '"""',
    '00_extract_difumo_cluster.py',
    'Extract DiFuMo-64 parcel timeseries from fMRIPrep BOLD.',
    'Uses standardize="psc" (percent signal change).',
    '"""',
    'import sys',
    'sys.stdout.reconfigure(line_buffering=True)',
    'import numpy as np',
    'import pandas as pd',
    'from pathlib import Path',
    'import os',
    'import warnings',
    'import nibabel as nib',
    '',
    '# ── Paths ─────────────────────────────────────────────',
    'FMRIPREP_ROOT = Path("' + FMRIPREP_ROOT + '")',
    'CLUSTER_BASE  = Path("' + CLUSTER_BASE + '")',
    'DIFUMO_CACHE  = Path("' + DIFUMO_CACHE + '")',
    'OUT_DIR       = CLUSTER_BASE / "difumo_timeseries"',
    'CACHE_DIR     = CLUSTER_BASE / "nilearn_cache"',
    'OUT_DIR.mkdir(parents=True, exist_ok=True)',
    'CACHE_DIR.mkdir(parents=True, exist_ok=True)',
    'os.environ["NILEARN_DATA"] = str(DIFUMO_CACHE)',
    '',
    'SUBJECTS = ' + str(SUBJECTS),
    'TASKS    = ' + str(TASKS),
    '',
    '# ── Load DiFuMo-64 atlas ──────────────────────────────',
    'print("=" * 55)',
    'print("Loading DiFuMo-64 atlas")',
    'print("=" * 55)',
    '',
    'from nilearn import datasets',
    'from nilearn.maskers import NiftiMapsMasker',
    '',
    'with warnings.catch_warnings():',
    '    warnings.simplefilter("ignore")',
    '    difumo = datasets.fetch_atlas_difumo(',
    '        dimension=64,',
    '        resolution_mm=2,',
    '        data_dir=str(DIFUMO_CACHE)',
    '    )',
    '',
    'atlas_img = difumo.maps',
    'print("DiFuMo maps: " + str(atlas_img))',
    '',
    '# Always use ROI_01..ROI_64 for consistency',
    'roi_cols = ["ROI_" + str(i+1).zfill(2) for i in range(64)]',
    'print("ROI columns: " + str(len(roi_cols)))',
    '',
    '# ── Build masker ──────────────────────────────────────',
    'masker = NiftiMapsMasker(',
    '    maps_img=atlas_img,',
    '    standardize="psc",',
    '    memory=str(CACHE_DIR),',
    '    verbose=0',
    ')',
    'masker.fit()',
    'print("Masker fitted  standardize=psc")',
    '',
    '# ── Main loop ─────────────────────────────────────────',
    'print()',
    'print("=" * 55)',
    'print("Extracting DiFuMo timeseries  (overwrite=True)")',
    'print("=" * 55)',
    '',
    'n_extracted = 0',
    'n_missing   = 0',
    '',
    'for subject in SUBJECTS:',
    '    for task, runs in TASKS.items():',
    '        for run in runs:',
    '',
    '            tsv_name = (subject + "_ses-dmnelf"',
    '                        + "_task-" + task',
    '                        + "_run-"  + run',
    '                        + "_desc-difumo64_timeseries.tsv")',
    '            tsv_path = OUT_DIR / tsv_name',
    '',
    '            bold_name = (subject + "_ses-dmnelf"',
    '                         + "_task-" + task',
    '                         + "_run-"  + run',
    '                         + "_space-MNI152NLin6Asym_res-2"',
    '                         + "_desc-preproc_bold.nii.gz")',
    '            bold_path = (FMRIPREP_ROOT / subject',
    '                         / "ses-dmnelf" / "func" / bold_name)',
    '',
    '            if not bold_path.exists():',
    '                print("  MISSING: " + bold_name)',
    '                n_missing += 1',
    '                continue',
    '',
    '            print("  Extracting: " + subject',
    '                  + "  task-" + task + "  run-" + run)',
    '',
    '            try:',
    '                with warnings.catch_warnings():',
    '                    warnings.simplefilter("ignore")',
    '                    ts = masker.transform(str(bold_path))',
    '            except Exception as e:',
    '                print("  ERROR: " + str(e))',
    '                n_missing += 1',
    '                continue',
    '',
    '            n_vols = ts.shape[0]',
    '            img    = nib.load(str(bold_path))',
    '            tr     = float(img.header.get_zooms()[3])',
    '            times  = np.arange(n_vols) * tr',
    '',
    '            df = pd.DataFrame(ts, columns=roi_cols)',
    '            df.insert(0, "volume", np.arange(n_vols))',
    '            df.insert(1, "time",   times.round(4))',
    '',
    '            df.to_csv(str(tsv_path), sep="\\t", index=False)',
    '            print("    shape=" + str(ts.shape)',
    '                  + "  TR=" + str(round(tr, 3))',
    '                  + "  PSC_mean=" + str(round(float(ts.mean()), 4)))',
    '            n_extracted += 1',
    '',
    'print()',
    'print("=" * 55)',
    'print("DONE")',
    'print("  Extracted: " + str(n_extracted))',
    'print("  Missing:   " + str(n_missing))',
    'print("=" * 55)',
]

# ── 2. Save cluster script locally ─────────────────────────
script_name = "00_extract_difumo_cluster.py"
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
    "#SBATCH --job-name=difumo_extract",
    "#SBATCH --output=" + CLUSTER_BASE + "/logs/difumo_extract_%j.out",
    "#SBATCH --error="  + CLUSTER_BASE + "/logs/difumo_extract_%j.err",
    "#SBATCH --partition=short",
    "#SBATCH --time=06:00:00",
    "#SBATCH --cpus-per-task=4",
    "#SBATCH --mem=32G",
    "#SBATCH --account=" + SLURM_ACCOUNT,
    "",
    PYTHON + " " + CLUSTER_BASE + "/scripts/" + script_name,
]

sbatch_name = "00_extract_difumo.sh"
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
                    "tail -30 " + CLUSTER_BASE
                    + "/logs/difumo_extract_" + job_id
                    + ".out 2>/dev/null",
                    verbose=False
                )
                print(log.stdout)
                break
            time.sleep(15)
    except KeyboardInterrupt:
        print("\nStopped watching.")
        print("  tail -f " + CLUSTER_BASE
              + "/logs/difumo_extract_" + job_id + ".out")