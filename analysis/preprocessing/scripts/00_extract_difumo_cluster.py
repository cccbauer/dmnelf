#!/usr/bin/env python3
"""
00_extract_difumo_cluster.py
Extract DiFuMo-64 parcel timeseries from fMRIPrep BOLD.
Uses standardize="psc" (percent signal change).
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
from pathlib import Path
import os
import warnings
import nibabel as nib

# ── Paths ─────────────────────────────────────────────
FMRIPREP_ROOT = Path("/projects/swglab/data/DMNELF/derivatives/fmriprep_25.2.5_fmap")
CLUSTER_BASE  = Path("/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3")
DIFUMO_CACHE  = Path("/projects/swglab/software/nilearn_data")
OUT_DIR       = CLUSTER_BASE / "difumo_timeseries"
CACHE_DIR     = CLUSTER_BASE / "nilearn_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["NILEARN_DATA"] = str(DIFUMO_CACHE)

SUBJECTS = ['sub-dmnelf001', 'sub-dmnelf002', 'sub-dmnelf003', 'sub-dmnelf004', 'sub-dmnelf005', 'sub-dmnelf006', 'sub-dmnelf007', 'sub-dmnelf008', 'sub-dmnelf009', 'sub-dmnelf010', 'sub-dmnelf011', 'sub-dmnelf1001', 'sub-dmnelf1002', 'sub-dmnelf1003', 'sub-dmnelf999']
TASKS    = {'rest': ['01', '02'], 'shortrest': ['01'], 'feedback': ['01', '02', '03', '04']}

# ── Load DiFuMo-64 atlas ──────────────────────────────
print("=" * 55)
print("Loading DiFuMo-64 atlas")
print("=" * 55)

from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    difumo = datasets.fetch_atlas_difumo(
        dimension=64,
        resolution_mm=2,
        data_dir=str(DIFUMO_CACHE)
    )

atlas_img = difumo.maps
print("DiFuMo maps: " + str(atlas_img))

# Always use ROI_01..ROI_64 for consistency
roi_cols = ["ROI_" + str(i+1).zfill(2) for i in range(64)]
print("ROI columns: " + str(len(roi_cols)))

# ── Build masker ──────────────────────────────────────
masker = NiftiMapsMasker(
    maps_img=atlas_img,
    standardize="psc",
    memory=str(CACHE_DIR),
    verbose=0
)
masker.fit()
print("Masker fitted  standardize=psc")

# ── Main loop ─────────────────────────────────────────
print()
print("=" * 55)
print("Extracting DiFuMo timeseries  (overwrite=True)")
print("=" * 55)

n_extracted = 0
n_missing   = 0

for subject in SUBJECTS:
    for task, runs in TASKS.items():
        for run in runs:

            tsv_name = (subject + "_ses-dmnelf"
                        + "_task-" + task
                        + "_run-"  + run
                        + "_desc-difumo64_timeseries.tsv")
            tsv_path = OUT_DIR / tsv_name

            bold_name = (subject + "_ses-dmnelf"
                         + "_task-" + task
                         + "_run-"  + run
                         + "_space-MNI152NLin6Asym_res-2"
                         + "_desc-preproc_bold.nii.gz")
            bold_path = (FMRIPREP_ROOT / subject
                         / "ses-dmnelf" / "func" / bold_name)

            if not bold_path.exists():
                print("  MISSING: " + bold_name)
                n_missing += 1
                continue

            print("  Extracting: " + subject
                  + "  task-" + task + "  run-" + run)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ts = masker.transform(str(bold_path))
            except Exception as e:
                print("  ERROR: " + str(e))
                n_missing += 1
                continue

            n_vols = ts.shape[0]
            img    = nib.load(str(bold_path))
            tr     = float(img.header.get_zooms()[3])
            times  = np.arange(n_vols) * tr

            df = pd.DataFrame(ts, columns=roi_cols)
            df.insert(0, "volume", np.arange(n_vols))
            df.insert(1, "time",   times.round(4))

            df.to_csv(str(tsv_path), sep="\t", index=False)
            print("    shape=" + str(ts.shape)
                  + "  TR=" + str(round(tr, 3))
                  + "  PSC_mean=" + str(round(float(ts.mean()), 4)))
            n_extracted += 1

print()
print("=" * 55)
print("DONE")
print("  Extracted: " + str(n_extracted))
print("  Missing:   " + str(n_missing))
print("=" * 55)