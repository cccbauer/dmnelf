# Validation Pipeline: Cross-Subject EEG → fMRI Decoding

## Overview

Train the ElasticNet decoder on N existing subjects, predict on a held-out
new subject's feedback runs. True out-of-sample validation — the test subject
is never seen during training.

First validated: dmnelf016 (2026-06-25).

---

## Prerequisites

Before starting, the new subject needs:
- **fMRIPrep** completed (BOLD + confounds + brain masks for all tasks)
- **BVA gradient correction** done in BrainVision Analyzer (produces `.edf` exports)
- BVA exports placed in `sourcedata/eeg_data/eeg_preprocessed/` on the cluster

---

## Pipeline Steps

### Step 1 — Organize raw EEG (BVA export → BIDS naming)

Renames BVA-exported EDFs from `sourcedata/eeg_data/eeg_preprocessed/` into
BIDS-compliant `rawdata_eeg/` structure.

**Script:** `organize_raw_eeg.py` (already deployed on cluster)  
**Env:** `eeg_preproc`  
**Runs:** on cluster login node (fast, no SLURM needed)

```bash
ssh cccbauer@explorer.northeastern.edu '
  /home/cccbauer/.conda/envs/eeg_preproc/bin/python \
    /projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/scripts/organize_raw_eeg.py \
    --subject sub-dmnelfXXX --move
'
```

**Verify:**
```bash
ssh cccbauer@explorer.northeastern.edu \
  'ls /projects/swglab/data/DMNELF/rawdata_eeg/sub-dmnelfXXX/ses-dmnelf/eeg/*feedback*edf | wc -l'
# Expect: 4
```

**Source:** `/projects/swglab/data/DMNELF/sourcedata/eeg_data/eeg_preprocessed/sub-dmnelfXXX_task_*`  
**Target:** `/projects/swglab/data/DMNELF/rawdata_eeg/sub-dmnelfXXX/ses-dmnelf/eeg/*.edf`

---

### Step 2 — EEG preprocessing (500 Hz .fif)

Runs the full 11-step EEG-fMRI preprocessing pipeline: gradient artifact
correction → BCG → ICA → interpolation → average reference.

**Script:** `eeg_preproc.py` (on cluster)  
**Env:** `eeg_preproc` → `/home/cccbauer/.conda/envs/eeg_preproc/bin/python`  
**SLURM:** `--partition=short --mem=32G --cpus-per-task=4 --time=08:00:00`  
**Runtime:** ~1–2 hours for all runs

```bash
ssh cccbauer@explorer.northeastern.edu << 'EOF'
sbatch --job-name=eeg_XXX --partition=short --mem=32G --cpus-per-task=4 --time=08:00:00 \
  --output=/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/validation/logs/eeg_XXX.out \
  --error=/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/validation/logs/eeg_XXX.err \
  --wrap="/home/cccbauer/.conda/envs/eeg_preproc/bin/python \
    /projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/scripts/eeg_preproc.py \
    --subject sub-dmnelfXXX --sfreq 500"
EOF
```

**Verify:**
```bash
ssh cccbauer@explorer.northeastern.edu \
  'ls /projects/swglab/data/DMNELF/derivatives/eeg_preprocessed/sub-dmnelfXXX/ses-dmnelf/eeg/*feedback*500Hz*fif | wc -l'
# Expect: 4
```

**Output:** `derivatives/eeg_preprocessed/sub-dmnelfXXX/ses-dmnelf/eeg/*desc-preproc500Hz_eeg.fif`

**Gotcha:** The shortrest run may fail (OK=6 FAILED=1 is normal — shortrest
sometimes lacks an R128 trigger). Only the 4 feedback + 2 rest runs are needed.

---

### Step 3 — Extract personalized DMN/CEN masks

Runs ICA on resting-state BOLD to identify subject-specific DMN and CEN networks.

**Script:** `mask_extraction.py`  
**Location:** `/projects/swglab/data/DMNELF/analysis/fmri_preprocessing/mask_extraction/`  
**Env:** `pineuro` → `/home/cccbauer/.conda/envs/pineuro/bin/python`  
**SLURM:** `--partition=short --mem=32G --cpus-per-task=8 --time=02:00:00`  
**Runtime:** ~20 min

⚠️ **CRITICAL:** This step requires the `pineuro` conda environment, NOT `eeg_preproc`.
Using the wrong env gives `ERROR: pineuro not importable`. Use the direct Python path
to avoid `conda activate` issues inside SLURM jobs.

```bash
ssh cccbauer@explorer.northeastern.edu << 'EOF'
sbatch --job-name=mask_XXX --partition=short --mem=32G --cpus-per-task=8 --time=02:00:00 \
  --output=/projects/swglab/data/DMNELF/analysis/fmri_preprocessing/mask_extraction/logs/mask_XXX_%j.out \
  --error=/projects/swglab/data/DMNELF/analysis/fmri_preprocessing/mask_extraction/logs/mask_XXX_%j.err \
  --wrap="/home/cccbauer/.conda/envs/pineuro/bin/python \
    /projects/swglab/data/DMNELF/analysis/fmri_preprocessing/mask_extraction/mask_extraction.py \
    --subject dmnelfXXX \
    --config /projects/swglab/data/DMNELF/analysis/fmri_preprocessing/mask_extraction/config.yaml"
EOF
```

Alternatively, from your Mac:
```bash
cd /Users/cccbauer/Documents/GitHub/dmnelf/analysis/fingerprint
bash run_mask_extraction.sh dmnelfXXX
```
(This uses `conda activate fingerprint` which may or may not work depending on
SLURM's shell init. If it fails with "pineuro not importable", use the direct
Python path method above.)

**Verify:**
```bash
ssh cccbauer@explorer.northeastern.edu \
  'ls /projects/swglab/data/DMNELF/derivatives/network_masks/sub-dmnelfXXX/*mask*'
# Expect: sub-dmnelfXXX_space-MNI152NLin6Asym_res-2_dmn_mask.nii.gz
#         sub-dmnelfXXX_space-MNI152NLin6Asym_res-2_cen_mask.nii.gz
```

---

### Step 4 — Extract cyclic_features .npz

Extracts DiFuMo-64 parcel timeseries + personalized DMN/CEN ROI timeseries
from fMRIPrep BOLD, and EEG block-averages. Produces the `.npz` files that
the decoding pipeline reads.

**Script:** `extract_features.py`  
**Location:** `/projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder/data/`  
**Env:** `eeg_preproc` → `/home/cccbauer/.conda/envs/eeg_preproc/bin/python`  
**SLURM:** `--partition=short --mem=16G --time=02:00:00`  
**Runtime:** ~30 min  
**Depends on:** Steps 2 (EEG .fif) + 3 (masks)

```bash
ssh cccbauer@explorer.northeastern.edu << 'EOF'
sbatch --job-name=feat_XXX --partition=short --mem=16G --time=02:00:00 \
  --output=/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/validation/logs/feat_XXX.out \
  --error=/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/validation/logs/feat_XXX.err \
  --wrap="cd /projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder && \
    /home/cccbauer/.conda/envs/eeg_preproc/bin/python data/extract_features.py \
    --subject dmnelfXXX --config config.yaml"
EOF
```

**Verify:**
```bash
ssh cccbauer@explorer.northeastern.edu \
  'ls /projects/swglab/data/DMNELF/derivatives/cyclic_features/sub-dmnelfXXX/*feedback*features.npz | wc -l'
# Expect: 4
```

**Output:** `derivatives/cyclic_features/sub-dmnelfXXX/sub-dmnelfXXX_task-feedback_run-{1,2,3,4}_features.npz`  
Each .npz contains: `fmri_features` (125×66), `eeg_block` (125×31), `pda` (125,)

---

### Step 5 — Cross-subject prediction

Trains ElasticNet on all existing subjects (pooled), predicts on the new subject's
4 feedback runs. 10K circular-shift null for significance.

**Script:** `cross_subject_predict.py`  
**Location:** `eeg_bold_coupling/validation/`  
**Env:** `eeg_preproc` → `/home/cccbauer/.conda/envs/eeg_preproc/bin/python`  
**SLURM:** `--partition=short --mem=16G --time=04:00:00`  
**Runtime:** ~1–2 hours  
**Depends on:** Step 4 (features for test subject) + existing subjects' cached bandpower

```bash
ssh cccbauer@explorer.northeastern.edu << 'EOF'
sbatch --job-name=xsub_XXX --partition=short --mem=16G --time=04:00:00 \
  --output=/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/validation/logs/xsub_XXX.out \
  --error=/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/validation/logs/xsub_XXX.err \
  --wrap="cd /projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/validation && \
    /home/cccbauer/.conda/envs/eeg_preproc/bin/python cross_subject_predict.py \
    --test_subject dmnelfXXX --n_shuffles 10000"
EOF
```

**Output:**
```
results/validation/
  dmnelfXXX_cross_subject_results.csv    — per-target r, p, per-run r
  dmnelfXXX_GSR_CEN_predictions.npz     — predicted + true timeseries
  dmnelfXXX_GSR_DMN_predictions.npz
  dmnelfXXX_PDA_predictions.npz
  dmnelfXXX_RAW_DMN_predictions.npz
```

### Step 6 — Pull results

```bash
PROJ=/Users/cccbauer/Documents/GitHub/dmnelf/analysis/fingerprint/eeg_bold_coupling
rsync -avhP cccbauer@explorer.northeastern.edu:/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/results/validation/ \
  $PROJ/results/validation/
```

---

## Environment Reference

| Step | Conda env | Python path | Key packages |
|------|-----------|-------------|--------------|
| 1. Organize EEG | `eeg_preproc` | `/home/cccbauer/.conda/envs/eeg_preproc/bin/python` | mne, numpy |
| 2. EEG preproc | `eeg_preproc` | same | mne, neurokit2, mne_icalabel |
| 3. Mask extraction | **`pineuro`** | `/home/cccbauer/.conda/envs/pineuro/bin/python` | pineuro, nilearn, nibabel |
| 4. Feature extraction | `eeg_preproc` | `/home/cccbauer/.conda/envs/eeg_preproc/bin/python` | mne, nilearn, nibabel |
| 5. Cross-subject predict | `eeg_preproc` | same | sklearn, mne, scipy, pandas |

⚠️ **Step 3 is the only step that needs a different env (`pineuro`).** All other
steps use `eeg_preproc`. Always use the full Python path in SLURM `--wrap` commands
to avoid `conda activate` failures.

---

## Cluster Paths

| Data | Path |
|------|------|
| Raw EEG (BVA export) | `/projects/swglab/data/DMNELF/sourcedata/eeg_data/eeg_preprocessed/` |
| Raw EEG (BIDS) | `/projects/swglab/data/DMNELF/rawdata_eeg/sub-*/ses-dmnelf/eeg/` |
| EEG preprocessed | `/projects/swglab/data/DMNELF/derivatives/eeg_preprocessed/sub-*/` |
| fMRIPrep | `/projects/swglab/data/DMNELF/derivatives/fmriprep_25.2.5_fmap/sub-*/` |
| Network masks | `/projects/swglab/data/DMNELF/derivatives/network_masks/sub-*/` |
| Cyclic features | `/projects/swglab/data/DMNELF/derivatives/cyclic_features/sub-*/` |
| Validation logs | `.../eeg_bold_coupling/validation/logs/` |
| Validation results | `.../eeg_bold_coupling/results/validation/` |
| Bandpower cache | `.../eeg_bold_coupling/results/multivariate/cache/` |
| Mask extraction script | `.../analysis/fmri_preprocessing/mask_extraction/mask_extraction.py` |
| Feature extraction script | `.../analysis/fingerprint/cyclic_transcoder/data/extract_features.py` |
| EEG preproc script | `.../analysis/MNE/jupyter/microstate_pda_v3/scripts/eeg_preproc.py` |
| Organize EEG script | `.../analysis/MNE/jupyter/microstate_pda_v3/scripts/organize_raw_eeg.py` |

---

## Run Log: dmnelf016 (2026-06-25)

| Step | Job ID | Status | Notes |
|------|--------|--------|-------|
| 1. Organize EEG | (inline) | ✅ | 8 EDFs copied to rawdata_eeg |
| 2. EEG preproc | 7866728 | ✅ | OK=6 FAILED=1 (shortrest failed, expected) |
| 3. Mask extraction | 7867310 | ✅ | Used `pineuro` env. First attempt (7866729) failed with wrong env |
| 4. Feature extraction | 7867367 | ✅ | 4 feedback .npz created |
| 5. Cross-subject predict | (pending) | ⏳ | Awaiting step 4 |

---

## Lessons Learned (dmnelf016)

1. **`conda activate` often fails inside SLURM jobs** — always use the full path
   to the Python binary (e.g., `/home/cccbauer/.conda/envs/pineuro/bin/python`)
   instead of relying on `conda activate` + `python`.

2. **Step 3 (masks) needs `pineuro` env**, not `eeg_preproc`. The `fingerprint` env
   also has pineuro but `conda activate fingerprint` inside SLURM didn't work
   reliably. Direct path to `/home/cccbauer/.conda/envs/pineuro/bin/python` is safest.

3. **EEG preprocessing may report OK=6 FAILED=1** — the shortrest run sometimes
   fails due to missing R128 trigger. This is fine; only feedback + rest runs are
   needed for the decoding pipeline.

6. **Within-subject decoding with 10K nulls needs >2h** — the 10K circular-shift
   permutations per target × 2 models are expensive. Use `--time=08:00:00` or more.
   The `short` partition allows up to 24 hours.

4. **rawdata_eeg must exist before EEG preprocessing** — the BVA exports in
   `sourcedata/eeg_data/eeg_preprocessed/` need to be organized into BIDS
   format first via `organize_raw_eeg.py`. Without this, eeg_preproc.py
   reports "MISSING EDF".

5. **Feature extraction depends on BOTH EEG + masks** — submit with
   `--dependency=afterok:EEG_JOB:MASK_JOB` if chaining via SLURM, or just
   verify both outputs exist before submitting.
