# Adding a New Subject to the Cyclic Transcoder Pipeline

## Overview

When a new DMNELF subject is collected, follow these steps to include them
in the cyclic transcoder pipeline. All commands run from your Mac unless
otherwise noted.

---

## Prerequisites

- fMRIPrep has been run for the new subject
- EEG has been preprocessed (500Hz `.fif` files in BIDS format)
- Subject ID follows the convention: `dmnelf0XX` or `dmnelf1XXX`

---

## Step 1 — Verify raw data on Explorer

```bash
# Check fMRIPrep outputs exist
ssh cccbauer@explorer.northeastern.edu \
  'ls /projects/swglab/data/DMNELF/derivatives/fmriprep_25.2.5_fmap/sub-dmnelfXXX/ses-dmnelf/func/*task-rest*bold.nii.gz'

# Check EEG preprocessed files exist
ssh cccbauer@explorer.northeastern.edu \
  'ls /projects/swglab/data/DMNELF/derivatives/eeg_preprocessed/sub-dmnelfXXX/ses-dmnelf/eeg/*task-rest*500Hz*'
```

Both should return files. If fMRIPrep or EEG preprocessing hasn't been run yet, do that first.

---

## Step 2 — Extract personal DMN/CEN masks

This runs ICA on the resting-state fMRI and identifies the subject's personal DMN and CEN networks.

```bash
cd /Users/cccbauer/Documents/GitHub/dmnelf/analysis/fingerprint/cyclic_transcoder
bash run_mask_extraction.sh dmnelfXXX
```

Monitor progress (~20 min):
```bash
ssh cccbauer@explorer.northeastern.edu \
  'tail -f /projects/swglab/data/DMNELF/analysis/fmri_preprocessing/mask_extraction/logs/mask_ext_dmnelfXXX_*.out'
```

When done, verify masks were created:
```bash
ssh cccbauer@explorer.northeastern.edu \
  'ls /projects/swglab/data/DMNELF/derivatives/network_masks/sub-dmnelfXXX/'
```

You should see:
```
sub-dmnelfXXX_space-MNI152NLin6Asym_res-2_dmn_mask.nii.gz
sub-dmnelfXXX_space-MNI152NLin6Asym_res-2_cen_mask.nii.gz
```

---

## Step 3 — Add subject to config.yaml

Open `config.yaml` and add the new subject to the `all` list:

```yaml
data:
  subjects:
    all:
      - dmnelf001
      - dmnelf004
      ...
      - dmnelfXXX    # ← add here
```

Save and commit:
```bash
cd /Users/cccbauer/Documents/GitHub/dmnelf
git add analysis/fingerprint/cyclic_transcoder/config.yaml
git commit -m "add dmnelfXXX to cyclic transcoder pipeline"
git push
```

---

## Step 4 — Run data audit

Verify all data is in place before extracting features:

```bash
cd analysis/fingerprint/cyclic_transcoder
bash scripts/deploy_and_run.sh --check
```

The new subject should appear in the table with green BOLD, EEG, masks, and features=0.

If anything is red, fix it before proceeding.

---

## Step 5 — Extract features

Extract DiFuMo-64 + personal mask time series and EEG block means:

```bash
bash scripts/deploy_and_run.sh --extract
```

Monitor (~2h):
```bash
ssh cccbauer@explorer.northeastern.edu 'squeue -u cccbauer'
```

When done, rerun the audit — `features_extracted` should be 7 for the new subject.

---

## Step 6 — Retrain with the new subject

Since LOOCV trains one model per subject, adding a new subject means:
- One new model trained with the new subject held out
- All existing models need retraining with the new subject included

Resubmit all training jobs:

```bash
bash scripts/deploy_and_run.sh --train
```

This runs 14+ parallel GPU jobs (~12-24h each). Monitor:
```bash
ssh cccbauer@explorer.northeastern.edu \
  'tail -f /projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder/logs/train_*_0.out'
```

---

## Step 7 — Predict PDA on feedback runs

> **IMPORTANT:** `predict_job.sh` has a hardcoded `SUBJECTS=(...)` array and a
> matching `#SBATCH --array=0-N`. It does **not** read `config.yaml`. When you
> add a subject you must append it to the `SUBJECTS` array **and** bump the
> `--array` upper bound, then re-deploy the script to Explorer (`scp`). If you
> skip this, `sbatch predict_job.sh` silently re-predicts only the old subjects
> and writes nothing for the new one. After deploying, you can predict just the
> new indices, e.g. `sbatch --array=14,15 predict_job.sh`.

Once training is complete, submit the prediction jobs to Explorer:

```bash
ssh cccbauer@explorer.northeastern.edu \
  'cd /projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder && \
   sbatch predict_job.sh'
```

Monitor progress (~2h):
```bash
ssh cccbauer@explorer.northeastern.edu 'squeue -u cccbauer'
```

Output saved to:
```
/projects/swglab/data/DMNELF/derivatives/cyclic_features/sub-*/predictions/
  sub-*_task-feedback_pda_prediction.npz
```

---

## Quick Reference

| Task | Command |
|---|---|
| Extract masks | `bash run_mask_extraction.sh dmnelfXXX` |
| Audit data | `bash scripts/deploy_and_run.sh --check` |
| Extract features | `bash scripts/deploy_and_run.sh --extract` |
| Train models | `bash scripts/deploy_and_run.sh --train` |
| Predict PDA | `ssh cccbauer@explorer.northeastern.edu 'cd /projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder && sbatch predict_job.sh'` |
| Full pipeline | `bash scripts/deploy_and_run.sh --all` |
| Monitor jobs | `ssh cccbauer@explorer.northeastern.edu 'squeue -u cccbauer'` |

---

## File locations on Explorer

| Data | Path |
|---|---|
| fMRIPrep BOLD | `/projects/swglab/data/DMNELF/derivatives/fmriprep_25.2.5_fmap/sub-{subject}/` |
| EEG preprocessed | `/projects/swglab/data/DMNELF/derivatives/eeg_preprocessed/sub-{subject}/` |
| Personal masks | `/projects/swglab/data/DMNELF/derivatives/network_masks/sub-{subject}/` |
| Extracted features | `/projects/swglab/data/DMNELF/derivatives/cyclic_features/sub-{subject}/` |
| Model checkpoints | `/projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder/checkpoints/` |
| PDA predictions | `/projects/swglab/data/DMNELF/derivatives/cyclic_features/sub-{subject}/predictions/` |
| Job logs | `/projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder/logs/` |
