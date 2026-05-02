---

## Step-by-Step Instructions

### 0. Setup (first time only)

**On local machine:**
```bash
# Clone repo
git clone https://github.com/cccbauer/microstate_pda.git
cd microstate_pda

# Verify config loads correctly
python -c "import config; print(config.LOCAL_BASE)"

# Set up passwordless SSH
ssh-keygen -t ed25519 -C "dmnelf_pipeline"
ssh-copy-id cccbauer@explorer.northeastern.edu
```

**Fix PATH if ssh not found (anitya machine):**
```bash
export PATH="/usr/bin:/usr/local/bin:/bin:/usr/sbin:/sbin:$PATH"
```
Add this to `~/.zshrc` to make it permanent.

---

### 1. Check data availability

**For DMNELF:**
```bash
ssh cccbauer@explorer.northeastern.edu "bash /projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/scripts/data_available.sh dmnelf 2>/dev/null"
```

**For rtBPD:**
```bash
ssh cccbauer@explorer.northeastern.edu "bash /projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/scripts/data_available.sh rtbpd 2>/dev/null"
```

The script outputs:
- Per-subject, per-task, per-run availability of RAW and fMRIPrep BOLD
- Summary of complete vs incomplete subjects
- Ready-to-paste Python list of complete subjects for `config.py`

**After running:** copy the `SUBJECTS = [...]` list into `config.py` if it has changed.

---

### 2. Rerun fMRIPrep for incomplete subjects

**DMNELF:**
```bash
ssh cccbauer@explorer.northeastern.edu
cd /projects/swglab/data/DMNELF/scripts/run_fmriprep
sh submit_dmnelf_short.sh 001 002 005 006 1002 1003
```
Replace subject numbers with whoever is incomplete per `data_available.sh`.

**rtBPD:**
```bash
ssh cccbauer@explorer.northeastern.edu
cd /projects/swglab/data/rtBPD/scripts/run_fmriprep_explorer
bash process_new_rtbpd_subjects.sh 001 002 004 012 013
```
Replace subject numbers with whoever is incomplete per `data_available.sh`.

**Monitor fMRIPrep jobs:**
```bash
ssh cccbauer@explorer.northeastern.edu 'bash -l -c "squeue -u cccbauer --format=\"%.8i %.20j %.8T %.10M\""'
```

**Check if fMRIPrep succeeded for a subject:**
```bash
ssh cccbauer@explorer.northeastern.edu 'bash -l -c "
ls /projects/swglab/data/DMNELF/derivatives/fmriprep_25.2.5_fmap/sub-dmnelf001.html
"'
```

---

### 3. Extract DiFuMo-64 timeseries

Run locally — deploys and submits SLURM job automatically:
```bash
python 00_extract_difumo.py
```

This extracts DiFuMo-64 parcel timeseries from fMRIPrep BOLD output.  
**Input:** `fmriprep_25.2.5_fmap/{subject}/ses-dmnelf/func/*preproc_bold.nii.gz`  
**Output:** `difumo_timeseries/{subject}_ses-dmnelf_task-{task}_run-{run}_desc-difumo64_timeseries.tsv`

---

### 4. Fit microstate maps

Run locally — deploys and submits SLURM job automatically:
```bash
python 01_fit_microstates.py
```

- Pools all rest EEG across all subjects
- Fits 7 polarity-invariant k-means maps (Custo 2017)
- **Expected runtime:** 20–40 minutes on CPU
- **Expected GEV:** ~57% (31-channel dataset, lower than 64-ch studies)
- **Output:** `microstates/templates.npy` (7, 31)

**Check results:**
```bash
ssh cccbauer@explorer.northeastern.edu 'bash -l -c "tail -20 /projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/logs/fit_microstates_JOBID.out"'
```

---

### 5. Compute TESS features

Run locally — deploys and submits SLURM job automatically:
```bash
python 02_tess_features.py
```

- Requires `templates.npy` from step 4
- Computes continuous T-hat coefficients via TESS Stage 1 spatial GLM
- Convolves with canonical HRF, downsamples to TR
- Appends GFP and GMD
- **Output:** `features/{subject}_task-{task}_run-{run}_features.npy` (n_vols, 9)
- **Feature columns:** T-hat_A, T-hat_B, T-hat_C, T-hat_D, T-hat_E, T-hat_F, T-hat_G, GFP, GMD

---

### 6. Compute PDA targets

```bash
python 03_compute_pda.py
```

- MURFI-aligned: z-scores each parcel relative to first 25 volumes (30s baseline)
- **Output:** `targets/{subject}_task-{task}_run-{run}_pda.npy` (n_vols,)

---

### 7. Train decoder

```bash
python 04_train_decoder.py
```

- ElasticNet, leave-one-run-out CV on feedback runs
- **Expected:** MS C weight negative, MS D weight positive
- **Output:** `models/{subject}_decoder.pkl`

---

### 8. Evaluate decoder

```bash
python 05_evaluate_decoder.py
```

- Tests on shortrest (near transfer) and feedback (in-distribution)
- **Output:** `results/evaluation.csv`

---

## When New Subjects Are Added

1. Run `data_available.sh` to identify what's missing
2. Rerun fMRIPrep for incomplete subjects
3. Rerun `00_extract_difumo.py` — skips already-processed files
4. Update `SUBJECTS` in `config.py`
5. Rerun `01_fit_microstates.py` with expanded subject list
6. Rerun `02_tess_features.py` — skips already-processed files
7. Rerun `03_compute_pda.py` — skips already-processed files
8. Rerun `04_train_decoder.py`
9. Rerun `05_evaluate_decoder.py`

---

## Key Paths

### Cluster
| Data | Path |
|------|------|
| Project | `/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/` |
| EEG preprocessed | `/projects/swglab/data/DMNELF/analysis/MNE/bids/derivatives/preprocessed/` |
| DiFuMo timeseries | `/projects/swglab/data/DMNELF/analysis/MNE/jupyter/neurobolt/difumo_timeseries/` |
| fMRIPrep DMNELF | `/projects/swglab/data/DMNELF/derivatives/fmriprep_25.2.5_fmap/` |
| fMRIPrep rtBPD | `/projects/swglab/data/rtBPD/derivatives/fmriprep_24.1.1/` |
| SLURM logs | `/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/logs/` |

### Local (Dropbox synced)
| Machine | Path |
|---------|------|
| cccbauer | `/Users/cccbauer/Dropbox (MIT)/00_2022/ResearchScientist/01_grants/01_R21-2021/Analysis/MNE/microstate_pda_v3/` |
| anitya | `/Users/anitya/Dropbox (MIT)/00_2022/ResearchScientist/01_grants/01_R21-2021/Analysis/MNE/microstate_pda_v3/` |

---

## Troubleshooting

### ssh: command not found
```bash
export PATH="/usr/bin:/usr/local/bin:/bin:/usr/sbin:/sbin:$PATH"
```

### SCP transferred 0 bytes
Always verify after SCP:
```bash
ssh cccbauer@explorer.northeastern.edu "ls -la REMOTE_PATH || echo MISSING"
```

### Job pending with (BeginTime)
Normal — SLURM scheduled it for a future slot. Wait or check:
```bash
ssh cccbauer@explorer.northeastern.edu 'bash -l -c "scontrol show job JOB_ID | grep StartTime"'
```

### Job pending with (Priority)
Normal — waiting for available nodes. Check queue depth:
```bash
ssh cccbauer@explorer.northeastern.edu 'bash -l -c "squeue -p short | wc -l"'
```

### fMRIPrep did not complete
Check the error log:
```bash
ssh cccbauer@explorer.northeastern.edu 'bash -l -c "
tail -30 /projects/swglab/data/DMNELF/scripts/run_fmriprep/logs/dmnelf_001_JOBID.err
"'
```

### templates.npy not found when running 02_tess_features.py
Step 4 (fit microstates) must complete first. Check:
```bash
ssh cccbauer@explorer.northeastern.edu 'bash -l -c "ls /projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/microstates/"'
```

---

## Git Workflow

```bash
# Pull latest before starting work
git pull

# After making changes
git add .
git commit -m "description of changes"
git push

# Check status
git log --oneline -5
```

**Repo:** https://github.com/cccbauer/microstate_pda (private)




PART 1 — HOW TO RUN
===================

Prerequisites

Local machine:
  conda activate base
  pip install mne numpy pandas nibabel nilearn

Cluster environments:
  Main pipeline: $HOME/my_anaconda/bin/python
  EEG preprocessing: /home/cccbauer/.conda/envs/eeg_preproc/bin/python


Full pipeline run order

  0a  BVA preprocessing          (manual, done once per subject)
  0b  EEG full preprocessing     python eeg_preproc_deploy.py
  0c  DiFuMo extraction          python 00_extract_difumo.py
  0d  Personalized masks         python 00b_extract_personal_masks.py
  0e  Add personal parcels       python 00c_add_personal_parcels.py
  1   Fit microstates            python 01_fit_microstates.py
  2   TESS features              python 02_tess_features.py
  3   Compute PDA                python 03_compute_pda.py
  4   Train decoder              python 04_train_decoder.py
  5   Evaluate decoder           python 05_evaluate_decoder.py


Step 0b — EEG preprocessing (250Hz and 500Hz run in parallel)

  python eeg_preproc_deploy.py
  python eeg_preproc_deploy.py --subject sub-dmnelf009
  python eeg_preproc_deploy.py --subject sub-dmnelf009 --task rest --run 01
  python eeg_preproc_deploy.py --overwrite

  Output: derivatives/eeg_preprocessed/{subject}/ses-dmnelf/eeg/
  QC images: .../eeg/qc/{subject}_..._qc_raw.png and _qc_preproc.png
  Expected: 82 FIF files per sfreq (12 subjects x 7 runs minus 2 missing)

  Check:
  find /projects/swglab/data/DMNELF/derivatives/eeg_preprocessed
    -name "*_desc-preproc250Hz_eeg.fif" | wc -l


Step 0c — DiFuMo extraction

  python 00_extract_difumo.py
  Expected: 103 TSV files (105 minus 2 permanently missing raw data runs)

  Check:
  ls microstate_pda_v3/difumo_timeseries/*.tsv | wc -l


Step 0d — Personalized masks

  python 00b_extract_personal_masks.py
  Expected: 15 parcel_weights.json files (one per subject)

  Check:
  ls /projects/swglab/data/DMNELF/derivatives/network_masks/*/*_parcel_weights.json | wc -l


Step 0e — Add personal parcels to TSVs

  python 00c_add_personal_parcels.py
  Expected: all TSVs updated to 68 columns
  (volume + time + 64 ROIs + DMN_personal + CEN_personal)


Steps 1 through 5

  python 01_fit_microstates.py
  python 02_tess_features.py
  python 03_compute_pda.py
  python 04_train_decoder.py
  python 05_evaluate_decoder.py


Key paths

  Raw EEG (BVA preprocessed):  /projects/swglab/data/DMNELF/rawdata_eeg/
  EEG preprocessed FIF:        /projects/swglab/data/DMNELF/derivatives/eeg_preprocessed/
  fMRIPrep BOLD:               /projects/swglab/data/DMNELF/derivatives/fmriprep_25.2.5_fmap/
  DiFuMo timeseries:           microstate_pda_v3/difumo_timeseries/
  Personalized masks:          /projects/swglab/data/DMNELF/derivatives/network_masks/
  Microstate templates:        microstate_pda_v3/microstates/
  TESS features:               microstate_pda_v3/features/
  PDA targets:                 microstate_pda_v3/targets/
  Decoder models:              microstate_pda_v3/models/
  Results:                     microstate_pda_v3/results/
  QC images:                   .../eeg_preprocessed/{subject}/.../eeg/qc/
  SLURM logs:                  microstate_pda_v3/logs/


Subject lists

  Complete EEG+fMRI (10 subjects):
    sub-dmnelf001, 004, 005, 006, 007, 008, 009, 010, 011, 1001

  Partial EEG+fMRI — matched but fewer runs (2 subjects):
    sub-dmnelf1002 (missing rest-02)
    sub-dmnelf1003 (missing feedback-04)

  fMRI only — no EEG acquired, no R128 marker (3 subjects):
    sub-dmnelf002, sub-dmnelf003, sub-dmnelf999


Troubleshooting

  SCP fails with spaces in path: scp_to in utils.py wraps local path in
  single quotes. Verify the function is up to date from git.

  squeue format error: use --format=%.8i_%.8T_%.10M (no quotes, underscores
  as separators instead of spaces).

  ICLabel fails (no montage): eeg_preproc.py sets standard_1020 montage
  before ICA automatically. If it fails, check that channel names match
  the standard 10-20 system.

  CEN/DMN swapped in masks: check cen_skipped_pcc field in the subject's
  parcel_weights.json. The spatial constraint excludes components with
  centroid |x| < 15mm AND y < -45mm from CEN selection to prevent
  PCC/precuneus contamination.

  fMRIPrep BOLD not found: verify fMRIPrep completed for that subject.
  sacct -u cccbauer --format=JobID,State,ExitCode -S 2026-01-01 | grep fmriprep

