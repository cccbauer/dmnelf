# fMRI Preprocessing Pipeline

This document describes the fMRI preprocessing pipeline for DMNELF data.

## Overview

The fMRI preprocessing pipeline includes:
1. **DiFuMo-64 extraction** — Extract parcel timeseries from fMRIPrep BOLD
2. **Microstate fitting** — Fit microstate maps to the data
3. **TESS features** — Compute tessellation-based features
4. **PDA** — Pattern Distinctiveness Analysis

## Quick Start

### From Local Machine

Deploy and run fMRI preprocessing for sub-dmnelf012:

```bash
cd /Users/cccbauer/Documents/GitHub/dmnelf/analysis/preprocessing

# Activate environment
conda activate eeg_preproc

# Deploy for single subject
python deploy_scripts/fmri_preproc_deploy.py --subject sub-dmnelf012

# Or deploy for all configured subjects
python deploy_scripts/fmri_preproc_deploy.py --all

# With overwrite flag
python deploy_scripts/fmri_preproc_deploy.py --subject sub-dmnelf012 --overwrite
```

### On Cluster (Direct Submission)

If you prefer to run directly on the cluster:

```bash
ssh cccbauer@explorer.northeastern.edu

cd /projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3

# Make script executable
chmod +x scripts/fmri_preproc.sh

# Submit to SLURM
sbatch scripts/fmri_preproc.sh
```

## Pipeline Steps

### Step 0: DiFuMo-64 Extraction
**Script:** `deploy_scripts/00_extract_difumo.py`

Extracts parcel timeseries from fMRIPrep preprocessed BOLD using DiFuMo-64 probabilistic parcels.

- **Input:** fMRIPrep BOLD (MNI152NLin6Asym res-2)
- **Output:** TSV files with 64 parcel timeseries
- **Time:** ~5-10 min per subject

```bash
python deploy_scripts/00_extract_difumo.py --subject sub-dmnelf012 --all
```

### Step 1: Microstate Fitting
**Script:** `deploy_scripts/01_fit_microstates.py`

Fits microstate maps to the extracted timeseries.

- **Input:** DiFuMo-64 timeseries
- **Output:** Microstate maps and segmentation
- **Time:** ~10-15 min per subject

```bash
python deploy_scripts/01_fit_microstates.py --subject sub-dmnelf012 --all
```

### Step 2: TESS Features
**Script:** `deploy_scripts/02_tess_features.py`

Computes tessellation-based features from microstates.

- **Input:** Microstate segmentation
- **Output:** TESS features (TSV)
- **Time:** ~5 min per subject

```bash
python deploy_scripts/02_tess_features.py --subject sub-dmnelf012 --all
```

### Step 3: Pattern Distinctiveness Analysis
**Script:** `deploy_scripts/03_compute_pda.py`

Computes Pattern Distinctiveness Analysis scores.

- **Input:** DiFuMo timeseries + microstate segmentation
- **Output:** PDA scores (TSV)
- **Time:** ~10 min per subject

```bash
python deploy_scripts/03_compute_pda.py --subject sub-dmnelf012 --all
```

## Output Locations

All outputs are stored on cluster:

```
/projects/swglab/data/DMNELF/analysis/MNE/jupyter/neurobolt/
├── difumo_timeseries/
│   └── {subject}_ses-dmnelf_task-{task}_run-{run}_desc-difumo64_timeseries.tsv
├── microstate_maps/
│   └── {subject}_microstate_maps.nii.gz
├── microstate_segmentation/
│   └── {subject}_microstate_segmentation.tsv
├── tess_features/
│   └── {subject}_tess_features.tsv
└── pda_features/
    └── {subject}_pda_scores.tsv
```

## Monitoring Jobs

### Check job status
```bash
ssh cccbauer@explorer.northeastern.edu 'squeue -u cccbauer --format="%.8i %.20j %.8T %.10M"'
```

### Watch output in real-time
```bash
ssh cccbauer@explorer.northeastern.edu \
  'tail -f /projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/logs/fmri_pipeline_*.out'
```

### Check specific job details
```bash
ssh cccbauer@explorer.northeastern.edu 'sstat --jobs=JOBID --format=JobID,MaxVMSize,MaxRSS'
```

## Command Reference

### Run individual steps

```bash
# DiFuMo extraction only
python deploy_scripts/00_extract_difumo.py --subject sub-dmnelf012 --all

# Microstate fitting for specific task
python deploy_scripts/01_fit_microstates.py --subject sub-dmnelf012 --task rest

# All steps with overwrite
python deploy_scripts/00_extract_difumo.py --subject sub-dmnelf012 --all --overwrite
python deploy_scripts/01_fit_microstates.py --subject sub-dmnelf012 --all --overwrite
python deploy_scripts/02_tess_features.py --subject sub-dmnelf012 --all --overwrite
python deploy_scripts/03_compute_pda.py --subject sub-dmnelf012 --all --overwrite
```

### Run for multiple subjects

```bash
# In a loop
for subj in sub-dmnelf012 sub-dmnelf013; do
    python deploy_scripts/fmri_preproc_deploy.py --subject $subj
done
```

## Configuration

**Config file:** `config.py`

Key settings:
- `CLUSTER_BASE`: Root path for cluster scripts
- `FMRIPREP_ROOT`: Location of fMRIPrep outputs
- `DIFUMO_CACHE`: Cache for DiFuMo atlases
- `SUBJECTS`: List of subjects to process

## Troubleshooting

### Issue: "fMRIPrep BOLD not found"
```bash
# Check fMRIPrep outputs exist
ssh cccbauer@explorer.northeastern.edu \
  'ls /projects/swglab/data/DMNELF/derivatives/fmriprep_25.2.5_fmap/sub-dmnelf012/'
```

### Issue: "Out of memory" error
Increase SLURM memory in SBATCH header:
```bash
# In fmri_preproc.sh
#SBATCH --mem=128G  # Increase from 64G
```

### Issue: "Module not found"
Check cluster Python environment:
```bash
ssh cccbauer@explorer.northeastern.edu \
  '/home/cccbauer/.conda/envs/eeg_preproc/bin/python -c "import nilearn; print(nilearn.__version__)"'
```

## Typical Workflow

```bash
# Step 1: Deploy EEG preprocessing (if needed)
cd /Users/cccbauer/Documents/GitHub/dmnelf/analysis/preprocessing
conda activate eeg_preproc
python deploy_scripts/eeg_preproc_deploy.py --subject sub-dmnelf012

# Step 2: Deploy fMRI preprocessing
python deploy_scripts/fmri_preproc_deploy.py --subject sub-dmnelf012

# Step 3: Monitor both on cluster
ssh cccbauer@explorer.northeastern.edu 'squeue -u cccbauer'

# Step 4: Check outputs when complete
ssh cccbauer@explorer.northeastern.edu \
  'ls /projects/swglab/data/DMNELF/derivatives/eeg_preprocessed/sub-dmnelf012/ses-dmnelf/eeg/*.fif'
ssh cccbauer@explorer.northeastern.edu \
  'ls /projects/swglab/data/DMNELF/analysis/MNE/jupyter/neurobolt/difumo_timeseries/ | grep sub-dmnelf012'
```

## References

- **DiFuMo atlas:** Schaefer et al., NeuroImage (2018)
- **Microstate analysis:** Koenig & Lehmann (1997)
- **Pattern Distinctiveness:** Kayhan et al. (2019)
