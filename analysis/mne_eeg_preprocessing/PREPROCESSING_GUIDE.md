# EEG Preprocessing Guide for sub-dmnelf012 & 013

## Overview
This guide walks through running EEG preprocessing locally, which automatically deploys to the cluster and processes at both 250Hz and 500Hz sampling rates.

---

## Prerequisites

### 1. Verify Local Setup
```bash
cd /Users/cccbauer/Documents/GitHub/dmnelf/analysis/preprocessing

# Verify config loads
python -c "import config; print('LOCAL_BASE:', config.LOCAL_BASE)"

# Verify SSH connection
ssh -o ConnectTimeout=5 cccbauer@explorer.northeastern.edu "echo 'SSH OK'"
```

### 2. Check Cluster Environment
Verify the EEG preprocessing environment exists on cluster:
```bash
ssh cccbauer@explorer.northeastern.edu \
  "/home/cccbauer/.conda/envs/eeg_preproc/bin/python --version"
```

Should output: `Python 3.x.x`

---

## EEG Preprocessing Steps

### Step 1: Verify Raw EEG Data on Cluster

```bash
ssh cccbauer@explorer.northeastern.edu << 'EOF'
echo "=== Checking sub-dmnelf012 raw data ==="
ls -lh /projects/swglab/data/DMNELF/rawdata_eeg/sub-dmnelf012/ses-dmnelf/eeg/ 2>/dev/null || echo "NOT FOUND"

echo "=== Checking sub-dmnelf013 raw data ==="
ls -lh /projects/swglab/data/DMNELF/rawdata_eeg/sub-dmnelf013/ses-dmnelf/eeg/ 2>/dev/null || echo "NOT FOUND"
EOF
```

Expected files: `sub-dmnelfXXX_ses-dmnelf_task-{rest,shortrest,feedback}_run-XX_desc-bvaAC1kHz_eeg.edf`

---

### Step 2: Deploy & Run Preprocessing (LOCAL)

Run from your local machine. This will deploy scripts to cluster and submit SLURM jobs.

#### Option A: Process individual subjects

```bash
cd /Users/cccbauer/Documents/GitHub/dmnelf/analysis/preprocessing

# Deploy for sub-dmnelf012
python deploy_scripts/eeg_preproc_deploy.py --subject sub-dmnelf012

# Deploy for sub-dmnelf013
python deploy_scripts/eeg_preproc_deploy.py --subject sub-dmnelf013
```

#### Option B: Process both at once (all configured subjects)

```bash
cd /Users/cccbauer/Documents/GitHub/dmnelf/analysis/preprocessing

python deploy_scripts/eeg_preproc_deploy.py
```

**What happens:**
- Deploys `deploy_scripts/eeg_preproc.py` to cluster
- Creates SBATCH scripts for 250Hz and 500Hz variants
- Submits 2 jobs per subject (4 jobs total if doing both subjects)
- Returns SLURM job IDs

---

### Step 3: Monitor Jobs on Cluster

```bash
# Check job queue
ssh cccbauer@explorer.northeastern.edu 'squeue -u cccbauer --format="%.8i %.20j %.8T %.10M"'

# Watch logs in real-time
ssh cccbauer@explorer.northeastern.edu \
  'tail -f /projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/logs/eeg_preproc_*.out'

# Check specific job
ssh cccbauer@explorer.northeastern.edu 'sstat --jobs=JOBID --format=JobID,MaxVMSize,MaxRSS'
```

---

### Step 4: Verify Output

Once jobs complete, check outputs:

```bash
# Check sub-dmnelf012 preprocessed files
ssh cccbauer@explorer.northeastern.edu << 'EOF'
echo "=== sub-dmnelf012 preprocessed outputs ==="
ls -lh /projects/swglab/data/DMNELF/derivatives/eeg_preprocessed/sub-dmnelf012/ses-dmnelf/eeg/*.fif 2>/dev/null || echo "No FIF files yet"

# Check for 250Hz
ls -lh /projects/swglab/data/DMNELF/derivatives/eeg_preprocessed/sub-dmnelf012/ses-dmnelf/eeg/*preproc250Hz*.fif 2>/dev/null

# Check for 500Hz
ls -lh /projects/swglab/data/DMNELF/derivatives/eeg_preprocessed/sub-dmnelf012/ses-dmnelf/eeg/*preproc500Hz*.fif 2>/dev/null

echo ""
echo "=== sub-dmnelf013 preprocessed outputs ==="
ls -lh /projects/swglab/data/DMNELF/derivatives/eeg_preprocessed/sub-dmnelf013/ses-dmnelf/eeg/*.fif 2>/dev/null || echo "No FIF files yet"

# Check for 250Hz
ls -lh /projects/swglab/data/DMNELF/derivatives/eeg_preprocessed/sub-dmnelf013/ses-dmnelf/eeg/*preproc250Hz*.fif 2>/dev/null

# Check for 500Hz
ls -lh /projects/swglab/data/DMNELF/derivatives/eeg_preprocessed/sub-dmnelf013/ses-dmnelf/eeg/*preproc500Hz*.fif 2>/dev/null

echo ""
echo "=== Quality Control Images ==="
ls -lh /projects/swglab/data/DMNELF/derivatives/eeg_preprocessed/sub-dmnelf012/ses-dmnelf/eeg/qc/ 2>/dev/null
EOF
```

---

## Output Locations

### Preprocessed EEG Data (FIF format)
```
/projects/swglab/data/DMNELF/derivatives/eeg_preprocessed/
├── sub-dmnelf012/ses-dmnelf/eeg/
│   ├── *_desc-preproc250Hz_eeg.fif    (250 Hz)
│   ├── *_desc-preproc500Hz_eeg.fif    (500 Hz)
│   └── qc/                             (Quality control images)
└── sub-dmnelf013/ses-dmnelf/eeg/
    ├── *_desc-preproc250Hz_eeg.fif
    ├── *_desc-preproc500Hz_eeg.fif
    └── qc/
```

### Preprocessing Pipeline Steps
1. Load minimally preprocessed EDF (BVA gradient correction, 1kHz)
2. Detect ECG channel, compute R-peaks (NeuroKit2)
3. Auto-detect bad channels (variance + HF noise z-score)
4. Annotate noisy edges (scanner ramp artifact)
5. Bandpass filter (1-40 Hz)
6. BCG correction using MNE create_ecg_epochs
7. Downsample to target sampling rate (250Hz or 500Hz)
8. ICA (29 components, ICLabel + cardiac + EOG correlation)
9. Interpolate bad channels
10. Average reference
11. Save FIF + QC images

---

## Troubleshooting

### Issue: "SSH connection refused"
```bash
# Verify SSH key
ssh-keygen -t ed25519 -C "dmnelf_pipeline"
ssh-copy-id cccbauer@explorer.northeastern.edu
```

### Issue: "Python environment not found on cluster"
```bash
# Check available environments
ssh cccbauer@explorer.northeastern.edu \
  'conda env list | grep eeg_preproc'

# If missing, ask admin to create or create manually:
ssh cccbauer@explorer.northeastern.edu \
  'conda create -n eeg_preproc python=3.10 mne neurokit2 scipy numpy pandas matplotlib -y'
```

### Issue: "Raw EEG data not found"
Verify files are in correct location:
```bash
ssh cccbauer@explorer.northeastern.edu \
  'ls -R /projects/swglab/data/DMNELF/rawdata_eeg/sub-dmnelf01*/ses-dmnelf/eeg/ | head -30'
```

Should show `.edf` files with naming: `sub-dmnelfXXX_ses-dmnelf_task-{rest,shortrest,feedback}_run-XX_desc-bvaAC1kHz_eeg.edf`

### Issue: Jobs fail with memory error
Increase memory in SLURM config. Edit `deploy_scripts/eeg_preproc_deploy.py`:
```python
# Line ~30: increase --mem from 32G to 64G
"#SBATCH --mem=64G",
```

### Issue: Check job error logs
```bash
ssh cccbauer@explorer.northeastern.edu \
  'cat /projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/logs/eeg_preproc_250Hz_*.err | head -50'
```

---

## Configuration Reference

**Local config:** [`config.py`](config.py)
- `LOCAL_BASE`: Local Dropbox path for scripts
- `CLUSTER_BASE`: Cluster path for deployed scripts
- `EEG_ROOT`: Output root for preprocessed data
- `SUBJECTS`: List of subjects (includes sub-dmnelf012, sub-dmnelf013)

**Deployment script:** [`deploy_scripts/eeg_preproc_deploy.py`](deploy_scripts/eeg_preproc_deploy.py)
- Handles SSH/SCP to cluster
- Generates SBATCH scripts
- Submits jobs in parallel (250Hz + 500Hz)

**Preprocessing script:** [`deploy_scripts/eeg_preproc.py`](deploy_scripts/eeg_preproc.py)
- Core 11-step preprocessing pipeline
- Accepts `--subject`, `--task`, `--run`, `--sfreq`, `--all`, `--overwrite`

---

## Example: Full Workflow

```bash
# 1. Navigate to preprocessing directory
cd /Users/cccbauer/Documents/GitHub/dmnelf/analysis/preprocessing

# 2. Verify setup
python -c "import config; print('OK')"
ssh cccbauer@explorer.northeastern.edu "echo 'OK'"

# 3. Deploy preprocessing for sub-dmnelf012
echo "Deploying sub-dmnelf012..."
python deploy_scripts/eeg_preproc_deploy.py --subject sub-dmnelf012

# 4. Deploy preprocessing for sub-dmnelf013
echo "Deploying sub-dmnelf013..."
python deploy_scripts/eeg_preproc_deploy.py --subject sub-dmnelf013

# 5. Monitor on cluster
sleep 5
ssh cccbauer@explorer.northeastern.edu 'squeue -u cccbauer'

# 6. Wait for completion, then check outputs
# (Can take 30-60 min per subject)
ssh cccbauer@explorer.northeastern.edu \
  'ls /projects/swglab/data/DMNELF/derivatives/eeg_preprocessed/sub-dmnelf012/ses-dmnelf/eeg/'
```

---

## Notes

- Each subject gets 2 jobs (250Hz and 500Hz) submitted in parallel
- Processing time: ~30-60 min per subject per sampling rate
- Output format: MNE FIF files (can be opened with MNE-Python)
- QC images generated in `qc/` subdirectory
- To reprocess, use `--overwrite` flag in deployment script
