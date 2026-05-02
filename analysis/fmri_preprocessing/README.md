# fMRI Preprocessing

Unified fMRI preprocessing pipeline for DMNELF neuroimaging data on HPC clusters.

## Structure

```
fmri_preprocessing/
├── deploy_scripts/
│   ├── fmri_preproc_deploy.py     # Deploy jobs to cluster via SSH
│   └── [future analysis scripts]
├── scripts/
│   └── fmri_preproc.sh            # SBATCH script for cluster execution
├── config.py                       # Cluster paths, credentials, subjects
├── utils.py                        # SSH/SCP utilities
├── environment.yml                 # Conda dependencies
└── setup_local_environment.sh      # One-command local setup
```

## Quick Start

### 1. Set up local environment

```bash
cd /Users/cccbauer/Documents/GitHub/dmnelf/analysis/fmri_preprocessing
chmod +x setup_local_environment.sh
bash setup_local_environment.sh
```

### 2. Deploy preprocessing to cluster

```bash
conda activate fmri_preproc

# Deploy all subjects
python deploy_scripts/fmri_preproc_deploy.py --all

# Deploy specific subject
python deploy_scripts/fmri_preproc_deploy.py --subject sub-dmnelf013
```

### 3. Monitor jobs on cluster

```bash
# Check queue
ssh cccbauer@explorer.northeastern.edu "squeue -u cccbauer"

# View job output
ssh cccbauer@explorer.northeastern.edu "tail -100 /projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/logs/fmri_preproc_*.out"
```

## Pipeline Details

See `FMRI_PREPROCESSING.md` for full pipeline documentation including:
- Data flow (fMRI input → derivatives)
- Processing steps (DiFuMo, Microstate, TESS, PDA)
- Output locations
- Troubleshooting

## Configuration

Edit `config.py` to customize:
- `CLUSTER_SSH` - SSH connection string
- `CLUSTER_BASE` - Base path on cluster
- `SUBJECTS` - List of subject IDs to process
- `SLURM_*` - Job submission parameters

## Parallel Structure

Works in parallel with `mne_eeg_preprocessing/`:
- **EEG**: Raw→FIF files, ICA cleanup, 250/500 Hz variants
- **fMRI**: Parcellation→Microstates→Features→Distinctiveness

Both share:
- Same conda environments
- SSH utilities for cluster access
- Configuration management via config.py
