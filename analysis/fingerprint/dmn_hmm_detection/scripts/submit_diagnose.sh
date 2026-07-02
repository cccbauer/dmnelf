#!/bin/bash
#SBATCH --job-name=hmm_diag
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/dmn_hmm_detection/logs/hmm_diag.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/dmn_hmm_detection/logs/hmm_diag.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=short

PYTHON=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
SCRIPT_DIR=/projects/swglab/data/DMNELF/analysis/fingerprint/dmn_hmm_detection/scripts
${PYTHON} ${SCRIPT_DIR}/diagnose_hmm.py
