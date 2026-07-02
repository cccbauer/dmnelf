#!/bin/bash
#SBATCH --job-name=hmm_group
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/dmn_hmm_detection/logs/hmm_group.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/dmn_hmm_detection/logs/hmm_group.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=short

PYTHON=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
SCRIPT_DIR=/projects/swglab/data/DMNELF/analysis/fingerprint/dmn_hmm_detection/scripts

${PYTHON} ${SCRIPT_DIR}/fit_hmm.py \
    --group \
    --n_epochs 20 \
    --out_name group_k12
