#!/bin/bash
#SBATCH --job-name=hmm_val
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/dmn_hmm_detection/logs/hmm_val.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/dmn_hmm_detection/logs/hmm_val.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=short

PYTHON=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
SCRIPT_DIR=/projects/swglab/data/DMNELF/analysis/fingerprint/dmn_hmm_detection/scripts
${PYTHON} ${SCRIPT_DIR}/validate_hmm_dmn.py --model group_k12 --dmn_state 7
