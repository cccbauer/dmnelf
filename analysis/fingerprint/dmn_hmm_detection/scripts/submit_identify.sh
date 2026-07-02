#!/bin/bash
#SBATCH --job-name=hmm_dmnid
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/dmn_hmm_detection/logs/hmm_dmnid.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/dmn_hmm_detection/logs/hmm_dmnid.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=short

PYTHON=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
SCRIPT_DIR=/projects/swglab/data/DMNELF/analysis/fingerprint/dmn_hmm_detection/scripts
${PYTHON} ${SCRIPT_DIR}/identify_dmn_state.py --model group_k12
