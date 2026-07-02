#!/bin/bash
#SBATCH --job-name=hmm_spec
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/dmn_hmm_detection/logs/hmm_spec.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/dmn_hmm_detection/logs/hmm_spec.err
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=short

PYTHON=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
SCRIPT_DIR=/projects/swglab/data/DMNELF/analysis/fingerprint/dmn_hmm_detection/scripts
${PYTHON} ${SCRIPT_DIR}/compute_state_spectra.py --model group_k12 --dmn_state 7
