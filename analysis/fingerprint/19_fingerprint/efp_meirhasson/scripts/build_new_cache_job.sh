#!/bin/bash
#SBATCH --job-name=efp_cache_new
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/efp_meirhasson/logs/build_cache_%j.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/efp_meirhasson/logs/build_cache_%j.err

PYTHON=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
cd /projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/efp_meirhasson/scripts
${PYTHON} efp_features.py --subjects dmnelf002 dmnelf003 --cache /projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/efp_meirhasson/results/features_cache --config /projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/efp_meirhasson/config.yaml
