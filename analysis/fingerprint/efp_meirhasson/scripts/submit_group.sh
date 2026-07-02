#!/bin/bash
#SBATCH --job-name=efp_group
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/efp_group.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/efp_group.err
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=short

PYTHON=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
BASE=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson
cd $BASE/scripts
${PYTHON} efp_group.py --outdir full --cache $BASE/results/features_cache
