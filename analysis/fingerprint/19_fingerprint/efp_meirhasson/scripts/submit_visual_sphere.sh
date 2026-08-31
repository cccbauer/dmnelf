#!/bin/bash
#SBATCH --job-name=efp_vissphere
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/vissphere.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/vissphere.err
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=short

PYTHON=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
BASE=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson
cd $BASE/scripts
${PYTHON} extract_visual_sphere.py --group
