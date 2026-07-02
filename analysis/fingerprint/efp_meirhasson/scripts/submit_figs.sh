#!/bin/bash
#SBATCH --job-name=efp_figs
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/efp_figs.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/efp_figs.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=short

PYTHON=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
BASE=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson
cd $BASE/scripts
${PYTHON} paper_figures.py --res tr --targets PDA CEN GSR_CEN
