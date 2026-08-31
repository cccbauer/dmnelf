#!/bin/bash
#SBATCH --job-name=efp_panel
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/panel.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/panel.err
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
cd /projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/scripts
/home/cccbauer/.conda/envs/eeg_preproc/bin/python same_electrode_panel.py --res tr
