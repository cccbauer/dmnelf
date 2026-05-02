#!/bin/bash
#SBATCH --job-name=personal_masks
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/logs/personal_masks_%j.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/logs/personal_masks_%j.err
#SBATCH --partition=sharing
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=suewhit

/home/cccbauer/.conda/envs/eeg_preproc/bin/python /projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/scripts/00b_extract_personal_masks_cluster.py