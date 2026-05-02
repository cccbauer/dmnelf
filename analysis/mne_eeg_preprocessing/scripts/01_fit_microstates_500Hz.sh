#!/bin/bash
#SBATCH --job-name=fit_ms_500Hz
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/logs/fit_ms_500Hz_%j.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/logs/fit_ms_500Hz_%j.err
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=suewhit

/home/cccbauer/.conda/envs/eeg_preproc/bin/python /projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/scripts/01_fit_microstates_500Hz_cluster.py