#!/bin/bash
#SBATCH --job-name=fit_microstates
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/logs/fit_microstates_%j.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/logs/fit_microstates_%j.err
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=suewhit

$HOME/my_anaconda/bin/python /projects/swglab/data/DMNELF/analysis/MNE/jupyter/microstate_pda_v3/scripts/01_fit_microstates_cluster.py