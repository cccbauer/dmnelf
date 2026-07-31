#!/bin/bash
#SBATCH --job-name=rtbpd_fit_band_k4_theta
#SBATCH --output=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates/band_k4/logs/rtbpd_fit_band_k4_theta_%j.out
#SBATCH --error=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates/band_k4/logs/rtbpd_fit_band_k4_theta_%j.err
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=suewhit

/home/cccbauer/.conda/envs/eeg_preproc/bin/python /projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates/band_k4/scripts/01_fit_microstates_band_k4_theta_k4_500Hz_cluster.py