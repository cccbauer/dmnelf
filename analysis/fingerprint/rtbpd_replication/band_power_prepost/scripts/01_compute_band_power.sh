#!/bin/bash
#SBATCH --job-name=rtbpd_band_power
#SBATCH --output=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/band_power_prepost/logs/rtbpd_band_power_%j.out
#SBATCH --error=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/band_power_prepost/logs/rtbpd_band_power_%j.err
#SBATCH --partition=sharing
#SBATCH --time=00:55:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=suewhit

/home/cccbauer/.conda/envs/eeg_preproc/bin/python /projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/band_power_prepost/scripts/01_compute_band_power_cluster.py