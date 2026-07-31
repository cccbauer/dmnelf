#!/bin/bash
#SBATCH --job-name=rtbpd_stats_bandpower
#SBATCH --output=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/band_power_prepost/logs/rtbpd_stats_bandpower_%j.out
#SBATCH --error=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/band_power_prepost/logs/rtbpd_stats_bandpower_%j.err
#SBATCH --partition=sharing
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=suewhit

/home/cccbauer/.conda/envs/eeg_preproc/bin/python /projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/band_power_prepost/scripts/02_stats_bandpower_cluster.py