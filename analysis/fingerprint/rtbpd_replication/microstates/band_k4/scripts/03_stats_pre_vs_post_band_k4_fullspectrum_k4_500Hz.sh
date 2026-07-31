#!/bin/bash
#SBATCH --job-name=rtbpd_stats_band_k4_fullspectrum
#SBATCH --output=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates/band_k4/logs/rtbpd_stats_band_k4_fullspectrum_%j.out
#SBATCH --error=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates/band_k4/logs/rtbpd_stats_band_k4_fullspectrum_%j.err
#SBATCH --partition=short
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=suewhit

/home/cccbauer/.conda/envs/eeg_preproc/bin/python /projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates/band_k4/scripts/03_stats_pre_vs_post_band_k4_fullspectrum_k4_500Hz_cluster.py