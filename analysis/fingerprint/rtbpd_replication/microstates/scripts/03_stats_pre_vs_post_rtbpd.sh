#!/bin/bash
#SBATCH --job-name=rtbpd_ms_stats
#SBATCH --output=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates/logs/rtbpd_ms_stats_%j.out
#SBATCH --error=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates/logs/rtbpd_ms_stats_%j.err
#SBATCH --partition=short
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=suewhit

/home/cccbauer/.conda/envs/eeg_preproc/bin/python /projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates/scripts/03_stats_pre_vs_post_rtbpd_cluster.py