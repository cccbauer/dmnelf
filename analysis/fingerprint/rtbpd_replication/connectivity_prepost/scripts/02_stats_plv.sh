#!/bin/bash
#SBATCH --job-name=rtbpd_stats_plv
#SBATCH --output=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/connectivity_prepost/logs/rtbpd_stats_plv_%j.out
#SBATCH --error=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/connectivity_prepost/logs/rtbpd_stats_plv_%j.err
#SBATCH --partition=sharing
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=suewhit

/home/cccbauer/.conda/envs/eeg_preproc/bin/python /projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/connectivity_prepost/scripts/02_stats_plv_cluster.py