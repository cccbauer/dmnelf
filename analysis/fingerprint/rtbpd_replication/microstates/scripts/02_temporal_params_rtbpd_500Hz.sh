#!/bin/bash
#SBATCH --job-name=rtbpd_ms_temporal_500Hz
#SBATCH --output=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates/logs/rtbpd_ms_temporal_500Hz_%j.out
#SBATCH --error=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates/logs/rtbpd_ms_temporal_500Hz_%j.err
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=suewhit

/home/cccbauer/.conda/envs/eeg_preproc/bin/python /projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates/scripts/02_temporal_params_rtbpd_500Hz_cluster.py