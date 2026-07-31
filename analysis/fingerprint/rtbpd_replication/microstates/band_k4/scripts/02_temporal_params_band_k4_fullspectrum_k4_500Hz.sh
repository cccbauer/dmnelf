#!/bin/bash
#SBATCH --job-name=rtbpd_temporal_band_k4_fullspectrum
#SBATCH --output=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates/band_k4/logs/rtbpd_temporal_band_k4_fullspectrum_%j.out
#SBATCH --error=/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates/band_k4/logs/rtbpd_temporal_band_k4_fullspectrum_%j.err
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=suewhit

/home/cccbauer/.conda/envs/eeg_preproc/bin/python /projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates/band_k4/scripts/02_temporal_params_band_k4_fullspectrum_k4_500Hz_cluster.py