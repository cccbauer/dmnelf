#!/bin/bash
#SBATCH --job-name=cc_efp
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/cc_efp.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/cc_efp.err
#SBATCH --partition=short
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

# Cross-cohort EFP: train DMNELF general fingerprint, predict rtBPD. Pass the rtBPD
# config + cache dir; defaults target nf1. For nf2 pass config_rtbpd_nf2.yaml and
# --rtbpd-cache results/features_cache_rtbpd_nf2.
BASE=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson
CONFIG=${1:-$BASE/config_rtbpd.yaml}
CACHE=${2:-$BASE/results/features_cache_rtbpd}
cd $BASE/scripts
/home/cccbauer/.conda/envs/eeg_preproc/bin/python cross_cohort_efp.py \
  --rtbpd-config $CONFIG \
  --dmnelf-cache $BASE/results/features_cache \
  --rtbpd-cache $CACHE --res tr
