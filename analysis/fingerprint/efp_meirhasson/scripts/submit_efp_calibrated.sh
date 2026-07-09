#!/bin/bash
#SBATCH --job-name=efp_cal
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/efp_cal.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/logs/efp_cal.err
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# Pseudo-target calibration of the general EFP fingerprint.
# LOSO within DMNELF (primary) + cross-cohort rtBPD nf1/nf2. Features already cached.
BASE=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson
PY=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
cd "$BASE/scripts"
mkdir -p "$BASE/logs"

echo "=== LOSO-DMNELF ==="
$PY efp_calibrated.py --mode loso
echo "=== cross-cohort nf1 ==="
$PY efp_calibrated.py --mode cross --rtbpd-config "$BASE/config_rtbpd.yaml" \
    --rtbpd-cache "$BASE/results/features_cache_rtbpd" --tag ""
echo "=== cross-cohort nf2 ==="
$PY efp_calibrated.py --mode cross --rtbpd-config "$BASE/config_rtbpd_nf2.yaml" \
    --rtbpd-cache "$BASE/results/features_cache_rtbpd_nf2" --tag _nf2
echo "=== LOSO-DMNELF frontal ==="
$PY efp_calibrated.py --mode loso --frontal
echo "=== cross-cohort nf1 frontal-multi ==="
$PY efp_calibrated.py --mode cross --frontal-multi --rtbpd-config "$BASE/config_rtbpd.yaml" \
    --rtbpd-cache "$BASE/results/features_cache_rtbpd" --tag ""
echo "=== cross-cohort nf2 frontal-multi ==="
$PY efp_calibrated.py --mode cross --frontal-multi --rtbpd-config "$BASE/config_rtbpd_nf2.yaml" \
    --rtbpd-cache "$BASE/results/features_cache_rtbpd_nf2" --tag _nf2
echo "=== DONE ==="
