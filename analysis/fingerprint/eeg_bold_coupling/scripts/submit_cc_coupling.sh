#!/bin/bash
#SBATCH --job-name=cc_coupling
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/logs/cc_coupling.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/logs/cc_coupling.err
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Cross-cohort band-power double replication: train DMNELF group model, predict
# rtBPD nf1 then nf2. rtBPD band-power features are extracted+cached on first run.
BASE=/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling
PY=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
cd "$BASE/scripts"
mkdir -p "$BASE/logs"

echo "=== nf1 ==="
$PY cross_cohort_coupling.py --rtbpd-config "$BASE/config_rtbpd.yaml" --tag ""
echo "=== nf2 ==="
$PY cross_cohort_coupling.py --rtbpd-config "$BASE/config_rtbpd_nf2.yaml" --tag "_nf2"
echo "=== DONE ==="
