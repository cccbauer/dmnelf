#!/bin/bash
#SBATCH --job-name=within_fair
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/logs/within_fair.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/logs/within_fair.err
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# Fair nested within-subject band-power re-run (OOF r + inner-CV alpha), matched to EFP v3.
BASE=/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling
PY=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
cd "$BASE/scripts"
mkdir -p "$BASE/logs"
$PY within_fair.py --cv_folds 5
echo "=== DONE ==="
