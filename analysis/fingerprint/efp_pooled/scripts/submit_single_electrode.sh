#!/bin/bash
#SBATCH --job-name=single_el
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled/results/logs/single_%j.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled/results/logs/single_%j.err
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# Controlled test of the alternative explanation for the PDA failure: model COMPLEXITY, not
# electrode coverage. Same harness, same nested-CV LOSO, but ONE electrode at a time (110 features)
# instead of a multivariate montage (1320-3410). If single-electrode PDA recovers here, the
# portable decoder's failure is dimensionality, not missing channels.
set -euo pipefail
P=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
D=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled
cd $D/scripts
$P montage_pz_ablation.py --single --out $D/results/single_electrode_loso.csv
echo DONE
