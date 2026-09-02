#!/bin/bash
#SBATCH --job-name=pz_ablate
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled/results/logs/pz_%j.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled/results/logs/pz_%j.err
#SBATCH --partition=short
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
P=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
D=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled
mkdir -p $D/results/logs
cd $D/scripts
$P montage_pz_ablation.py --out $D/results/montage_pz_ablation_loso.csv
echo DONE
