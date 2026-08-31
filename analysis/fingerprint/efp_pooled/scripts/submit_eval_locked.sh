#!/bin/bash
#SBATCH --job-name=efp_locked
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled/results/logs/locked_%j.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled/results/logs/locked_%j.err
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#
# Score three models on the LOCKED external test set (cohort_split.json), which no arm has seen.
# Policy: each arm is scored exactly once. Do not iterate against these numbers.
#   nf1-only locked subjects: 004 012 013 018 020 021 024 034 038 040   (cohort rtbpd)
#   nf2-only locked subjects: 027 028                                    (cohort rtbpd_nf2)

set -euo pipefail
P=/home/cccbauer/.conda/envs/eeg_preproc/bin/python
D=/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled
R=$D/results
mkdir -p $R/logs
cd $D/scripts

LOCKED_NF1=rtbpd004,rtbpd012,rtbpd013,rtbpd018,rtbpd020,rtbpd021,rtbpd024,rtbpd034,rtbpd038,rtbpd040
LOCKED_NF2=rtbpd027,rtbpd028

score () {   # $1 = label, $2 = model path
  echo ""
  echo "############################################################"
  echo "### MODEL: $1"
  echo "############################################################"
  $P eval_holdout.py --model "$2" --cohort rtbpd     --subs $LOCKED_NF1 \
     --label "LOCKED nf1-only | $1" --out $R/locked_nf1_$1.csv
  $P eval_holdout.py --model "$2" --cohort rtbpd_nf2 --subs $LOCKED_NF2 \
     --label "LOCKED nf2-only | $1" --out $R/locked_nf2_$1.csv
}

score shipped        /projects/swglab/data/DMNELF/analysis/fingerprint/efp_epoc/efp_epoc_model.npz
score pooled28       $D/models/pooled_epoc12_clean_ridge.npz
score dmnelfonly19   $D/models/dmnelfonly_epoc12_clean_ridge.npz

echo ""
echo "DONE"
