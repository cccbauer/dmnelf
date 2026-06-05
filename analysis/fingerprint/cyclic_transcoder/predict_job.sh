#!/bin/bash
#SBATCH --job-name=cyclic_predict
#SBATCH --account=suewhit
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/predict_%A_%a.out
#SBATCH --error=logs/predict_%A_%a.err

SUBJECTS=(
    dmnelf001
    dmnelf004
    dmnelf005
    dmnelf006
    dmnelf007
    dmnelf008
    dmnelf009
    dmnelf010
    dmnelf011
    dmnelf012
    dmnelf013
    dmnelf1001
    dmnelf1002
    dmnelf1003
    dmnelf014
    dmnelf015
)

SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
CONFIG=/projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder/config.yaml

echo "Predicting PDA for: $SUBJECT"

source /shared/EL9/explorer/anaconda3/2024.06-root/etc/profile.d/conda.sh
conda activate fingerprint

cd /projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder

python predict_pda.py \
    --subject "$SUBJECT" \
    --task feedback \
    --config "$CONFIG"

echo "Done: $SUBJECT"
