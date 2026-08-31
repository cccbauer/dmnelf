#!/bin/bash
#SBATCH --job-name=cyclic_predict
#SBATCH --account=suewhit
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-18
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/cyclic_transcoder_legacyloss/logs/predict_%A_%a.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/cyclic_transcoder_legacyloss/logs/predict_%A_%a.err

BASE=/projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/cyclic_transcoder_legacyloss
CONFIG=$BASE/config.yaml
FP_PY=/home/cccbauer/.conda/envs/fingerprint/bin/python

mapfile -t SUBJECTS < <("$FP_PY" -c "import yaml; print('\n'.join(yaml.safe_load(open('$CONFIG'))['data']['subjects']['all']))")

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SUBJECTS[@]}" ]; then
    echo "Array index $SLURM_ARRAY_TASK_ID >= ${#SUBJECTS[@]} subjects — nothing to do."
    exit 0
fi

SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

echo "Predicting PDA for: $SUBJECT"

source /shared/EL9/explorer/anaconda3/2024.06-root/etc/profile.d/conda.sh
conda activate fingerprint

cd "$BASE"

python predict_pda.py \
    --subject "$SUBJECT" \
    --task feedback \
    --config "$CONFIG"

echo "Done: $SUBJECT"
