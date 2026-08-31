#!/bin/bash
#SBATCH --job-name=cyclic_train
#SBATCH --account=suewhit
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-29
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/cyclic_transcoder_diag17/logs/train_%A_%a.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/cyclic_transcoder_diag17/logs/train_%A_%a.err

# LOOCV training — one job per left-out subject.
# Depends on extract_job completing first:
#   sbatch --dependency=afterok:<extract_jobid> scripts/train_job.sh
#
# The left-out subject list is derived from config.yaml (data.subjects.all) at
# runtime, so adding subjects there requires NO edits here. The --array upper
# bound is generous; indices past the end of the list exit as no-ops.
#
# Optional reproducibility seed: sbatch --export=ALL,SEED=1 scripts/train_job.sh

BASE=/projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/cyclic_transcoder_diag17
CONFIG=$BASE/config.yaml
# FSL prepends its python to PATH, so call the env python by absolute path.
FP_PY=/home/cccbauer/.conda/envs/fingerprint/bin/python

cd "$BASE"
mkdir -p logs

mapfile -t SUBJECTS < <("$FP_PY" -c "import yaml; print('\n'.join(yaml.safe_load(open('$CONFIG'))['data']['subjects']['all']))")

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SUBJECTS[@]}" ]; then
    echo "Array index $SLURM_ARRAY_TASK_ID >= ${#SUBJECTS[@]} subjects — nothing to do."
    exit 0
fi

LEFT_OUT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
echo "Training LOOCV, left-out: $LEFT_OUT  (index $SLURM_ARRAY_TASK_ID of ${#SUBJECTS[@]})"
# Report accelerator if present; prior runs trained on CPU (partition=short).
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "No GPU allocated — training on CPU (partition=short, as in prior runs)."
fi

source /shared/EL9/explorer/anaconda3/2024.06-root/etc/profile.d/conda.sh
conda activate fingerprint

"$FP_PY" -c "import py_compile; py_compile.compile('train.py')"

SEED_ARGS=()
if [ -n "${SEED:-}" ]; then
    echo "Seed: $SEED"
    SEED_ARGS=(--seed "$SEED")
fi

"$FP_PY" train.py \
    --left-out "$LEFT_OUT" \
    --config "$CONFIG" \
    "${SEED_ARGS[@]}"

echo "Done: $LEFT_OUT"
