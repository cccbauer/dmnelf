#!/bin/bash
#SBATCH --job-name=cyclic_train
#SBATCH --account=suewhit
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-13
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

# LOOCV training — one GPU job per left-out subject
# Depends on extract_job completing first:
#   sbatch --dependency=afterok:<extract_jobid> slurm/train_job.sh

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
)

LEFT_OUT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
CONFIG=/projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder/config.yaml

echo "Training LOOCV, left-out: $LEFT_OUT"
echo "GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source /shared/EL9/explorer/anaconda3/2024.06-root/etc/profile.d/conda.sh
conda activate fingerprint

cd /projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder

mkdir -p logs

python -c "import py_compile; py_compile.compile('train.py')"

python train.py \
    --left-out "$LEFT_OUT" \
    --config "$CONFIG"

echo "Done: $LEFT_OUT"