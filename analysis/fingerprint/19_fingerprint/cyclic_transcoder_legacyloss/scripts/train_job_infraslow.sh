#!/bin/bash
#SBATCH --job-name=cyc_train_is
#SBATCH --account=suewhit
#SBATCH --partition=sharing
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-15
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder/logs/train_is_%A_%a.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder/logs/train_is_%A_%a.err

BASE=/projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder
CONFIG=$BASE/config_infraslow.yaml
FP_PY=/home/cccbauer/.conda/envs/fingerprint/bin/python
cd "$BASE"; mkdir -p logs
mapfile -t SUBJECTS < <("$FP_PY" -c "import yaml; print('\n'.join(yaml.safe_load(open('$CONFIG'))['data']['subjects']['all']))")
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SUBJECTS[@]}" ]; then echo "idx past end"; exit 0; fi
LEFT_OUT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
echo "INFRASLOW LOOCV, left-out: $LEFT_OUT (idx $SLURM_ARRAY_TASK_ID)"
source /shared/EL9/explorer/anaconda3/2024.06-root/etc/profile.d/conda.sh
conda activate fingerprint
"$FP_PY" train.py --left-out "$LEFT_OUT" --config "$CONFIG"
echo "Done: $LEFT_OUT"
