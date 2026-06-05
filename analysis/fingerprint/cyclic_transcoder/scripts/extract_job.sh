#!/bin/bash
#SBATCH --job-name=cyclic_extract
#SBATCH --account=suewhit
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-29
#SBATCH --output=logs/extract_%A_%a.out
#SBATCH --error=logs/extract_%A_%a.err

# Feature extraction — no GPU needed.
# The subject list is derived from config.yaml (data.subjects.all) at runtime,
# so adding subjects there requires NO edits here. The --array upper bound is
# intentionally generous; indices past the end of the list exit as no-ops.

BASE=/projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder
CONFIG=$BASE/config.yaml
# FSL prepends its python to PATH, so call the env python by absolute path.
FP_PY=/home/cccbauer/.conda/envs/fingerprint/bin/python

cd "$BASE"

mapfile -t SUBJECTS < <("$FP_PY" -c "import yaml; print('\n'.join(yaml.safe_load(open('$CONFIG'))['data']['subjects']['all']))")

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SUBJECTS[@]}" ]; then
    echo "Array index $SLURM_ARRAY_TASK_ID >= ${#SUBJECTS[@]} subjects — nothing to do."
    exit 0
fi

SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
echo "Extracting subject: $SUBJECT  (index $SLURM_ARRAY_TASK_ID of ${#SUBJECTS[@]})"
echo "Config: $CONFIG"

source /shared/EL9/explorer/anaconda3/2024.06-root/etc/profile.d/conda.sh
conda activate fingerprint

"$FP_PY" -c "import py_compile; py_compile.compile('data/extract_features.py')"

"$FP_PY" data/extract_features.py \
    --subject "$SUBJECT" \
    --config "$CONFIG"

echo "Done: $SUBJECT"
