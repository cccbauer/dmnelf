#!/bin/bash
#SBATCH --job-name=cyclic_extract
#SBATCH --partition=short
#SBATCH --array=0-16
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/extract_%A_%a.out
#SBATCH --error=logs/extract_%A_%a.err

# Feature extraction — no GPU needed
# Array index 0-16 covers all 17 subjects (including dmnelf999)

SUBJECTS=(
    dmnelf999
    dmnelf001
    dmnelf002
    dmnelf003
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

SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
CONFIG=/work/gablab/dmnelf/analysis/fingerprint/cyclic_transcoder/config.yaml

echo "Extracting subject: $SUBJECT"
echo "Config: $CONFIG"

source /home/$USER/.bashrc
conda activate microstate_pda

cd /work/gablab/dmnelf/analysis/fingerprint/cyclic_transcoder

python -c "import py_compile; py_compile.compile('data/extract_features.py')"

python data/extract_features.py \
    --subject "$SUBJECT" \
    --config "$CONFIG"

echo "Done: $SUBJECT"
