#!/bin/bash
#SBATCH --job-name=cyclic_extract_new
#SBATCH --partition=short
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/cyclic_transcoder/logs/extract_new_%A_%a.out
#SBATCH --error=/projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/cyclic_transcoder/logs/extract_new_%A_%a.err

SUBJECTS=(dmnelf002 dmnelf003)
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
CONFIG=/projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/cyclic_transcoder/config.yaml

source /shared/EL9/explorer/anaconda3/2024.06-root/etc/profile.d/conda.sh
conda activate fingerprint

cd /projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/cyclic_transcoder
python -c "import py_compile; py_compile.compile(\"data/extract_features.py\")"
python data/extract_features.py --subject "$SUBJECT" --config "$CONFIG"
echo "Done: $SUBJECT"
