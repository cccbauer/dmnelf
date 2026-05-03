#!/bin/bash
# run_mask_extraction.sh
# ----------------------
# Run mask extraction for a subject on Explorer from your Mac.
#
# Usage:
#   bash run_mask_extraction.sh dmnelf012
#   bash run_mask_extraction.sh dmnelf014 --overwrite

SUBJECT=$1
OVERWRITE=${2:-""}

if [ -z "$SUBJECT" ]; then
    echo "Usage: bash run_mask_extraction.sh <subject_id> [--overwrite]"
    echo "Example: bash run_mask_extraction.sh dmnelf012"
    exit 1
fi

EXPLORER="cccbauer@explorer.northeastern.edu"
REMOTE_DIR="/projects/swglab/data/DMNELF/analysis/fmri_preprocessing/mask_extraction"
LOG_DIR="$REMOTE_DIR/logs"

echo "══════════════════════════════════════════════"
echo " Mask Extraction: $SUBJECT"
echo " Remote: $EXPLORER"
echo "══════════════════════════════════════════════"

# Submit job on Explorer via SSH
ssh "$EXPLORER" bash << EOF
mkdir -p $LOG_DIR

sbatch \\
  --job-name=mask_ext_${SUBJECT} \\
  --partition=short \\
  --mem=32G \\
  --cpus-per-task=8 \\
  --time=02:00:00 \\
  --output=$LOG_DIR/mask_ext_${SUBJECT}_%j.out \\
  --error=$LOG_DIR/mask_ext_${SUBJECT}_%j.err \\
  --wrap="
    source /home/cccbauer/.bashrc
    conda activate pineuro
    cd $REMOTE_DIR
    python mask_extraction.py --subject $SUBJECT --config config.yaml $OVERWRITE
  "
EOF

echo ""
echo "Job submitted. Monitor with:"
echo "  ssh $EXPLORER 'squeue -u cccbauer'"
echo ""
echo "Watch log:"
echo "  ssh $EXPLORER 'tail -f $LOG_DIR/mask_ext_${SUBJECT}_*.out'"