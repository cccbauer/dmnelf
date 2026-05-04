#!/bin/bash
# deploy_and_run.sh
# -----------------
# Deploy cyclic_transcoder to Explorer and optionally run pipeline stages.
#
# Usage:
#   bash scripts/deploy_and_run.sh                    # deploy only
#   bash scripts/deploy_and_run.sh --check            # deploy + run audit
#   bash scripts/deploy_and_run.sh --extract          # deploy + submit feature extraction
#   bash scripts/deploy_and_run.sh --train            # deploy + submit training
#   bash scripts/deploy_and_run.sh --extract --train  # deploy + extract + train (chained)
#   bash scripts/deploy_and_run.sh --all              # deploy + check + extract + train

set -e

# ── Config ────────────────────────────────────────────────────────────────────
EXPLORER="cccbauer@explorer.northeastern.edu"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # cyclic_transcoder root
REMOTE_BASE="/projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder"
CONDA_INIT="source /shared/EL9/explorer/anaconda3/2024.06-root/etc/profile.d/conda.sh"
CONDA_ENV="fingerprint"

# ── Parse args ────────────────────────────────────────────────────────────────
DO_CHECK=false
DO_EXTRACT=false
DO_TRAIN=false
DO_PREDICT=false

for arg in "$@"; do
    case $arg in
        --check)   DO_CHECK=true ;;
        --extract) DO_EXTRACT=true ;;
        --train)   DO_TRAIN=true ;;
        --predict) DO_PREDICT=true ;;
        --all)     DO_CHECK=true; DO_EXTRACT=true; DO_TRAIN=true; DO_PREDICT=true ;;
    esac
done

echo "══════════════════════════════════════════════"
echo " Cyclic Transcoder — Deploy & Run"
echo " Local : $LOCAL_DIR"
echo " Remote: $EXPLORER:$REMOTE_BASE"
echo "══════════════════════════════════════════════"
echo ""

# ── 1. Sync code to Explorer ──────────────────────────────────────────────────
echo "[1/4] Syncing code to Explorer..."
rsync -avz --progress \
    --exclude "__pycache__" \
    --exclude "*.pyc" \
    --exclude ".DS_Store" \
    --exclude "*.egg-info" \
    --exclude ".git" \
    --exclude "logs/" \
    --exclude "checkpoints/" \
    "$LOCAL_DIR/" \
    "$EXPLORER:$REMOTE_BASE/"

echo ""
echo "  ✓ Code synced"
echo ""

# ── 2. Create logs/checkpoints dirs on Explorer ───────────────────────────────
ssh "$EXPLORER" "mkdir -p $REMOTE_BASE/logs $REMOTE_BASE/checkpoints"

# ── 3. Run audit ─────────────────────────────────────────────────────────────
if [ "$DO_CHECK" = true ]; then
    echo "[2/4] Running data audit..."
    ssh "$EXPLORER" "
        $CONDA_INIT
        conda activate $CONDA_ENV
        cd $REMOTE_BASE
        python check_data.py --config config.yaml 2>&1 | tee logs/check_data.txt
    "
    echo ""
    ssh "$EXPLORER" "cat $REMOTE_BASE/check_data_report.tsv"
    echo ""
fi

# ── 4. Submit feature extraction ─────────────────────────────────────────────
EXTRACT_JOB_ID=""
if [ "$DO_EXTRACT" = true ]; then
    echo "[3/4] Submitting feature extraction..."
    EXTRACT_JOB_ID=$(ssh "$EXPLORER" "
        $CONDA_INIT
        conda activate $CONDA_ENV
        cd $REMOTE_BASE
        sbatch --parsable scripts/extract_job.sh
    ")
    echo "  ✓ Extract job submitted: $EXTRACT_JOB_ID"
    echo "  Monitor: ssh $EXPLORER 'squeue -u cccbauer'"
    echo ""
fi

# ── 5. Submit training (optionally after extract) ─────────────────────────────
if [ "$DO_TRAIN" = true ]; then
    echo "[4/4] Submitting training (LOOCV)..."
    if [ -n "$EXTRACT_JOB_ID" ]; then
        # Chain: wait for extraction to finish first
        TRAIN_JOB_ID=$(ssh "$EXPLORER" "
            $CONDA_INIT
            conda activate $CONDA_ENV
            cd $REMOTE_BASE
            sbatch --parsable --dependency=afterok:$EXTRACT_JOB_ID scripts/train_job.sh
        ")
        echo "  ✓ Train job submitted: $TRAIN_JOB_ID (depends on extract job $EXTRACT_JOB_ID)"
    else
        TRAIN_JOB_ID=$(ssh "$EXPLORER" "
            $CONDA_INIT
            conda activate $CONDA_ENV
            cd $REMOTE_BASE
            sbatch --parsable scripts/train_job.sh
        ")
        echo "  ✓ Train job submitted: $TRAIN_JOB_ID"
    fi
    echo "  Monitor: ssh $EXPLORER 'squeue -u cccbauer'"
    echo ""
fi

# ── 6. Submit prediction ─────────────────────────────────────────────────────
if [ "$DO_PREDICT" = true ]; then
    echo "[5/5] Submitting PDA prediction..."
    if [ -n "$TRAIN_JOB_ID" ]; then
        PREDICT_JOB_ID=$(ssh "$EXPLORER" "
            $CONDA_INIT
            conda activate $CONDA_ENV
            cd $REMOTE_BASE
            sbatch --parsable --dependency=afterok:$TRAIN_JOB_ID scripts/predict_job.sh
        ")
        echo "  ✓ Predict job submitted: $PREDICT_JOB_ID (depends on train job $TRAIN_JOB_ID)"
    else
        PREDICT_JOB_ID=$(ssh "$EXPLORER" "
            $CONDA_INIT
            conda activate $CONDA_ENV
            cd $REMOTE_BASE
            sbatch --parsable scripts/predict_job.sh
        ")
        echo "  ✓ Predict job submitted: $PREDICT_JOB_ID"
    fi
    echo ""
fi
echo "══════════════════════════════════════════════"
echo " Done."
echo ""
echo " Useful commands:"
echo "   Monitor jobs  : ssh $EXPLORER 'squeue -u cccbauer'"
echo "   Watch logs    : ssh $EXPLORER 'tail -f $REMOTE_BASE/logs/train_*_0.out'"
echo "   Check results : bash scripts/deploy_and_run.sh --check"
echo ""
echo " Pipeline stages:"
echo "   --check    audit data"
echo "   --extract  extract features"
echo "   --train    train LOOCV models"
echo "   --predict  predict PDA on feedback"
echo "   --all      run everything"
echo "══════════════════════════════════════════════"