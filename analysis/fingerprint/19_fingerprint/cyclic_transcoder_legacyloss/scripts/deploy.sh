#!/bin/bash
# deploy.sh
# ---------
# Deploys cyclic_transcoder from your Mac to Explorer.
# Run from the cyclic_transcoder directory:
#   bash scripts/deploy.sh
#
# First run:  creates remote directory and copies everything
# Subsequent: rsync only changed files (fast)

# ── Config ────────────────────────────────────────────────────────────────────
EXPLORER_USER="cccbauer"
EXPLORER_HOST="login.discovery.neu.edu"
REMOTE_BASE="/projects/swglab/data/DMNELF/analysis/fingerprint"
REMOTE_DIR="$REMOTE_BASE/cyclic_transcoder"

# Root of the project on your Mac (directory containing config.yaml)
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ── Checks ────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════"
echo " Cyclic Transcoder — Deploy to Explorer"
echo "══════════════════════════════════════════════"
echo ""
echo "  Local  : $LOCAL_DIR"
echo "  Remote : $EXPLORER_USER@$EXPLORER_HOST:$REMOTE_DIR"
echo ""

# Warn if FILL_IN placeholders still in config
if grep -q "FILL_IN" "$LOCAL_DIR/config.yaml"; then
    echo "  ⚠  config.yaml still has FILL_IN placeholders."
    echo "     Deploy will continue — fill them in after SSHing to Explorer."
    echo ""
fi

# ── Create remote directory ───────────────────────────────────────────────────
echo "  [1/3] Creating remote directory..."
ssh "$EXPLORER_USER@$EXPLORER_HOST" "mkdir -p $REMOTE_DIR/logs $REMOTE_DIR/checkpoints"

# ── rsync ─────────────────────────────────────────────────────────────────────
echo "  [2/3] Syncing files..."
rsync -avz --progress \
    --exclude "__pycache__" \
    --exclude "*.pyc" \
    --exclude ".DS_Store" \
    --exclude "*.egg-info" \
    --exclude ".git" \
    --exclude "logs/" \
    --exclude "checkpoints/" \
    "$LOCAL_DIR/" \
    "$EXPLORER_USER@$EXPLORER_HOST:$REMOTE_DIR/"

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "  [3/3] Remote file listing:"
ssh "$EXPLORER_USER@$EXPLORER_HOST" "find $REMOTE_DIR -not -path '*/\.*' -not -path '*/logs/*' -not -path '*/checkpoints/*' | sort"

echo ""
echo "══════════════════════════════════════════════"
echo " Done. Next steps on Explorer:"
echo ""
echo "  ssh $EXPLORER_USER@$EXPLORER_HOST"
echo "  cd $REMOTE_DIR"
echo ""
echo "  # 1. Check what's in derivatives/ to fill FILL_IN paths:"
echo "  ls /projects/swglab/data/DMNELF/derivatives/"
echo ""
echo "  # 2. Edit config.yaml with correct eeg_preproc_dir and personal_masks_dir"
echo "  nano config.yaml"
echo ""
echo "  # 3. Run the data audit:"
echo "  conda activate microstate_pda"
echo "  python check_data.py --config config.yaml 2>&1 | tee logs/check_data.txt"
echo "══════════════════════════════════════════════"
