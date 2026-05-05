#!/bin/bash
# evaluate_on_explorer.sh
# -----------------------
# Run evaluation directly on Explorer cluster.
# Uploads evaluate_predictions.py and runs it with the config.
#
# Usage:
#   bash evaluate_on_explorer.sh
#   bash evaluate_on_explorer.sh --plot       # generate plots
#   bash evaluate_on_explorer.sh --download   # download results after

set -e

# ── Config ────────────────────────────────────────────────────────────────────
EXPLORER="cccbauer@explorer.northeastern.edu"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_BASE="/projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder"
EVAL_SCRIPT="evaluate_predictions.py"
CONFIG_FILE="config.yaml"

PLOT=false
DOWNLOAD=false

for arg in "$@"; do
    case $arg in
        --plot)     PLOT=true ;;
        --download) DOWNLOAD=true ;;
    esac
done

echo "══════════════════════════════════════════════"
echo " Run Evaluation on Explorer"
echo "══════════════════════════════════════════════"
echo ""
echo "Local:  $LOCAL_DIR"
echo "Remote: $EXPLORER:$REMOTE_BASE"
echo ""

# ── Step 1: Upload evaluation script ───────────────────────────────────────────
echo "[1/3] Uploading evaluation script..."
scp "$LOCAL_DIR/$EVAL_SCRIPT" "$EXPLORER:$REMOTE_BASE/" > /dev/null
echo "  ✓ $EVAL_SCRIPT uploaded"

# ── Step 2: Run evaluation on Explorer ─────────────────────────────────────────
echo ""
echo "[2/3] Running evaluation on Explorer..."

EVAL_ARGS="--config $CONFIG_FILE"
if [ "$PLOT" = "true" ]; then
    EVAL_ARGS="$EVAL_ARGS --plot"
fi

ssh "$EXPLORER" << EOF
source /shared/EL9/explorer/anaconda3/2024.06-root/etc/profile.d/conda.sh
conda activate fingerprint
cd $REMOTE_BASE
python $EVAL_SCRIPT $EVAL_ARGS
EOF

echo ""
echo "  ✓ Evaluation complete"

# ── Step 3: Download results ──────────────────────────────────────────────────
RESULTS_DIR="$LOCAL_DIR/results"
mkdir -p "$RESULTS_DIR"

if [ "$DOWNLOAD" = "true" ]; then
    echo ""
    echo "[3/3] Downloading results..."
    
    # Download results CSV
    scp "$EXPLORER:$REMOTE_BASE/evaluation_results.csv" "$RESULTS_DIR/" 2>/dev/null && \
        echo "  ✓ evaluation_results.csv downloaded" || echo "  [skip] evaluation_results.csv not found"
    
    # Download plots if they exist
    if [ "$PLOT" = "true" ]; then
        scp -r "$EXPLORER:$REMOTE_BASE/evaluation_plots" "$RESULTS_DIR/" 2>/dev/null && \
            echo "  ✓ evaluation_plots/ downloaded" || echo "  [skip] evaluation_plots/ not found"
    fi
    
    echo ""
    echo "Results downloaded to: $RESULTS_DIR"
else
    echo ""
    echo "[3/3] Skipping download (use --download to fetch results)"
fi

echo ""
echo "═════════════════════════════════════════════════════"
echo "✓ COMPLETE"
echo ""
echo "Results:"
if [ "$DOWNLOAD" = "true" ]; then
    echo "  Location: $RESULTS_DIR/"
    echo "  CSV: $RESULTS_DIR/evaluation_results.csv"
    if [ "$PLOT" = "true" ]; then
        echo "  Plots: $RESULTS_DIR/evaluation_plots/"
    fi
else
    echo "  Remote: $REMOTE_BASE/"
    echo "  CSV: $REMOTE_BASE/evaluation_results.csv"
    if [ "$PLOT" = "true" ]; then
        echo "  Plots: $REMOTE_BASE/evaluation_plots/"
    fi
    echo ""
    echo "To download results:"
    echo "  bash evaluate_on_explorer.sh --download"
fi
echo ""
