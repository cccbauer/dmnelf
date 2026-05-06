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
SMOOTH_WINDOW=1
SMOOTH_BOTH=false
RESULT_TAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --plot)
            PLOT=true
            shift
            ;;
        --download)
            DOWNLOAD=true
            shift
            ;;
        --smooth-window)
            SMOOTH_WINDOW="$2"
            shift 2
            ;;
        --smooth-both)
            SMOOTH_BOTH=true
            shift
            ;;
        --result-tag)
            RESULT_TAG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
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
if [ "$SMOOTH_WINDOW" != "1" ]; then
    EVAL_ARGS="$EVAL_ARGS --smooth-window $SMOOTH_WINDOW"
fi
if [ "$SMOOTH_BOTH" = "true" ]; then
    EVAL_ARGS="$EVAL_ARGS --smooth-both"
fi
if [ -n "$RESULT_TAG" ]; then
    EVAL_ARGS="$EVAL_ARGS --result-tag $RESULT_TAG"
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

    RESULTS_FILE="evaluation_results.csv"
    PLOTS_DIR="evaluation_plots"
    if [ -n "$RESULT_TAG" ]; then
        RESULTS_FILE="evaluation_results_${RESULT_TAG}.csv"
        PLOTS_DIR="evaluation_plots_${RESULT_TAG}"
    fi
    
    # Download results CSV
    scp "$EXPLORER:$REMOTE_BASE/$RESULTS_FILE" "$RESULTS_DIR/" 2>/dev/null && \
        echo "  ✓ $RESULTS_FILE downloaded" || echo "  [skip] $RESULTS_FILE not found"
    
    # Download plots if they exist
    if [ "$PLOT" = "true" ]; then
        scp -r "$EXPLORER:$REMOTE_BASE/$PLOTS_DIR" "$RESULTS_DIR/" 2>/dev/null && \
            echo "  ✓ $PLOTS_DIR/ downloaded" || echo "  [skip] $PLOTS_DIR/ not found"
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
    RESULTS_FILE="evaluation_results.csv"
    PLOTS_DIR="evaluation_plots"
    if [ -n "$RESULT_TAG" ]; then
        RESULTS_FILE="evaluation_results_${RESULT_TAG}.csv"
        PLOTS_DIR="evaluation_plots_${RESULT_TAG}"
    fi

    echo "  Location: $RESULTS_DIR/"
    echo "  CSV: $RESULTS_DIR/$RESULTS_FILE"
    if [ "$PLOT" = "true" ]; then
        echo "  Plots: $RESULTS_DIR/$PLOTS_DIR/"
    fi
else
    RESULTS_FILE="evaluation_results.csv"
    PLOTS_DIR="evaluation_plots"
    if [ -n "$RESULT_TAG" ]; then
        RESULTS_FILE="evaluation_results_${RESULT_TAG}.csv"
        PLOTS_DIR="evaluation_plots_${RESULT_TAG}"
    fi

    echo "  Remote: $REMOTE_BASE/"
    echo "  CSV: $REMOTE_BASE/$RESULTS_FILE"
    if [ "$PLOT" = "true" ]; then
        echo "  Plots: $REMOTE_BASE/$PLOTS_DIR/"
    fi
    echo ""
    echo "To download results:"
    echo "  bash evaluate_on_explorer.sh --download"
fi
echo ""
