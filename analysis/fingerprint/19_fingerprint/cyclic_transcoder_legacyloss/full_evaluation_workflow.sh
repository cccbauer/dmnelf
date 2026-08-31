#!/bin/bash
# full_evaluation_workflow.sh
# ---------------------------
# Complete workflow: run evaluation on Explorer, download, and summarize
#
# Usage:
#   bash full_evaluation_workflow.sh           # evaluate + summarize
#   bash full_evaluation_workflow.sh --plot    # + generate plots

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLOT=false
SMOOTH_WINDOW=1
SMOOTH_BOTH=false
RESULT_TAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --plot)
            PLOT=true
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

echo "════════════════════════════════════════════════"
echo " Full Evaluation Workflow"
echo "════════════════════════════════════════════════"
echo ""

EVAL_EXTRA_ARGS=""
if [ "$SMOOTH_WINDOW" != "1" ]; then
    EVAL_EXTRA_ARGS="$EVAL_EXTRA_ARGS --smooth-window $SMOOTH_WINDOW"
fi
if [ "$SMOOTH_BOTH" = "true" ]; then
    EVAL_EXTRA_ARGS="$EVAL_EXTRA_ARGS --smooth-both"
fi
if [ -n "$RESULT_TAG" ]; then
    EVAL_EXTRA_ARGS="$EVAL_EXTRA_ARGS --result-tag $RESULT_TAG"
fi

RESULTS_FILE="evaluation_results.csv"
SUMMARY_FILE="summary_correlations.png"
if [ -n "$RESULT_TAG" ]; then
    RESULTS_FILE="evaluation_results_${RESULT_TAG}.csv"
    SUMMARY_FILE="summary_correlations_${RESULT_TAG}.png"
fi

# Step 1: Run evaluation on Explorer
echo "[1/3] Running evaluation on Explorer..."
if [ "$PLOT" = "true" ]; then
    bash "$SCRIPT_DIR/evaluate_on_explorer.sh" --plot --download $EVAL_EXTRA_ARGS
else
    bash "$SCRIPT_DIR/evaluate_on_explorer.sh" --download $EVAL_EXTRA_ARGS
fi

# Step 2: Summarize results
echo ""
echo "[2/3] Generating summary report..."
cd "$SCRIPT_DIR"
python summarize_results.py --csv "results/$RESULTS_FILE" --visualize --output-dir results --result-tag "$RESULT_TAG"

# Step 3: Display summary
echo ""
echo "[3/3] Summary complete!"
echo ""
echo "Results saved to: $SCRIPT_DIR/results/"
echo ""
echo "Files generated:"
echo "  • results/$RESULTS_FILE          (per-subject metrics)"
echo "  • results/$SUMMARY_FILE        (bar chart of performance)"
if [ "$PLOT" = "true" ]; then
    PLOTS_DIR="evaluation_plots"
    if [ -n "$RESULT_TAG" ]; then
        PLOTS_DIR="evaluation_plots_${RESULT_TAG}"
    fi
    echo "  • results/$PLOTS_DIR/               (detailed evaluation plots)"
fi
echo ""
echo "Open results:"
open results 2>/dev/null || echo "  results/"
echo ""
