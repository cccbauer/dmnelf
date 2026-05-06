"""
evaluate_predictions.py
-----------------------
Evaluate cyclic transcoder predictions on feedback runs.

Computes:
  - Pearson correlation (predicted vs true PDA)
  - RMSE
  - Mean Absolute Error
  - ROC AUC (if thresholded as binary classification)
  - Per-subject and group-level statistics

Generates:
  - Summary CSV with per-subject metrics
  - Aggregate plots (correlation scatter, timeseries samples)
  - Statistical report

Usage:
    python evaluate_predictions.py --config config.yaml
    python evaluate_predictions.py --config config.yaml --plot
    python evaluate_predictions.py --config config.yaml --subject dmnelf001
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_predictions(pred_path):
    """Load prediction .npz file."""
    data = np.load(pred_path, allow_pickle=True)
    return {
        "pda_pred": data["pda_predicted"],
        "pda_true": data["fmri_true"][:, data["dmn_idx"]:data["dmn_idx"]+1].mean(axis=1) - 
                    data["fmri_true"][:, data["cen_idx"]:data["cen_idx"]+1].mean(axis=1)
                    if "fmri_true" in data else None,
        "fmri_pred": data["fmri_predicted"],
        "fmri_true": data["fmri_true"],
        "subject": str(data["subject"]),
        "task": str(data["task"]),
    }


def moving_average(x, window):
    """Centered moving-average smoothing with edge padding."""
    if window <= 1:
        return x

    kernel = np.ones(window, dtype=float) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    x_padded = np.pad(x, (pad_left, pad_right), mode="edge")
    return np.convolve(x_padded, kernel, mode="valid")


def compute_metrics(pda_pred, pda_true, smooth_window=1, smooth_both=False):
    """Compute evaluation metrics."""
    if pda_true is None or len(pda_pred) == 0:
        return None
    
    # Ensure same length
    min_len = min(len(pda_pred), len(pda_true))
    pda_pred = pda_pred[:min_len]
    pda_true = pda_true[:min_len]

    if smooth_window > 1:
        pda_pred = moving_average(pda_pred, smooth_window)
        if smooth_both:
            pda_true = moving_average(pda_true, smooth_window)
    
    metrics = {}
    
    # Correlation
    r, p_corr = pearsonr(pda_pred, pda_true)
    metrics["pearson_r"] = r
    metrics["pearson_p"] = p_corr
    
    # Spearman correlation
    rho, p_spear = spearmanr(pda_pred, pda_true)
    metrics["spearman_rho"] = rho
    metrics["spearman_p"] = p_spear
    
    # Error metrics
    rmse = np.sqrt(mean_squared_error(pda_true, pda_pred))
    mae = mean_absolute_error(pda_true, pda_pred)
    metrics["rmse"] = rmse
    metrics["mae"] = mae
    
    # Normalize to [-1, 1] for ROC AUC
    pred_range = pda_pred.max() - pda_pred.min()
    if pred_range == 0:
        pda_pred_norm = np.zeros_like(pda_pred)
    else:
        pda_pred_norm = (pda_pred - pda_pred.min()) / pred_range
    pda_true_binary = (pda_true > np.median(pda_true)).astype(int)
    
    try:
        auc = roc_auc_score(pda_true_binary, pda_pred_norm)
        metrics["roc_auc"] = auc
    except:
        metrics["roc_auc"] = np.nan
    
    # Variance explained (R²)
    ss_res = np.sum((pda_true - pda_pred) ** 2)
    ss_tot = np.sum((pda_true - pda_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    metrics["r2"] = r2
    
    return metrics


def evaluate_all_predictions(cfg, smooth_window=1, smooth_both=False, result_tag=""):
    """Evaluate all prediction files."""
    pred_base = Path(cfg["data"]["features_dir"])
    
    excluded = set(cfg["data"]["subjects"].get("exclude", []))
    all_subjects = [s for s in cfg["data"]["subjects"]["all"] if s not in excluded]
    
    results = []
    metrics_by_subject = defaultdict(dict)
    
    print("\n" + "="*80)
    print("CYCLIC TRANSCODER — PREDICTION EVALUATION")
    print("="*80)
    
    for subject in sorted(all_subjects):
        pred_path = pred_base / f"sub-{subject}" / "predictions" / f"sub-{subject}_task-feedback_pda_prediction.npz"
        
        if not pred_path.exists():
            print(f"\n[SKIP] {subject}: prediction file not found")
            continue
        
        try:
            pred_data = load_predictions(pred_path)
            
            # Compute PDA from fMRI true if not available
            if pred_data["pda_true"] is None:
                dmn_idx = pred_data["fmri_true"].shape[0] // 2
                cen_idx = pred_data["fmri_true"].shape[0] - 1
                pda_true = pred_data["fmri_true"][cen_idx, :] - pred_data["fmri_true"][dmn_idx, :]
            else:
                pda_true = pred_data["pda_true"]
            
            metrics = compute_metrics(
                pred_data["pda_pred"],
                pda_true,
                smooth_window=smooth_window,
                smooth_both=smooth_both,
            )
            
            if metrics:
                metrics_by_subject[subject] = metrics
                results.append({
                    "subject": subject,
                    **metrics,
                    "n_timepoints": len(pred_data["pda_pred"]),
                })
                
                print(f"\n{subject}:")
                print(f"  Pearson r  = {metrics['pearson_r']:7.4f}  (p = {metrics['pearson_p']:.3e})")
                print(f"  Spearman ρ = {metrics['spearman_rho']:7.4f}  (p = {metrics['spearman_p']:.3e})")
                print(f"  R²         = {metrics['r2']:7.4f}")
                print(f"  RMSE       = {metrics['rmse']:7.4f}")
                print(f"  MAE        = {metrics['mae']:7.4f}")
                print(f"  ROC AUC    = {metrics['roc_auc']:7.4f}")
                print(f"  N timepoints = {results[-1]['n_timepoints']}")
        
        except Exception as e:
            print(f"\n[ERROR] {subject}: {e}")
            continue
    
    # Group-level statistics
    if results:
        print("\n" + "="*80)
        print("GROUP-LEVEL STATISTICS")
        print("="*80)
        
        pearson_rs = [r["pearson_r"] for r in results if not np.isnan(r["pearson_r"])]
        r2s = [r["r2"] for r in results if not np.isnan(r["r2"])]
        rmses = [r["rmse"] for r in results if not np.isnan(r["rmse"])]
        
        if pearson_rs:
            print(f"\nPearson r:  mean = {np.mean(pearson_rs):.4f} ± {np.std(pearson_rs):.4f}")
            print(f"            min = {np.min(pearson_rs):.4f}, max = {np.max(pearson_rs):.4f}")
        
        if r2s:
            print(f"\nR²:         mean = {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
            print(f"            min = {np.min(r2s):.4f}, max = {np.max(r2s):.4f}")
        
        if rmses:
            print(f"\nRMSE:       mean = {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
            print(f"            min = {np.min(rmses):.4f}, max = {np.max(rmses):.4f}")
        
        # Save summary CSV
        results_df = None
        try:
            import pandas as pd
            results_df = pd.DataFrame(results)
            csv_name = "evaluation_results.csv"
            if result_tag:
                csv_name = f"evaluation_results_{result_tag}.csv"
            csv_path = Path(cfg["project"]["base_dir"]) / csv_name
            results_df.to_csv(csv_path, index=False)
            print(f"\nResults saved to: {csv_path}")
        except ImportError:
            print("\n[NOTE] pandas not available; skipping CSV export")
    
    return results, metrics_by_subject


def plot_results(cfg, results, metrics_by_subject, result_tag=""):
    """Generate evaluation plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[NOTE] matplotlib/seaborn not available; skipping plots")
        return
    
    if not results:
        return
    
    pred_base = Path(cfg["data"]["features_dir"])
    
    # Create output directory
    plot_dir_name = "evaluation_plots"
    if result_tag:
        plot_dir_name = f"evaluation_plots_{result_tag}"
    plot_dir = Path(cfg["project"]["base_dir"]) / plot_dir_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Correlation distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    pearson_rs = [r["pearson_r"] for r in results if not np.isnan(r["pearson_r"])]
    r2s = [r["r2"] for r in results if not np.isnan(r["r2"])]
    rmses = [r["rmse"] for r in results if not np.isnan(r["rmse"])]
    
    axes[0, 0].hist(pearson_rs, bins=10, edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Pearson r Distribution")
    axes[0, 0].set_xlabel("Correlation")
    axes[0, 0].set_ylabel("Count")
    
    axes[0, 1].hist(r2s, bins=10, edgecolor="black", alpha=0.7)
    axes[0, 1].set_title("R² Distribution")
    axes[0, 1].set_xlabel("Variance Explained")
    axes[0, 1].set_ylabel("Count")
    
    axes[1, 0].hist(rmses, bins=10, edgecolor="black", alpha=0.7)
    axes[1, 0].set_title("RMSE Distribution")
    axes[1, 0].set_xlabel("RMSE")
    axes[1, 0].set_ylabel("Count")
    
    # Summary stats
    axes[1, 1].axis("off")
    summary_text = (
        f"N subjects: {len(results)}\n"
        f"Pearson r: {np.mean(pearson_rs):.3f} ± {np.std(pearson_rs):.3f}\n"
        f"R²: {np.mean(r2s):.3f} ± {np.std(r2s):.3f}\n"
        f"RMSE: {np.mean(rmses):.3f} ± {np.std(rmses):.3f}"
    )
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family="monospace")
    
    plt.tight_layout()
    plot_path = plot_dir / "01_metrics_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {plot_path}")
    plt.close()
    
    # 2. Per-subject metrics heatmap
    subjects = [r["subject"] for r in results]
    corrs = [r["pearson_r"] for r in results]
    
    fig, ax = plt.subplots(figsize=(8, max(6, len(subjects)*0.3)))
    ax.barh(subjects, corrs, color="steelblue", edgecolor="black")
    ax.set_xlabel("Pearson Correlation")
    ax.set_title("Per-Subject Prediction Correlation")
    ax.axvline(np.mean(corrs), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(corrs):.3f}")
    ax.legend()
    
    plt.tight_layout()
    plot_path = plot_dir / "02_subject_correlations.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--subject", type=str, default=None, help="Evaluate single subject")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Centered moving-average window for PDA smoothing (1 = disabled)",
    )
    parser.add_argument(
        "--smooth-both",
        action="store_true",
        help="Apply smoothing to both predicted and true PDA before metric calculation",
    )
    parser.add_argument(
        "--result-tag",
        type=str,
        default="",
        help="Optional suffix tag for output artifact names (e.g. smooth_w11)",
    )
    args = parser.parse_args()

    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be >= 1")
    
    cfg = load_config(args.config)
    
    results, metrics_by_subject = evaluate_all_predictions(
        cfg,
        smooth_window=args.smooth_window,
        smooth_both=args.smooth_both,
        result_tag=args.result_tag,
    )
    
    if args.plot:
        plot_results(cfg, results, metrics_by_subject, result_tag=args.result_tag)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
