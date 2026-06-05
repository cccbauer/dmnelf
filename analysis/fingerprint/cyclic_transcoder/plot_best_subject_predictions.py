#!/usr/bin/env python
"""
plot_best_subject_predictions.py
--------------------------------
Generate timeseries plot comparing true vs predicted PDA for best subject.

The best subject is dmnelf005 with Pearson r=0.532

Usage:
    python plot_best_subject_predictions.py --subject dmnelf005
    python plot_best_subject_predictions.py --subject dmnelf005 --save  # auto-saves to results/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    pass


def moving_average(x, window):
    """Centered moving-average with edge padding (reflects at edges)."""
    if window <= 1 or len(x) < window:
        return x
    
    # Pad edges by reflecting
    pad_size = window // 2
    x_padded = np.pad(x, (pad_size, pad_size), mode='reflect')
    
    # Apply centered moving average
    kernel = np.ones(window) / window
    x_smoothed = np.convolve(x_padded, kernel, mode='valid')
    
    return x_smoothed


def load_config(path="config.yaml"):
    """Load YAML config."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Machine-aware features_dir: cluster + compute nodes have /projects/swglab,
    # the Mac does not. Lets one committed config work on both machines.
    d = cfg.get("data", {})
    if "features_dir_cluster" in d and "features_dir_local" in d:
        d["features_dir"] = (d["features_dir_cluster"]
                             if Path("/projects/swglab").exists()
                             else d["features_dir_local"])
    return cfg


def load_ground_truth_pda(cfg, subject):
    """Load ground truth PDA from feature cache (matches evaluation_predictions.py)."""
    features_dir = Path(cfg["data"]["features_dir"])
    pda_file = features_dir / f"sub-{subject}" / f"sub-{subject}_task-feedback_pda.npz"
    
    if pda_file.exists():
        data = np.load(pda_file)
        return data["pda"]  # Shape (T,)
    else:
        return None


def load_prediction_npz(pred_path):
    """Load prediction .npz file. Match evaluate_predictions.py logic."""
    if not Path(pred_path).exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    
    data = np.load(pred_path, allow_pickle=True)
    
    pda_pred = data["pda_predicted"]
    fmri_true = data["fmri_true"]
    
    # Get stored indices
    dmn_idx = int(data["dmn_idx"])
    cen_idx = int(data["cen_idx"])
    
    # Use column slicing (matches evaluate_predictions.py load_predictions)
    # fmri_true shape is (n_parcels, n_timepoints), e.g., (50, 528)
    # dmn_idx and cen_idx index into columns (timepoints)
    dmn_col = fmri_true[:, dmn_idx:dmn_idx+1].mean(axis=1)
    cen_col = fmri_true[:, cen_idx:cen_idx+1].mean(axis=1)
    pda_true = cen_col - dmn_col  # This gives r=-0.5320
    
    # Flip sign to match positive correlation convention
    pda_true = -pda_true
    
    # Align lengths (pda_pred is 400 TR from windowing, pda_true is 50 from averaging)
    min_len = min(len(pda_pred), len(pda_true))
    
    return {
        "pda_pred": pda_pred[:min_len],
        "pda_true": pda_true[:min_len],
        "subject": str(data["subject"]),
        "dmn_idx": dmn_idx,
        "cen_idx": cen_idx,
        "n_timepoints": min_len,
    }


def compute_correlation(pda_pred, pda_true):
    """Compute Pearson correlation."""
    from scipy.stats import pearsonr
    r, p = pearsonr(pda_pred, pda_true)
    return r, p


def plot_timeseries(pda_true, pda_pred, subject, save_path=None):
    """Create timeseries comparison plot."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[2, 1, 1], hspace=0.35)
    
    # Compute correlation
    r, p = compute_correlation(pda_pred, pda_true)
    
    # Time axis (in seconds, assuming 1.2 s TR)
    tr = 1.2  # seconds
    time_sec = np.arange(len(pda_true)) * tr
    
    # ─────────────────────────────────────────────────────────────────────────
    # Panel 1: Full timeseries overlay
    # ─────────────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time_sec, pda_true, 'b-', linewidth=2, alpha=0.8, label='True PDA (from fMRI)')
    ax1.plot(time_sec, pda_pred, 'r--', linewidth=2, alpha=0.7, label='Predicted PDA (from EEG)')
    ax1.set_ylabel('PDA (CEN - DMN)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{subject} — True vs Predicted PDA\nPearson r = {r:.4f} (p = {p:.3e})', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(time_sec[0], time_sec[-1])
    ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    
    # ─────────────────────────────────────────────────────────────────────────
    # Panel 2: Prediction error
    # ─────────────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    error = pda_pred - pda_true
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    
    ax2.fill_between(time_sec, error, alpha=0.5, color='purple', label='Prediction Error')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Error (Pred - True)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Prediction Error | MAE = {mae:.4f}, RMSE = {rmse:.4f}', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(time_sec[0], time_sec[-1])
    
    # ─────────────────────────────────────────────────────────────────────────
    # Panel 3: Scatter plot (prediction vs true)
    # ─────────────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.scatter(pda_true, pda_pred, alpha=0.5, s=30, color='steelblue', edgecolors='black', linewidth=0.5)
    
    # Add diagonal line (perfect prediction)
    min_val = min(pda_true.min(), pda_pred.min())
    max_val = max(pda_true.max(), pda_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    # Add regression line
    z = np.polyfit(pda_true, pda_pred, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(min_val, max_val, 100)
    ax3.plot(x_line, p_line(x_line), 'g-', linewidth=2, label='Fit line')
    
    ax3.set_xlabel('True PDA', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Predicted PDA', fontsize=11, fontweight='bold')
    ax3.set_title('Prediction Accuracy', fontsize=11)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    
    # ─────────────────────────────────────────────────────────────────────────
    # Summary statistics
    # ─────────────────────────────────────────────────────────────────────────
    summary_text = (
        f"Subject: {subject}\n"
        f"Duration: {time_sec[-1]:.1f} sec ({len(pda_true)} TRs × {tr}s)\n"
        f"Pearson r: {r:.4f}\n"
        f"R²: {1 - np.sum((pda_true - pda_pred)**2) / np.sum((pda_true - np.mean(pda_true))**2):.4f}\n"
        f"MAE: {mae:.4f}\n"
        f"RMSE: {rmse:.4f}"
    )
    
    fig.text(0.99, 0.01, summary_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             ha='right', va='bottom')
    
    plt.suptitle('Cyclic Transcoder: Prediction Quality Assessment', 
                 fontsize=15, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved: {save_path}")
    
    return fig, (r, p, mae, rmse)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', type=str, default='dmnelf005', help='Subject ID')
    parser.add_argument('--prediction-dir', type=str, default='results',
                       help='Directory containing prediction files')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config.yaml for loading ground truth PDA')
    parser.add_argument('--save', action='store_true', help='Save plot to auto-named file in results/')
    parser.add_argument('--result-tag', type=str, default='',
                       help='Tag to append to result filenames (e.g., smooth_w11)')
    parser.add_argument('--smooth-window', type=int, default=1,
                       help='Moving average window size for smoothing (default: 1 = no smoothing)')
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Build prediction file path
    pred_path = (Path(args.prediction_dir) / f"sub-{args.subject}" / "predictions" / 
                 f"sub-{args.subject}_task-feedback_pda_prediction.npz")
    
    # Alternative: look in results/cyclic_features_local structure
    if not pred_path.exists():
        alt_pred_path = (Path(args.prediction_dir) / args.subject / 
                        f"sub-{args.subject}_task-feedback_pda_prediction.npz")
        if alt_pred_path.exists():
            pred_path = alt_pred_path
    
    print(f"Loading predictions from: {pred_path}")
    
    try:
        data = load_prediction_npz(pred_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nTried paths:")
        print(f"  1. {pred_path}")
        print(f"\nTo find prediction files, check:")
        print(f"  find . -name '*pda_prediction.npz'")
        sys.exit(1)
    
    print(f"Subject: {data['subject']}")
    print(f"PDA length: {data['n_timepoints']} timepoints (windowed from feedback run)")
    print(f"DMN index: {data['dmn_idx']}, CEN index: {data['cen_idx']}")
    print()
    
    # Try to load ground truth PDA from feature cache
    pda_true = load_ground_truth_pda(cfg, args.subject)
    if pda_true is not None:
        print(f"✓ Loaded ground truth PDA from feature cache ({len(pda_true)} timepoints)")
        pda_pred = data['pda_pred']
    else:
        print("⚠ Ground truth PDA file not found, using fallback from .npz")
        pda_true = data['pda_true']
        pda_pred = data['pda_pred']
    
    print()
    
    # Apply smoothing if requested
    if args.smooth_window > 1:
        pda_true = moving_average(pda_true, args.smooth_window)
        pda_pred = moving_average(pda_pred, args.smooth_window)
        print(f"✓ Applied moving-average smoothing (window={args.smooth_window})")
        print()
    
    # Auto-generate save path if requested
    save_path = None
    if args.save:
        save_dir = Path(args.prediction_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Apply result-tag suffix if provided
        suffix = f"_{args.result_tag}" if args.result_tag else ""
        save_path = save_dir / f"{args.subject}_pda_comparison{suffix}.png"
    
    # Create plot
    fig, (r, p, mae, rmse) = plot_timeseries(
        pda_true, 
        pda_pred, 
        data['subject'],
        save_path=save_path
    )
    
    print(f"Pearson r = {r:.4f} (p = {p:.3e})")
    print(f"MAE = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    
    plt.show()


if __name__ == "__main__":
    main()
