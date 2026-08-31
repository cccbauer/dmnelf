#!/usr/bin/env python3
"""
Analyze feedback run data structure and prepare for full visualization.

The downloaded prediction file contains:
- pda_predicted (400 points): Model output from windowed processing
- fmri_true (50×528): Parcel recordings, but WITHOUT personal DMN/CEN ROIs

To visualize full 528-timepoint ground truth PDA, cluster data needed.

Usage:
    python plot_all_feedback_runs.py --subject dmnelf005 --save
    python plot_all_feedback_runs.py --subject dmnelf010 --analyze
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
import yaml
import warnings

warnings.filterwarnings('ignore')


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


def load_prediction_npz(pred_path):
    """
    Load prediction NPZ. Note: ground truth PDA computation limited.
    
    The saved fmri_true contains only 50 DiFuMo parcels (out of original 64),
    but DMN/CEN personal ROIs (indices 64-65) are not included. For full
    ground truth PDA with all 528 timepoints, cluster data needed.
    
    Currently returns:
    - pda_pred: Original 400-timepoint model predictions
    - fmri_true: (50, 528) parcel × timepoint data (without personal ROIs)
    - Window information explaining 400 vs 528 mismatch
    
    Args:
        pred_path: Path to prediction NPZ file
    
    Returns:
        dict with available data and explanatory metadata
    """
    data = np.load(pred_path, allow_pickle=True)
    
    pda_pred = data['pda_predicted']  # (400,)
    fmri_true = data['fmri_true']      # (50, 528) - without personal ROIs!
    dmn_idx = int(data['dmn_idx'])     # 64 - refers to original 66-feature space
    cen_idx = int(data['cen_idx'])     # 65 - refers to original 66-feature space
    
    return {
        'pda_pred': pda_pred,
        'fmri_true': fmri_true,
        'n_original_pred': len(pda_pred),
        'n_timepoints_fmri': fmri_true.shape[1],
        'n_parcels': fmri_true.shape[0],
        'dmn_idx': dmn_idx,
        'cen_idx': cen_idx,
        'data_limitation': 'Personal DMN/CEN ROIs not in downloaded NPZ - need cluster data for full ground truth',
    }


def compute_metrics(pda_true, pda_pred):
    """Compute correlation and error metrics."""
    r, p = pearsonr(pda_true, pda_pred)
    mae = np.mean(np.abs(pda_true - pda_pred))
    rmse = np.sqrt(np.mean((pda_true - pda_pred)**2))
    r_squared = r**2
    
    # ROC AUC: treat as binary classification at median
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score((pda_true > np.median(pda_true)).astype(int),
                           pda_pred)
    except:
        auc = np.nan
    
    return {
        'r': r,
        'p': p,
        'mae': mae,
        'rmse': rmse,
        'r_squared': r_squared,
        'auc': auc
    }


def plot_timeseries(pda_true, pda_pred, subject, tr=1.2, save_path=None):
    """
    Create 3-panel visualization of full feedback run.
    
    Panels:
    1. Overlay timeseries (true vs predicted)
    2. Error magnitude with statistics
    3. Scatter accuracy plot
    
    Args:
        pda_true: Ground truth PDA (528,)
        pda_pred: Predicted PDA (528,)
        subject: Subject ID
        tr: Repetition time (seconds)
        save_path: Optional path to save figure
    
    Returns:
        (fig, metrics_dict)
    """
    metrics = compute_metrics(pda_true, pda_pred)
    
    # Time axis in seconds
    time_sec = np.arange(len(pda_true)) * tr
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(
        f'Subject {subject}: Full Feedback Run PDA Comparison\n'
        f'Pearson r = {metrics["r"]:.4f} (p = {metrics["p"]:.3e})',
        fontsize=14, fontweight='bold', y=0.995
    )
    
    # Panel 1: Overlay
    ax1 = axes[0]
    ax1.plot(time_sec, pda_true, 'b-', linewidth=2, label='True PDA', alpha=0.8)
    ax1.plot(time_sec, pda_pred, 'r--', linewidth=2, label='Predicted PDA', alpha=0.8)
    ax1.set_ylabel('PDA (CEN - DMN)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add statistics box to panel 1
    stats_text = (
        f'R² = {metrics["r_squared"]:.4f}\n'
        f'MAE = {metrics["mae"]:.4f}\n'
        f'RMSE = {metrics["rmse"]:.4f}'
    )
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 2: Error with fill_between
    ax2 = axes[1]
    error = pda_true - pda_pred
    ax2.fill_between(time_sec, 0, error, alpha=0.6, color='coral', label='Error')
    ax2.plot(time_sec, error, 'r-', linewidth=1, alpha=0.8)
    ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Residual', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    
    # Panel 3: Scatter with regression line
    ax3 = axes[2]
    ax3.scatter(pda_true, pda_pred, alpha=0.5, s=20, color='steelblue')
    
    # Regression line
    z = np.polyfit(pda_true, pda_pred, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(pda_true.min(), pda_true.max(), 100)
    ax3.plot(x_line, p_line(x_line), 'r-', linewidth=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
    
    # Diagonal (perfect prediction)
    lim = [min(pda_true.min(), pda_pred.min()), max(pda_true.max(), pda_pred.max())]
    ax3.plot(lim, lim, 'k--', linewidth=1, alpha=0.5, label='Perfect prediction')
    
    ax3.set_xlabel('True PDA', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Predicted PDA', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: {save_path}")
    
    return fig, metrics


def save_csv_data(pda_true, pda_pred, subject, output_dir='results', tr=1.2):
    """Save timeseries data as CSV for external analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    time_sec = np.arange(len(pda_true)) * tr
    
    # Combined CSV
    combined_path = output_dir / f'{subject}_feedback_pda_full.csv'
    np.savetxt(
        combined_path,
        np.column_stack([time_sec, pda_true, pda_pred]),
        delimiter=',',
        header='time_sec,pda_true,pda_predicted',
        fmt='%.6f'
    )
    print(f"✓ Combined data saved: {combined_path}")
    
    # Separate CSVs for true and predicted
    true_path = output_dir / f'{subject}_feedback_pda_true.csv'
    np.savetxt(true_path, np.column_stack([time_sec, pda_true]),
              delimiter=',', header='time_sec,pda_true', fmt='%.6f')
    
    pred_path = output_dir / f'{subject}_feedback_pda_predicted.csv'
    np.savetxt(pred_path, np.column_stack([time_sec, pda_pred]),
              delimiter=',', header='time_sec,pda_predicted', fmt='%.6f')
    
    print(f"✓ Individual CSVs saved: {true_path}, {pred_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize model predictions with available data.'
    )
    parser.add_argument('--subject', default='dmnelf005',
                       help='Subject ID (default: dmnelf005)')
    parser.add_argument('--prediction-dir', default='results',
                       help='Directory containing prediction files')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--save', action='store_true',
                       help='Save plots and data to disk')
    parser.add_argument('--result-tag', type=str, default='',
                       help='Tag to append to result filenames (e.g., smooth_w11)')
    parser.add_argument('--smooth-window', type=int, default=1,
                       help='Moving average window size for smoothing (default: 1 = no smoothing)')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"✗ Config not found: {config_path}")
        return
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Construct prediction file path
    pred_path = (Path(args.prediction_dir) / 
                f'sub-{args.subject}' / 'predictions' /
                f'sub-{args.subject}_task-feedback_pda_prediction.npz')
    
    if not pred_path.exists():
        print(f"✗ Prediction file not found: {pred_path}")
        return
    
    print(f"Loading predictions from: {pred_path}")
    
    # Load data
    data = load_prediction_npz(pred_path)
    pda_pred = data['pda_pred']
    
    print(f"\nSubject: {args.subject}")
    print(f"\n{'='*80}")
    print(f"DATA STRUCTURE EXPLANATION")
    print(f"{'='*80}")
    print(f"\nModel predictions (pda_predicted):")
    print(f"  - {data['n_original_pred']} timepoints")
    print(f"  - Output of windowed processing (50 TR window, 10 TR stride)")
    print(f"  - From 125 TR feedback run → ~{125/10} windows")
    
    print(f"\nfMRI recordings (fmri_true):")
    print(f"  - Shape: {data['n_parcels']} parcels × {data['n_timepoints_fmri']} timepoints")
    print(f"  - {data['n_timepoints_fmri']} timepoints = {data['n_timepoints_fmri'] * cfg['data']['fmri']['tr']:.1f} seconds")
    print(f"  - Contains: DiFuMo 64 parcels, reduced to 50 in this file")
    
    print(f"\n⚠️  DATA LIMITATION:")
    print(f"  {data['data_limitation']}")
    print(f"  - Indices stored (DMN=64, CEN=65) refer to original 66-feature space")
    print(f"  - Personal ROI masks are NOT in the downloaded NPZ")
    print(f"  - Cannot compute ground truth PDA from this data")
    
    print(f"\n{'='*80}")
    print(f"SOLUTION OPTIONS:")
    print(f"{'='*80}")
    print(f"\n1. USE CLUSTER DATA:")
    print(f"   - Download full 66-feature fMRI with DMN/CEN ROIs")
    print(f"   - Would enable full 528-timepoint ground truth PDA")
    print(f"   - Command: scp -r explorer:/projects/swglab/data/DMNELF/analysis/...")
    
    print(f"\n2. USE AVAILABLE DATA:")
    print(f"   - Plot 400-point pda_predicted as-is")
    print(f"   - Show parcel-level fMRI timeseries")
    print(f"   - Explain windowing architecture")
    
    print(f"\n3. RECONSTRUCT FROM TRAINING:")
    print(f"   - Run inference with full model to get all per-parcel predictions")
    print(f"   - Takes ~5 min per subject on GPU")
    
    print(f"\n{'='*80}\n")
    
    # Save raw prediction data
    if args.save:
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save raw predictions
        time_sec = np.arange(len(pda_pred)) * cfg['data']['fmri']['tr']
        pred_path_out = output_dir / f'{args.subject}_feedback_pda_predicted_raw.csv'
        np.savetxt(
            pred_path_out,
            np.column_stack([time_sec, pda_pred]),
            delimiter=',',
            header='time_sec,pda_predicted',
            fmt='%.6f'
        )
        print(f"✓ Raw predictions saved: {pred_path_out}")
        print(f"\nTo get full ground truth visualization, fetch cluster data using option 1 above.")


if __name__ == '__main__':
    main()
