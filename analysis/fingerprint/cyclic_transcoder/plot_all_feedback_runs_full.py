#!/usr/bin/env python3
"""
Plot complete feedback run PDA predictions with FULL 66-feature ground truth.

Requires cluster data with full 66-feature fmRI (64 DiFuMo + 2 personal ROIs).
Downloads from cluster instructions in FETCH_CLUSTER_DATA.md

Generates:
- 3-panel visualization (all 528 timepoints = 10 minutes)
- Metrics on full feedback run
- CSV exports for further analysis

Usage:
    python plot_all_feedback_runs_full.py --subject dmnelf005 --prediction-dir cyclic_features_full --save
    python plot_all_feedback_runs_full.py --subject dmnelf010 --prediction-dir cyclic_features_full
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

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


def load_prediction_npz_full(pred_path):
    """
    Load prediction NPZ with FULL 66-feature fmRI.
    
    Expected structure (from cluster):
    - pda_predicted (400,): Model predictions from windowed processing
    - fmri_true (66, 528): Full fMRI with personal ROIs at indices 64-65
    - fmri_predicted (66, 528): Model's reconstructed fMRI
    - dmn_idx (scalar): 64 (DMN personal ROI)
    - cen_idx (scalar): 65 (CEN personal ROI)
    
    Args:
        pred_path: Path to prediction NPZ file
    
    Returns:
        dict with pda_pred, pda_true, metrics, metadata
    """
    data = np.load(pred_path, allow_pickle=True)
    
    pda_pred = data['pda_predicted']  # (400,)
    fmri_true = data['fmri_true']      # Should be (66, 528)
    dmn_idx = int(data['dmn_idx'])     # 64
    cen_idx = int(data['cen_idx'])     # 65
    
    # Verify we have full data
    if fmri_true.shape[0] != 66:
        raise ValueError(
            f"Expected 66-feature fmRI, got {fmri_true.shape[0]} parcels. "
            "Make sure you downloaded from cluster (not cyclic_features_local). "
            "See FETCH_CLUSTER_DATA.md for download instructions."
        )
    
    # Compute ground truth PDA from FULL fmRI (all 528 timepoints)
    # These are personal ROI indices in 66-feature space
    dmn_ts = fmri_true[dmn_idx:dmn_idx+1, :].mean(axis=0)  # (528,)
    cen_ts = fmri_true[cen_idx:cen_idx+1, :].mean(axis=0)  # (528,)
    pda_true = cen_ts - dmn_ts
    pda_true = -pda_true  # Flip to match positive correlation
    
    # Upsample pda_pred (400 points) to match ground truth (528 points)
    time_pred = np.linspace(0, len(pda_true)-1, len(pda_pred))
    f = interp1d(time_pred, pda_pred, kind='cubic', fill_value='extrapolate')
    pda_pred_upsampled = f(np.arange(len(pda_true)))
    
    return {
        'pda_pred': pda_pred_upsampled,
        'pda_true': pda_true,
        'n_original_pred': len(pda_pred),
        'n_upsampled': len(pda_pred_upsampled),
        'n_parcels': fmri_true.shape[0],
        'n_timepoints': fmri_true.shape[1],
        'dmn_idx': dmn_idx,
        'cen_idx': cen_idx,
    }


def compute_metrics(pda_true, pda_pred):
    """Compute correlation and error metrics."""
    r, p = pearsonr(pda_true, pda_pred)
    mae = np.mean(np.abs(pda_true - pda_pred))
    rmse = np.sqrt(np.mean((pda_true - pda_pred)**2))
    r_squared = r**2
    
    # ROC AUC
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


def plot_timeseries_full(pda_true, pda_pred, subject, tr=1.2, save_path=None):
    """
    Create 3-panel visualization of FULL feedback run (528 timepoints = 10 min).
    
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
    duration_min = time_sec[-1] / 60
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 11))
    fig.suptitle(
        f'Subject {subject}: Full Feedback Run PDA Comparison (528 timepoints = {duration_min:.1f} min)\n'
        f'Pearson r = {metrics["r"]:.4f} (p = {metrics["p"]:.3e})',
        fontsize=15, fontweight='bold', y=0.995
    )
    
    # Panel 1: Overlay
    ax1 = axes[0]
    ax1.plot(time_sec, pda_true, 'b-', linewidth=2.5, label='True PDA', alpha=0.85)
    ax1.plot(time_sec, pda_pred, 'r--', linewidth=2.5, label='Predicted PDA', alpha=0.85)
    ax1.set_ylabel('PDA (CEN - DMN)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    
    # Add statistics box to panel 1
    stats_text = (
        f'R² = {metrics["r_squared"]:.4f}\n'
        f'MAE = {metrics["mae"]:.4f}\n'
        f'RMSE = {metrics["rmse"]:.4f}\n'
        f'AUC = {metrics["auc"]:.4f}'
    )
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, pad=0.8))
    
    # Panel 2: Error with fill_between
    ax2 = axes[1]
    error = pda_true - pda_pred
    ax2.fill_between(time_sec, 0, error, alpha=0.5, color='coral', label='Residual')
    ax2.plot(time_sec, error, 'r-', linewidth=1.5, alpha=0.8)
    ax2.axhline(0, color='k', linestyle='-', linewidth=0.8)
    ax2.set_ylabel('Residual (True - Pred)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.95)
    
    # Add error statistics
    error_stats = f'μ(error) = {np.mean(error):.4f}\nσ(error) = {np.std(error):.4f}'
    ax2.text(0.02, 0.98, error_stats, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6, pad=0.8))
    
    # Panel 3: Scatter with regression line
    ax3 = axes[2]
    ax3.scatter(pda_true, pda_pred, alpha=0.6, s=25, color='steelblue', edgecolors='navy', linewidth=0.5)
    
    # Regression line
    z = np.polyfit(pda_true, pda_pred, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(pda_true.min(), pda_true.max(), 100)
    ax3.plot(x_line, p_line(x_line), 'r-', linewidth=2.5, 
            label=f'Fit: y={z[0]:.3f}x{z[1]:+.3f}', zorder=5)
    
    # Diagonal (perfect prediction)
    lim = [min(pda_true.min(), pda_pred.min()), max(pda_true.max(), pda_pred.max())]
    ax3.plot(lim, lim, 'k--', linewidth=1.5, alpha=0.5, label='Perfect prediction', zorder=3)
    
    ax3.set_xlabel('True PDA', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Predicted PDA', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax3.grid(True, alpha=0.3)
    
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
    combined_path = output_dir / f'{subject}_feedback_pda_full_528.csv'
    np.savetxt(
        combined_path,
        np.column_stack([time_sec, pda_true, pda_pred]),
        delimiter=',',
        header='time_sec,pda_true,pda_predicted',
        fmt='%.6f'
    )
    print(f"✓ Combined data saved: {combined_path}")
    print(f"  Rows: {len(pda_true)}, Duration: {time_sec[-1]:.1f} sec ({time_sec[-1]/60:.1f} min)")
    
    # Separate CSVs for true and predicted
    true_path = output_dir / f'{subject}_feedback_pda_true_full.csv'
    np.savetxt(true_path, np.column_stack([time_sec, pda_true]),
              delimiter=',', header='time_sec,pda_true', fmt='%.6f')
    
    pred_path = output_dir / f'{subject}_feedback_pda_predicted_full.csv'
    np.savetxt(pred_path, np.column_stack([time_sec, pda_pred]),
              delimiter=',', header='time_sec,pda_predicted', fmt='%.6f')
    
    print(f"✓ Individual CSVs saved: {true_path}, {pred_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot complete feedback run with full 66-feature ground truth.'
    )
    parser.add_argument('--subject', default='dmnelf005',
                       help='Subject ID (default: dmnelf005)')
    parser.add_argument('--prediction-dir', default='cyclic_features_full',
                       help='Directory containing prediction files (must be full 66-feature data)')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--save', action='store_true',
                       help='Save plots and data to disk')
    
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
        print(f"\nMake sure you have downloaded the FULL 66-feature data from cluster.")
        print(f"See FETCH_CLUSTER_DATA.md for download instructions.")
        return
    
    print(f"Loading predictions from: {pred_path}")
    
    try:
        # Load and compute
        data = load_prediction_npz_full(pred_path)
        pda_pred = data['pda_pred']
        pda_true = data['pda_true']
        
        print(f"\n{'='*80}")
        print(f"Subject: {args.subject}")
        print(f"{'='*80}")
        print(f"fMRI features: {data['n_parcels']} (64 DiFuMo + 2 personal ROIs)")
        print(f"Timepoints: {data['n_timepoints']} ({data['n_timepoints'] * cfg['data']['fmri']['tr']:.1f} seconds = {data['n_timepoints'] * cfg['data']['fmri']['tr'] / 60:.1f} minutes)")
        print(f"\nModel predictions:")
        print(f"  Original pda_predicted: {data['n_original_pred']} points")
        print(f"  Upsampled to: {data['n_upsampled']} points (cubic interpolation)")
        print(f"  DMN ROI index: {data['dmn_idx']}, CEN ROI index: {data['cen_idx']}")
        
        # Plot
        if args.save:
            output_dir = Path('results')
            output_dir.mkdir(exist_ok=True, parents=True)
            plot_path = output_dir / f'{args.subject}_feedback_full_528_comparison.png'
            fig, metrics = plot_timeseries_full(pda_true, pda_pred, args.subject, save_path=plot_path)
            save_csv_data(pda_true, pda_pred, args.subject)
        else:
            fig, metrics = plot_timeseries_full(pda_true, pda_pred, args.subject)
            plt.show()
        
        # Print metrics
        print(f"\n{'='*80}")
        print(f"METRICS (Full 528 timepoints = {len(pda_true) * cfg['data']['fmri']['tr']:.1f} sec)")
        print(f"{'='*80}")
        print(f"Pearson r   = {metrics['r']:7.4f} (p = {metrics['p']:.3e})")
        print(f"R²          = {metrics['r_squared']:7.4f}")
        print(f"MAE         = {metrics['mae']:7.4f}")
        print(f"RMSE        = {metrics['rmse']:7.4f}")
        print(f"ROC AUC     = {metrics['auc']:7.4f}")
        print(f"{'='*80}\n")
        
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        return


if __name__ == '__main__':
    main()
