#!/usr/bin/env python
"""
summarize_results.py
--------------------
Generate a Prediction Performance Summary from evaluation_results.csv

This script reads the evaluation results downloaded from Explorer and generates:
  - Group-level statistics (mean, std, range)
  - Per-subject ranking
  - Performance tiers (excellent, good, poor)
  - Visual report

Usage:
    python summarize_results.py
    python summarize_results.py --csv evaluation_results.csv
    python summarize_results.py --detailed
    python summarize_results.py --visualize
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_results(csv_path):
    """Load evaluation results CSV."""
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)
    
    if HAS_PANDAS:
        return pd.read_csv(csv_path)
    else:
        # Minimal CSV parsing if pandas unavailable
        with open(csv_path) as f:
            lines = f.readlines()
        headers = lines[0].strip().split(',')
        rows = []
        for line in lines[1:]:
            vals = line.strip().split(',')
            rows.append(dict(zip(headers, vals)))
        return rows


def compute_group_stats(df):
    """Compute group-level statistics."""
    if HAS_PANDAS:
        metrics = {
            'pearson_r': df['pearson_r'].values,
            'spearman_rho': df['spearman_rho'].values,
            'r2': df['r2'].values,
            'rmse': df['rmse'].values,
            'mae': df['mae'].values,
            'roc_auc': df['roc_auc'].values,
        }
    else:
        metrics = {
            'pearson_r': [float(row['pearson_r']) for row in df],
            'spearman_rho': [float(row['spearman_rho']) for row in df],
            'r2': [float(row['r2']) for row in df],
            'rmse': [float(row['rmse']) for row in df],
            'mae': [float(row['mae']) for row in df],
            'roc_auc': [float(row['roc_auc']) for row in df],
        }
    
    stats = {}
    for metric_name, values in metrics.items():
        values_arr = np.array(values)
        stats[metric_name] = {
            'mean': np.nanmean(values_arr),
            'std': np.nanstd(values_arr),
            'min': np.nanmin(values_arr),
            'max': np.nanmax(values_arr),
            'median': np.nanmedian(values_arr),
        }
    
    return stats


def categorize_performance(r):
    """Categorize correlation performance."""
    if np.isnan(r):
        return "unknown"
    elif r >= 0.4:
        return "excellent"
    elif r >= 0.2:
        return "good"
    elif r >= 0:
        return "fair"
    elif r >= -0.2:
        return "poor"
    else:
        return "very_poor"


def print_summary(df, detailed=False):
    """Print formatted summary."""
    print("\n" + "="*80)
    print("CYCLIC TRANSCODER — PREDICTION PERFORMANCE SUMMARY")
    print("="*80)
    
    if HAS_PANDAS:
        n_subjects = len(df)
        n_complete = df[['pearson_r', 'r2', 'rmse']].notna().all(axis=1).sum()
    else:
        n_subjects = len(df)
        n_complete = sum(1 for row in df if row['pearson_r'] != 'nan' and row['r2'] != 'nan')
    
    # Extract metrics
    if HAS_PANDAS:
        pearson_rs = df['pearson_r'].values
        r2s = df['r2'].values
        rmses = df['rmse'].values
        maes = df['mae'].values
    else:
        pearson_rs = np.array([float(row['pearson_r']) for row in df])
        r2s = np.array([float(row['r2']) for row in df])
        rmses = np.array([float(row['rmse']) for row in df])
        maes = np.array([float(row['mae']) for row in df])
    
    # Group-level statistics
    print("\n" + "─"*80)
    print("GROUP-LEVEL STATISTICS")
    print("─"*80)
    
    print(f"\nTotal subjects: {n_subjects}")
    print(f"With complete metrics: {n_complete}\n")
    
    print(f"Pearson Correlation (r):")
    print(f"  Mean:   {np.nanmean(pearson_rs):7.4f}")
    print(f"  Std:    {np.nanstd(pearson_rs):7.4f}")
    print(f"  Median: {np.nanmedian(pearson_rs):7.4f}")
    print(f"  Range:  [{np.nanmin(pearson_rs):7.4f}, {np.nanmax(pearson_rs):7.4f}]")
    
    print(f"\nVariance Explained (R²):")
    print(f"  Mean:   {np.nanmean(r2s):7.4f}")
    print(f"  Std:    {np.nanstd(r2s):7.4f}")
    print(f"  Median: {np.nanmedian(r2s):7.4f}")
    print(f"  Range:  [{np.nanmin(r2s):7.4f}, {np.nanmax(r2s):7.4f}]")
    
    print(f"\nRMSE:")
    print(f"  Mean:   {np.nanmean(rmses):7.4f}")
    print(f"  Std:    {np.nanstd(rmses):7.4f}")
    print(f"  Median: {np.nanmedian(rmses):7.4f}")
    print(f"  Range:  [{np.nanmin(rmses):7.4f}, {np.nanmax(rmses):7.4f}]")
    
    print(f"\nMAE:")
    print(f"  Mean:   {np.nanmean(maes):7.4f}")
    print(f"  Std:    {np.nanstd(maes):7.4f}")
    print(f"  Median: {np.nanmedian(maes):7.4f}")
    print(f"  Range:  [{np.nanmin(maes):7.4f}, {np.nanmax(maes):7.4f}]")
    
    # Performance tiers
    print("\n" + "─"*80)
    print("PERFORMANCE TIERS")
    print("─"*80)
    
    if HAS_PANDAS:
        subjects = df['subject'].values
        correlations = df['pearson_r'].values
    else:
        subjects = [row['subject'] for row in df]
        correlations = [float(row['pearson_r']) for row in df]
    
    tiers = defaultdict(list)
    for subj, r in zip(subjects, correlations):
        tier = categorize_performance(r)
        tiers[tier].append((subj, r))
    
    for tier in ["excellent", "good", "fair", "poor", "very_poor"]:
        if tier in tiers:
            print(f"\n{tier.upper()} (r >= {['0.4', '0.2', '0', '-0.2', '-∞'][['excellent', 'good', 'fair', 'poor', 'very_poor'].index(tier)]}):")
            for subj, r in sorted(tiers[tier], key=lambda x: -x[1]):
                print(f"  {subj:12s}  r = {r:7.4f}")
    
    # Best and worst
    print("\n" + "─"*80)
    print("RANKING")
    print("─"*80)
    
    if HAS_PANDAS:
        sorted_df = df.sort_values('pearson_r', ascending=False)
        top_n = min(5, len(sorted_df))
        
        print(f"\nTop {top_n} performers:")
        for i, (_, row) in enumerate(sorted_df.head(top_n).iterrows(), 1):
            print(f"  {i}. {row['subject']:12s}  r = {row['pearson_r']:7.4f}  R² = {row['r2']:7.4f}")
        
        print(f"\nBottom {top_n} performers:")
        for i, (_, row) in enumerate(sorted_df.tail(top_n).iterrows(), 1):
            print(f"  {i}. {row['subject']:12s}  r = {row['pearson_r']:7.4f}  R² = {row['r2']:7.4f}")
    else:
        sorted_rows = sorted(zip(subjects, correlations), key=lambda x: -x[1])
        top_n = min(5, len(sorted_rows))
        
        print(f"\nTop {top_n} performers:")
        for i, (subj, r) in enumerate(sorted_rows[:top_n], 1):
            print(f"  {i}. {subj:12s}  r = {r:7.4f}")
        
        print(f"\nBottom {top_n} performers:")
        for i, (subj, r) in enumerate(sorted_rows[-top_n:], 1):
            print(f"  {i}. {subj:12s}  r = {r:7.4f}")
    
    # Interpretation
    print("\n" + "─"*80)
    print("INTERPRETATION")
    print("─"*80)
    
    n_positive = np.sum(pearson_rs > 0)
    pct_positive = 100 * n_positive / n_subjects
    
    print(f"\nPositive correlations: {n_positive}/{n_subjects} ({pct_positive:.1f}%)")
    
    if np.nanmean(pearson_rs) > 0.3:
        quality = "Good — Model generalizes well across subjects"
    elif np.nanmean(pearson_rs) > 0.1:
        quality = "Moderate — Model shows some predictive ability"
    elif np.nanmean(pearson_rs) > -0.1:
        quality = "Poor — Model predictions near random"
    else:
        quality = "Very Poor — Model may be overfitting or data issues"
    
    print(f"Overall quality: {quality}")
    
    if np.nanstd(pearson_rs) > 0.3:
        consistency = "High variability — Performance differs greatly across subjects"
    else:
        consistency = "Consistent — Performance stable across subjects"
    
    print(f"Consistency: {consistency}")


def visualize_results(df, output_dir=None, result_tag=""):
    """Generate visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[NOTE] matplotlib not available; skipping visualizations")
        return
    
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if HAS_PANDAS:
        subjects = df['subject'].values
        pearson_rs = df['pearson_r'].values
        r2s = df['r2'].values
    else:
        subjects = [row['subject'] for row in df]
        pearson_rs = np.array([float(row['pearson_r']) for row in df])
        r2s = np.array([float(row['r2']) for row in df])
    
    # Bar plot of correlations
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['green' if r > 0.2 else 'orange' if r > 0 else 'red' for r in pearson_rs]
    ax.barh(subjects, pearson_rs, color=colors, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(np.nanmean(pearson_rs), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.nanmean(pearson_rs):.3f}')
    ax.set_xlabel('Pearson Correlation')
    ax.set_title('Per-Subject Prediction Correlation')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_name = 'summary_correlations.png'
    if result_tag:
        plot_name = f'summary_correlations_{result_tag}.png'
    plot_path = output_dir / plot_name
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='results/evaluation_results.csv', help='CSV results file')
    parser.add_argument('--detailed', action='store_true', help='Show detailed stats')
    parser.add_argument('--visualize', action='store_true', help='Generate plots')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for plots (default: results/)')
    parser.add_argument('--result-tag', type=str, default='', help='Optional suffix tag for output artifact names')
    args = parser.parse_args()
    
    df = load_results(args.csv)
    print_summary(df, detailed=args.detailed)
    
    if args.visualize:
        visualize_results(df, output_dir=args.output_dir, result_tag=args.result_tag)
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
