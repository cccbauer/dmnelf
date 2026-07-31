#!/usr/bin/env python3
"""
plot_topomaps.py
----------------
Topographic maps of Ridge model weights: which channels x bands drive decoding.
Group-level significance via one-sample t-test across subjects (FDR corrected).

Usage:
  python plot_topomaps.py                          # GSR_CEN, 5-fold, best/worst + group
  python plot_topomaps.py --target PDA --loro      # PDA, LORO
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import mne
from scipy.stats import ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bandpower import load_config, zscore
from multivariate_decode_pda import (load_subject_data, load_confounds_run,
    prepare_targets, car_and_flatten, contiguous_folds, loro_folds,
    run_cv, make_model, CONFIG_PATH)

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")


def get_eeg_info(cfg):
    eeg_dir = Path(cfg["data"]["eeg_preproc_dir"])
    fif = next(eeg_dir.glob("sub-dmnelf001/ses-dmnelf/eeg/*preproc500Hz_eeg.fif"))
    raw = mne.io.read_raw_fif(str(fif), preload=False, verbose=False)
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    return raw.info, picks


def get_subject_weights(cfg, sub, target, cache_dir, band_names, use_loro):
    runs_data, confounds, ch_names = load_subject_data(cfg, sub, cache_dir)
    if runs_data is None:
        return None, None, None

    X, run_boundaries = car_and_flatten(runs_data, band_names)
    targets_dict = prepare_targets(runs_data, confounds, [target])
    y = targets_dict[target]

    folds = loro_folds(run_boundaries) if use_loro else contiguous_folds(len(y), 5)
    model = make_model("ridge", 1.0)
    r_folds, pred, coefs = run_cv(X, y, model, folds)

    n_ch = len(ch_names)
    n_bands = len(band_names)
    w = coefs.reshape(coefs.shape[0], n_bands, n_ch)
    w_abs = np.mean(np.abs(w), axis=0)
    w_signed = np.mean(w, axis=0)

    return w_abs, w_signed, np.mean(r_folds)


def fdr_correct(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns boolean mask of significant tests."""
    pvals = np.asarray(pvals).ravel()
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    thresholds = alpha * np.arange(1, n + 1) / n
    below = np.where(sorted_p <= thresholds)[0]
    if len(below) == 0:
        return np.zeros_like(pvals, dtype=bool)
    max_idx = below[-1]
    sig = np.zeros(n, dtype=bool)
    sig[sorted_idx[:max_idx + 1]] = True
    return sig


def add_colorbar(fig, ax, cmap, vmin, vmax, label=""):
    """Add a small colorbar next to an axis."""
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.005, pos.y0 + 0.05, 0.008, pos.height - 0.1])
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=7)
    if label:
        cb.set_label(label, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--target", default="GSR_CEN", choices=["PDA","GSR_DMN","GSR_CEN","RAW_DMN"])
    ap.add_argument("--loro", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    subjects = cfg["data"]["subjects"]["all"]
    band_names = list(cfg["bands"].keys())
    n_bands = len(band_names)
    n_ch = 31

    proj_dir = CONFIG_PATH.parent
    cv_label = "loro" if args.loro else "5fold"
    results_subdir = "multivariate_loro" if args.loro else "multivariate"
    results_dir = proj_dir / "results" / results_subdir
    cache_dir = proj_dir / "results" / "multivariate" / "cache"
    fig_dir = proj_dir / "results" / "figures"

    info, picks = get_eeg_info(cfg)
    info_pick = mne.pick_info(info, picks)

    # Find best and worst
    rs = []
    for sub in subjects:
        f = results_dir / f"{sub}_{args.target}_ridge.csv"
        if f.exists():
            rs.append((sub, pd.read_csv(f)["mean_cv_r"].iloc[0]))
    rs.sort(key=lambda x: x[1], reverse=True)
    best_sub, best_r = rs[0]
    worst_sub, worst_r = rs[-1]
    print(f"Best: {best_sub} r={best_r:.3f}, Worst: {worst_sub} r={worst_r:.3f}")

    # Collect all subjects' signed weights
    print("  Computing weights for all subjects...")
    all_w_signed = []
    all_w_abs = []
    for sub in subjects:
        w_abs, w_signed, _ = get_subject_weights(cfg, sub, args.target, cache_dir, band_names, args.loro)
        if w_abs is not None:
            all_w_abs.append(w_abs)
            all_w_signed.append(w_signed)
    all_w_signed = np.array(all_w_signed)  # (n_subjects, n_bands, n_ch)
    all_w_abs = np.array(all_w_abs)
    n_sub = len(all_w_signed)

    # Group t-test at each (band, channel) — signed weights
    t_vals = np.zeros((n_bands, n_ch))
    p_vals = np.zeros((n_bands, n_ch))
    for bi in range(n_bands):
        for ci in range(n_ch):
            t, p = ttest_1samp(all_w_signed[:, bi, ci], 0)
            t_vals[bi, ci] = t
            p_vals[bi, ci] = p

    # Also for overall (sum across bands)
    overall_signed = np.sum(all_w_signed, axis=1)  # (n_sub, n_ch)
    t_overall = np.zeros(n_ch)
    p_overall = np.zeros(n_ch)
    for ci in range(n_ch):
        t, p = ttest_1samp(overall_signed[:, ci], 0)
        t_overall[ci] = t
        p_overall[ci] = p

    # FDR correction across all (5 bands * 31 ch) + 31 overall = 186 tests
    all_p = np.concatenate([p_vals.ravel(), p_overall])
    sig_mask_all = fdr_correct(all_p, alpha=0.05)
    sig_bands = sig_mask_all[:n_bands * n_ch].reshape(n_bands, n_ch)
    sig_overall = sig_mask_all[n_bands * n_ch:]

    n_sig = sig_mask_all.sum()
    print(f"  FDR-significant channels: {n_sig}/{len(all_p)} (alpha=0.05)")

    # Group means
    group_signed = np.mean(all_w_signed, axis=0)
    group_abs = np.mean(all_w_abs, axis=0)

    # ── Figure: Group signed weights with t-stats and significance ──
    fig, axes = plt.subplots(2, n_bands + 1, figsize=(18, 7),
                             gridspec_kw={"hspace": 0.4, "wspace": 0.15})

    # Row 1: group mean signed weights
    vmax_w = np.max(np.abs(group_signed)) * 1.1
    for bi, bname in enumerate(band_names):
        ax = axes[0, bi]
        mask = sig_bands[bi]
        mne.viz.plot_topomap(group_signed[bi], info_pick, axes=ax, show=False,
                             cmap="RdBu_r", vlim=(-vmax_w, vmax_w),
                             sensors=True, contours=0,
                             mask=mask, mask_params=dict(marker="*", markerfacecolor="lime",
                                                         markeredgecolor="green", markersize=10))
        ax.set_title(bname, fontsize=12)
    # Overall signed
    ax = axes[0, n_bands]
    overall_mean = np.mean(overall_signed, axis=0)
    ov_vmax = np.max(np.abs(overall_mean)) * 1.1
    mne.viz.plot_topomap(overall_mean, info_pick, axes=ax, show=False,
                         cmap="RdBu_r", vlim=(-ov_vmax, ov_vmax),
                         sensors=True, contours=0,
                         mask=sig_overall, mask_params=dict(marker="o", markerfacecolor="black",
                                                            markeredgecolor="black", markersize=6))
    ax.set_title("overall", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("signed\nweight", fontsize=11, fontweight="bold")
    add_colorbar(fig, axes[0, -1], "RdBu_r", -vmax_w, vmax_w, "weight")

    # Row 2: t-statistics
    vmax_t = np.max(np.abs(t_vals)) * 0.9
    for bi, bname in enumerate(band_names):
        ax = axes[1, bi]
        mask = sig_bands[bi]
        mne.viz.plot_topomap(t_vals[bi], info_pick, axes=ax, show=False,
                             cmap="RdBu_r", vlim=(-vmax_t, vmax_t),
                             sensors=True, contours=0,
                             mask=mask, mask_params=dict(marker="*", markerfacecolor="lime",
                                                         markeredgecolor="green", markersize=10))
        ax.set_title(bname, fontsize=12)
    ax = axes[1, n_bands]
    ov_tvmax = np.max(np.abs(t_overall)) * 0.9
    mne.viz.plot_topomap(t_overall, info_pick, axes=ax, show=False,
                         cmap="RdBu_r", vlim=(-ov_tvmax, ov_tvmax),
                         sensors=True, contours=0,
                         mask=sig_overall, mask_params=dict(marker="o", markerfacecolor="black",
                                                            markeredgecolor="black", markersize=6))
    ax.set_title("overall", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("t-statistic\n(n=16)", fontsize=11, fontweight="bold")
    add_colorbar(fig, axes[1, -1], "RdBu_r", -vmax_t, vmax_t, "t")

    target_label = args.target.replace("_", " ")
    fig.suptitle(f"Group Ridge Weights — {target_label} ({cv_label.upper()}, n={n_sub})\n"
                 f"Top: mean signed weight  |  Bottom: t-statistic  |  "
                 f"Green * = FDR-significant (p<0.05, {n_sig} of {len(all_p)} tests)",
                 fontsize=14, fontweight="bold")
    out_group = fig_dir / f"topomap_group_{args.target}_{cv_label}.png"
    fig.savefig(out_group, dpi=200, bbox_inches="tight")
    print(f"Saved {out_group.name}")

    # ── Figure: Best vs Worst (magnitude, with colorbars) ──
    fig, axes = plt.subplots(2, n_bands + 1, figsize=(18, 6),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.15})

    for row, (sub, label, r_val) in enumerate([
        (best_sub, "Best", best_r), (worst_sub, "Worst", worst_r)
    ]):
        idx = subjects.index(sub)
        w_abs = all_w_abs[idx]
        vmax = np.max(w_abs) * 1.1

        for bi, bname in enumerate(band_names):
            ax = axes[row, bi]
            mne.viz.plot_topomap(w_abs[bi], info_pick, axes=ax, show=False,
                                 cmap="hot", vlim=(0, vmax), sensors=True, contours=0)
            ax.set_title(bname if row == 0 else "", fontsize=12)

        overall = np.sum(w_abs, axis=0)
        ax = axes[row, n_bands]
        mne.viz.plot_topomap(overall, info_pick, axes=ax, show=False,
                             cmap="hot", vlim=(0, np.max(overall) * 1.1), sensors=True, contours=0)
        ax.set_title("overall" if row == 0 else "", fontsize=12, fontweight="bold")
        add_colorbar(fig, ax, "hot", 0, vmax, "|w|")

        short = sub.replace("dmnelf", "")
        axes[row, 0].set_ylabel(f"{label} ({short})\nr={r_val:+.3f}", fontsize=11, fontweight="bold")

    fig.suptitle(f"Ridge |Weight| Topomaps — {target_label} ({cv_label.upper()})\n"
                 f"Mean |weight| across CV folds per band × channel",
                 fontsize=14, fontweight="bold")
    out_bw = fig_dir / f"topomap_best_worst_{args.target}_{cv_label}.png"
    fig.savefig(out_bw, dpi=200, bbox_inches="tight")
    print(f"Saved {out_bw.name}")


if __name__ == "__main__":
    main()
