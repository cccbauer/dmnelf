#!/usr/bin/env python3
"""
plot_methods_rich.py — Publication-quality methods figure with real data panels.

Panels:
  A. DMN / CEN masks on MNI brain slice
  B. Raw EEG snippet (multi-channel)
  C. Band-power features (heatmap, 31ch × 5 bands, one TR window)
  D. HRF kernel
  E. fMRI target timeseries (DMN, CEN, PDA for one run)
  F. Predicted vs true timeseries (best subject, best run)
  G. Pipeline arrows connecting everything
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import mne
import nibabel as nib

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

DATA = Path.home() / "Documents/GitHub/dmnelf/data/DMNELF/derivatives"
PROJ = Path(__file__).resolve().parent.parent
MASKS = Path.home() / "Documents/GitHub/dmnelf/murfi-rt-PyProject/scripts"
CACHE = PROJ / "results" / "multivariate_cluster" / "cache"
RESULTS_5F = PROJ / "results" / "multivariate_cluster"
FIG_DIR = PROJ / "results" / "figures"

SUB = "dmnelf008"  # best decoder
RUN = 1
TR = 1.2
BANDS = {"delta": [1, 4], "theta": [4, 8], "alpha": [8, 13], "beta": [13, 30], "gamma": [30, 40]}
BAND_NAMES = list(BANDS.keys())
BAND_LABELS = ["δ", "θ", "α", "β", "γ"]

SUBJECTS = [
    "dmnelf001", "dmnelf004", "dmnelf005", "dmnelf006", "dmnelf007", "dmnelf008",
    "dmnelf009", "dmnelf010", "dmnelf011", "dmnelf012", "dmnelf013", "dmnelf014",
    "dmnelf015", "dmnelf1001", "dmnelf1002", "dmnelf1003",
]
CH_NAMES = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz", "Oz",
            "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5", "CP6",
            "TP9", "TP10", "POz"]


def load_eeg_snippet():
    """Load ~10s of raw EEG for display."""
    eeg_dir = DATA / "eeg_preprocessed" / f"sub-{SUB}" / "ses-dmnelf" / "eeg"
    fif = eeg_dir / f"sub-{SUB}_ses-dmnelf_task-feedback_run-0{RUN}_desc-preproc500Hz_eeg.fif"
    if not fif.exists():
        fif = list(eeg_dir.glob(f"*feedback*run-0{RUN}*preproc500Hz_eeg.fif"))[0]
    raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw.get_data(picks=picks)  # (n_ch, n_samples)
    sfreq = raw.info["sfreq"]
    ch_names = [raw.ch_names[p] for p in picks]
    # Take 10 seconds starting at t=30s
    start = int(30 * sfreq)
    end = int(40 * sfreq)
    return data[:, start:end], sfreq, ch_names


def load_fmri_targets():
    """Load DMN, CEN, PDA timeseries for one run."""
    feat_dir = DATA / "cyclic_features" / f"sub-{SUB}"
    npz = np.load(feat_dir / f"sub-{SUB}_task-feedback_run-{RUN}_features.npz")
    fmri = npz["fmri_features"]  # (125, 66)
    dmn = fmri[:, 64]
    cen = fmri[:, 65]
    pda = cen - dmn

    # Load confounds for global signal
    conf_dir = DATA / "fmriprep_25.2.5_fmap" / f"sub-{SUB}" / "ses-dmnelf" / "func"
    tsv = conf_dir / f"sub-{SUB}_ses-dmnelf_task-feedback_run-0{RUN}_desc-confounds_timeseries.tsv"
    df = pd.read_csv(tsv, sep="\t")
    gs = df["global_signal"].values.astype(float)
    gs[0] = gs[1]
    return dmn, cen, pda, gs


def load_group_weight_matrix(target="GSR_CEN"):
    """Load group mean ElasticNet weight matrix (n_ch, n_bands)."""
    n_ch = len(CH_NAMES)
    n_bands = len(BAND_NAMES)
    all_w = []
    for sub in SUBJECTS:
        wf = RESULTS_5F / f"{sub}_{target}_elasticnet_weights.npz"
        if wf.exists():
            w = np.load(wf, allow_pickle=True)
            all_w.append(w["coefs"].mean(axis=0))
    all_w = np.array(all_w)
    group_mean = all_w.mean(axis=0).reshape(n_bands, n_ch).T  # (n_ch, n_bands)
    return group_mean


def canonical_hrf(tr=1.2, length_s=32):
    """Double-gamma HRF."""
    from scipy.stats import gamma as gamma_dist
    t = np.arange(0, length_s, tr)
    h = gamma_dist.pdf(t, 6) - 0.35 * gamma_dist.pdf(t, 16)
    return h / h.sum()


def plot_brain_masks(ax):
    """Plot DMN and CEN masks on axial MNI slices (3 slices stacked)."""
    dmn_img = nib.load(MASKS / "DMNa_brainmaskero2.nii")
    cen_img = nib.load(MASKS / "CENa_brainmaskero2.nii")
    bold_img = nib.load(
        DATA / "fmriprep_25.2.5_fmap" / f"sub-{SUB}" / "ses-dmnelf" / "func" /
        f"sub-{SUB}_ses-dmnelf_task-feedback_run-0{RUN}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
    )

    bold_data = bold_img.get_fdata()[:, :, :, 0]
    dmn_data = dmn_img.get_fdata()
    cen_data = cen_img.get_fdata()

    ax.axis("off")
    slices = [33, 48, 56]
    n_sl = len(slices)

    for i, z_slice in enumerate(slices):
        bg = bold_data[:, :, z_slice].T
        dmn_sl = dmn_data[:, :, z_slice].T
        cen_sl = cen_data[:, :, z_slice].T

        # Normalize background for better contrast
        p2, p98 = np.percentile(bg[bg > 0], [2, 98]) if np.any(bg > 0) else (0, 1)
        bg_norm = np.clip((bg - p2) / (p98 - p2 + 1e-6), 0, 1)

        pos = ax.get_position()
        h_each = pos.height / n_sl * 0.9
        gap = pos.height / n_sl * 0.1
        y_bot = pos.y0 + (n_sl - 1 - i) * (h_each + gap)
        inax = ax.figure.add_axes([pos.x0 + 0.02, y_bot, pos.width * 0.85, h_each])
        inax.imshow(bg_norm, cmap="gray", origin="lower", aspect="equal", vmin=0, vmax=1)
        # Solid-color overlays
        if dmn_sl.max() > 0:
            overlay_dmn = np.zeros((*dmn_sl.shape, 4))
            overlay_dmn[dmn_sl > 0] = [0.29, 0.53, 0.78, 0.6]  # blue
            inax.imshow(overlay_dmn, origin="lower", aspect="equal")
        if cen_sl.max() > 0:
            overlay_cen = np.zeros((*cen_sl.shape, 4))
            overlay_cen[cen_sl > 0] = [0.91, 0.58, 0.23, 0.6]  # orange
            inax.imshow(overlay_cen, origin="lower", aspect="equal")
        inax.set_xticks([])
        inax.set_yticks([])
        inax.spines[:].set_visible(False)

    ax.set_title("fMRI ROIs", fontsize=10, fontweight="bold", pad=5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#4a86c8", alpha=0.6, label="DMN"),
                       Patch(facecolor="#e8943a", alpha=0.6, label="CEN")],
              loc="upper center", fontsize=7, framealpha=0.9,
              bbox_to_anchor=(0.5, 1.0))


def plot_eeg_traces(ax, data, sfreq, ch_names):
    """Plot multi-channel EEG waterfall."""
    n_ch = min(data.shape[0], 15)
    # Show 5 seconds for clarity
    n_samp = int(5 * sfreq)
    data = data[:, :n_samp]
    t = np.arange(data.shape[1]) / sfreq
    subset_idx = np.linspace(0, data.shape[0] - 1, n_ch, dtype=int)

    # Auto-scale spacing
    traces_uv = data[subset_idx] * 1e6
    spacing = np.percentile(np.abs(traces_uv), 95) * 2.5

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_ch))
    for i, ci in enumerate(subset_idx):
        trace = data[ci] * 1e6
        ax.plot(t, trace + i * spacing, linewidth=0.5, color=colors[i], alpha=0.9)
    ax.set_yticks([i * spacing for i in range(n_ch)])
    ax.set_yticklabels([ch_names[ci] for ci in subset_idx], fontsize=6)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_xlim(0, t[-1])
    ax.set_title("EEG (31 ch, 500 Hz)", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_weight_matrix(ax, weight_matrix):
    """Plot group mean ElasticNet weight matrix as heatmap."""
    vmax = np.max(np.abs(weight_matrix)) * 1.05
    im = ax.imshow(weight_matrix, cmap="RdBu_r", aspect="auto", interpolation="nearest",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(BAND_LABELS)))
    ax.set_xticklabels(BAND_LABELS, fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(CH_NAMES)))
    ax.set_yticklabels(CH_NAMES, fontsize=5.5)
    ax.set_title("ElasticNet Weights\n(31 ch × 5 bands)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Channel", fontsize=8)
    cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
    cb.set_label("weight", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    # Highlight F4 alpha (top feature)
    f4_idx = CH_NAMES.index("F4")
    alpha_idx = BAND_NAMES.index("alpha")
    rect = plt.Rectangle((alpha_idx - 0.5, f4_idx - 0.5), 1, 1,
                          linewidth=2.5, edgecolor="#00cc00", facecolor="none",
                          zorder=10)
    ax.add_patch(rect)
    ax.annotate("F4 α\n(top feature)", xy=(alpha_idx + 0.5, f4_idx),
                xytext=(alpha_idx + 2.2, f4_idx - 5),
                fontsize=7, fontweight="bold", color="#006600",
                arrowprops=dict(arrowstyle="-|>", color="#006600", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#006600", alpha=0.9),
                zorder=11)


def plot_hrf(ax):
    """Plot HRF kernel."""
    hrf = canonical_hrf(tr=0.05, length_s=32)
    t = np.arange(len(hrf)) * 0.05
    ax.fill_between(t, hrf, alpha=0.3, color="#e07b7b")
    ax.plot(t, hrf, color="#990000", linewidth=2)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_title("HRF convolution", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 30)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([])


def plot_fmri_timeseries(ax, dmn, cen, pda, gs):
    """Plot fMRI target timeseries for one run."""
    t = np.arange(len(dmn)) * TR
    z = lambda x: (x - x.mean()) / x.std()
    ax.plot(t, z(dmn), label="DMN", color="#4a86c8", linewidth=1.5, alpha=0.85)
    ax.plot(t, z(cen), label="CEN", color="#e8943a", linewidth=1.5, alpha=0.85)
    ax.plot(t, z(pda) - 5, label="PDA (CEN−DMN)", color="#38761d", linewidth=1.8)
    ax.axhline(-5, color="#38761d", linewidth=0.3, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.3, alpha=0.3)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_title("fMRI Targets (1 run)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([])
    ax.set_xlim(0, t[-1])
    ax.text(t[-1] + 2, 0, "z-scored", fontsize=7, color="gray", va="center", rotation=90)


def plot_predicted_vs_true(ax):
    """Plot predicted vs true for best subject, best run (GSR_CEN, ElasticNet)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bandpower import load_config, canonical_hrf as bp_hrf, gather_subject, zscore
    from multivariate_decode_pda import (
        load_subject_data, load_confounds_run, prepare_targets,
        car_and_flatten, contiguous_folds, run_cv, make_model, CONFIG_PATH
    )

    cfg = load_config(str(PROJ / "config.yaml"))
    # Override to local paths
    cfg["data"]["eeg_preproc_dir"] = str(DATA / "eeg_preprocessed")
    cfg["data"]["features_dir"] = str(DATA / "cyclic_features")
    cfg["data"]["confounds_dir"] = str(DATA / "fmriprep_25.2.5_fmap")

    runs_data, confounds, ch_names = load_subject_data(cfg, SUB, CACHE)
    if runs_data is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    X, run_boundaries = car_and_flatten(runs_data, BAND_NAMES)
    targets = prepare_targets(runs_data, confounds, ["GSR_CEN"])
    y = targets["GSR_CEN"]

    folds = contiguous_folds(len(y), 5)
    model = make_model("elasticnet", 1.0)
    r_folds, pred, coefs = run_cv(X, y, model, folds)

    # Find best run
    from scipy.stats import pearsonr
    best_run_r = -1
    best_ri = 0
    for ri, (s, e) in enumerate(run_boundaries):
        r_run, _ = pearsonr(y[s:e], pred[s:e])
        if r_run > best_run_r:
            best_run_r = r_run
            best_ri = ri
    start, end = run_boundaries[best_ri]
    t = np.arange(end - start) * TR

    ax.plot(t, y[start:end], color="#2c5f8a", linewidth=1.5, label="True GSR'd CEN", alpha=0.85)
    ax.plot(t, pred[start:end], color="#cc0000", linewidth=1.5, label="Predicted", alpha=0.85)
    ax.fill_between(t, y[start:end], pred[start:end], alpha=0.08, color="gray")
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_title(f"EEG-predicted vs True fMRI   (GSR'd CEN, run {best_ri+1}, r = {best_run_r:.2f})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([])
    ax.set_xlim(0, t[-1])


def add_arrow(fig, xy_from, xy_to, **kwargs):
    """Add an arrow in figure coordinates."""
    defaults = dict(arrowstyle="-|>", color="black", lw=1.8,
                    mutation_scale=15, connectionstyle="arc3,rad=0")
    defaults.update(kwargs)
    arrow = FancyArrowPatch(xy_from, xy_to, transform=fig.transFigure,
                            **defaults)
    fig.patches.append(arrow)


def main():
    print("Loading data...")

    eeg_data, sfreq, eeg_ch_names = load_eeg_snippet()
    weight_matrix = load_group_weight_matrix("GSR_CEN")
    dmn, cen, pda, gs = load_fmri_targets()

    print("Building figure...")

    fig = plt.figure(figsize=(20, 12))

    # Layout: 3 rows
    # Row 0: [Brain masks] [EEG traces] [Band power heatmap] [HRF]
    # Row 1: [fMRI timeseries] [arrow: model box]
    # Row 2: [Predicted vs True — wide]
    gs_main = gridspec.GridSpec(3, 4, figure=fig,
                                height_ratios=[1.1, 0.9, 0.9],
                                hspace=0.45, wspace=0.35,
                                left=0.05, right=0.95, top=0.92, bottom=0.06)

    # Row 0
    ax_masks = fig.add_subplot(gs_main[0, 0])
    ax_eeg = fig.add_subplot(gs_main[0, 1:3])
    ax_bp = fig.add_subplot(gs_main[0, 3])

    # Row 1
    ax_fmri = fig.add_subplot(gs_main[1, 0:2])
    ax_hrf = fig.add_subplot(gs_main[1, 2])
    ax_model = fig.add_subplot(gs_main[1, 3])

    # Row 2
    ax_pred = fig.add_subplot(gs_main[2, :])

    # ── Populate panels ──
    print("  Panel A: Brain masks...")
    plot_brain_masks(ax_masks)

    print("  Panel B: EEG traces...")
    plot_eeg_traces(ax_eeg, eeg_data, sfreq, eeg_ch_names)

    print("  Panel C: ElasticNet weight matrix...")
    plot_weight_matrix(ax_bp, weight_matrix)

    print("  Panel D: fMRI timeseries...")
    plot_fmri_timeseries(ax_fmri, dmn, cen, pda, gs)

    print("  Panel E: HRF...")
    plot_hrf(ax_hrf)

    print("  Panel F: Model box...")
    ax_model.axis("off")
    model_box = dict(boxstyle="round,pad=0.6", facecolor="#fce5cd",
                     edgecolor="#b45f06", linewidth=2)
    ax_model.text(0.5, 0.7, "ElasticNet\nRegression", fontsize=13,
                  ha="center", va="center", bbox=model_box, fontweight="bold",
                  transform=ax_model.transAxes)
    val_box = dict(boxstyle="round,pad=0.4", facecolor="#f4cccc",
                   edgecolor="#990000", linewidth=1.5)
    ax_model.text(0.5, 0.2, "5-fold CV + LORO\n10K null permutations",
                  fontsize=9, ha="center", va="center", bbox=val_box,
                  transform=ax_model.transAxes)

    print("  Panel G: Predicted vs True...")
    plot_predicted_vs_true(ax_pred)

    # ── Panel labels ──
    for label, ax_target in [("A", ax_masks), ("B", ax_eeg), ("C", ax_bp),
                             ("D", ax_fmri), ("E", ax_hrf), ("F", ax_model),
                             ("G", ax_pred)]:
        pos = ax_target.get_position()
        fig.text(pos.x0 - 0.02, pos.y1 + 0.01, label, fontsize=16,
                 fontweight="bold", va="bottom", ha="right")

    # ── Connecting arrows ──
    # EEG → Band power
    eeg_pos = ax_eeg.get_position()
    bp_pos = ax_bp.get_position()
    add_arrow(fig, (eeg_pos.x1 + 0.005, eeg_pos.y0 + eeg_pos.height/2),
              (bp_pos.x0 - 0.005, bp_pos.y0 + bp_pos.height/2))

    # Weight matrix → HRF convolution
    hrf_pos = ax_hrf.get_position()
    bp_bottom = (bp_pos.x0 + bp_pos.width/2, bp_pos.y0 - 0.01)
    hrf_top = (hrf_pos.x0 + hrf_pos.width/2, hrf_pos.y1 + 0.01)
    add_arrow(fig, bp_bottom, hrf_top)
    mid_y = (bp_bottom[1] + hrf_top[1]) / 2
    fig.text(bp_bottom[0] + 0.02, mid_y, "HRF\nconv.", fontsize=7, color="#990000",
             fontweight="bold", va="center", ha="left", style="italic")

    # HRF → Model
    model_pos = ax_model.get_position()
    add_arrow(fig, (hrf_pos.x1 + 0.005, hrf_pos.y0 + hrf_pos.height * 0.6),
              (model_pos.x0 - 0.005, model_pos.y0 + model_pos.height * 0.7))

    # Masks → fMRI targets
    masks_pos = ax_masks.get_position()
    fmri_pos = ax_fmri.get_position()
    add_arrow(fig, (masks_pos.x0 + masks_pos.width/2, masks_pos.y0 - 0.01),
              (fmri_pos.x0 + fmri_pos.width * 0.15, fmri_pos.y1 + 0.01))

    # fMRI → Model
    add_arrow(fig, (fmri_pos.x1 + 0.005, fmri_pos.y0 + fmri_pos.height/2),
              (hrf_pos.x0 - 0.005, hrf_pos.y0 + hrf_pos.height * 0.3),
              connectionstyle="arc3,rad=0.1")

    # Model → Predicted
    pred_pos = ax_pred.get_position()
    add_arrow(fig, (model_pos.x0 + model_pos.width/2, model_pos.y0 - 0.01),
              (pred_pos.x0 + pred_pos.width * 0.75, pred_pos.y1 + 0.01),
              connectionstyle="arc3,rad=-0.15")

    fig.suptitle(f"Methods: Multivariate EEG Band Power → fMRI Network Decoding   "
                 f"(sub-{SUB}, run {RUN})",
                 fontsize=15, fontweight="bold", y=0.97)

    out = FIG_DIR / "methods_rich.png"
    fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
