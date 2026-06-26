#!/usr/bin/env python3
"""
plot_results.py — Generate all PPT figures from pulled cluster results.
Reads CSVs + weight .npz files only (no raw EEG data needed).

Figures generated:
  1. group_bar_ridge_vs_enet.png      — Ridge vs ElasticNet bar plot (5-fold + LORO)
  2. subject_heatmap_enet.png         — Per-subject ElasticNet r heatmap (5-fold + LORO)
  3. topomap_enet_GSR_CEN.png         — ElasticNet weight topomaps for GSR_CEN
  4. topomap_enet_PDA.png             — ElasticNet weight topomaps for PDA
  5. methods_diagram.png              — Methods pipeline schematic
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import mne

mne.set_log_level("ERROR")

PROJ = Path(__file__).resolve().parent.parent
RESULTS_5F = PROJ / "results" / "multivariate_cluster"
RESULTS_LORO = PROJ / "results" / "multivariate_loro_cluster"
FIG_DIR = PROJ / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = [
    "dmnelf001", "dmnelf004", "dmnelf005", "dmnelf006", "dmnelf007", "dmnelf008",
    "dmnelf009", "dmnelf010", "dmnelf011", "dmnelf012", "dmnelf013", "dmnelf014",
    "dmnelf015", "dmnelf016", "dmnelf1001", "dmnelf1002", "dmnelf1003",
]
TARGETS = ["RAW_DMN", "GSR_DMN", "GSR_CEN", "PDA"]
TARGET_LABELS = {"RAW_DMN": "RAW\nDMN", "GSR_DMN": "GSR'd\nDMN",
                 "GSR_CEN": "GSR'd\nCEN", "PDA": "PDA\n(CEN−DMN)"}
TARGET_COLORS = {"RAW_DMN": "#e07b7b", "GSR_DMN": "#6fa8dc",
                 "GSR_CEN": "#f6b26b", "PDA": "#93c47d"}
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_LABELS = ["δ (1–4)", "θ (4–8)", "α (8–13)", "β (13–30)", "γ (30–40)"]
CH_NAMES = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz", "Oz",
            "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5", "CP6",
            "TP9", "TP10", "POz"]


def load_r_values(results_dir, model):
    """Load per-subject r values from CSVs. Returns dict target -> list of (sub, r, p)."""
    out = {}
    for tname in TARGETS:
        vals = []
        for sub in SUBJECTS:
            f = results_dir / f"{sub}_{tname}_{model}.csv"
            if f.exists():
                df = pd.read_csv(f)
                vals.append((sub, df["mean_cv_r"].iloc[0], df["p_circ_shift"].iloc[0]))
        out[tname] = vals
    return out


def sign_flip_test(r_vals, n_flips=10000):
    r_arr = np.array(r_vals)
    n = len(r_arr)
    obs_mean = np.mean(r_arr)
    obs_t = obs_mean / (np.std(r_arr, ddof=1) / np.sqrt(n))
    rng = np.random.default_rng(42)
    null = np.zeros(n_flips)
    for i in range(n_flips):
        null[i] = np.mean(rng.choice([-1, 1], size=n) * r_arr)
    p_right = (np.sum(null >= obs_mean) + 1) / (n_flips + 1)
    p_two = 2 * min(p_right, (np.sum(null <= obs_mean) + 1) / (n_flips + 1))
    return obs_mean, obs_t, p_two


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    if p < 0.1: return "†"
    return "n.s."


def make_eeg_info():
    """Create MNE Info from standard 10-20 montage."""
    info = mne.create_info(CH_NAMES, sfreq=500, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)
    return info


# ── Figure 1: Group bar plot — Ridge vs ElasticNet, 5-fold + LORO ──

def plot_group_bars():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    bar_width = 0.35

    for ax_idx, (label, rdir) in enumerate([("5-fold contiguous CV", RESULTS_5F),
                                             ("LORO (leave-one-run-out)", RESULTS_LORO)]):
        ax = axes[ax_idx]
        x = np.arange(len(TARGETS))

        for mi, (model, offset, hatch, alpha) in enumerate([
            ("ridge", -bar_width/2, "", 0.7),
            ("elasticnet", bar_width/2, "//", 0.85),
        ]):
            rdata = load_r_values(rdir, model)
            means, ps = [], []
            for tname in TARGETS:
                rs = [v[1] for v in rdata[tname]]
                if len(rs) >= 3:
                    m, t, p = sign_flip_test(rs)
                else:
                    m, p = 0, 1
                means.append(m)
                ps.append(p)

            color = [TARGET_COLORS[t] for t in TARGETS]
            bars = ax.bar(x + offset, means, bar_width, color=color,
                          alpha=alpha, edgecolor="black", linewidth=0.8,
                          hatch=hatch, label=model.replace("elasticnet", "ElasticNet").capitalize())

            # Individual subject dots
            for ti, tname in enumerate(TARGETS):
                rs = [v[1] for v in rdata[tname]]
                jitter = np.random.default_rng(ti).uniform(-0.08, 0.08, len(rs))
                ax.scatter(x[ti] + offset + jitter, rs, s=15, color="gray",
                           alpha=0.5, zorder=5, edgecolors="none")

            # Significance stars
            for ti in range(len(TARGETS)):
                stars = sig_stars(ps[ti])
                y_pos = max(means[ti] + 0.02, 0.03)
                ax.text(x[ti] + offset, y_pos, stars, ha="center", va="bottom",
                        fontsize=11, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([TARGET_LABELS[t] for t in TARGETS], fontsize=10)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylim(-0.08, 0.62)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ax_idx == 0:
            # Arousal divider
            ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
            ax.text(-0.15, -0.06, "arousal\nincluded", fontsize=8, color="gray",
                    ha="center", style="italic")
            ax.text(1.5, -0.06, "arousal removed", fontsize=8, color="gray",
                    ha="center", style="italic")

    axes[0].set_ylabel("Mean CV correlation (r)", fontsize=12)
    axes[0].legend(loc="upper left", fontsize=10, framealpha=0.9)

    fig.suptitle("Multivariate EEG Band Power → fMRI Network Decoding\n"
                 "(n=17, group sign-flip test, 10K nulls)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = FIG_DIR / "group_bar_ridge_vs_enet.png"
    fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── Figure 2: Subject heatmap (ElasticNet, 5-fold + LORO) ──

def plot_subject_heatmap():
    fig, axes = plt.subplots(2, 1, figsize=(16, 5.5), gridspec_kw={"hspace": 0.5})

    for ax_idx, (cv_label, rdir) in enumerate([("5-fold", RESULTS_5F),
                                                 ("LORO", RESULTS_LORO)]):
        ax = axes[ax_idx]
        rdata = load_r_values(rdir, "elasticnet")
        sub_labels = [s.replace("dmnelf", "") for s in SUBJECTS]

        mat = np.full((len(TARGETS), len(SUBJECTS)), np.nan)
        pmat = np.full((len(TARGETS), len(SUBJECTS)), np.nan)
        for ti, tname in enumerate(TARGETS):
            for sub, r, p in rdata[tname]:
                si = SUBJECTS.index(sub)
                mat[ti, si] = r
                pmat[ti, si] = p

        vmax = max(0.6, np.nanmax(np.abs(mat)))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

        for ti in range(len(TARGETS)):
            for si in range(len(SUBJECTS)):
                if np.isnan(mat[ti, si]):
                    continue
                r = mat[ti, si]
                p = pmat[ti, si]
                color = "white" if abs(r) > 0.3 else "black"
                star = "*" if p < 0.05 else ""
                ax.text(si, ti, f"{r:.2f}{star}", ha="center", va="center",
                        fontsize=7.5, color=color, fontweight="bold" if star else "normal")

        ax.set_xticks(range(len(SUBJECTS)))
        ax.set_xticklabels(sub_labels, fontsize=9)
        ax.set_yticks(range(len(TARGETS)))
        ax.set_yticklabels([t.replace("_", " ") for t in TARGETS], fontsize=10, fontweight="bold")
        ax.set_title(f"ElasticNet — {cv_label}", fontsize=12, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("r", fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Subject", fontsize=11)
    fig.suptitle("Per-Subject CV Correlation (ElasticNet, * = p<0.05 circular-shift)",
                 fontsize=13, fontweight="bold", y=1.02)
    out = FIG_DIR / "subject_heatmap_enet.png"
    fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── Figure 3 & 4: ElasticNet weight topomaps ──

def plot_weight_topomap(target, cv_label="5fold", results_dir=None):
    if results_dir is None:
        results_dir = RESULTS_5F

    info = make_eeg_info()
    n_ch = len(CH_NAMES)
    n_bands = len(BAND_NAMES)

    # Load all subjects' weights
    all_w = []
    sub_rs = []
    for sub in SUBJECTS:
        wf = results_dir / f"{sub}_{target}_elasticnet_weights.npz"
        rf = results_dir / f"{sub}_{target}_elasticnet.csv"
        if wf.exists() and rf.exists():
            w = np.load(wf, allow_pickle=True)
            mean_coef = w["coefs"].mean(axis=0)
            all_w.append(mean_coef)
            r = pd.read_csv(rf)["mean_cv_r"].iloc[0]
            sub_rs.append((sub, r))

    if not all_w:
        print(f"  No weight files for {target}, skipping.")
        return

    all_w = np.array(all_w)  # (n_sub, 155)
    n_sub = len(all_w)
    group_mean = all_w.mean(axis=0).reshape(n_bands, n_ch)  # (n_bands, n_ch)

    # t-test at each (band, ch)
    from scipy.stats import ttest_1samp
    all_w_reshaped = all_w.reshape(n_sub, n_bands, n_ch)
    t_vals = np.zeros((n_bands, n_ch))
    p_vals = np.zeros((n_bands, n_ch))
    for bi in range(n_bands):
        for ci in range(n_ch):
            t, p = ttest_1samp(all_w_reshaped[:, bi, ci], 0)
            t_vals[bi, ci] = t
            p_vals[bi, ci] = p

    # FDR correction
    from statsmodels.stats.multitest import multipletests
    _, p_fdr, _, _ = multipletests(p_vals.ravel(), alpha=0.05, method="fdr_bh")
    sig_mask = (p_fdr < 0.05).reshape(n_bands, n_ch)
    n_sig = sig_mask.sum()

    # Best and worst subjects
    sub_rs.sort(key=lambda x: x[1], reverse=True)
    best_sub, best_r = sub_rs[0]
    worst_sub, worst_r = sub_rs[-1]
    best_idx = SUBJECTS.index(best_sub)
    worst_idx = SUBJECTS.index(worst_sub)

    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(4, n_bands + 1, figure=fig,
                           hspace=0.45, wspace=0.12,
                           width_ratios=[1]*n_bands + [0.08])

    target_label = target.replace("_", " ")

    # Row 0: group mean signed weights
    vmax_w = np.max(np.abs(group_mean)) * 1.05
    for bi in range(n_bands):
        ax = fig.add_subplot(gs[0, bi])
        mask = sig_mask[bi]
        mne.viz.plot_topomap(group_mean[bi], info, axes=ax, show=False,
                             cmap="RdBu_r", vlim=(-vmax_w, vmax_w),
                             sensors=True, contours=0,
                             mask=mask,
                             mask_params=dict(marker="o", markerfacecolor="black",
                                              markeredgecolor="black", markersize=7))
        ax.set_title(BAND_LABELS[bi], fontsize=11, fontweight="bold")
    ax_cb = fig.add_subplot(gs[0, -1])
    norm = Normalize(-vmax_w, vmax_w)
    sm = cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb)
    cb.set_label("weight", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.text(0.02, 0.82, "Group\nmean\nweight", fontsize=11, fontweight="bold",
             va="center", ha="center")

    # Row 1: t-statistics
    vmax_t = np.max(np.abs(t_vals)) * 0.85
    for bi in range(n_bands):
        ax = fig.add_subplot(gs[1, bi])
        mask = sig_mask[bi]
        mne.viz.plot_topomap(t_vals[bi], info, axes=ax, show=False,
                             cmap="RdBu_r", vlim=(-vmax_t, vmax_t),
                             sensors=True, contours=0,
                             mask=mask,
                             mask_params=dict(marker="o", markerfacecolor="black",
                                              markeredgecolor="black", markersize=7))
    ax_cb = fig.add_subplot(gs[1, -1])
    norm = Normalize(-vmax_t, vmax_t)
    sm = cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb)
    cb.set_label("t", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.text(0.02, 0.6, f"t-stat\n(n={n_sub})", fontsize=11, fontweight="bold",
             va="center", ha="center")

    # Row 2: Best subject |weights|
    best_w = np.abs(all_w_reshaped[best_idx])
    vmax_best = np.max(best_w) * 1.05
    for bi in range(n_bands):
        ax = fig.add_subplot(gs[2, bi])
        mne.viz.plot_topomap(best_w[bi], info, axes=ax, show=False,
                             cmap="YlOrRd", vlim=(0, vmax_best),
                             sensors=True, contours=0)
    ax_cb = fig.add_subplot(gs[2, -1])
    norm = Normalize(0, vmax_best)
    sm = cm.ScalarMappable(norm=norm, cmap="YlOrRd")
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb)
    cb.set_label("|w|", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    short_best = best_sub.replace("dmnelf0", "").replace("dmnelf", "")
    fig.text(0.02, 0.38, f"Best\n({short_best})\nr={best_r:+.2f}",
             fontsize=10, fontweight="bold", va="center", ha="center")

    # Row 3: Worst subject |weights|
    worst_w = np.abs(all_w_reshaped[worst_idx])
    vmax_worst = np.max(worst_w) * 1.05
    for bi in range(n_bands):
        ax = fig.add_subplot(gs[3, bi])
        mne.viz.plot_topomap(worst_w[bi], info, axes=ax, show=False,
                             cmap="YlOrRd", vlim=(0, vmax_worst),
                             sensors=True, contours=0)
    ax_cb = fig.add_subplot(gs[3, -1])
    norm = Normalize(0, vmax_worst)
    sm = cm.ScalarMappable(norm=norm, cmap="YlOrRd")
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb)
    cb.set_label("|w|", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    short_worst = worst_sub.replace("dmnelf0", "").replace("dmnelf", "")
    fig.text(0.02, 0.15, f"Worst\n({short_worst})\nr={worst_r:+.2f}",
             fontsize=10, fontweight="bold", va="center", ha="center")

    fdr_note = f"FDR-sig (●): {n_sig}/{n_bands*n_ch}" if n_sig > 0 else f"No FDR-sig channels (0/{n_bands*n_ch})"
    fig.suptitle(f"ElasticNet Weights — {target_label} ({cv_label.upper()}, n={n_sub})\n"
                 f"{fdr_note}",
                 fontsize=14, fontweight="bold", y=0.98)

    out = FIG_DIR / f"topomap_enet_{target}_{cv_label}.png"
    fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── Figure 5: 5-fold vs LORO comparison (paired) ──

def plot_5fold_vs_loro():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

    for ti, tname in enumerate(TARGETS):
        ax = axes[ti]
        r5f = load_r_values(RESULTS_5F, "elasticnet")
        rloro = load_r_values(RESULTS_LORO, "elasticnet")

        r5_dict = {s: r for s, r, p in r5f[tname]}
        rl_dict = {s: r for s, r, p in rloro[tname]}

        paired_subs = [s for s in SUBJECTS if s in r5_dict and s in rl_dict]
        r5_arr = np.array([r5_dict[s] for s in paired_subs])
        rl_arr = np.array([rl_dict[s] for s in paired_subs])

        for i in range(len(paired_subs)):
            color = TARGET_COLORS[tname]
            ax.plot([0, 1], [r5_arr[i], rl_arr[i]], "o-", color=color,
                    alpha=0.4, markersize=5, linewidth=1)

        ax.plot([0, 1], [np.mean(r5_arr), np.mean(rl_arr)], "s-",
                color="black", markersize=8, linewidth=2.5, zorder=10)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["5-fold", "LORO"], fontsize=11)
        ax.set_title(TARGET_LABELS[tname].replace("\n", " "), fontsize=12, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("CV correlation (r)", fontsize=11)
    fig.suptitle("Within-run (5-fold) vs Cross-run (LORO) — ElasticNet\n"
                 "Lines = individual subjects, ■ = group mean",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = FIG_DIR / "5fold_vs_loro_paired.png"
    fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── Figure 6: Methods pipeline schematic ──

def plot_methods():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", linewidth=1.5)
    blue_box = dict(boxstyle="round,pad=0.4", facecolor="#d6e4f0", edgecolor="#2c5f8a", linewidth=1.5)
    green_box = dict(boxstyle="round,pad=0.4", facecolor="#d9ead3", edgecolor="#38761d", linewidth=1.5)
    orange_box = dict(boxstyle="round,pad=0.4", facecolor="#fce5cd", edgecolor="#b45f06", linewidth=1.5)
    red_box = dict(boxstyle="round,pad=0.4", facecolor="#f4cccc", edgecolor="#990000", linewidth=1.5)

    # Top: Data
    ax.text(2, 6, "Simultaneous EEG-fMRI\n16 subjects × 4 runs\n31 ch, TR=1.2s",
            fontsize=10, ha="center", va="center", bbox=blue_box)

    # EEG branch
    ax.annotate("", xy=(2, 4.8), xytext=(2, 5.3),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(2, 4.3, "EEG Band Power\nδ θ α β γ × 31ch\nHilbert → log → HRF",
            fontsize=9, ha="center", va="center", bbox=blue_box)

    ax.annotate("", xy=(2, 3.2), xytext=(2, 3.7),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(2, 2.7, "CAR\n(remove spatial mean\nper band per TR)",
            fontsize=9, ha="center", va="center", bbox=blue_box)

    ax.annotate("", xy=(4.5, 2.7), xytext=(3.5, 2.7),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(6.2, 2.7, "155 features\n(31 ch × 5 bands)",
            fontsize=9, ha="center", va="center", bbox=blue_box)

    # fMRI branch
    ax.text(12, 6, "fMRI ROI Timeseries\nDMN, CEN (personalized)\nGlobal signal (fMRIPrep)",
            fontsize=10, ha="center", va="center", bbox=green_box)

    ax.annotate("", xy=(12, 4.8), xytext=(12, 5.3),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(12, 4.3, "GSR: regress global\nsignal per run\n→ GSR'd DMN, GSR'd CEN",
            fontsize=9, ha="center", va="center", bbox=green_box)

    ax.annotate("", xy=(12, 3.2), xytext=(12, 3.7),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(12, 2.7, "Targets:\nGSR'd CEN | GSR'd DMN\nPDA = CEN−DMN | RAW DMN",
            fontsize=9, ha="center", va="center", bbox=green_box)

    ax.annotate("", xy=(9.5, 2.7), xytext=(10.5, 2.7),
                arrowprops=dict(arrowstyle="->", lw=1.5))

    # Model
    ax.text(8, 1.5, "ElasticNet / Ridge Regression\nWithin-subject, per-run z-scored",
            fontsize=10, ha="center", va="center", bbox=orange_box, fontweight="bold")

    ax.annotate("", xy=(8, 2.0), xytext=(6.2, 2.2),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(8, 2.0), xytext=(9.5, 2.2),
                arrowprops=dict(arrowstyle="->", lw=1.5))

    # Validation
    ax.annotate("", xy=(8, 0.3), xytext=(8, 0.9),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(8, -0.15, "5-fold contiguous CV + LORO\n10K circular-shift null (per-subject)\n"
            "10K sign-flip null (group)",
            fontsize=9, ha="center", va="center", bbox=red_box)

    ax.text(8, 6.8, "Methods: Multivariate EEG → fMRI Network Decoding",
            fontsize=14, ha="center", va="center", fontweight="bold")

    out = FIG_DIR / "methods_diagram.png"
    fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── Figure 7: Feature importance bar (top 15 features for GSR_CEN) ──

def plot_feature_importance(target="GSR_CEN"):
    n_ch = len(CH_NAMES)
    n_bands = len(BAND_NAMES)
    feature_names = [f"{c} {b}" for b in BAND_NAMES for c in CH_NAMES]

    all_w = []
    for sub in SUBJECTS:
        wf = RESULTS_5F / f"{sub}_{target}_elasticnet_weights.npz"
        if wf.exists():
            w = np.load(wf, allow_pickle=True)
            all_w.append(w["coefs"].mean(axis=0))
    all_w = np.array(all_w)
    group_mean = all_w.mean(axis=0)

    top_idx = np.argsort(np.abs(group_mean))[::-1][:15]

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [feature_names[i] for i in top_idx]
    vals = [group_mean[i] for i in top_idx]

    # Color by band
    band_colors = {"delta": "#1f77b4", "theta": "#ff7f0e", "alpha": "#2ca02c",
                   "beta": "#d62728", "gamma": "#9467bd"}
    colors = []
    for i in top_idx:
        bi = i // n_ch
        colors.append(band_colors[BAND_NAMES[bi]])

    # Count cross-subject sign agreement
    labels = []
    for i in top_idx:
        nonzero = all_w[:, i][all_w[:, i] != 0]
        if len(nonzero) == 0:
            labels.append("")
            continue
        if group_mean[i] > 0:
            agree = np.sum(nonzero > 0)
        else:
            agree = np.sum(nonzero < 0)
        labels.append(f"{agree}/{len(nonzero)}")

    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Group mean ElasticNet weight", fontsize=11)

    # Add agreement labels
    for i, (val, label) in enumerate(zip(vals, labels)):
        offset = 0.003 if val >= 0 else -0.003
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, i, label, va="center", ha=ha, fontsize=8, color="gray")

    # Legend for bands
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=band_colors[b], edgecolor="black", label=b) for b in BAND_NAMES]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9, title="Band")

    target_label = target.replace("_", " ")
    ax.set_title(f"Top 15 ElasticNet Features — {target_label}\n"
                 f"(group mean weight, numbers = cross-subject sign agreement)",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = FIG_DIR / f"feature_importance_{target}.png"
    fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out.name}")


if __name__ == "__main__":
    print("Generating PPT figures...")

    print("\n1. Group bar plot (Ridge vs ElasticNet)...")
    plot_group_bars()

    print("\n2. Subject heatmap (ElasticNet)...")
    plot_subject_heatmap()

    print("\n3. Weight topomaps...")
    plot_weight_topomap("GSR_CEN", "5fold", RESULTS_5F)
    plot_weight_topomap("PDA", "5fold", RESULTS_5F)

    print("\n4. 5-fold vs LORO paired plot...")
    plot_5fold_vs_loro()

    print("\n5. Methods diagram...")
    plot_methods()

    print("\n6. Feature importance bars...")
    plot_feature_importance("GSR_CEN")
    plot_feature_importance("PDA")

    print("\nAll figures saved to:", FIG_DIR)
