#!/usr/bin/env python3
"""
paper_figures.py
----------------
Recreate the signature Meir-Hasson 2014 figures for the DMNELF EFP results:

  A. Group EFP [frequency x time-delay] fingerprints, Hz axis   (paper Fig 5c/7c/9c)
  B. Actual fMRI vs EFP-predicted timeseries overlay, with R    (paper Fig 3d)
  C. Per-electrode NMSE scalp topomap, best electrode marked    (paper Fig 3e / 5b/7b)

A uses the group-averaged fingerprints; B and C use the best-performing subject for
each target (like the paper's representative panels), recomputing out-of-fold
predictions from the cached features.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, yaml, mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.stats import zscore, pearsonr
from sklearn.linear_model import RidgeCV

from efp_features import (load_config, load_subject_features, load_eeg_run,
                          equal_energy_edges, bin_average)
from efp_decode import assemble, mk_block_folds, nmse
from stockwell import stockwell_power

mne.set_log_level("ERROR")
PROJ = Path(__file__).resolve().parent.parent
RES = PROJ / "results" / "full"
CACHE = PROJ / "results" / "features_cache"

# representative Hz labels (median band edges across subjects, PDA tr)
BAND_HZ = ["1-4", "5-7", "8-9", "10-11", "12-13", "14-16", "17-19", "20-22", "23-28", "29-40"]
JET = "jet"

# network ROI overlay colors (match 00b_extract_personal_masks.py)
_CMAP_DMN = LinearSegmentedColormap.from_list("dmn", ["#0077FF", "#0077FF"])
_CMAP_CEN = LinearSegmentedColormap.from_list("cen", ["#FF1A00", "#FF1A00"])
_CMAP_VIS = LinearSegmentedColormap.from_list("vis", ["#00A000", "#00A000"])
VIS_MASK = PROJ / "results" / "visual_sphere_mask.nii.gz"  # 6mm calcarine sphere (paper Fig 5a)


# ── small shared helpers ─────────────────────────────────────────────────────
def band_labels(band_hz):
    """['1-4', '5-7', ...] from a list of (lo, hi) Hz band edges."""
    return [f"{int(lo)}-{int(hi)}" for lo, hi in band_hz]


def n_delays_for(cfg, res):
    e = cfg["efp"]; tr = cfg["data"]["fmri"]["tr"]
    return (int(round(e["delay_window_s"] / tr)) + 1 if res == "tr"
            else int(round(e["delay_window_s"] * e["hz4"])) + 1)


def alphas_from(cfg):
    e = cfg["efp"]
    return np.logspace(np.log10(e["alpha_grid_lo"]), np.log10(e["alpha_grid_hi"]),
                       e["alpha_grid_n"])


def per_channel_r(cfg, runs, chs, target, res, n_delays, alphas, best_ch):
    """Per-electrode out-of-fold CV correlation r; also return best electrode's pred/y."""
    e = cfg["efp"]
    r_by_ch = {}
    best_pred = best_y = None
    for ci, ch in enumerate(chs):
        X, y = assemble(runs, ci, target, res, n_delays)
        if X is None:
            continue
        folds = mk_block_folds(len(y), e["cv_outer_k"], e["cv_outer_m"])
        pred, nm, rch = cv_predict(X, y, alphas, folds)
        r_by_ch[ch] = rch
        if ch == best_ch:
            best_pred, best_y = pred, y
    return r_by_ch, best_pred, best_y


def render_r_topomap(ax, cfg, sub, r_by_ch, best_ch, vmax=None):
    """Draw a per-electrode CV-r scalp topomap into ax; star marks best electrode.

    Uses MNE's internal 2D projection for both the map and the star so they align.
    Returns the image handle (for a colorbar)."""
    d = cfg["data"]
    fif = (Path(d["eeg_preproc_dir"]) / f"sub-{sub}" / d["session"] / "eeg" /
           f"sub-{sub}_{d['session']}_task-{d['task']}_run-01_desc-{d['eeg']['desc']}_eeg.fif")
    raw = mne.io.read_raw_fif(str(fif), preload=False, verbose=False)
    raw.pick(mne.pick_types(raw.info, eeg=True, exclude=[]))
    raw.set_montage("standard_1020", match_case=False, on_missing="ignore")
    have = [i for i, c in enumerate(raw.ch_names)
            if raw.info["chs"][i]["loc"][:3].any()
            and c in r_by_ch and np.isfinite(r_by_ch[c])]
    info = mne.pick_info(raw.info, have)
    vals = np.array([r_by_ch[raw.ch_names[i]] for i in have])
    from mne.channels.layout import _find_topomap_coords
    pos2d = _find_topomap_coords(info, picks=list(range(len(have))))
    if vmax is None:
        vmax = float(np.abs(vals).max()) if vals.size else 1.0
    im, _ = mne.viz.plot_topomap(vals, pos2d, axes=ax, show=False, cmap="RdBu_r",
                                 vlim=(-vmax, vmax), contours=6, sensors=True)
    have_chs = [raw.ch_names[i] for i in have]
    if best_ch in have_chs:
        bi = have_chs.index(best_ch)
        ax.plot(pos2d[bi, 0], pos2d[bi, 1], "k*", markersize=14)
    return im


def render_roi_masks(ax, cfg, sub, target, cut_coords=5):
    """Overlay the subject's personalized network mask(s) as axial slices into ax.

    DMN/GSR_DMN -> DMN mask (blue); CEN/GSR_CEN -> CEN mask (red);
    PDA/GSR_PDA -> both (CEN red + DMN blue). Degrades to a placeholder if masks
    or nilearn are unavailable, so the composite figure still renders."""
    d = cfg["data"]
    mroot = d.get("network_masks_dir")
    base_t = target.replace("GSR_", "")

    # VIS is a group ROI (same mask for all subjects)
    if base_t == "VIS":
        try:
            from nilearn import plotting
            if VIS_MASK.exists():
                plotting.plot_roi(str(VIS_MASK), axes=ax, display_mode="z",
                                  cut_coords=5, cmap=_CMAP_VIS, colorbar=False,
                                  annotate=False)
                ax.set_title("visual ROI: 6mm calcarine sphere (V1)", fontsize=10, fontweight="bold")
                return
        except Exception as ex:
            print(f"  render_roi_masks VIS failed: {ex}")
        ax.text(0.5, 0.5, "visual ROI N/A", ha="center", va="center",
                fontsize=9, transform=ax.transAxes); ax.axis("off")
        return

    show_dmn = base_t in ("DMN", "PDA")
    show_cen = base_t in ("CEN", "PDA")
    roi_title = {"DMN": "DMN ROI", "CEN": "CEN ROI",
                 "PDA": "CEN (red) − DMN (blue) ROI"}.get(base_t, f"{base_t} ROI")
    ok = False
    if mroot:
        prefix = f"sub-{sub}_space-MNI152NLin6Asym_res-2"
        sdir = Path(mroot) / f"sub-{sub}"
        dmn_p = sdir / f"{prefix}_dmn_mask.nii.gz"
        cen_p = sdir / f"{prefix}_cen_mask.nii.gz"
        try:
            from nilearn import plotting
            disp = None
            if show_cen and cen_p.exists():
                disp = plotting.plot_roi(str(cen_p), axes=ax, display_mode="z",
                                         cut_coords=cut_coords, cmap=_CMAP_CEN,
                                         colorbar=False, annotate=False)
                ok = True
            if show_dmn and dmn_p.exists():
                if disp is None:
                    disp = plotting.plot_roi(str(dmn_p), axes=ax, display_mode="z",
                                             cut_coords=cut_coords, cmap=_CMAP_DMN,
                                             colorbar=False, annotate=False)
                else:
                    disp.add_overlay(str(dmn_p), cmap=_CMAP_DMN)
                ok = True
        except Exception as ex:
            print(f"  render_roi_masks failed for {sub} {target}: {ex}")
            ok = False
    if ok:
        ax.set_title(roi_title, fontsize=10, fontweight="bold")
    else:
        ax.text(0.5, 0.5, f"ROI mask N/A\n{sub} {target}", ha="center", va="center",
                fontsize=9, transform=ax.transAxes)
        ax.axis("off")


def cv_predict(X, y, alphas, folds):
    """Out-of-fold predictions + mean NMSE + correlation r (over concatenated OOF)."""
    pred = np.full(len(y), np.nan)
    nm = []
    for tr, te in folds:
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        m = RidgeCV(alphas=alphas).fit((X[tr] - mu) / sd, y[tr])
        p = m.predict((X[te] - mu) / sd)
        pred[te] = p
        if np.std(p) > 1e-9 and np.std(y[te]) > 1e-9:
            nm.append(nmse(y[te], p))
    ok = ~np.isnan(pred)
    r = (pearsonr(y[ok], pred[ok])[0]
         if ok.sum() > 2 and np.std(pred[ok]) > 1e-9 else np.nan)
    return pred, (np.mean(nm) if nm else np.nan), r


# ── A. group fingerprints ────────────────────────────────────────────────────
def fig_fingerprints(cfg, res="tr"):
    gf = np.load(RES / "group_fingerprints.npz", allow_pickle=True)
    targets = ["DMN", "CEN", "PDA", "GSR_DMN", "GSR_CEN", "GSR_PDA"]
    tr = cfg["data"]["fmri"]["tr"]
    step = tr if res == "tr" else 1.0 / cfg["efp"]["hz4"]
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25)
    vmax = max(np.abs(gf[f"{t}__{res}"]).max() for t in targets if f"{t}__{res}" in gf)
    for i, t in enumerate(targets):
        key = f"{t}__{res}"
        if key not in gf:
            continue
        mat = gf[key]
        n_bands, n_delays = mat.shape
        delays = -np.arange(n_delays) * step
        ax = fig.add_subplot(gs[i // 3, i % 3])
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=[delays[-1], delays[0], n_bands + 0.5, 0.5])
        ax.set_yticks(range(1, n_bands + 1)); ax.set_yticklabels(BAND_HZ, fontsize=8)
        ax.set_title(t, fontsize=14, fontweight="bold")
        if i // 3 == 1:
            ax.set_xlabel("Time delay (s)")
        if i % 3 == 0:
            ax.set_ylabel("Frequency band (Hz)")
    cax = fig.add_axes([0.93, 0.25, 0.015, 0.5])
    fig.colorbar(im, cax=cax, label="norm. ridge weight (EFP)")
    fig.suptitle(f"Group EEG Finger-Prints  [frequency × time-delay]  ({res})",
                 fontsize=17, fontweight="bold")
    out = RES / f"paper_fig_fingerprints_{res}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


# ── best subject per target ──────────────────────────────────────────────────
def best_subject(target, res="tr"):
    df = pd.read_csv(RES / "efp_persubject_all.csv")
    g = df[(df.target == target) & (df.resolution == res) & (df.method == "EFP")]
    row = g.loc[g.mean_r.idxmax()]
    return row["subject"], row["best_ch"], row["mean_r"]


# ── B + C for a representative subject ───────────────────────────────────────
def fig_predictor_and_topomap(cfg, target, res="tr"):
    tr = cfg["data"]["fmri"]["tr"]
    alphas = alphas_from(cfg)
    n_delays = n_delays_for(cfg, res)
    step = tr if res == "tr" else 1 / cfg["efp"]["hz4"]

    sub, best_ch, _ = best_subject(target, res)
    runs, chs = load_subject_features(CACHE, sub)
    r_by_ch, best_pred, best_y = per_channel_r(cfg, runs, chs, target, res,
                                               n_delays, alphas, best_ch)
    r = pearsonr(best_y[~np.isnan(best_pred)], best_pred[~np.isnan(best_pred)])[0]

    # ---- B. predictor overlay ----
    fig, ax = plt.subplots(figsize=(10, 3.4))
    t_axis = np.arange(len(best_y)) * step
    pred_z = np.clip(zscore(np.nan_to_num(best_pred)), -4, 4)  # clip fold-boundary spikes for display
    ax.plot(t_axis, zscore(best_y), color="#111", lw=1.4, label=f"fMRI {target}")
    ax.plot(t_axis, pred_z, color="#2E86C1", lw=1.4, alpha=0.85,
            label=f"EFP predictor (electrode {best_ch})")
    ax.set_ylim(-4.2, 4.2)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("z-score")
    ax.set_title(f"{target}: EFP predictor vs fMRI — {sub}  (R = {r:.2f})", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9); fig.tight_layout()
    out = RES / f"paper_fig_predictor_{target}_{res}.png"
    fig.savefig(out, dpi=150); plt.close(fig); print("saved", out)

    # ---- C. correlation (r) topomap ----
    fig, ax = plt.subplots(figsize=(5.2, 5))
    im = render_r_topomap(ax, cfg, sub, r_by_ch, best_ch)
    ax.set_title(f"{target}: per-electrode CV r — {sub}\n(best: {best_ch}★, red=better)",
                 fontweight="bold", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, label="CV correlation r")
    fig.tight_layout()
    out = RES / f"paper_fig_r_topomap_{target}_{res}.png"
    fig.savefig(out, dpi=150); plt.close(fig); print("saved", out)


# ── Fig 2 (post-processing schematic) + Fig 3 (prediction input/output) ───────
def fig2_postprocessing_schematic(cfg, target="PDA", res="tr"):
    """Paper Fig 2 analog: EEG -> S-transform -> 4 Hz -> data-driven bands ->
    band-averaged TF -> sliding window per fMRI TR; fMRI ROI -> 4 Hz + normalized."""
    d = cfg["data"]; e = cfg["efp"]
    sf = d["eeg"]["sfreq"]; tr = d["fmri"]["tr"]
    sub, best_ch, _ = best_subject(target, res)
    runs, chs = load_subject_features(CACHE, sub)
    ci = chs.index(best_ch) if best_ch in chs else 0
    rd = max(runs, key=lambda r: r["n_tr"])          # representative (longest) run
    n_hz4 = rd["n_hz4"]

    # (a) recompute the full 1-Hz S-transform from the raw fif (not cached)
    eeg, _ = load_eeg_run(cfg, sub, rd["run"])
    power4 = freqs = spec = edges = None
    if eeg is not None and ci < eeg.shape[0]:
        freqs, power = stockwell_power(eeg[ci], sf, e["freq_min"], e["freq_max"])
        power4 = bin_average(power, n_hz4)           # "reduced to 4 Hz"
        spec = power.mean(axis=1)
        edges = equal_energy_edges(spec, freqs, e["n_bands"])

    bp_hz4_ci = rd["bp_hz4"][ci]                      # (n_bands, n_hz4)
    band_hz = rd["band_hz"]
    labels = band_labels(band_hz)
    dur_s = rd["n_tr"] * tr
    t_hz4 = np.linspace(0, dur_s, n_hz4)

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.28,
                  width_ratios=[1.25, 1.0])

    # (a) spectrogram
    ax_a = fig.add_subplot(gs[0, 0])
    if power4 is not None:
        ax_a.imshow(power4, aspect="auto", origin="lower", cmap=JET,
                    extent=[0, dur_s, freqs[0], freqs[-1]])
        ax_a.set_ylabel("Freq (Hz)")
    else:
        ax_a.text(0.5, 0.5, "raw EEG N/A", ha="center", va="center",
                  transform=ax_a.transAxes); ax_a.axis("off")
    ax_a.set_title(f"(a) EEG {best_ch}: S-transform → 4 Hz", fontsize=11, fontweight="bold")

    # (b) mean spectrum + equal-energy band edges
    ax_b = fig.add_subplot(gs[1, 0])
    if spec is not None:
        ax_b.plot(freqs, spec, color="#333", lw=1.5)
        for k, (lo, hi) in enumerate(edges):
            ax_b.axvspan(freqs[lo], freqs[hi], alpha=0.18,
                         color=("#4C72B0" if k % 2 == 0 else "#DD8452"))
        ax_b.set_xlabel("Freq (Hz)"); ax_b.set_ylabel("power")
    else:
        ax_b.axis("off")
    ax_b.set_title("(b) data-driven equal-energy bands", fontsize=11, fontweight="bold")

    # (c) band-averaged TF matrix
    ax_c = fig.add_subplot(gs[2, 0])
    ax_c.imshow(bp_hz4_ci, aspect="auto", origin="lower", cmap=JET,
                extent=[0, dur_s, 0.5, e["n_bands"] + 0.5])
    ax_c.set_yticks(range(1, e["n_bands"] + 1)); ax_c.set_yticklabels(labels, fontsize=7)
    ax_c.set_xlabel("Time (s)"); ax_c.set_ylabel("Band (Hz)")
    ax_c.set_title("(c) band-averaged TF (10 × time)", fontsize=11, fontweight="bold")

    # (d) sliding-window schematic
    ax_d = fig.add_subplot(gs[0, 1])
    n_del = n_delays_for(cfg, res)
    grid = np.random.default_rng(0).random((e["n_bands"], n_del + 6)) * 0.3
    ax_d.imshow(grid, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax_d.add_patch(Rectangle((-0.5, -0.5), n_del, e["n_bands"], fill=False,
                             edgecolor="crimson", lw=2.5))
    ax_d.annotate("EEG window\n0..−12 s", xy=(n_del - 0.5, e["n_bands"] / 2),
                  xytext=(n_del + 2, e["n_bands"] / 2), va="center", fontsize=9,
                  arrowprops=dict(arrowstyle="->", color="crimson"))
    ax_d.set_xticks([]); ax_d.set_yticks([])
    ax_d.set_title("(d) sliding window → one fMRI TR", fontsize=11, fontweight="bold")

    # fMRI (b) ROI brain image
    ax_roi = fig.add_subplot(gs[1, 1])
    render_roi_masks(ax_roi, cfg, sub, target)

    # fMRI (c) ROI signal upsampled to 4 Hz + normalized
    ax_ts = fig.add_subplot(gs[2, 1])
    ax_ts.plot(t_hz4, zscore(rd["tgt_hz4"][target]), color="#111", lw=1.2)
    ax_ts.set_xlabel("Time (s)"); ax_ts.set_ylabel("z-score")
    ax_ts.set_title(f"fMRI {target}: up-sampled 4 Hz + normalized", fontsize=11, fontweight="bold")

    fig.suptitle(f"Post-processing schematic — {sub}, electrode {best_ch} ({res})",
                 fontsize=16, fontweight="bold")
    out = RES / f"paper_fig2_schematic_{target}_{res}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print("saved", out)


def fig3_composite(cfg, target, res="tr", metric="r"):
    """Paper Fig 3 analog (a-e): ROI signal, bottom/top-25% EEG TF matrices, EFP,
    predictor vs measured, per-electrode error map — plus the ROI brain image."""
    e = cfg["efp"]; tr = cfg["data"]["fmri"]["tr"]
    alphas = alphas_from(cfg)
    n_delays = n_delays_for(cfg, res)
    n_bands = e["n_bands"]
    step = tr if res == "tr" else 1 / e["hz4"]
    delays = -np.arange(n_delays) * step

    sub, best_ch, _ = best_subject(target, res)
    runs, chs = load_subject_features(CACHE, sub)
    ci = chs.index(best_ch) if best_ch in chs else 0
    r_by_ch, best_pred, best_y = per_channel_r(cfg, runs, chs, target, res,
                                               n_delays, alphas, best_ch)
    X, y = assemble(runs, ci, target, res, n_delays)
    r = pearsonr(best_y[~np.isnan(best_pred)], best_pred[~np.isnan(best_pred)])[0]

    # (b) bottom/top-25% quartile-averaged design matrices  (n_bands x n_delays).
    # Standardize each band×delay cell over time first, else 1/f power dominates and
    # the low- vs high-ROI contrast is invisible; z-scoring is also what the ridge sees.
    lo_q, hi_q = np.quantile(y, 0.25), np.quantile(y, 0.75)
    lo_m, hi_m = y <= lo_q, y >= hi_q
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)

    def _mat(mask):
        if mask.sum() < 5 or X.shape[1] != n_bands * n_delays:
            return None
        return Xz[mask].mean(0).reshape(n_delays, n_bands).T

    lo_mat, hi_mat = _mat(lo_m), _mat(hi_m)
    q_vmax = max([np.abs(m).max() for m in (lo_mat, hi_mat) if m is not None] or [1.0])

    # (c) EFP coefficient matrix (already computed by the pipeline)
    zf = np.load(RES / f"efp_{sub}_{target}_{res}.npz", allow_pickle=True)
    efp = zf["efp"]; labels = band_labels(list(zf["band_hz"]))
    efp_vmax = float(np.abs(efp).max()) or 1.0

    fig = plt.figure(figsize=(17, 11))
    gs = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)
    ext = [delays[-1], delays[0], n_bands + 0.5, 0.5]

    # (a) ROI signal
    ax_a = fig.add_subplot(gs[0, :2])
    ax_a.plot(np.arange(len(best_y)) * step, zscore(best_y), color="#111", lw=1.2)
    ax_a.set_xlabel("Time (s)"); ax_a.set_ylabel("z-score")
    ax_a.set_title(f"(a) processed ROI signal — {target}", fontsize=11, fontweight="bold")

    # ROI brain image
    ax_roi = fig.add_subplot(gs[0, 2])
    render_roi_masks(ax_roi, cfg, sub, target)

    # (b) quartile matrices
    for j, (mat, ttl) in enumerate([(lo_mat, "(b) lower 25% ROI  (mean z-EEG)"),
                                    (hi_mat, "(b) upper 25% ROI  (mean z-EEG)")]):
        ax = fig.add_subplot(gs[1, j])
        if mat is not None:
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-q_vmax, vmax=q_vmax, extent=ext)
            ax.set_yticks(range(1, n_bands + 1)); ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel("Time delay (s)")
            fig.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.text(0.5, 0.5, "n<5", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(ttl, fontsize=11, fontweight="bold")

    # (c) EFP
    ax_c = fig.add_subplot(gs[1, 2])
    im = ax_c.imshow(efp, aspect="auto", cmap="RdBu_r", vmin=-efp_vmax, vmax=efp_vmax, extent=ext)
    ax_c.set_yticks(range(1, n_bands + 1)); ax_c.set_yticklabels(labels, fontsize=7)
    ax_c.set_xlabel("Time delay (s)")
    ax_c.set_title("(c) EFP (red=increase, blue=decrease)", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax_c, fraction=0.046)

    # (d) predictor vs measured
    ax_d = fig.add_subplot(gs[2, :2])
    t_axis = np.arange(len(best_y)) * step
    ax_d.plot(t_axis, zscore(best_y), color="#111", lw=1.3, label=f"fMRI {target}")
    ax_d.plot(t_axis, np.clip(zscore(np.nan_to_num(best_pred)), -4, 4), color="#2E86C1",
              lw=1.3, alpha=0.85, label=f"EFP predictor ({best_ch})")
    ax_d.set_ylim(-4.2, 4.2); ax_d.set_xlabel("Time (s)"); ax_d.set_ylabel("z-score")
    ax_d.legend(loc="upper right", fontsize=8)
    ax_d.set_title(f"(d) EFP predictor vs fMRI  (R = {r:.2f})", fontsize=11, fontweight="bold")

    # (e) per-electrode error map
    ax_e = fig.add_subplot(gs[2, 2])
    im = render_r_topomap(ax_e, cfg, sub, r_by_ch, best_ch)
    ax_e.set_title(f"(e) per-electrode CV r\n(best: {best_ch}★)", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax_e, fraction=0.046)

    fig.suptitle(f"Prediction input & output — {target}, {sub}, electrode {best_ch} ({res})",
                 fontsize=16, fontweight="bold")
    out = RES / f"paper_fig3_composite_{target}_{res}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print("saved", out)


def fig_group_electrode_topomap(cfg, target, res="tr"):
    """Group-averaged per-electrode CV r across all subjects (paper Fig 5b analog).

    The definitive electrode-topography test: unlike the single-best-subject topomap,
    this averages every electrode's CV r over subjects, so it shows where the signal
    lives on the scalp on average (e.g. occipital for a visual ROI)."""
    alphas = alphas_from(cfg)
    n_delays = n_delays_for(cfg, res)
    acc, n_used, ref_sub = {}, 0, None
    for sub in cfg["data"]["subjects"]["all"]:
        try:
            runs, chs = load_subject_features(CACHE, sub)
        except Exception:
            continue
        rbc, _, _ = per_channel_r(cfg, runs, chs, target, res, n_delays, alphas, chs[0])
        if not rbc:
            continue
        if ref_sub is None:
            ref_sub = sub
        n_used += 1
        for ch, rr in rbc.items():
            if np.isfinite(rr):
                acc.setdefault(ch, []).append(rr)
    gmap = {ch: float(np.mean(v)) for ch, v in acc.items() if len(v) >= 10}
    if not gmap:
        print(f"  group topomap {target}: no data"); return
    best_ch = max(gmap, key=gmap.get)
    fig, ax = plt.subplots(figsize=(5.4, 5))
    im = render_r_topomap(ax, cfg, ref_sub, gmap, best_ch)
    ax.set_title(f"{target}: group-mean per-electrode CV r (n={n_used})\n"
                 f"(peak: {best_ch}, red=better)", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, label="mean CV r")
    fig.tight_layout()
    out = RES / f"paper_fig_group_topomap_{target}_{res}.png"
    fig.savefig(out, dpi=150); plt.close(fig); print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="tr")
    ap.add_argument("--targets", nargs="+", default=["PDA", "CEN", "GSR_CEN"])
    ap.add_argument("--fig2-target", default="PDA")
    ap.add_argument("--group-topo", nargs="*", default=None,
                    help="targets to also render a group-averaged electrode topomap for")
    args = ap.parse_args()
    cfg = load_config()
    fig_fingerprints(cfg, args.res)
    for t in args.targets:
        fig_predictor_and_topomap(cfg, t, args.res)
    fig2_postprocessing_schematic(cfg, args.fig2_target, args.res)
    for t in args.targets:
        fig3_composite(cfg, t, args.res)
    for t in (args.group_topo or []):
        fig_group_electrode_topomap(cfg, t, args.res)


if __name__ == "__main__":
    main()
