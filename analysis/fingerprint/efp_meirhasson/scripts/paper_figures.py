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
from scipy.stats import zscore, pearsonr
from sklearn.linear_model import RidgeCV

from efp_features import load_config, load_subject_features
from efp_decode import assemble, mk_block_folds, nmse

mne.set_log_level("ERROR")
PROJ = Path(__file__).resolve().parent.parent
RES = PROJ / "results" / "full"
CACHE = PROJ / "results" / "features_cache"

# representative Hz labels (median band edges across subjects, PDA tr)
BAND_HZ = ["1-4", "5-7", "8-9", "10-11", "12-13", "14-16", "17-19", "20-22", "23-28", "29-40"]
JET = "jet"


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
    e = cfg["efp"]; tr = cfg["data"]["fmri"]["tr"]
    alphas = np.logspace(np.log10(e["alpha_grid_lo"]), np.log10(e["alpha_grid_hi"]), e["alpha_grid_n"])
    n_delays = int(round(e["delay_window_s"] / tr)) + 1 if res == "tr" \
        else int(round(e["delay_window_s"] * e["hz4"])) + 1

    sub, best_ch, _ = best_subject(target, res)
    runs, chs = load_subject_features(CACHE, sub)

    # per-channel correlation r for topomap + best-electrode predictions
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
    r = pearsonr(best_y[~np.isnan(best_pred)], best_pred[~np.isnan(best_pred)])[0]

    # ---- B. predictor overlay ----
    fig, ax = plt.subplots(figsize=(10, 3.4))
    t_axis = np.arange(len(best_y)) * (tr if res == "tr" else 1 / cfg["efp"]["hz4"])
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
    d = cfg["data"]
    fif = (Path(d["eeg_preproc_dir"]) / f"sub-{sub}" / d["session"] / "eeg" /
           f"sub-{sub}_{d['session']}_task-{d['task']}_run-01_desc-{d['eeg']['desc']}_eeg.fif")
    raw = mne.io.read_raw_fif(str(fif), preload=False, verbose=False)
    raw.pick(mne.pick_types(raw.info, eeg=True, exclude=[]))
    raw.set_montage("standard_1020", match_case=False, on_missing="ignore")
    # keep only channels with a montage position and a finite r
    have = [i for i, c in enumerate(raw.ch_names)
            if raw.info["chs"][i]["loc"][:3].any()
            and c in r_by_ch and np.isfinite(r_by_ch[c])]
    info = mne.pick_info(raw.info, have)
    vals = np.array([r_by_ch[raw.ch_names[i]] for i in have])
    # Use the SAME 2D projection MNE uses internally, so the star marker lines up
    # with the plotted sensors (raw loc[:2] is un-projected and drifts off-centre).
    from mne.channels.layout import _find_topomap_coords
    pos2d = _find_topomap_coords(info, picks=list(range(len(have))))
    # diverging scale centred on 0: red = higher r (better), blue = anticorrelated
    vmax = float(np.abs(vals).max()) if vals.size else 1.0
    fig, ax = plt.subplots(figsize=(5.2, 5))
    im, _ = mne.viz.plot_topomap(vals, pos2d, axes=ax, show=False, cmap="RdBu_r",
                                 vlim=(-vmax, vmax), contours=6, sensors=True)
    # mark best electrode (the one the EFP pipeline selected)
    have_chs = [raw.ch_names[i] for i in have]
    if best_ch in have_chs:
        bi = have_chs.index(best_ch)
        ax.plot(pos2d[bi, 0], pos2d[bi, 1], "k*", markersize=16)
    ax.set_title(f"{target}: per-electrode CV r — {sub}\n(best: {best_ch}★, red=better)",
                 fontweight="bold", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, label="CV correlation r")
    fig.tight_layout()
    out = RES / f"paper_fig_r_topomap_{target}_{res}.png"
    fig.savefig(out, dpi=150); plt.close(fig); print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="tr")
    ap.add_argument("--targets", nargs="+", default=["PDA", "CEN", "GSR_CEN"])
    args = ap.parse_args()
    cfg = load_config()
    fig_fingerprints(cfg, args.res)
    for t in args.targets:
        fig_predictor_and_topomap(cfg, t, args.res)


if __name__ == "__main__":
    main()
