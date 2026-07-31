#!/usr/bin/env python3
"""
export_epoc_dual_model.py  —  freeze the two-electrode CEN(P8)/DMN(O1) decoder for mindwear
-------------------------------------------------------------------------------------------
Replaces the 12-channel EPOC montage ridge with two independent single-electrode ridges, one
per target, at the specific electrodes validated in electrode_vs_montage_loso.py's zero-shot
LOSO comparison restricted to EPOC-X-available channels (no Cz/Pz on that headset):
  CEN <- P8   (LOSO r=0.118, p=0.001)
  DMN <- O1   (LOSO r=0.082, p=0.034)
PDA is NOT independently fit here — the deployed app derives it downstream as CEN_pred -
DMN_pred (an EPOC-restricted independent PDA fit is at chance, r=0.014, p=0.78; Pz was
essential and isn't available), matching the manuscript's own recommendation.

Reuses the EXACT validated pipeline (efp_decode.py::assemble() — raw DiFuMo CEN/DMN targets via
efp_features.py::load_targets_run, no baseline-TR masking, per-run z-scored design) and the
per-subject self-standardization convention efp_group.py's LOSO arms use (zscore(X, axis=0) per
subject before pooling, no further pooled-level rescaling) — NOT export_model.py's
clean_targets/subject_designs convention, which was never validated at these electrodes. Final
model is fit on ALL 16 locally-cached DMNELF subjects (no held-out subject — this is the
deployable fit, not another LOSO estimate).

Because each subject is self-z-scored before pooling, the live decoder's per-session Calibrator
(or RunningStats fallback) already reproduces this same self-standardization on a new subject's
own streamed features — so this model's feat_mean/feat_std are stored as 0/1 (a no-op second
stage), matching decoder.py's documented convention, not export_model.py's nontrivial pooled
mu/sd.

Output: mindwear/model/efp_epoc_dual_model.npz — schema documented in rt_features.py/decoder.py.

Usage: python export_epoc_dual_model.py
"""
from pathlib import Path
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import RidgeCV

from efp_features import load_config, load_subject_features, load_eeg_run, channel_bandpower
from efp_decode import assemble

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPT_DIR.parent
CACHE_DIR = PROJ_DIR / "results" / "features_cache"
OUT_PATH = PROJ_DIR.parent / "mindwear" / "model" / "efp_epoc_dual_model.npz"

TARGET_CHANNEL = {"CEN": "P8", "DMN": "O1"}


def frozen_band_edges(cfg, subs, channel):
    """This channel's own equal-energy Hz edges, per subject/run, frozen as the cross-subject
    median. Recomputed from raw EEG rather than read from the feature cache: build_subject_features
    overwrites its cached `band_hz` with whichever channel is processed LAST in that subject's
    full montage (a pre-existing cache quirk, unrelated to the per-channel band POWER values
    themselves, which ARE correct) — so it can't be trusted for a specific channel like P8/O1."""
    d = cfg["data"]; e = cfg["efp"]
    sf = d["eeg"]["sfreq"]; fmin, fmax, n_bands = e["freq_min"], e["freq_max"], e["n_bands"]
    edges_all = []
    for sub in subs:
        try:
            runs, chs = load_subject_features(CACHE_DIR, sub)
        except FileNotFoundError:
            continue
        if channel not in chs:
            continue
        for rd in runs:
            eeg, eeg_chs = load_eeg_run(cfg, sub, rd["run"])
            if eeg is None or channel not in eeg_chs:
                continue
            ci = eeg_chs.index(channel)
            _, band_hz, _ = channel_bandpower(eeg[ci], sf, fmin, fmax, n_bands)
            edges_all.append(np.array(band_hz))
    return np.median(np.array(edges_all), axis=0).round().astype(int)


def fit_target(subs, target, channel, n_delays, alphas):
    Xs, ys = [], []
    for sub in subs:
        try:
            runs, chs = load_subject_features(CACHE_DIR, sub)
        except FileNotFoundError:
            continue
        if channel not in chs:
            continue
        X, y = assemble(runs, chs.index(channel), target, "tr", n_delays)
        if X is None:
            continue
        Xs.append(zscore(X, axis=0)); ys.append(y)   # self-subject standardize before pooling
    Xall = np.vstack(Xs); yall = np.concatenate(ys)
    model = RidgeCV(alphas=alphas).fit(Xall, yall)
    r = np.corrcoef(model.predict(Xall), yall)[0, 1]
    print(f"{target}@{channel}: n_subjects={len(Xs)} n={len(yall)} feat={Xall.shape[1]} "
          f"alpha={model.alpha_:g} in-sample r={r:+.3f}")
    return model


def main():
    cfg = load_config()
    e = cfg["efp"]; tr = cfg["data"]["fmri"]["tr"]
    n_delays = int(round(e["delay_window_s"] / tr)) + 1
    alphas = np.logspace(np.log10(e["alpha_grid_lo"]), np.log10(e["alpha_grid_hi"]), e["alpha_grid_n"])
    subs = cfg["data"]["subjects"]["all"]

    model = {"n_bands": e["n_bands"], "n_delays": n_delays, "tr": tr,
             "sfreq": cfg["data"]["eeg"]["sfreq"], "fmin": e["freq_min"], "fmax": e["freq_max"],
             "window_tr": 1, "layout": "per-target single channel, delay-major, band-minor",
             "pda": "derived downstream as cen - dmn, not independently fit (see docstring)"}
    offset = 0
    for target, channel in TARGET_CHANNEL.items():
        key = target.lower()
        m = fit_target(subs, target, channel, n_delays, alphas)
        n_feat = len(m.coef_)
        model[f"{key}_channel"] = channel
        model[f"{key}_coef"] = m.coef_.astype(np.float32)
        model[f"{key}_intercept"] = float(m.intercept_)
        model[f"{key}_alpha"] = float(m.alpha_)
        model[f"{key}_feat_mean"] = np.zeros(n_feat, dtype=np.float32)
        model[f"{key}_feat_std"] = np.ones(n_feat, dtype=np.float32)
        model[f"{key}_offset"] = offset
        offset += n_feat
        band_edges = frozen_band_edges(cfg, subs, channel)
        model[f"{key}_band_edges_hz"] = band_edges
        print(f"  {channel} frozen bands (Hz): {band_edges.tolist()}")
    model["n_train_subjects"] = len(subs)
    model["total_features"] = offset

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH, **model)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
