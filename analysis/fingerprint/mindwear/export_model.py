#!/usr/bin/env python3
"""
export_model.py  (cluster, env eeg_preproc)  —  freeze the EFP decoder for a given montage
--------------------------------------------------------------------------------------------
Trains CEN and DMN ridges on ALL DMNELF (clean confound-regressed targets, feedback block,
[10 band x 11 delay] design) for one electrode montage and saves a single deployable model, e.g.:

  mindwear/model/efp_epoc12_model.npz   (--montage epoc12, 12 ch — Emotiv EPOC X)
  mindwear/model/efp_cap31_model.npz    (--montage cap31,  31 ch — full research cap)

Feature/design layout MUST match the online decoder (rt_features.py):
  per channel (montage order) -> make_delay_design row = [delay0: band0..9, delay1: band0..9, ...],
  channels concatenated channel-major -> n_ch * 11 * 10 features.
Ridge is fit on per-run z-scored designs (as in efp_cen_clean); the online decoder z-scores its
inputs with per-session calibration stats, so the coefficients apply in the same units.

Run: python export_model.py --montage {epoc12,cap31} --cache <features_cache> --cenmean <cenrel_out> --out <model.npz>
"""
import argparse
from pathlib import Path
import numpy as np
import sys
B = "/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson"
sys.path.insert(0, f"{B}/scripts")
from scipy.stats import zscore
from sklearn.linear_model import RidgeCV
from efp_features import load_config, load_subject_features, make_delay_design

EPOC12 = ["F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8"]
# Full research-cap montage (31 usable EEG channels of the 32-electrode cap; 1 is reference).
CAP31 = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T7", "T8",
         "P7", "P8", "Fz", "Cz", "Pz", "Oz", "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5",
         "CP6", "TP9", "TP10", "POz"]
MONTAGES = {"epoc12": EPOC12, "cap31": CAP31}
BASELINE_TR, HRF_DROP = 25, 5
ALPHAS = np.logspace(-2, 5, 15)


def clean_targets(cenmean_dir, sub, runs):
    z = np.load(f"{cenmean_dir}/cenmean_dmnelf_{sub}.npz", allow_pickle=True)
    tv = {}
    for rd in runs:
        r = rd["run"]
        if f"run{r}" in z.files and f"run{r}_dmn" in z.files:
            tv[r] = {"CEN": np.asarray(z[f"run{r}"], float), "DMN": np.asarray(z[f"run{r}_dmn"], float)}
    return tv


def subject_designs(runs, ch_names, eidx, n_delays, tv, target):
    """Feedback-masked, per-run z-scored montage multivariate design + z-scored clean target."""
    Xs, ys = [], []
    for rd in runs:
        if rd["run"] not in tv:
            continue
        per_ch, off = [], None
        for ci in eidx:                                   # montage order
            Xc, off = make_delay_design(rd["bp_tr"][ci], n_delays)
            per_ch.append((Xc - Xc.mean(0)) / (Xc.std(0) + 1e-12))
        nvalid = per_ch[0].shape[0]; t_idx = off + np.arange(nvalid)
        y = np.asarray(tv[rd["run"]][target], float)[off:off + nvalid]
        m = (t_idx >= BASELINE_TR + HRF_DROP) & np.isfinite(y)
        if m.sum() < 20:
            continue
        Xs.append(np.column_stack([Xc[m] for Xc in per_ch]))   # channel-major
        ys.append(zscore(y[m]))
    return Xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage", choices=list(MONTAGES), default="epoc12")
    ap.add_argument("--cache", default=f"{B}/results/features_cache")
    ap.add_argument("--cenmean", default="/home/cccbauer/cenrel_out")
    ap.add_argument("--out", default=None)
    ap.add_argument("--subs", default="/home/cccbauer/efp17_subs.txt")
    a = ap.parse_args()
    out = a.out or f"/home/cccbauer/efp_{a.montage}_model.npz"
    channels = MONTAGES[a.montage]
    cfg = load_config(); tr = cfg["data"]["fmri"]["tr"]
    n_delays = int(round(cfg["efp"]["delay_window_s"] / tr)) + 1
    n_bands = cfg["efp"]["n_bands"]; fmin, fmax = cfg["efp"]["freq_min"], cfg["efp"]["freq_max"]
    subs = [s.strip() for s in Path(a.subs).read_text().split() if s.strip()]

    fitX = {"CEN": [], "DMN": []}; fitY = {"CEN": [], "DMN": []}; band_hz_all = []
    for sub in subs:
        try:
            runs, ch_names = load_subject_features(Path(a.cache), sub)
        except FileNotFoundError:
            print(f"  {sub}: no cache, skip"); continue
        eidx = [ch_names.index(c) for c in channels if c in ch_names]
        if len(eidx) != len(channels):
            print(f"  {sub}: missing {a.montage} channels, skip"); continue
        tv = clean_targets(a.cenmean, sub, runs)
        if not tv:
            continue
        for tgt in ["CEN", "DMN"]:
            Xs, ys = subject_designs(runs, ch_names, eidx, n_delays, tv, tgt)
            fitX[tgt] += Xs; fitY[tgt] += ys
        for rd in runs:                                    # collect band edges for the frozen set
            if rd.get("band_hz"):
                band_hz_all.append(np.array(rd["band_hz"]))
        print(f"  {sub}: {sum(x.shape[0] for x in Xs)} feedback TRs", flush=True)

    band_edges = np.median(np.array(band_hz_all), axis=0).round().astype(int)   # [10,2] Hz, frozen
    model = {"channels": np.array(channels), "montage": a.montage, "n_bands": n_bands,
             "n_delays": n_delays, "tr": tr, "sfreq": cfg["data"]["eeg"]["sfreq"], "fmin": fmin,
             "fmax": fmax, "band_edges_hz": band_edges,
             "layout": "channel-major, delay-major, band-minor"}
    for tgt in ["CEN", "DMN"]:
        X = np.vstack(fitX[tgt]); y = np.concatenate(fitY[tgt])
        mu, sd = X.mean(0), X.std(0) + 1e-12               # pooled scaling (pre-calibration default)
        m = RidgeCV(alphas=ALPHAS).fit((X - mu) / sd, y)
        r = np.corrcoef(m.predict((X - mu) / sd), y)[0, 1]
        model[f"{tgt.lower()}_coef"] = m.coef_.astype(np.float32)
        model[f"{tgt.lower()}_intercept"] = float(m.intercept_)
        model[f"{tgt.lower()}_alpha"] = float(m.alpha_)
        model[f"{tgt.lower()}_feat_mean"] = mu.astype(np.float32)
        model[f"{tgt.lower()}_feat_std"] = sd.astype(np.float32)
        print(f"{tgt}: n={len(y)} feat={X.shape[1]} alpha={m.alpha_:g} in-sample r={r:+.3f}", flush=True)
    model["n_train_subjects"] = len(subs)
    np.savez_compressed(out, **model)
    print(f"saved {out}  (montage={a.montage}, {len(channels)} ch, bands Hz: {band_edges.tolist()})")


if __name__ == "__main__":
    main()
