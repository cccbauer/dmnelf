#!/usr/bin/env python3
"""
fig_extract_efp.py  (cluster, env eeg_preproc)  —  Fig 2 + Fig 3 assets
-----------------------------------------------------------------------
Reuses efp_features.py + the efp_cen_clean LORO ridge on CLEAN targets (feedback block).
Fig 2 (best subject): Stockwell spectrogram of one electrode, 10-band power, [band x delay]
  design example, and the fitted single-electrode ridge weight matrix (= the "fingerprint").
Fig 3 (best + worst): out-of-fold LORO predicted vs observed CEN/DMN/PDA timeseries.
Output: ~/figassets/efp_<sub>.npz
"""
import sys, argparse
from pathlib import Path
import numpy as np
B = "/projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson"
sys.path.insert(0, f"{B}/scripts")
from scipy.stats import zscore
from sklearn.linear_model import RidgeCV
from efp_features import (load_config, load_subject_features, make_delay_design,
                          load_eeg_run, channel_bandpower)
from stockwell import stockwell_power

CLEAN = "/home/cccbauer/cenrel_out"
OUT = Path.home() / "figassets"; OUT.mkdir(exist_ok=True)
BASELINE_TR, HRF_DROP = 25, 5
ALPHAS = np.logspace(-2, 5, 15)


def clean_tv(sub, runs):
    z = np.load(f"{CLEAN}/cenmean_dmnelf_{sub}.npz", allow_pickle=True)
    tv = {}
    for rd in runs:
        r = rd["run"]
        if f"run{r}" in z.files and f"run{r}_dmn" in z.files:
            cen = np.asarray(z[f"run{r}"], float); dmn = np.asarray(z[f"run{r}_dmn"], float)
            tv[r] = {"CEN": cen, "DMN": dmn, "PDA": cen - dmn}
    return tv


def run_designs(runs, n_delays, tv, target):
    """Per run -> (per-channel design list, y) feedback-masked, per-run z-scored."""
    out = []
    for rd in runs:
        if rd["run"] not in tv:
            continue
        nch = rd["bp_tr"].shape[0]; Xs = []; off = None
        for ci in range(nch):
            X, off = make_delay_design(rd["bp_tr"][ci], n_delays)
            Xs.append((X - X.mean(0)) / (X.std(0) + 1e-12))
        nvalid = Xs[0].shape[0]; t_idx = off + np.arange(nvalid)
        y = np.asarray(tv[rd["run"]][target], float)[off:off + nvalid]
        mask = (t_idx >= BASELINE_TR + HRF_DROP) & np.isfinite(y)
        if mask.sum() < 20:
            continue
        out.append(([X[mask] for X in Xs], zscore(y[mask])))
    return out


def loro_obs_pred(rd_list, cols):
    obs, pred = [], []
    Xr = [np.column_stack([r[0][ci] for ci in cols]) for r in rd_list]
    yr = [r[1] for r in rd_list]
    for i in range(len(rd_list)):
        tr = [j for j in range(len(rd_list)) if j != i]
        m = RidgeCV(alphas=ALPHAS).fit(np.vstack([Xr[j] for j in tr]),
                                       np.concatenate([yr[j] for j in tr]))
        pred.append(m.predict(Xr[i])); obs.append(yr[i])
    return np.concatenate(obs), np.concatenate(pred)


def best_electrode_cen(runs, ch_names, n_delays, tv):
    """Single-electrode CEN r ranking (feedback block, simple in-sample fit) -> best ci."""
    rd_list = run_designs(runs, n_delays, tv, "CEN")
    best, bci = -np.inf, 0
    for ci in range(len(ch_names)):
        o, p = loro_obs_pred(rd_list, [ci])
        r = np.corrcoef(o, p)[0, 1] if np.std(p) > 1e-9 else np.nan
        if np.isfinite(r) and r > best:
            best, bci = r, ci
    return bci, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", default="dmnelf1002"); ap.add_argument("--worst", default="dmnelf009")
    a = ap.parse_args()
    cfg = load_config(); tr = cfg["data"]["fmri"]["tr"]
    n_delays = int(round(cfg["efp"]["delay_window_s"] / tr)) + 1
    cache = Path(f"{B}/results/features_cache")

    for role, sub in [("best", a.best), ("worst", a.worst)]:
        runs, ch_names = load_subject_features(cache, sub)
        tv = clean_tv(sub, runs)
        blob = {"subject": sub, "role": role, "ch_names": np.array(ch_names),
                "n_delays": n_delays, "band_hz": np.array(runs[0]["band_hz"])}
        # Fig 3: predicted vs observed (multivariate LORO)
        for t in ["CEN", "DMN", "PDA"]:
            rd_list = run_designs(runs, n_delays, tv, t)
            o, p = loro_obs_pred(rd_list, list(range(len(ch_names))))
            r = np.corrcoef(o, p)[0, 1]
            blob[f"obs_{t}"] = o; blob[f"pred_{t}"] = p; blob[f"r_{t}"] = r
            print(f"{sub} {t}: r={r:+.3f} (n={len(o)})", flush=True)
        # Fig 2 (best only): spectrogram + bands + design + fingerprint weights
        if role == "best":
            bci, br = best_electrode_cen(runs, ch_names, n_delays, tv)
            blob["best_ci"] = bci; blob["best_ch"] = ch_names[bci]; blob["best_r"] = br
            print(f"  best electrode: {ch_names[bci]} (CEN r={br:+.3f})")
            # Stockwell spectrogram of best electrode, one run, 30 s window
            run0 = runs[0]["run"]
            eeg, chs = load_eeg_run(cfg, sub, run0)
            sf = cfg["data"]["eeg"]["sfreq"]
            x = eeg[chs.index(ch_names[bci])]
            w0, w1 = int(40 * sf), int(70 * sf)                 # 30 s mid-run
            freqs, power = stockwell_power(x[w0:w1], sf, cfg["efp"]["freq_min"], cfg["efp"]["freq_max"])
            dec = max(1, power.shape[1] // 1200)
            blob["spec_freqs"] = freqs; blob["spec_power"] = power[:, ::dec]
            blob["spec_t"] = np.arange(power.shape[1])[::dec] / sf
            # 10-band power timeseries (best electrode, run0, TR grid)
            r0 = runs[0]
            blob["band_power"] = r0["bp_tr"][bci]               # [10, n_tr]
            # [band x delay] design example (one TR row) reshaped
            X, off = make_delay_design(r0["bp_tr"][bci], n_delays)
            row = X[X.shape[0] // 2]
            blob["design_example"] = row.reshape(n_delays, -1)  # [delay, band]
            # single-electrode fingerprint weights: ridge on best-electrode design vs CEN
            rd_list = run_designs(runs, n_delays, tv, "CEN")
            Xall = np.vstack([r[0][bci] for r in rd_list]); yall = np.concatenate([r[1] for r in rd_list])
            m = RidgeCV(alphas=ALPHAS).fit(Xall, yall)
            blob["fingerprint"] = m.coef_.reshape(n_delays, -1)  # [delay, band]
        np.savez_compressed(OUT / f"efp_{sub}.npz", **blob)
        print(f"  saved {OUT}/efp_{sub}.npz")


if __name__ == "__main__":
    main()
