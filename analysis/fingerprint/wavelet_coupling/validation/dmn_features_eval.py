#!/usr/bin/env python3
"""
dmn_features_eval.py
--------------------
Compare feature types for DMN decoding: infraslow power, connectivity (PLV),
and combinations. Tests whether DMN can be decoded cross-subject.

Methods:
  1. dwt_stats:      DWT+stats 1-40Hz (baseline)
  2. dwt_stats_is:   DWT+stats 0.01-40Hz (infraslow bands added)
  3. plv:            PLV connectivity features 1-40Hz
  4. plv_is:         PLV connectivity 0.01-40Hz
  5. combined:       DWT+stats + PLV (all features)
  6. combined_is:    DWT+stats IS + PLV IS

Usage:
  python dmn_features_eval.py
  python dmn_features_eval.py --methods dwt_stats plv
"""
import argparse, sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPT_DIR.parent
EBC_SCRIPTS = PROJ_DIR.parent / "eeg_bold_coupling" / "scripts"
sys.path.insert(0, str(PROJ_DIR / "scripts"))
sys.path.insert(0, str(EBC_SCRIPTS))

from bandpower import load_config, canonical_hrf, zscore
from bandpower_wavelet import gather_subject_wavelet, hrf_convolve
from connectivity_features import gather_subject_plv
from multivariate_decode_pda import (
    load_confounds_run, prepare_targets, car_and_flatten, CONFIG_PATH
)

warnings.filterwarnings("ignore")
import mne; mne.set_log_level("ERROR")

CFG_PATH = PROJ_DIR / "config.yaml"
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_NAMES_IS = ["infraslow", "slow", "delta", "theta", "alpha", "beta", "gamma"]
TARGETS = ["GSR_DMN", "GSR_CEN", "PDA"]

BANDS_INFRASLOW = {
    "infraslow": [0.01, 0.1],
    "slow":      [0.1, 1],
    "delta":     [1, 4],
    "theta":     [4, 8],
    "alpha":     [8, 13],
    "beta":      [13, 30],
    "gamma":     [30, 40],
}


def sign_flip_test(r_vals, n_flips=10000):
    r_arr = np.array(r_vals)
    n = len(r_arr)
    if n < 3:
        return np.mean(r_arr), 0, 1.0
    obs_mean = np.mean(r_arr)
    obs_t = obs_mean / (np.std(r_arr, ddof=1) / np.sqrt(n))
    rng = np.random.default_rng(42)
    null = np.array([np.mean(rng.choice([-1, 1], size=n) * r_arr) for _ in range(n_flips)])
    p_right = (np.sum(null >= obs_mean) + 1) / (n_flips + 1)
    p_two = 2 * min(p_right, (np.sum(null <= obs_mean) + 1) / (n_flips + 1))
    return obs_mean, obs_t, p_two


def flatten_features(runs_data, band_names):
    """CAR + flatten for any feature type (power or PLV)."""
    pieces = []
    boundaries = []
    offset = 0
    for rd in runs_data:
        n_tr = rd["n_tr"]
        bands_list = []
        for bname in band_names:
            if bname in rd["bp"]:
                bp = rd["bp"][bname].copy()
                if bp.ndim == 2 and bp.shape[0] == n_tr:
                    # CAR: remove spatial mean per feature type
                    bp -= bp.mean(axis=1, keepdims=True)
                    bands_list.append(bp)
        if not bands_list:
            continue
        X_run = np.concatenate(bands_list, axis=1)
        X_run = zscore(X_run)
        pieces.append(X_run)
        boundaries.append((offset, offset + n_tr))
        offset += n_tr
    if not pieces:
        return np.array([]).reshape(0, 0), []
    return np.vstack(pieces), boundaries


def load_method_data(cfg, subjects, method, hrf):
    """Load features for all subjects for a given method."""
    data = {}
    for sub in subjects:
        try:
            if method == "dwt_stats":
                runs = gather_subject_wavelet(cfg, sub, hrf, method="dwt_stats")
                band_names = BAND_NAMES
            elif method == "dwt_stats_is":
                # Use infraslow EEG with extended bands
                cfg_is = dict(cfg)
                cfg_is = {**cfg, "bands": BANDS_INFRASLOW}
                # Override desc to use infraslow files
                runs = _gather_infraslow_dwt(cfg, sub, hrf)
                band_names = BAND_NAMES_IS
            elif method == "plv":
                runs = gather_subject_plv(cfg, sub, hrf, desc="preproc500Hz")
                band_names = BAND_NAMES
            elif method == "plv_is":
                runs = gather_subject_plv(cfg, sub, hrf, desc="preproc500HzISp01")
                band_names = BAND_NAMES
            elif method == "combined":
                runs_dwt = gather_subject_wavelet(cfg, sub, hrf, method="dwt_stats")
                runs_plv = gather_subject_plv(cfg, sub, hrf, desc="preproc500Hz")
                runs = _combine_runs(runs_dwt, runs_plv, BAND_NAMES)
                band_names = BAND_NAMES
            elif method == "combined_is":
                runs_dwt = _gather_infraslow_dwt(cfg, sub, hrf)
                runs_plv = gather_subject_plv(cfg, sub, hrf, desc="preproc500HzISp01")
                runs = _combine_runs(runs_dwt, runs_plv, BAND_NAMES_IS if runs_dwt else BAND_NAMES)
                band_names = BAND_NAMES_IS if runs_dwt else BAND_NAMES
            else:
                continue

            if runs:
                confounds = [load_confounds_run(cfg, sub, rd["run"])[:rd["n_tr"]] for rd in runs]
                data[sub] = (runs, confounds, band_names)
        except Exception as e:
            print(f"  WARNING: {sub} {method}: {e}")
    return data


def _gather_infraslow_dwt(cfg, sub, hrf):
    """Gather DWT+stats from infraslow EEG files with extended bands."""
    from bandpower_wavelet import wavelet_power_run
    d = cfg["data"]; ses = d["session"]; task = d["task"]
    spt = d["eeg"]["samples_per_tr"]
    dmn_i = d["fmri"]["dmn_idx"]; cen_i = d["fmri"]["cen_idx"]; ndi = d["fmri"]["n_difumo"]
    fdir = Path(d["features_dir"]) / f"sub-{sub}"
    eroot = Path(d["eeg_preproc_dir"]); runs = []

    for npz in sorted(fdir.glob(f"sub-{sub}_task-{task}_run-*_features.npz")):
        z = np.load(npz, allow_pickle=True)
        fm = np.asarray(z["fmri_features"], float)
        n_tr = fm.shape[0]; run = npz.name.split("run-")[1][0]
        fif = (eroot/f"sub-{sub}"/ses/"eeg" /
               f"sub-{sub}_{ses}_task-{task}_run-{int(run):02d}_desc-preproc500HzISp01_eeg.fif")
        if not fif.exists():
            continue
        raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
        bp, chs = wavelet_power_run(raw, BANDS_INFRASLOW, spt, n_tr, hrf, method="dwt_stats")
        targets = dict(DMN=fm[:, dmn_i], CEN=fm[:, cen_i], PDA=fm[:, cen_i] - fm[:, dmn_i])
        runs.append(dict(run=run, n_tr=n_tr, targets=targets,
                         bp=bp, parcels=fm[:, :ndi], chs=chs))
    return runs


def _combine_runs(runs_a, runs_b, band_names):
    """Combine two feature sets (e.g., DWT+stats + PLV) by concatenating per band."""
    if not runs_a or not runs_b:
        return runs_a or runs_b or []
    combined = []
    for ra, rb in zip(runs_a, runs_b):
        bp_combined = {}
        for bname in band_names:
            parts = []
            if bname in ra["bp"]:
                parts.append(ra["bp"][bname])
            if bname in rb["bp"]:
                parts.append(rb["bp"][bname])
            if parts:
                bp_combined[bname] = np.concatenate(parts, axis=1)
        combined.append(dict(run=ra["run"], n_tr=ra["n_tr"], targets=ra["targets"],
                             bp=bp_combined, parcels=ra["parcels"],
                             chs=list(ra.get("chs", [])) + list(rb.get("chs", []))))
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CFG_PATH))
    ap.add_argument("--methods", nargs="+",
                    default=["dwt_stats", "dwt_stats_is", "plv", "plv_is", "combined", "combined_is"])
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    args = ap.parse_args()

    cfg = load_config(args.config)
    subjects = cfg["data"]["subjects"]["all"] + cfg["data"]["subjects"].get("validation", [])
    hrf = canonical_hrf(tr=cfg["data"]["fmri"]["tr"], length_s=cfg["hrf"]["length_s"])
    out_dir = PROJ_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"DMN Features Evaluation")
    print(f"  Subjects: {len(subjects)}")
    print(f"  Methods: {args.methods}")
    print(f"  Targets: {args.targets}")
    print()

    t0 = time.time()
    results = []

    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"  Loading {method}...")
        method_data = load_method_data(cfg, subjects, method, hrf)
        n_loaded = len(method_data)
        print(f"  Loaded {n_loaded} subjects")

        if n_loaded < 3:
            print(f"  Too few subjects, skipping")
            continue

        for target in args.targets:
            print(f"\n  {method} / {target}:")
            within_rs, cross_rs = [], []

            for test_sub in subjects:
                if test_sub not in method_data:
                    continue
                runs, confounds, band_names = method_data[test_sub]
                X_test, test_bounds = flatten_features(runs, band_names)
                if X_test.shape[0] == 0:
                    continue
                y_test = prepare_targets(runs, confounds, [target])[target]

                # Within-subject (5-fold)
                folds = list(KFold(n_splits=5, shuffle=False).split(np.arange(len(y_test))))
                r_folds = []
                for train_idx, test_idx_cv in folds:
                    scaler = StandardScaler()
                    model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
                    model.fit(scaler.fit_transform(X_test[train_idx]), y_test[train_idx])
                    pred = model.predict(scaler.transform(X_test[test_idx_cv]))
                    r, _ = pearsonr(y_test[test_idx_cv], pred)
                    r_folds.append(r)
                within_rs.append(np.mean(r_folds))

                # Cross-subject (LOSO)
                train_subs = [s for s in subjects if s != test_sub and s in method_data]
                train_X_parts, train_y_parts = [], []
                for ts in train_subs:
                    rd, cf, bn = method_data[ts]
                    Xs, bs = flatten_features(rd, bn)
                    if Xs.shape[0] == 0:
                        continue
                    ys = prepare_targets(rd, cf, [target])[target]
                    train_X_parts.append(Xs)
                    train_y_parts.append(ys)

                if not train_X_parts:
                    continue
                X_train = np.vstack(train_X_parts)
                y_train = np.concatenate(train_y_parts)
                scaler = StandardScaler()
                model = Ridge(alpha=1000)
                model.fit(scaler.fit_transform(X_train), y_train)
                pred = model.predict(scaler.transform(X_test))
                r_cross, _ = pearsonr(y_test, pred)
                cross_rs.append(r_cross)

                results.append({
                    "method": method, "target": target, "subject": test_sub,
                    "within_r": np.mean(r_folds), "cross_r": r_cross,
                    "n_features": X_test.shape[1], "n_subjects": n_loaded,
                })

                print(f"    {test_sub}: within={np.mean(r_folds):+.3f}  cross={r_cross:+.3f}  "
                      f"({X_test.shape[1]}f)")

            if within_rs:
                w_mean, w_t, w_p = sign_flip_test(within_rs)
                c_mean, c_t, c_p = sign_flip_test(cross_rs)
                w_sig = "***" if w_p < 0.001 else "**" if w_p < 0.01 else "*" if w_p < 0.05 else ""
                c_sig = "***" if c_p < 0.001 else "**" if c_p < 0.01 else "*" if c_p < 0.05 else ""
                print(f"    GROUP (n={len(within_rs)}): within={w_mean:+.4f} p={w_p:.4f} {w_sig}  "
                      f"cross={c_mean:+.4f} p={c_p:.4f} {c_sig}")

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "dmn_features_eval.csv", index=False)

    # Summary
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    summary_rows = []
    for (method, target), grp in df.groupby(["method", "target"]):
        w_mean, w_t, w_p = sign_flip_test(grp["within_r"].values)
        c_mean, c_t, c_p = sign_flip_test(grp["cross_r"].values)
        w_sig = "***" if w_p < 0.001 else "**" if w_p < 0.01 else "*" if w_p < 0.05 else ""
        c_sig = "***" if c_p < 0.001 else "**" if c_p < 0.01 else "*" if c_p < 0.05 else ""
        n_feat = int(grp["n_features"].iloc[0])
        n_sub = int(grp["n_subjects"].iloc[0])
        print(f"  {method:14s} {target:8s}  within={w_mean:+.4f} {w_sig:3s}  "
              f"cross={c_mean:+.4f} {c_sig:3s}  ({n_feat}f, n={n_sub})")
        summary_rows.append({
            "method": method, "target": target, "n_features": n_feat,
            "n_subjects": n_sub,
            "within_mean_r": w_mean, "within_p": w_p,
            "cross_mean_r": c_mean, "cross_p": c_p,
        })

    pd.DataFrame(summary_rows).to_csv(out_dir / "dmn_features_summary.csv", index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
