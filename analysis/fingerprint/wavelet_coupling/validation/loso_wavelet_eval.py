#!/usr/bin/env python3
"""
loso_wavelet_eval.py
--------------------
Compare wavelet vs Hilbert features for EEG→fMRI decoding.
Within-subject (5-fold) and cross-subject (LOSO) evaluation.

Usage:
  python loso_wavelet_eval.py                           # all methods
  python loso_wavelet_eval.py --methods hilbert morlet   # subset
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

# Import from eeg_bold_coupling for Hilbert baseline
EBC_SCRIPTS = PROJ_DIR.parent / "eeg_bold_coupling" / "scripts"
sys.path.insert(0, str(EBC_SCRIPTS))
sys.path.insert(0, str(PROJ_DIR / "scripts"))

from bandpower import load_config, canonical_hrf, zscore, gather_subject
from bandpower_wavelet import gather_subject_wavelet
from multivariate_decode_pda import (
    load_subject_data, load_confounds_run, prepare_targets,
    car_and_flatten, make_model, residualize, contiguous_folds,
)

warnings.filterwarnings("ignore")
import mne; mne.set_log_level("ERROR")

CONFIG_PATH = PROJ_DIR / "config.yaml"
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
ALL_METHODS = ["hilbert", "morlet", "dwt", "dwt_stats"]
TARGETS = ["GSR_CEN", "PDA"]


def load_subject_wavelet(cfg, sub, method, hrf, cache_dir):
    """Load wavelet features for one subject, with caching."""
    cache_file = cache_dir / f"{sub}_{method}_bandpower.npz"
    if cache_file.exists():
        cached = np.load(cache_file, allow_pickle=True)
        return list(cached["runs_data"])

    runs_data = gather_subject_wavelet(cfg, sub, hrf, method=method)
    if runs_data:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, runs_data=np.array(runs_data, dtype=object))
    return runs_data


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


def run_within_subject(X, y, run_boundaries, n_folds=5):
    """Within-subject 5-fold contiguous CV. Returns mean r."""
    folds = list(KFold(n_splits=n_folds, shuffle=False).split(np.arange(len(y))))
    r_folds = []
    for train_idx, test_idx in folds:
        scaler = StandardScaler()
        model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
        model.fit(scaler.fit_transform(X[train_idx]), y[train_idx])
        pred = model.predict(scaler.transform(X[test_idx]))
        r, _ = pearsonr(y[test_idx], pred)
        r_folds.append(r)
    return np.mean(r_folds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    args = ap.parse_args()

    cfg = load_config(args.config)
    subjects = cfg["data"]["subjects"]["all"] + cfg["data"]["subjects"].get("validation", [])
    hrf = canonical_hrf(tr=cfg["data"]["fmri"]["tr"], length_s=cfg["hrf"]["length_s"])
    hilbert_cache = PROJ_DIR.parent / "eeg_bold_coupling" / "results" / "multivariate" / "cache"
    wavelet_cache = PROJ_DIR / "results" / "cache"
    wavelet_cache.mkdir(parents=True, exist_ok=True)
    out_dir = PROJ_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Wavelet vs Hilbert Evaluation")
    print(f"  Subjects: {len(subjects)}")
    print(f"  Methods: {args.methods}")
    print(f"  Targets: {args.targets}")
    print()

    # ── Load all subjects for each method ──
    t0 = time.time()
    all_data = {}  # method -> sub -> (runs_data, confounds, ch_names)

    for method in args.methods:
        print(f"Loading {method}...")
        all_data[method] = {}
        for sub in subjects:
            if method == "hilbert":
                runs_data, confounds, ch_names = load_subject_data(cfg, sub, hilbert_cache)
            else:
                runs_data = load_subject_wavelet(cfg, sub, method, hrf, wavelet_cache)
                if runs_data:
                    confounds = []
                    for rd in runs_data:
                        gs = load_confounds_run(cfg, sub, rd["run"])
                        confounds.append(gs[:rd["n_tr"]])
                    ch_names = runs_data[0]["chs"]
                else:
                    runs_data, confounds, ch_names = None, None, None

            if runs_data is not None:
                all_data[method][sub] = (runs_data, confounds, ch_names)
                print(f"  {sub}: {len(runs_data)} runs")
            else:
                print(f"  {sub}: SKIPPED")

    print(f"\nAll data loaded in {time.time()-t0:.0f}s\n")

    # ── Evaluate ──
    results = []

    for method in args.methods:
        for target in args.targets:
            print(f"\n{method} / {target}:")

            within_rs = []
            cross_rs = []

            for test_sub in subjects:
                if test_sub not in all_data[method]:
                    continue

                runs_data, confounds, ch_names = all_data[method][test_sub]
                band_names = list(cfg["bands"].keys())

                # Get features
                X_test, test_bounds = car_and_flatten(runs_data, band_names)
                y_test = prepare_targets(runs_data, confounds, [target])[target]

                # ── Within-subject (5-fold) ──
                r_within = run_within_subject(X_test, y_test, test_bounds)
                within_rs.append(r_within)

                # ── Cross-subject (LOSO) ──
                train_subs = [s for s in subjects if s != test_sub and s in all_data[method]]
                train_X_parts, train_y_parts = [], []
                for ts in train_subs:
                    rd, cf, ch = all_data[method][ts]
                    Xs, bs = car_and_flatten(rd, band_names)
                    ys = prepare_targets(rd, cf, [target])[target]
                    train_X_parts.append(Xs)
                    train_y_parts.append(ys)

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
                    "within_r": r_within, "cross_r": r_cross,
                    "n_features": X_test.shape[1],
                })

                print(f"  {test_sub}: within={r_within:+.3f}  cross={r_cross:+.3f}  "
                      f"({X_test.shape[1]} features)")

            # Group summary
            if within_rs:
                w_mean, w_t, w_p = sign_flip_test(within_rs)
                c_mean, c_t, c_p = sign_flip_test(cross_rs)
                w_sig = "***" if w_p < 0.001 else "**" if w_p < 0.01 else "*" if w_p < 0.05 else ""
                c_sig = "***" if c_p < 0.001 else "**" if c_p < 0.01 else "*" if c_p < 0.05 else ""
                print(f"  GROUP: within={w_mean:+.4f} p={w_p:.4f} {w_sig}  "
                      f"cross={c_mean:+.4f} p={c_p:.4f} {c_sig}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "wavelet_vs_hilbert_eval.csv", index=False)

    # Summary table
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
        print(f"  {method:12s} {target:8s}  within={w_mean:+.4f} p={w_p:.4f} {w_sig:3s}  "
              f"cross={c_mean:+.4f} p={c_p:.4f} {c_sig:3s}  ({n_feat} features)")
        summary_rows.append({
            "method": method, "target": target, "n_features": n_feat,
            "within_mean_r": w_mean, "within_p": w_p,
            "cross_mean_r": c_mean, "cross_p": c_p,
        })

    pd.DataFrame(summary_rows).to_csv(out_dir / "wavelet_vs_hilbert_summary.csv", index=False)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
