#!/usr/bin/env python3
"""
loso_fingerprint_eval.py
------------------------
Leave-One-Subject-Out evaluation of cross-subject EEG fingerprints.
Tests multiple fingerprint construction × normalization × target combinations
to find the best EEG-only predictor of fMRI network state.

Usage:
  python loso_fingerprint_eval.py                    # full eval
  python loso_fingerprint_eval.py --methods pooled   # single method
  python loso_fingerprint_eval.py --targets GSR_CEN  # single target
"""
import argparse, sys, warnings, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from scipy.stats import pearsonr, rankdata

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJ_DIR / "scripts"))
from bandpower import load_config, canonical_hrf, zscore
from multivariate_decode_pda import (
    load_subject_data, load_confounds_run, prepare_targets,
    car_and_flatten, make_model, residualize, CONFIG_PATH,
)

warnings.filterwarnings("ignore")
import mne; mne.set_log_level("ERROR")

BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]

# DiFuMo-64 parcel indices for DMN and CEN (0-indexed, from Yeo17 mapping)
DIFUMO_DMN = [0, 3, 21, 29, 38, 52, 58, 60, 61]
DIFUMO_CEN = [4, 13, 17, 31, 35, 37, 47, 48, 50, 51, 53]

ALL_METHODS = ["pooled", "weight_avg", "topK10", "topK12", "ridge100", "ridge1000"]
ALL_NORMS = ["runz", "runz_subz", "relative", "rank"]
ALL_TARGETS = ["GSR_CEN", "PDA", "RAW_DMN", "GSR_DMN", "ATLAS_CEN", "ATLAS_DMN", "ATLAS_PDA"]


def load_all_subjects(cfg, subjects, cache_dir):
    """Load bandpower + confounds for all subjects. Returns dict sub -> (runs_data, confounds, ch_names)."""
    data = {}
    for sub in subjects:
        runs_data, confounds, ch_names = load_subject_data(cfg, sub, cache_dir)
        if runs_data is not None:
            data[sub] = (runs_data, confounds, ch_names)
        else:
            print(f"  WARNING: skipping {sub}")
    return data


def prepare_atlas_targets(runs_data, target_name):
    """Build atlas-based DMN/CEN/PDA targets from DiFuMo parcels (cols 0-63)."""
    pieces = []
    for rd in runs_data:
        parcels = rd["targets"]["parcels"]  # (n_tr, 64) DiFuMo
        if target_name == "ATLAS_DMN":
            y = np.mean(parcels[:, DIFUMO_DMN], axis=1)
        elif target_name == "ATLAS_CEN":
            y = np.mean(parcels[:, DIFUMO_CEN], axis=1)
        elif target_name == "ATLAS_PDA":
            y = np.mean(parcels[:, DIFUMO_CEN], axis=1) - np.mean(parcels[:, DIFUMO_DMN], axis=1)
        else:
            raise ValueError(f"Unknown atlas target: {target_name}")
        pieces.append(zscore(y))
    return np.concatenate(pieces)


def apply_normalization(X, run_boundaries, norm):
    """Apply feature normalization."""
    if norm == "runz":
        return X  # already z-scored per run in car_and_flatten
    elif norm == "runz_subz":
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    elif norm == "relative":
        n_ch = X.shape[1] // len(BAND_NAMES)
        X_rel = X.copy()
        for bi in range(len(BAND_NAMES)):
            start_col = bi * n_ch
            end_col = (bi + 1) * n_ch
            total_power = np.abs(X[:, :]).sum(axis=1, keepdims=True) + 1e-10
            X_rel[:, start_col:end_col] = X[:, start_col:end_col] / total_power
        return X_rel
    elif norm == "rank":
        X_rank = X.copy()
        for ri, (start, end) in enumerate(run_boundaries):
            for col in range(X.shape[1]):
                X_rank[start:end, col] = rankdata(X[start:end, col]) / (end - start)
        return X_rank
    else:
        raise ValueError(f"Unknown norm: {norm}")


def get_target(runs_data, confounds, target_name):
    """Get target vector for any target type."""
    if target_name.startswith("ATLAS_"):
        return prepare_atlas_targets(runs_data, target_name)
    else:
        targets = prepare_targets(runs_data, confounds, [target_name])
        return targets[target_name]


def run_loso_method(all_data, subjects, method, norm, target_name, cfg):
    """Run one LOSO evaluation for a given method/norm/target combination."""
    results = []

    for test_sub in subjects:
        if test_sub not in all_data:
            continue
        train_subs = [s for s in subjects if s != test_sub and s in all_data]

        # ── Build training data ──
        train_X_parts, train_y_parts, train_bounds_parts = [], [], []
        offset = 0

        if method == "weight_avg":
            # Fit per-subject models, average weights
            sub_coefs = []
            for sub in train_subs:
                runs_data, confounds, ch_names = all_data[sub]
                X_sub, bounds_sub = car_and_flatten(runs_data, BAND_NAMES)
                X_sub = apply_normalization(X_sub, bounds_sub, norm)
                y_sub = get_target(runs_data, confounds, target_name)
                scaler = StandardScaler()
                X_sc = scaler.fit_transform(X_sub)
                model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
                model.fit(X_sc, y_sub)
                sub_coefs.append(model.coef_)

            avg_coef = np.mean(sub_coefs, axis=0)

            # Apply to test subject
            test_runs, test_conf, test_ch = all_data[test_sub]
            X_test, test_bounds = car_and_flatten(test_runs, BAND_NAMES)
            X_test = apply_normalization(X_test, test_bounds, norm)
            y_test = get_target(test_runs, test_conf, target_name)
            scaler = StandardScaler()
            X_test_sc = scaler.fit_transform(X_test)
            pred = X_test_sc @ avg_coef

        elif method.startswith("topK"):
            K = int(method.replace("topK", ""))
            # First pass: get per-subject weight signs
            sub_coefs = []
            for sub in train_subs:
                runs_data, confounds, ch_names = all_data[sub]
                X_sub, bounds_sub = car_and_flatten(runs_data, BAND_NAMES)
                X_sub = apply_normalization(X_sub, bounds_sub, norm)
                y_sub = get_target(runs_data, confounds, target_name)
                scaler = StandardScaler()
                model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
                model.fit(scaler.fit_transform(X_sub), y_sub)
                sub_coefs.append(model.coef_)

            sub_coefs = np.array(sub_coefs)
            # Count sign agreement
            n_pos = np.sum(sub_coefs > 0, axis=0)
            n_neg = np.sum(sub_coefs < 0, axis=0)
            agree = np.maximum(n_pos, n_neg)
            mask = agree >= K

            # Pool training data with feature mask
            for sub in train_subs:
                runs_data, confounds, ch_names = all_data[sub]
                X_sub, bounds_sub = car_and_flatten(runs_data, BAND_NAMES)
                X_sub = apply_normalization(X_sub, bounds_sub, norm)
                X_sub[:, ~mask] = 0
                y_sub = get_target(runs_data, confounds, target_name)
                train_X_parts.append(X_sub)
                train_y_parts.append(y_sub)

            X_train = np.vstack(train_X_parts)
            y_train = np.concatenate(train_y_parts)
            scaler = StandardScaler()
            model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
            model.fit(scaler.fit_transform(X_train), y_train)

            test_runs, test_conf, test_ch = all_data[test_sub]
            X_test, test_bounds = car_and_flatten(test_runs, BAND_NAMES)
            X_test = apply_normalization(X_test, test_bounds, norm)
            X_test[:, ~mask] = 0
            y_test = get_target(test_runs, test_conf, target_name)
            pred = model.predict(scaler.transform(X_test))

        else:
            # Pooled methods (pooled, ridge100, ridge1000)
            for sub in train_subs:
                runs_data, confounds, ch_names = all_data[sub]
                X_sub, bounds_sub = car_and_flatten(runs_data, BAND_NAMES)
                X_sub = apply_normalization(X_sub, bounds_sub, norm)
                y_sub = get_target(runs_data, confounds, target_name)
                shifted = [(s + offset, e + offset) for s, e in bounds_sub]
                train_X_parts.append(X_sub)
                train_y_parts.append(y_sub)
                train_bounds_parts.extend(shifted)
                offset += X_sub.shape[0]

            X_train = np.vstack(train_X_parts)
            y_train = np.concatenate(train_y_parts)

            scaler = StandardScaler()
            if method == "pooled":
                model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
            elif method == "ridge100":
                model = Ridge(alpha=100)
            elif method == "ridge1000":
                model = Ridge(alpha=1000)
            else:
                raise ValueError(f"Unknown method: {method}")

            model.fit(scaler.fit_transform(X_train), y_train)

            test_runs, test_conf, test_ch = all_data[test_sub]
            X_test, test_bounds = car_and_flatten(test_runs, BAND_NAMES)
            X_test = apply_normalization(X_test, test_bounds, norm)
            y_test = get_target(test_runs, test_conf, target_name)
            pred = model.predict(scaler.transform(X_test))

        # ── Evaluate ──
        r_all, _ = pearsonr(y_test, pred)
        run_rs = []
        for start, end in test_bounds:
            r_run, _ = pearsonr(y_test[start:end], pred[start:end])
            run_rs.append(r_run)

        row = {"method": method, "norm": norm, "target": target_name,
               "test_subject": test_sub, "r": r_all, "n_train": len(train_subs)}
        for ri, rr in enumerate(run_rs):
            row[f"run{ri+1}_r"] = rr
        results.append(row)

    return results


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--norms", nargs="+", default=ALL_NORMS)
    ap.add_argument("--targets", nargs="+", default=ALL_TARGETS)
    args = ap.parse_args()

    cfg = load_config(args.config)
    subjects = cfg["data"]["subjects"]["all"] + cfg["data"]["subjects"].get("validation", [])
    cache_dir = PROJ_DIR / "results" / "multivariate" / "cache"
    out_dir = PROJ_DIR / "results" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"LOSO Fingerprint Evaluation")
    print(f"  Subjects: {len(subjects)}")
    print(f"  Methods: {args.methods}")
    print(f"  Norms: {args.norms}")
    print(f"  Targets: {args.targets}")
    n_combos = len(args.methods) * len(args.norms) * len(args.targets)
    print(f"  Total combinations: {n_combos}")
    print()

    # Load all subjects once
    print("Loading all subjects...")
    t0 = time.time()
    all_data = load_all_subjects(cfg, subjects, cache_dir)
    print(f"  Loaded {len(all_data)} subjects in {time.time()-t0:.0f}s\n")

    # Store parcels for atlas targets
    for sub in all_data:
        runs_data, confounds, ch_names = all_data[sub]
        for rd in runs_data:
            fmri = rd["targets"].get("parcels")
            if fmri is None:
                # Load from npz
                feat_dir = Path(cfg["data"]["features_dir"])
                ses = cfg["data"]["session"]
                task = cfg["data"]["task"]
                npz = np.load(feat_dir / f"sub-{sub}" /
                              f"sub-{sub}_task-{task}_run-{rd['run']}_features.npz")
                rd["targets"]["parcels"] = npz["fmri_features"][:rd["n_tr"], :64]

    all_results = []
    combo_i = 0
    for method, norm, target in itertools.product(args.methods, args.norms, args.targets):
        combo_i += 1
        print(f"[{combo_i}/{n_combos}] {method} / {norm} / {target}...", end=" ", flush=True)
        t1 = time.time()
        try:
            results = run_loso_method(all_data, subjects, method, norm, target, cfg)
            all_results.extend(results)
            r_vals = [r["r"] for r in results]
            mean_r, t_stat, p = sign_flip_test(r_vals)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"mean r={mean_r:+.4f} t={t_stat:+.2f} p={p:.4f} {sig}  ({time.time()-t1:.0f}s)")
        except Exception as e:
            print(f"ERROR: {e}")

    # Save full results
    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "loso_fingerprint_eval.csv", index=False)

    # Summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY (sorted by mean r)")
    print(f"{'='*80}")
    summary_rows = []
    for (method, norm, target), grp in df.groupby(["method", "norm", "target"]):
        r_vals = grp["r"].values
        mean_r, t_stat, p = sign_flip_test(r_vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        summary_rows.append({"method": method, "norm": norm, "target": target,
                             "mean_r": mean_r, "t": t_stat, "p": p, "sig": sig,
                             "n": len(r_vals)})

    summary = pd.DataFrame(summary_rows).sort_values("mean_r", ascending=False)
    summary.to_csv(out_dir / "loso_fingerprint_summary.csv", index=False)

    for _, row in summary.head(20).iterrows():
        print(f"  {row['method']:12s} {row['norm']:12s} {row['target']:10s}  "
              f"r={row['mean_r']:+.4f}  t={row['t']:+.2f}  p={row['p']:.4f} {row['sig']}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Results: {out_dir / 'loso_fingerprint_eval.csv'}")
    print(f"Summary: {out_dir / 'loso_fingerprint_summary.csv'}")


if __name__ == "__main__":
    main()
