#!/usr/bin/env python3
"""
multivariate_decode_pda.py
--------------------------
Within-subject Ridge / ElasticNet: CAR'd HRF-band-power -> PDA / GSR'd DMN / GSR'd CEN / RAW DMN.
Uses bandpower.py for feature extraction. Runs locally or on cluster (auto-detect).

Usage:
  python multivariate_decode_pda.py --fast                  # local: PDA+RAW_DMN, ridge, no null (quick)
  python multivariate_decode_pda.py                         # full: all targets, ridge+enet, 1000 nulls
  python multivariate_decode_pda.py --subjects dmnelf001    # single subject
  python multivariate_decode_pda.py --group                 # group sign-flip test
  python multivariate_decode_pda.py --targets PDA GSR_DMN   # subset of targets
  python multivariate_decode_pda.py --models ridge          # ridge only
  python multivariate_decode_pda.py --n_shuffles 10000      # publication-grade nulls
"""
import argparse, sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import mne

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bandpower import load_config, canonical_hrf, gather_subject, zscore

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

ALL_TARGETS = ["PDA", "GSR_DMN", "GSR_CEN", "RAW_DMN"]
ALL_MODELS = ["ridge", "elasticnet"]
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_confounds_run(cfg, sub, run_idx):
    """Load global_signal from fMRIPrep confounds for one run. Returns (n_tr,) array."""
    cdir = Path(cfg["data"]["confounds_dir"])
    ses = cfg["data"]["session"]
    task = cfg["data"]["task"]
    tsv = (cdir / f"sub-{sub}" / ses / "func" /
           f"sub-{sub}_{ses}_task-{task}_run-{int(run_idx):02d}_desc-confounds_timeseries.tsv")
    df = pd.read_csv(tsv, sep="\t")
    gs = df["global_signal"].values.astype(float)
    gs[0] = gs[1]  # first row is NaN for FD; global_signal may also be n/a
    return gs


def residualize(y, confound):
    """OLS residualize y on [1, confound]."""
    X = np.column_stack([np.ones_like(confound), confound])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


def prepare_targets(runs_data, confounds, target_names):
    """Build target vectors for each target name, per-run z-scored then concatenated.

    runs_data: list of dicts from gather_subject (keys: targets, run, n_tr, ...)
    confounds: list of global_signal arrays, one per run (aligned with runs_data)
    target_names: list of target names to compute

    Returns dict target_name -> (total_trs,) array.
    """
    out = {}
    for tname in target_names:
        pieces = []
        for rd, gs in zip(runs_data, confounds):
            if tname == "PDA":
                y = rd["targets"]["PDA"].copy()
            elif tname == "RAW_DMN":
                y = rd["targets"]["DMN"].copy()
            elif tname == "GSR_DMN":
                y = residualize(rd["targets"]["DMN"].copy(), gs)
            elif tname == "GSR_CEN":
                y = residualize(rd["targets"]["CEN"].copy(), gs)
            else:
                raise ValueError(f"Unknown target: {tname}")
            pieces.append(zscore(y))
        out[tname] = np.concatenate(pieces)
    return out


def car_and_flatten(runs_data, band_names):
    """CAR per band, z-score per run, concatenate, return X (total_trs, n_ch*n_bands) and run_boundaries."""
    pieces = []
    boundaries = []
    offset = 0
    for rd in runs_data:
        n_tr = rd["n_tr"]
        bands_list = []
        for bname in band_names:
            bp = rd["bp"][bname].copy()  # (n_tr, n_ch)
            bp -= bp.mean(axis=1, keepdims=True)  # CAR: remove spatial mean
            bands_list.append(bp)
        X_run = np.concatenate(bands_list, axis=1)  # (n_tr, n_ch * n_bands)
        X_run = zscore(X_run)
        pieces.append(X_run)
        boundaries.append((offset, offset + n_tr))
        offset += n_tr
    return np.vstack(pieces), boundaries


def contiguous_folds(n, k):
    return list(KFold(n_splits=k, shuffle=False).split(np.arange(n)))


def make_model(name, alpha):
    if name == "ridge":
        return Ridge(alpha=alpha)
    elif name == "elasticnet":
        return ElasticNet(alpha=alpha * 0.01, l1_ratio=0.5, max_iter=10000)
    raise ValueError(f"Unknown model: {name}")


def run_cv(X, y, model, folds):
    """Return per-fold r values and concatenated predictions."""
    r_folds = []
    pred_all = np.full(len(y), np.nan)
    coefs = []
    for train_idx, test_idx in folds:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        model.fit(X_tr, y[train_idx])
        pred = model.predict(X_te)
        pred_all[test_idx] = pred
        r, _ = pearsonr(y[test_idx], pred)
        r_folds.append(r)
        coefs.append(model.coef_.copy())
    return r_folds, pred_all, np.array(coefs)


def circular_shift_null(X, y, model_name, alpha, folds, run_boundaries, n_shuffles, rng):
    """Null distribution: circular-shift y within each run, run CV, return null mean-r."""
    r_null = np.zeros(n_shuffles)
    for i in range(n_shuffles):
        y_shifted = np.empty_like(y)
        for start, end in run_boundaries:
            n = end - start
            shift = rng.integers(5, n - 5)
            y_shifted[start:end] = np.roll(y[start:end], shift)
        r_fold = []
        model = make_model(model_name, alpha)
        for train_idx, test_idx in folds:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            model.fit(X_tr, y_shifted[train_idx])
            pred = model.predict(X_te)
            r, _ = pearsonr(y_shifted[test_idx], pred)
            r_fold.append(r)
        r_null[i] = np.mean(r_fold)
    return r_null


def load_subject_data(cfg, sub, cache_dir):
    """Load (or extract + cache) bandpower and confounds for one subject."""
    band_names = list(cfg["bands"].keys())
    hrf = canonical_hrf(tr=cfg["data"]["fmri"]["tr"], length_s=cfg["hrf"]["length_s"])

    cache_file = cache_dir / f"{sub}_bandpower.npz"
    if cache_file.exists():
        print(f"  Loading cached features from {cache_file.name}")
        cached = np.load(cache_file, allow_pickle=True)
        runs_data = list(cached["runs_data"])
        ch_names = list(cached["ch_names"])
    else:
        print(f"  Extracting band power (this takes a few minutes)...")
        t0 = time.time()
        runs_data = gather_subject(cfg, sub, hrf)
        if not runs_data:
            print(f"  WARNING: no runs found for {sub}, skipping.")
            return None, None, None
        ch_names = runs_data[0]["chs"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, runs_data=np.array(runs_data, dtype=object),
                            ch_names=ch_names)
        print(f"  Band power extracted in {time.time()-t0:.0f}s, cached to {cache_file.name}")

    if not runs_data:
        return None, None, None

    confounds = []
    for rd in runs_data:
        gs = load_confounds_run(cfg, sub, rd["run"])
        confounds.append(gs[:rd["n_tr"]])

    return runs_data, confounds, ch_names


def loro_folds(run_boundaries):
    """Leave-one-run-out: each run is a test fold, rest are training."""
    n_total = run_boundaries[-1][1]
    folds = []
    for i, (start, end) in enumerate(run_boundaries):
        test_idx = np.arange(start, end)
        train_idx = np.concatenate([np.arange(s, e) for j, (s, e) in enumerate(run_boundaries) if j != i])
        folds.append((train_idx, test_idx))
    return folds


def process_subject(cfg, sub, target_names, model_names, n_shuffles, alpha, cv_folds, out_dir, cache_dir, use_loro=False):
    """Run decode for one subject. Returns list of result dicts."""
    print(f"\n{'='*60}")
    print(f"  {sub}")
    print(f"{'='*60}")

    band_names = list(cfg["bands"].keys())
    runs_data, confounds, ch_names = load_subject_data(cfg, sub, cache_dir)
    if runs_data is None:
        return []

    X, run_boundaries = car_and_flatten(runs_data, band_names)
    total_trs = X.shape[0]
    targets = prepare_targets(runs_data, confounds, target_names)

    cv_label = "LORO" if use_loro else f"{cv_folds}-fold"
    print(f"  Features: {X.shape[1]} (={len(ch_names)}ch x {len(band_names)}bands), {total_trs} TRs pooled, CV={cv_label}")

    if use_loro:
        folds = loro_folds(run_boundaries)
    else:
        folds = contiguous_folds(total_trs, cv_folds)
    rng = np.random.default_rng(42)

    results = []
    for tname in target_names:
        y = targets[tname]
        for mname in model_names:
            model = make_model(mname, alpha)
            r_folds, pred, coefs = run_cv(X, y, model, folds)
            mean_r = np.mean(r_folds)

            # Circular-shift null (skip if n_shuffles == 0)
            if n_shuffles > 0:
                r_null = circular_shift_null(X, y, mname, alpha, folds, run_boundaries, n_shuffles, rng)
                p = (np.sum(r_null >= mean_r) + 1) / (n_shuffles + 1)
            else:
                p = np.nan

            sig = "*" if p < 0.05 else "" if np.isnan(p) else ""
            p_str = f"p={p:.4f}" if not np.isnan(p) else "p=--"
            print(f"  {tname:8s} {mname:11s}  r={mean_r:+.4f}  {p_str} {sig}")

            n_folds_actual = len(folds)
            row = {"subject": sub, "target": tname, "model": mname,
                   "mean_cv_r": mean_r, "p_circ_shift": p,
                   "n_features": X.shape[1], "n_trs": total_trs,
                   "cv_folds": n_folds_actual, "cv_type": "loro" if use_loro else "contiguous",
                   "n_shuffles": n_shuffles, "alpha": alpha}
            for fi, rv in enumerate(r_folds):
                row[f"fold{fi}_r"] = rv
            results.append(row)

            pd.DataFrame([row]).to_csv(out_dir / f"{sub}_{tname}_{mname}.csv", index=False)

            if mname == "elasticnet":
                np.savez_compressed(out_dir / f"{sub}_{tname}_{mname}_weights.npz",
                                    coefs=coefs, ch_names=ch_names, band_names=band_names)

    return results


def group_test(out_dir, target_names, model_names, subjects, n_flips=10000):
    """Group sign-flip test on per-subject mean CV correlations."""
    print(f"\n{'='*60}")
    print(f"  GROUP SIGN-FLIP TEST")
    print(f"{'='*60}")

    all_rows = []
    for tname in target_names:
        for mname in model_names:
            r_vals = []
            for sub in subjects:
                f = out_dir / f"{sub}_{tname}_{mname}.csv"
                if f.exists():
                    df = pd.read_csv(f)
                    r_vals.append(df["mean_cv_r"].iloc[0])
            if len(r_vals) < 3:
                print(f"  {tname:8s} {mname:11s}  too few subjects ({len(r_vals)})")
                continue

            r_vals = np.array(r_vals)
            n_sub = len(r_vals)
            obs_mean = np.mean(r_vals)
            obs_t = obs_mean / (np.std(r_vals, ddof=1) / np.sqrt(n_sub))

            rng = np.random.default_rng(42)
            null_means = np.zeros(n_flips)
            for i in range(n_flips):
                signs = rng.choice([-1, 1], size=n_sub)
                null_means[i] = np.mean(signs * r_vals)

            p_right = (np.sum(null_means >= obs_mean) + 1) / (n_flips + 1)
            p_two = 2 * min(p_right, (np.sum(null_means <= obs_mean) + 1) / (n_flips + 1))

            sig = "***" if p_two < 0.001 else "**" if p_two < 0.01 else "*" if p_two < 0.05 else ""
            print(f"  {tname:8s} {mname:11s}  mean r={obs_mean:+.4f}  t={obs_t:+.2f}  "
                  f"p_right={p_right:.4f}  p_two={p_two:.4f} {sig}  (n={n_sub})")

            row = {"target": tname, "model": mname, "mean_subject_r": obs_mean,
                   "t_stat": obs_t, "n_subjects": n_sub,
                   "p_right_tailed": p_right, "p_two_tailed": p_two}
            all_rows.append(row)

            pd.DataFrame([row]).to_csv(out_dir / f"group_{tname}_{mname}.csv", index=False)

    if all_rows:
        summary = pd.DataFrame(all_rows)
        summary.to_csv(out_dir / "multivariate_summary.csv", index=False)
        print(f"\n  Summary saved to {out_dir / 'multivariate_summary.csv'}")
    return all_rows


def main():
    ap = argparse.ArgumentParser(description="Multivariate EEG band-power -> fMRI target decoding")
    ap.add_argument("--config", type=str, default=str(CONFIG_PATH))
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="Subject IDs (default: all from config)")
    ap.add_argument("--targets", nargs="+", default=ALL_TARGETS,
                    choices=ALL_TARGETS, help="Targets to decode")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS,
                    choices=ALL_MODELS, help="Models to fit")
    ap.add_argument("--n_shuffles", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=1.0, help="Regularization strength")
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--group", action="store_true", help="Run group test only (from existing per-subject CSVs)")
    ap.add_argument("--no_cache", action="store_true", help="Skip feature cache")
    ap.add_argument("--fast", action="store_true",
                    help="Quick local run: PDA+RAW_DMN, ridge only, no per-subject null")
    ap.add_argument("--loro", action="store_true",
                    help="Leave-one-run-out CV (train 3 runs, test 1) instead of contiguous folds")
    args = ap.parse_args()

    if args.fast:
        args.targets = ["PDA", "RAW_DMN"]
        args.models = ["ridge"]
        args.n_shuffles = 0

    cfg = load_config(args.config)
    subjects = args.subjects or cfg["data"]["subjects"]["all"]
    proj_dir = CONFIG_PATH.parent  # always resolve relative to config.yaml location
    subdir = "multivariate_loro" if args.loro else "multivariate"
    out_dir = proj_dir / "results" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = proj_dir / "results" / "multivariate" / "cache"  # shared cache

    if args.group:
        group_test(out_dir, args.targets, args.models, subjects)
        return

    cv_label = "LORO (leave-one-run-out)" if args.loro else f"{args.cv_folds}-fold contiguous"
    print(f"Multivariate decode: {len(subjects)} subjects, targets={args.targets}, models={args.models}")
    print(f"CV={cv_label}, {args.n_shuffles} circular-shift nulls, alpha={args.alpha}")
    print(f"Output: {out_dir}")

    t0 = time.time()
    all_results = []
    for sub in subjects:
        res = process_subject(cfg, sub, args.targets, args.models,
                              args.n_shuffles, args.alpha, args.cv_folds,
                              out_dir, cache_dir if not args.no_cache else Path("/dev/null"),
                              use_loro=args.loro)
        all_results.extend(res)

    elapsed = time.time() - t0
    print(f"\nAll subjects done in {elapsed/60:.1f} min.")

    # Auto-run group test
    group_test(out_dir, args.targets, args.models, subjects)


if __name__ == "__main__":
    main()
