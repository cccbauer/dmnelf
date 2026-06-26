#!/usr/bin/env python3
"""
cross_subject_predict.py
------------------------
Cross-subject validation: train ElasticNet on N existing subjects,
predict on held-out dmnelf016's 4 feedback runs.

Usage:
  python cross_subject_predict.py                          # full run (10K nulls)
  python cross_subject_predict.py --n_shuffles 0           # quick (no null)
  python cross_subject_predict.py --test_subject dmnelf016 # explicit test subject
"""
import argparse, sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJ_DIR / "scripts"))
from bandpower import load_config, canonical_hrf, gather_subject, zscore
from multivariate_decode_pda import (
    load_subject_data, load_confounds_run, prepare_targets,
    car_and_flatten, make_model, residualize, CONFIG_PATH,
)

warnings.filterwarnings("ignore")
import mne; mne.set_log_level("ERROR")

ALL_TARGETS = ["PDA", "GSR_DMN", "GSR_CEN", "RAW_DMN"]
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]


def circular_shift_null_simple(X, y, model_name, alpha, run_boundaries, n_shuffles, rng):
    """Null: circular-shift y within each run, fit+predict, return null r values."""
    r_null = np.zeros(n_shuffles)
    for i in range(n_shuffles):
        y_shifted = np.empty_like(y)
        for start, end in run_boundaries:
            n = end - start
            shift = rng.integers(5, n - 5)
            y_shifted[start:end] = np.roll(y[start:end], shift)
        model = make_model(model_name, alpha)
        scaler = StandardScaler()
        model.fit(scaler.fit_transform(X), y_shifted)
        pred = model.predict(scaler.transform(X))
        r, _ = pearsonr(y_shifted, pred)
        r_null[i] = r
    return r_null


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--test_subject", default="dmnelf016")
    ap.add_argument("--n_shuffles", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    train_subjects = cfg["data"]["subjects"]["all"]
    test_sub = args.test_subject

    if test_sub in train_subjects:
        train_subjects = [s for s in train_subjects if s != test_sub]

    out_dir = PROJ_DIR / "results" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = PROJ_DIR / "results" / "multivariate" / "cache"

    hrf = canonical_hrf(tr=cfg["data"]["fmri"]["tr"], length_s=cfg["hrf"]["length_s"])

    # ── Load training data (all existing subjects) ──
    print(f"Training subjects: {len(train_subjects)}")
    print(f"Test subject: {test_sub}")
    print(f"Targets: {ALL_TARGETS}")
    print()

    train_X_pieces = []
    train_targets_pieces = {t: [] for t in ALL_TARGETS}
    train_run_boundaries_all = []
    offset = 0

    t0 = time.time()
    for sub in train_subjects:
        print(f"  Loading train subject {sub}...")
        runs_data, confounds, ch_names = load_subject_data(cfg, sub, cache_dir)
        if runs_data is None:
            print(f"    WARNING: skipping {sub}")
            continue

        X_sub, run_bounds_sub = car_and_flatten(runs_data, BAND_NAMES)
        targets_sub = prepare_targets(runs_data, confounds, ALL_TARGETS)

        train_X_pieces.append(X_sub)
        shifted_bounds = [(s + offset, e + offset) for s, e in run_bounds_sub]
        train_run_boundaries_all.extend(shifted_bounds)
        offset += X_sub.shape[0]

        for t in ALL_TARGETS:
            train_targets_pieces[t].append(targets_sub[t])

    X_train = np.vstack(train_X_pieces)
    y_train = {t: np.concatenate(train_targets_pieces[t]) for t in ALL_TARGETS}
    print(f"\n  Training set: {X_train.shape[0]} TRs, {X_train.shape[1]} features "
          f"({len(train_subjects)} subjects)")

    # ── Load test data (dmnelf016) ──
    print(f"\n  Loading test subject {test_sub}...")
    test_runs, test_confounds, test_ch = load_subject_data(cfg, test_sub, cache_dir)
    if test_runs is None:
        print(f"ERROR: no data for test subject {test_sub}")
        sys.exit(1)

    X_test, test_run_bounds = car_and_flatten(test_runs, BAND_NAMES)
    y_test = prepare_targets(test_runs, test_confounds, ALL_TARGETS)
    n_test_runs = len(test_run_bounds)
    print(f"  Test set: {X_test.shape[0]} TRs, {n_test_runs} runs")

    # ── Fit and predict ──
    print(f"\n{'='*70}")
    print(f"  CROSS-SUBJECT PREDICTION")
    print(f"{'='*70}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    results = []
    for tname in ALL_TARGETS:
        model = make_model("elasticnet", args.alpha)
        model.fit(X_train_sc, y_train[tname])
        pred_all = model.predict(X_test_sc)

        # Overall r
        r_all, _ = pearsonr(y_test[tname], pred_all)

        # Per-run r
        run_rs = []
        for ri, (start, end) in enumerate(test_run_bounds):
            r_run, _ = pearsonr(y_test[tname][start:end], pred_all[start:end])
            run_rs.append(r_run)

        # Circular-shift null on TEST subject data
        if args.n_shuffles > 0:
            rng = np.random.default_rng(42)
            r_null = np.zeros(args.n_shuffles)
            for i in range(args.n_shuffles):
                y_shifted = np.empty_like(y_test[tname])
                for start, end in test_run_bounds:
                    n = end - start
                    shift = rng.integers(5, n - 5)
                    y_shifted[start:end] = np.roll(y_test[tname][start:end], shift)
                r_null[i], _ = pearsonr(y_shifted, pred_all)
            p = (np.sum(r_null >= r_all) + 1) / (args.n_shuffles + 1)
        else:
            p = np.nan

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        p_str = f"p={p:.4f}" if not np.isnan(p) else "p=--"
        print(f"\n  {tname:8s}  overall r={r_all:+.4f}  {p_str} {sig}")
        for ri, rr in enumerate(run_rs):
            print(f"           run {ri+1}: r={rr:+.4f}")

        row = {
            "test_subject": test_sub,
            "target": tname,
            "model": "elasticnet",
            "overall_r": r_all,
            "p_circ_shift": p,
            "n_train_subjects": len(train_subjects),
            "n_train_trs": X_train.shape[0],
            "n_test_trs": X_test.shape[0],
            "n_test_runs": n_test_runs,
            "n_shuffles": args.n_shuffles,
        }
        for ri, rr in enumerate(run_rs):
            row[f"run{ri+1}_r"] = rr
        results.append(row)

        # Save predictions for plotting
        np.savez_compressed(
            out_dir / f"{test_sub}_{tname}_predictions.npz",
            pred=pred_all, true=y_test[tname],
            run_boundaries=np.array(test_run_bounds),
            coefs=model.coef_, ch_names=test_ch, band_names=BAND_NAMES,
        )

    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Done in {elapsed/60:.1f} min")
    print(f"{'='*70}")

    df = pd.DataFrame(results)
    df.to_csv(out_dir / f"{test_sub}_cross_subject_results.csv", index=False)
    print(f"\n  Results saved to {out_dir / f'{test_sub}_cross_subject_results.csv'}")
    print(f"  Predictions saved to {out_dir / f'{test_sub}_*_predictions.npz'}")


if __name__ == "__main__":
    main()
