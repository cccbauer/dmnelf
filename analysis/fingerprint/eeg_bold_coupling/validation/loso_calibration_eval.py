#!/usr/bin/env python3
"""
loso_calibration_eval.py
------------------------
LOSO evaluation of calibration-based fingerprints.
Start from a group model (trained on N-1 subjects' feedback data),
then fine-tune with the held-out subject's rest/shortrest data,
then predict on the held-out subject's feedback runs.

Calibration strategies:
  1. no_cal:     group model only (baseline — same as loso_fingerprint_eval)
  2. rest1:      fine-tune on rest run 1 (~350 TRs, ~7 min)
  3. rest2:      fine-tune on rest run 2
  4. rest_both:  fine-tune on rest runs 1+2 (~700 TRs)
  5. shortrest:  fine-tune on shortrest (~100 TRs, ~2 min)
  6. scratch_rest1: train from scratch on rest run 1 only (no group prior)

Fine-tuning = re-fit Ridge on [group_prediction, calibration_features] → calibration_target
(learns a subject-specific affine transform of the group prediction).

Usage:
  python loso_calibration_eval.py
  python loso_calibration_eval.py --targets GSR_CEN ATLAS_CEN
"""
import argparse, sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from scipy.stats import pearsonr

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
DIFUMO_DMN = [0, 3, 21, 29, 38, 52, 58, 60, 61]
DIFUMO_CEN = [4, 13, 17, 31, 35, 37, 47, 48, 50, 51, 53]

ALL_TARGETS = ["GSR_CEN", "PDA", "RAW_DMN", "ATLAS_CEN"]
CAL_STRATEGIES = ["no_cal", "shortrest", "rest1", "rest2", "rest_both",
                  "scratch_rest1", "scratch_rest_both"]


def load_rest_data(cfg, sub, cache_dir, task="rest", run_num=None):
    """Load rest/shortrest bandpower + targets for calibration."""
    band_names = list(cfg["bands"].keys())
    hrf = canonical_hrf(tr=cfg["data"]["fmri"]["tr"], length_s=cfg["hrf"]["length_s"])

    feat_dir = Path(cfg["data"]["features_dir"])
    ses = cfg["data"]["session"]
    eeg_dir = Path(cfg["data"]["eeg_preproc_dir"])

    runs_data = []
    confounds = []

    if run_num is not None:
        run_list = [run_num]
    elif task == "shortrest":
        run_list = [1]
    else:
        run_list = [1, 2]

    for run in run_list:
        npz_path = feat_dir / f"sub-{sub}" / f"sub-{sub}_task-{task}_run-{run}_features.npz"
        if not npz_path.exists():
            continue

        npz = np.load(npz_path)
        fmri = npz["fmri_features"]
        n_tr = fmri.shape[0]

        # Load EEG bandpower — use the gather_subject machinery but for rest task
        eeg_path = eeg_dir / f"sub-{sub}" / ses / "eeg"
        desc = cfg["data"]["eeg"].get("desc", "preproc500Hz")
        fif_pattern = f"sub-{sub}_{ses}_task-{task}_run-{run:02d}_desc-{desc}_eeg.fif"
        fif = eeg_path / fif_pattern
        if not fif.exists():
            continue

        raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        data = raw.get_data(picks=picks)
        ch_names = [raw.ch_names[p] for p in picks]
        sfreq = raw.info["sfreq"]
        samples_per_tr = int(cfg["data"]["eeg"]["samples_per_tr"])

        n_tr_eeg = min(n_tr, data.shape[1] // samples_per_tr)
        n_tr = n_tr_eeg

        bp = {}
        for bname, (lo, hi) in cfg["bands"].items():
            from scipy.signal import hilbert as sci_hilbert
            filtered = mne.filter.filter_data(data, sfreq, lo, hi, verbose=False)
            envelope = np.abs(sci_hilbert(filtered, axis=1))
            n_ch = envelope.shape[0]
            blocked = envelope[:, :n_tr * samples_per_tr].reshape(n_ch, n_tr, samples_per_tr).mean(axis=2).T
            blocked = np.log1p(blocked)
            convolved = np.zeros_like(blocked)
            for ci in range(n_ch):
                convolved[:, ci] = np.convolve(blocked[:, ci], hrf, mode="full")[:n_tr]
            bp[bname] = convolved

        targets_dict = {
            "DMN": fmri[:n_tr, 64],
            "CEN": fmri[:n_tr, 65],
            "PDA": fmri[:n_tr, 65] - fmri[:n_tr, 64],
            "parcels": fmri[:n_tr, :64],
        }

        rd = {"run": run, "n_tr": n_tr, "bp": bp, "targets": targets_dict, "chs": ch_names}
        runs_data.append(rd)

        # Load confounds
        conf_dir = Path(cfg["data"]["confounds_dir"])
        tsv = conf_dir / f"sub-{sub}" / ses / "func" / \
              f"sub-{sub}_{ses}_task-{task}_run-{run:02d}_desc-confounds_timeseries.tsv"
        if tsv.exists():
            df = pd.read_csv(tsv, sep="\t")
            gs = df["global_signal"].values.astype(float)
            gs[0] = gs[1]
            confounds.append(gs[:n_tr])
        else:
            confounds.append(np.zeros(n_tr))

    return runs_data, confounds


def get_target(runs_data, confounds, target_name):
    """Get target vector for any target type."""
    if target_name == "ATLAS_CEN":
        pieces = []
        for rd in runs_data:
            pieces.append(zscore(np.mean(rd["targets"]["parcels"][:, DIFUMO_CEN], axis=1)))
        return np.concatenate(pieces)
    elif target_name == "ATLAS_DMN":
        pieces = []
        for rd in runs_data:
            pieces.append(zscore(np.mean(rd["targets"]["parcels"][:, DIFUMO_DMN], axis=1)))
        return np.concatenate(pieces)
    elif target_name == "ATLAS_PDA":
        pieces = []
        for rd in runs_data:
            cen = np.mean(rd["targets"]["parcels"][:, DIFUMO_CEN], axis=1)
            dmn = np.mean(rd["targets"]["parcels"][:, DIFUMO_DMN], axis=1)
            pieces.append(zscore(cen - dmn))
        return np.concatenate(pieces)
    else:
        targets = prepare_targets(runs_data, confounds, [target_name])
        return targets[target_name]


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
    ap.add_argument("--targets", nargs="+", default=ALL_TARGETS)
    ap.add_argument("--strategies", nargs="+", default=CAL_STRATEGIES)
    args = ap.parse_args()

    cfg = load_config(args.config)
    subjects = cfg["data"]["subjects"]["all"] + cfg["data"]["subjects"].get("validation", [])
    cache_dir = PROJ_DIR / "results" / "multivariate" / "cache"
    out_dir = PROJ_DIR / "results" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"LOSO Calibration Evaluation")
    print(f"  Subjects: {len(subjects)}")
    print(f"  Targets: {args.targets}")
    print(f"  Strategies: {args.strategies}")
    print()

    # Load all subjects' feedback data once
    print("Loading feedback data...")
    t0 = time.time()
    all_feedback = {}
    for sub in subjects:
        runs_data, confounds, ch_names = load_subject_data(cfg, sub, cache_dir)
        if runs_data is not None:
            # Also load parcels for atlas targets
            feat_dir = Path(cfg["data"]["features_dir"])
            ses = cfg["data"]["session"]
            task = cfg["data"]["task"]
            for rd in runs_data:
                if "parcels" not in rd["targets"]:
                    npz = np.load(feat_dir / f"sub-{sub}" /
                                  f"sub-{sub}_task-{task}_run-{rd['run']}_features.npz")
                    rd["targets"]["parcels"] = npz["fmri_features"][:rd["n_tr"], :64]
            all_feedback[sub] = (runs_data, confounds, ch_names)
    print(f"  Loaded {len(all_feedback)} subjects in {time.time()-t0:.0f}s\n")

    all_results = []

    for target in args.targets:
        for strategy in args.strategies:
            print(f"  {target:10s} / {strategy:18s} ...", end=" ", flush=True)
            t1 = time.time()
            strat_results = []

            for test_sub in subjects:
                if test_sub not in all_feedback:
                    continue
                train_subs = [s for s in subjects if s != test_sub and s in all_feedback]

                # ── Build group model from training subjects' feedback ──
                train_X_parts, train_y_parts = [], []
                for sub in train_subs:
                    runs_data, confounds, ch_names = all_feedback[sub]
                    X_sub, bounds_sub = car_and_flatten(runs_data, BAND_NAMES)
                    y_sub = get_target(runs_data, confounds, target)
                    train_X_parts.append(X_sub)
                    train_y_parts.append(y_sub)

                X_train = np.vstack(train_X_parts)
                y_train = np.concatenate(train_y_parts)

                scaler_group = StandardScaler()
                X_train_sc = scaler_group.fit_transform(X_train)
                group_model = Ridge(alpha=1000)
                group_model.fit(X_train_sc, y_train)

                # ── Test subject feedback data ──
                test_runs, test_conf, test_ch = all_feedback[test_sub]
                X_test, test_bounds = car_and_flatten(test_runs, BAND_NAMES)
                y_test = get_target(test_runs, test_conf, target)
                X_test_sc = scaler_group.transform(X_test)

                if strategy == "no_cal":
                    pred = group_model.predict(X_test_sc)

                elif strategy.startswith("scratch_"):
                    # Train from scratch on rest only (no group model)
                    if strategy == "scratch_rest1":
                        cal_runs, cal_conf = load_rest_data(cfg, test_sub, cache_dir, "rest", 1)
                    elif strategy == "scratch_rest_both":
                        cal_runs, cal_conf = load_rest_data(cfg, test_sub, cache_dir, "rest", None)
                    else:
                        continue

                    if not cal_runs:
                        pred = group_model.predict(X_test_sc)  # fallback
                    else:
                        # Add parcels for atlas targets
                        feat_dir = Path(cfg["data"]["features_dir"])
                        for rd in cal_runs:
                            if "parcels" not in rd["targets"]:
                                npz = np.load(feat_dir / f"sub-{test_sub}" /
                                              f"sub-{test_sub}_task-rest_run-{rd['run']}_features.npz")
                                rd["targets"]["parcels"] = npz["fmri_features"][:rd["n_tr"], :64]

                        X_cal, cal_bounds = car_and_flatten(cal_runs, BAND_NAMES)
                        y_cal = get_target(cal_runs, cal_conf, target)
                        scaler_cal = StandardScaler()
                        scratch_model = Ridge(alpha=100)
                        scratch_model.fit(scaler_cal.fit_transform(X_cal), y_cal)
                        pred = scratch_model.predict(scaler_cal.transform(X_test))

                else:
                    # Fine-tune: load calibration data
                    if strategy == "rest1":
                        cal_runs, cal_conf = load_rest_data(cfg, test_sub, cache_dir, "rest", 1)
                    elif strategy == "rest2":
                        cal_runs, cal_conf = load_rest_data(cfg, test_sub, cache_dir, "rest", 2)
                    elif strategy == "rest_both":
                        cal_runs, cal_conf = load_rest_data(cfg, test_sub, cache_dir, "rest", None)
                    elif strategy == "shortrest":
                        cal_runs, cal_conf = load_rest_data(cfg, test_sub, cache_dir, "shortrest", 1)
                    else:
                        continue

                    if not cal_runs:
                        pred = group_model.predict(X_test_sc)  # fallback
                    else:
                        # Add parcels for atlas targets
                        feat_dir = Path(cfg["data"]["features_dir"])
                        for rd in cal_runs:
                            task_name = "rest" if "rest" in strategy else "shortrest"
                            if "parcels" not in rd["targets"]:
                                npz = np.load(feat_dir / f"sub-{test_sub}" /
                                              f"sub-{test_sub}_task-{task_name}_run-{rd['run']}_features.npz")
                                rd["targets"]["parcels"] = npz["fmri_features"][:rd["n_tr"], :64]

                        X_cal, cal_bounds = car_and_flatten(cal_runs, BAND_NAMES)
                        y_cal = get_target(cal_runs, cal_conf, target)

                        # Fine-tune: use group prediction as feature + raw EEG features
                        X_cal_sc = scaler_group.transform(X_cal)
                        group_pred_cal = group_model.predict(X_cal_sc)

                        # Augmented features: [group_prediction, EEG_features]
                        X_aug_cal = np.column_stack([group_pred_cal, X_cal_sc])
                        X_aug_test = np.column_stack([group_model.predict(X_test_sc), X_test_sc])

                        scaler_aug = StandardScaler()
                        finetune_model = Ridge(alpha=100)
                        finetune_model.fit(scaler_aug.fit_transform(X_aug_cal), y_cal)
                        pred = finetune_model.predict(scaler_aug.transform(X_aug_test))

                # ── Evaluate ──
                r_all, _ = pearsonr(y_test, pred)
                run_rs = []
                for start, end in test_bounds:
                    r_run, _ = pearsonr(y_test[start:end], pred[start:end])
                    run_rs.append(r_run)

                row = {"target": target, "strategy": strategy, "test_subject": test_sub,
                       "r": r_all, "n_train": len(train_subs)}
                for ri, rr in enumerate(run_rs):
                    row[f"run{ri+1}_r"] = rr
                strat_results.append(row)

            all_results.extend(strat_results)
            r_vals = [r["r"] for r in strat_results]
            mean_r, t_stat, p = sign_flip_test(r_vals)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"mean r={mean_r:+.4f} t={t_stat:+.2f} p={p:.4f} {sig}  ({time.time()-t1:.0f}s)")

    # Save
    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "loso_calibration_eval.csv", index=False)

    # Summary
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    summary_rows = []
    for (target, strategy), grp in df.groupby(["target", "strategy"]):
        r_vals = grp["r"].values
        mean_r, t_stat, p = sign_flip_test(r_vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        summary_rows.append({"target": target, "strategy": strategy,
                             "mean_r": mean_r, "t": t_stat, "p": p, "sig": sig, "n": len(r_vals)})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "loso_calibration_summary.csv", index=False)

    for target in args.targets:
        sub = summary[summary["target"] == target].sort_values("mean_r", ascending=False)
        print(f"\n  {target}:")
        for _, row in sub.iterrows():
            print(f"    {row['strategy']:18s}  r={row['mean_r']:+.4f}  t={row['t']:+.2f}  p={row['p']:.4f} {row['sig']}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
