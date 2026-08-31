#!/usr/bin/env python
"""
within_subject_decode.py
------------------------
Within-subject PDA decoding sanity test: train an ElasticNet on a subject's OWN
rest, predict their OWN feedback PDA. Compares two EEG feature sets:
  - baseline  : 1-40 Hz block-means (from existing cyclic_features npz, eeg_block)
  - infraslow : 0.01-40 Hz block-means recomputed from the desc-preproc500HzISp01
                .fif files (the DC-coupled infraslow band the 1 Hz HP had removed)

PDA target and TR alignment come from the existing npz (pda array); infraslow
block-means are computed with the SAME block-average over the same n_volumes, so
they align 1:1 (IS and baseline .fif have identical sample counts).

A small EEG->BOLD lag scan (EEG leads BOLD) is included; best lag is chosen on a
rest-internal split, then applied to the held-out feedback runs.

Usage:
    python within_subject_decode.py --subject dmnelf007 --config config.yaml
"""
import argparse, glob, warnings
from pathlib import Path
import numpy as np, yaml
from scipy.stats import pearsonr
warnings.simplefilter("ignore")

import mne
mne.set_log_level("ERROR")
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    d = cfg.get("data", {})
    if "features_dir_cluster" in d and "features_dir_local" in d:
        d["features_dir"] = (d["features_dir_cluster"]
                             if Path("/projects/swglab").exists()
                             else d["features_dir_local"])
    return cfg


def block_mean(raw, samples_per_tr, n_vol):
    x = raw.get_data(picks="eeg")            # (nch, nsamp)
    need = n_vol * samples_per_tr
    x = x[:, :need].reshape(x.shape[0], n_vol, samples_per_tr).mean(2).T  # (nvol,nch)
    mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-8
    return ((x - mu) / sd).astype(np.float32)


def is_fif_path(cfg, subject, task, run, desc):
    root = Path(cfg["data"]["eeg_preproc_dir"]); ses = cfg["data"]["session"]
    sub = f"sub-{subject}"
    return (root / sub / ses / "eeg" /
            f"{sub}_{ses}_task-{task}_run-{run}_desc-{desc}_eeg.fif")


def gather(cfg, subject, task, desc_is):
    """Return list of (pda, X_base, X_is) per run for a task."""
    fdir = Path(cfg["data"]["features_dir"]) / f"sub-{subject}"
    spt = cfg["data"]["eeg"]["samples_per_tr"]
    runs = []
    for npz in sorted(fdir.glob(f"sub-{subject}_task-{task}_run-*_features.npz")):
        d = np.load(npz, allow_pickle=True)
        pda = np.asarray(d["pda"], float)
        Xb = np.asarray(d["eeg_block"], float)              # baseline (nvol,31)
        run = npz.name.split("run-")[1][0]                   # '1'..'4'
        fif = is_fif_path(cfg, subject, task, f"{int(run):02d}", desc_is)
        if not fif.exists():
            print(f"  [skip {task} run-{run}] no IS fif: {fif.name}")
            continue
        raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
        Xi = block_mean(raw, spt, len(pda))
        runs.append((pda, Xb, Xi))
    return runs


def apply_lag(X, y, lag):
    """EEG leads BOLD by `lag` TRs: pair X[t] with y[t+lag]."""
    if lag == 0:
        return X, y
    return X[:-lag], y[lag:]


def fit_eval(train_runs, test_runs, which, lags=(0, 2, 4, 6)):
    # build train/test matrices for the chosen feature set
    idx = 1 if which == "base" else 2
    def stack(runs, lag):
        Xs, ys = [], []
        for r in runs:
            X, y = apply_lag(r[idx], r[0], lag)
            Xs.append(X); ys.append(y)
        return np.vstack(Xs), np.concatenate(ys)

    # choose lag by internal rest split (last run held out within train)
    best_lag, best_cv = 0, -np.inf
    for lag in lags:
        if len(train_runs) >= 2:
            tr = train_runs[:-1]; va = train_runs[-1:]
            Xtr, ytr = stack(tr, lag); Xva, yva = stack(va, lag)
            sc = StandardScaler().fit(Xtr)
            m = ElasticNetCV(l1_ratio=[.1,.5,.9], cv=3, max_iter=5000)
            m.fit(sc.transform(Xtr), ytr)
            r = pearsonr(m.predict(sc.transform(Xva)), yva)[0]
        else:
            r = 0.0
        if r > best_cv:
            best_cv, best_lag = r, lag

    Xtr, ytr = stack(train_runs, best_lag)
    Xte, yte = stack(test_runs, best_lag)
    sc = StandardScaler().fit(Xtr)
    m = ElasticNetCV(l1_ratio=[.1,.5,.9], cv=3, max_iter=5000)
    m.fit(sc.transform(Xtr), ytr)
    pred = m.predict(sc.transform(Xte))
    r, p = pearsonr(pred, yte)
    return dict(r=r, p=p, lag=best_lag, cv_r=best_cv,
                pred_std=float(pred.std()), true_std=float(yte.std()), n=len(yte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--desc-is", default="preproc500HzISp01")
    ap.add_argument("--train-task", default="rest")
    ap.add_argument("--test-task", default="feedback")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"=== within-subject decode: {args.subject} "
          f"(train={args.train_task} -> test={args.test_task}) ===")
    train_runs = gather(cfg, args.subject, args.train_task, args.desc_is)
    test_runs  = gather(cfg, args.subject, args.test_task, args.desc_is)
    print(f"train runs: {len(train_runs)}  test runs: {len(test_runs)}")
    if not train_runs or not test_runs:
        print("  insufficient runs"); return

    for which in ("base", "infraslow"):
        res = fit_eval(train_runs, test_runs, which)
        print(f"\n[{which:9s}] feedback Pearson r = {res['r']:+.4f} (p={res['p']:.2e})  "
              f"lag={res['lag']}TR  rest-CV r={res['cv_r']:+.3f}  "
              f"pred_std={res['pred_std']:.3f} true_std={res['true_std']:.3f}  n={res['n']}")


if __name__ == "__main__":
    main()
