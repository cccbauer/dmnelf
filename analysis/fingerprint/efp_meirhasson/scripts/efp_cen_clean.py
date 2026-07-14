#!/usr/bin/env python3
"""
efp_cen_clean.py  —  NEW VERSION: honest EFP re-scoring (CEN/DMN/PDA) + electrode modes
---------------------------------------------------------------------------------------
Honest bar: leave-one-run-out (LORO), FEEDBACK block only, no within-run block-CV leakage.
Targets: CEN/DMN/PDA (from the cache, orig) + clean confound-regressed CEN (from cenmean) as a
reference (in-block clean~orig for CEN). Electrode modes:
  best     single best electrode, inner-LORO nested selection (the classic EFP)
  frontal  MULTIVARIATE over frontal electrodes (concatenated sliding-delay designs), ridge
  all      MULTIVARIATE over all 31 electrodes, ridge
Each electrode contributes a [10 band x 11 delay] EFP design; frontal/all concatenate them.

Output: efp_cen_clean_<sub>.csv  (subject, target, ttype, mode, window, r)
Usage (cluster, from efp scripts dir): python efp_cen_clean.py --subject SUB --cenmean-dir DIR --cache DIR --out DIR
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import zscore, pearsonr
from sklearn.linear_model import RidgeCV
from efp_features import load_config, build_subject_features, load_subject_features, make_delay_design

BASELINE_TR, HRF_DROP = 25, 5
ALPHAS = np.logspace(-2, 5, 15)
FRONTAL = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "FC1", "FC2", "FC5", "FC6"]


def nmse(y, p):
    v = np.var(y)
    return np.mean((y - p) ** 2) / v if v > 0 else np.nan


def run_designs(runs, n_delays, target_vec_by_run, window):
    """Per run -> (per-channel design list, y). Feedback-block masked, per-run z-scored y."""
    out = []
    for rd in runs:
        nch = rd["bp_tr"].shape[0]; Xs, off = [], None
        for ci in range(nch):
            X, off = make_delay_design(rd["bp_tr"][ci], n_delays)
            Xs.append((X - X.mean(0)) / (X.std(0) + 1e-12))
        nvalid = Xs[0].shape[0]; t_idx = off + np.arange(nvalid)
        y_full = target_vec_by_run[rd["run"]]
        y = y_full[off:off + nvalid]
        mask = (t_idx >= BASELINE_TR + HRF_DROP) if window == "fb" else np.ones(nvalid, bool)
        m2 = mask & np.isfinite(y)
        if m2.sum() < 20:
            continue
        out.append(([X[m2] for X in Xs], zscore(y[m2])))
    return out


def loro_multivar(rd_list, cols):
    """LORO ridge over the concatenated designs of channels `cols` (multivariate)."""
    if len(rd_list) < 2:
        return np.nan
    obs, pred = [], []
    Xr = [np.column_stack([r[0][ci] for ci in cols]) for r in rd_list]
    yr = [r[1] for r in rd_list]
    for i in range(len(rd_list)):
        tr = [j for j in range(len(rd_list)) if j != i]
        m = RidgeCV(alphas=ALPHAS).fit(np.vstack([Xr[j] for j in tr]),
                                       np.concatenate([yr[j] for j in tr]))
        pred.append(m.predict(Xr[i])); obs.append(yr[i])
    o, p = np.concatenate(obs), np.concatenate(pred)
    return float(pearsonr(o, p)[0]) if np.std(p) > 1e-9 else np.nan


def loro_best(rd_list, nch):
    """LORO with inner-LORO single-electrode selection (classic EFP)."""
    if len(rd_list) < 2:
        return np.nan
    obs, pred = [], []
    for i in range(len(rd_list)):
        tr = [j for j in range(len(rd_list)) if j != i]
        best_ci, best = None, np.inf
        for ci in range(nch):
            errs = []
            for h in tr:
                inner = [j for j in tr if j != h]
                mdl = RidgeCV(alphas=ALPHAS).fit(np.vstack([rd_list[j][0][ci] for j in inner]),
                                                 np.concatenate([rd_list[j][1] for j in inner]))
                pp = mdl.predict(rd_list[h][0][ci])
                if np.std(pp) > 1e-9:
                    errs.append(nmse(rd_list[h][1], pp))
            if errs and np.mean(errs) < best:
                best, best_ci = np.mean(errs), ci
        if best_ci is None:
            continue
        mdl = RidgeCV(alphas=ALPHAS).fit(np.vstack([rd_list[j][0][best_ci] for j in tr]),
                                         np.concatenate([rd_list[j][1] for j in tr]))
        pred.append(mdl.predict(rd_list[i][0][best_ci])); obs.append(rd_list[i][1])
    o, p = np.concatenate(obs), np.concatenate(pred)
    return float(pearsonr(o, p)[0]) if np.std(p) > 1e-9 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--cenmean-dir", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config()
    tr = cfg["data"]["fmri"]["tr"]; n_delays = int(round(cfg["efp"]["delay_window_s"] / tr)) + 1
    cache_dir = Path(a.cache); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); sub = a.subject
    if not (cache_dir / f"{sub}_efp.npz").exists():
        build_subject_features(cfg, sub, cache_dir)
    runs, ch_names = load_subject_features(cache_dir, sub)
    nch = len(ch_names); fidx = [i for i, c in enumerate(ch_names) if c in FRONTAL]
    cen_clean = np.load(Path(a.cenmean_dir) / f"cenmean_dmnelf_{sub}.npz", allow_pickle=True)

    # target vectors per run: orig CEN/DMN/PDA (cache) + clean CEN (cenmean)
    tvecs = {("CEN", "orig"): {rd["run"]: np.asarray(rd["tgt_tr"]["CEN"], float) for rd in runs},
             ("DMN", "orig"): {rd["run"]: np.asarray(rd["tgt_tr"]["DMN"], float) for rd in runs},
             ("PDA", "orig"): {rd["run"]: np.asarray(rd["tgt_tr"]["PDA"], float) for rd in runs},
             ("CEN", "clean"): {rd["run"]: np.asarray(cen_clean[f"run{rd['run']}"], float) for rd in runs}}

    rows = []
    for (tgt, ttype), tvec in tvecs.items():
        rd_list = run_designs(runs, n_delays, tvec, "fb")
        for mode, cols in [("best", None), ("frontal", fidx), ("all", list(range(nch)))]:
            r = loro_best(rd_list, nch) if mode == "best" else loro_multivar(rd_list, cols)
            rows.append(dict(subject=sub, target=tgt, ttype=ttype, mode=mode, window="fb", r=r))
            print(f"  {sub} {tgt}/{ttype} {mode}: r={r:+.3f}", flush=True)
    pd.DataFrame(rows).to_csv(out / f"efp_cen_clean_{sub}.csv", index=False)


if __name__ == "__main__":
    main()
