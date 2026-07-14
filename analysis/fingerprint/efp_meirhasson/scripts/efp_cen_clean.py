#!/usr/bin/env python3
"""
efp_cen_clean.py  —  NEW VERSION: honest EFP re-scoring for CEN
---------------------------------------------------------------
The frozen EFP gives CEN r=0.279, but that number is (a) full-run (rest+feedback, so
inflated by the state step) and (b) on the un-cleaned CEN target (no confound regression).
This re-scores the EFP single-electrode sliding-delay decoder against the CLEAN
(confound-regressed) CEN mask-mean, restricted to the FEEDBACK block, with leave-one-run-out
(LORO) nested-electrode selection — the same honest bar we hold every other decoder to.

For each subject it reports LORO OOF r for CEN under 4 conditions (2 targets x 2 windows):
  clean/fb   = clean CEN, feedback block only        <- the honest number
  clean/full = clean CEN, whole run
  orig/fb    = cached (contaminated) CEN, feedback
  orig/full  = cached (contaminated) CEN, whole run  <- closest to the frozen 0.279

Reuses efp_features (build/load, make_delay_design). Clean CEN from cenmean_dmnelf_<sub>.npz.
Usage (cluster): python efp_cen_clean.py --subject dmnelf001 --cenmean-dir DIR --cache DIR --out DIR
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import zscore, pearsonr
from sklearn.linear_model import RidgeCV
from efp_features import load_config, build_subject_features, load_subject_features, make_delay_design

BASELINE_TR, HRF_DROP = 25, 5
ALPHAS = np.logspace(-2, 5, 15)


def nmse(y, p):
    v = np.var(y)
    return np.mean((y - p) ** 2) / v if v > 0 else np.nan


def run_designs(runs, n_delays, cen_clean, target, window):
    """Per run -> (list of per-channel feedback/full designs, y). Aligned, per-run z-scored."""
    out = []
    for rd in runs:
        nch = rd["bp_tr"].shape[0]
        Xs, off = [], None
        for ci in range(nch):
            X, off = make_delay_design(rd["bp_tr"][ci], n_delays)
            X = (X - X.mean(0)) / (X.std(0) + 1e-12)
            Xs.append(X)
        nvalid = Xs[0].shape[0]
        t_idx = off + np.arange(nvalid)                 # original TR of each design row
        if target == "clean":
            y_full = np.asarray(cen_clean[f"run{rd['run']}"], float)
        else:
            y_full = np.asarray(rd["tgt_tr"]["CEN"], float)
        y = y_full[off:off + nvalid]
        mask = (t_idx >= BASELINE_TR + HRF_DROP) if window == "fb" else np.ones(nvalid, bool)
        m2 = mask & np.isfinite(y)
        if m2.sum() < 20:
            continue
        yy = zscore(y[m2])
        out.append(([X[m2] for X in Xs], yy))
    return out


def nested_loro(rd_list, nch):
    """LORO with inner-LORO electrode selection (leak-free). Returns OOF r."""
    if len(rd_list) < 2:
        return np.nan
    obs, pred = [], []
    for i in range(len(rd_list)):
        train = [j for j in range(len(rd_list)) if j != i]
        # inner electrode selection: inner-LORO NMSE within training runs
        best_ci, best = None, np.inf
        for ci in range(nch):
            errs = []
            for h in train:
                inner_tr = [j for j in train if j != h]
                Xtr = np.vstack([rd_list[j][0][ci] for j in inner_tr])
                ytr = np.concatenate([rd_list[j][1] for j in inner_tr])
                mdl = RidgeCV(alphas=ALPHAS).fit(Xtr, ytr)
                p = mdl.predict(rd_list[h][0][ci])
                if np.std(p) > 1e-9:
                    errs.append(nmse(rd_list[h][1], p))
            if errs and np.mean(errs) < best:
                best, best_ci = np.mean(errs), ci
        if best_ci is None:
            continue
        Xtr = np.vstack([rd_list[j][0][best_ci] for j in train])
        ytr = np.concatenate([rd_list[j][1] for j in train])
        mdl = RidgeCV(alphas=ALPHAS).fit(Xtr, ytr)
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
    cache_dir = Path(a.cache); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sub = a.subject
    if not (cache_dir / f"{sub}_efp.npz").exists():
        build_subject_features(cfg, sub, cache_dir)
    runs, ch_names = load_subject_features(cache_dir, sub)
    nch = len(ch_names)
    cen = np.load(Path(a.cenmean_dir) / f"cenmean_dmnelf_{sub}.npz", allow_pickle=True)

    rows = []
    for target in ["clean", "orig"]:
        for window in ["fb", "full"]:
            rd_list = run_designs(runs, n_delays, cen, target, window)
            r = nested_loro(rd_list, nch)
            rows.append(dict(subject=sub, target=target, window=window, efp_cen_r=r))
            print(f"  {sub} EFP CEN {target}/{window}: r={r:+.3f}", flush=True)
    pd.DataFrame(rows).to_csv(out / f"efp_cen_clean_{sub}.csv", index=False)


if __name__ == "__main__":
    main()
