#!/usr/bin/env python
"""
within_fair.py
--------------
Fair, nested within-subject re-run of the multivariate band-power decoder, matched
to the EFP nested-CV v3 estimator so the head-to-head within-subject tier is
apples-to-apples:

  - Score = concatenated OUT-OF-FOLD Pearson r (not mean-of-per-fold-r).
  - Regularization tuned by INNER CV (RidgeCV / ElasticNetCV) on each outer-train
    fold — matches EFP's RidgeCV-GCV lambda selection (no fixed alpha=1.0).
  - Standardization fit on the outer-train fold only.
  - ALL subjects included for EVERY target (complete, fixed subject set).
  - Full target set incl. RAW_CEN and GSR_PDA (the old run omitted these).

Reuses the cached band-power features (results/multivariate/cache).

Usage (cluster):
  python within_fair.py --cv_folds 5
"""
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.model_selection import KFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bandpower import load_config, canonical_hrf, gather_subject, zscore
from multivariate_decode_pda import residualize, load_confounds_run

warnings.filterwarnings("ignore")
PROJ = Path(__file__).resolve().parent.parent
CACHE = PROJ / "results" / "multivariate" / "cache"
RES = PROJ / "results"
TARGETS = ["PDA", "RAW_DMN", "RAW_CEN", "GSR_DMN", "GSR_CEN", "GSR_PDA"]
MODELS = ["ridge", "elasticnet"]
RIDGE_ALPHAS = np.logspace(-2, 4, 30)


def get_runs(cfg, sub, hrf):
    cf = CACHE / f"{sub}_bandpower.npz"
    if cf.exists():
        z = np.load(cf, allow_pickle=True)
        return list(z["runs_data"]), list(z["ch_names"])
    runs = gather_subject(cfg, sub, hrf)
    return (runs, runs[0]["chs"]) if runs else (None, None)


def car_flatten(runs, band_names):
    pieces, bnds, off = [], [], 0
    for rd in runs:
        bl = []
        for b in band_names:
            bp = rd["bp"][b].copy()
            bp -= bp.mean(axis=1, keepdims=True)   # CAR
            bl.append(bp)
        pieces.append(zscore(np.concatenate(bl, axis=1)))
        bnds.append((off, off + rd["n_tr"])); off += rd["n_tr"]
    return np.vstack(pieces), bnds


def target_vec(runs, conf, tname):
    out = []
    for rd, gs in zip(runs, conf):
        t = rd["targets"]
        if tname == "PDA":       y = t["PDA"].copy()
        elif tname == "RAW_DMN": y = t["DMN"].copy()
        elif tname == "RAW_CEN": y = t["CEN"].copy()
        elif tname == "GSR_DMN": y = residualize(t["DMN"].copy(), gs)
        elif tname == "GSR_CEN": y = residualize(t["CEN"].copy(), gs)
        elif tname == "GSR_PDA": y = residualize(t["CEN"].copy(), gs) - residualize(t["DMN"].copy(), gs)
        else: raise ValueError(tname)
        out.append(zscore(y))
    return np.concatenate(out)


def make_cv(name):
    if name == "ridge":
        return RidgeCV(alphas=RIDGE_ALPHAS)
    return ElasticNetCV(l1_ratio=0.5, n_alphas=20, cv=3, max_iter=10000, n_jobs=1)


def oof_r(X, y, model_name, k):
    """Concatenated out-of-fold Pearson r; inner-CV regularization per outer fold."""
    folds = list(KFold(n_splits=k, shuffle=False).split(np.arange(len(y))))
    pred = np.full(len(y), np.nan)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        m = make_cv(model_name).fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    ok = np.isfinite(pred)
    return pearsonr(y[ok], pred[ok])[0] if np.std(pred[ok]) > 1e-9 else np.nan


def sign_flip(rs, n=10000, seed=42):
    rs = np.asarray([r for r in rs if np.isfinite(r)])
    if len(rs) < 3:
        return np.nan, np.nan
    obs = rs.mean(); rng = np.random.default_rng(seed)
    null = (rng.choice([-1, 1], size=(n, len(rs))) * np.abs(rs)).mean(1)
    p = (np.sum(null >= obs) + 1) / (n + 1)
    return float(obs), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJ / "config.yaml"))
    ap.add_argument("--cv_folds", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(args.config)
    band_names = list(cfg["bands"].keys())
    hrf = canonical_hrf(tr=cfg["data"]["fmri"]["tr"], length_s=cfg["hrf"]["length_s"])
    subs = cfg["data"]["subjects"]["all"]

    per, summ = [], []
    for sub in subs:
        runs, chs = get_runs(cfg, sub, hrf)
        if runs is None:
            print(f"{sub}: no runs, skip"); continue
        conf = [load_confounds_run(cfg, sub, rd["run"])[:rd["n_tr"]] for rd in runs]
        X, _ = car_flatten(runs, band_names)
        for t in TARGETS:
            y = target_vec(runs, conf, t)
            for m in MODELS:
                r = oof_r(X, y, m, args.cv_folds)
                per.append(dict(subject=sub, target=t, model=m, oof_r=r))
        print(f"{sub}: done")

    pdf = pd.DataFrame(per)
    pdf.to_csv(RES / "within_fair_persubject.csv", index=False)
    for t in TARGETS:
        for m in MODELS:
            rs = pdf[(pdf.target == t) & (pdf.model == m)].oof_r.values
            mean_r, p = sign_flip(rs)
            summ.append(dict(target=t, model=m, n=int(np.isfinite(rs).sum()),
                             mean_oof_r=mean_r, sign_flip_p=p))
            print(f"{t:8s} {m:11s} n={int(np.isfinite(rs).sum()):2d} mean_oof_r={mean_r:+.3f} p={p:.3f}")
    pd.DataFrame(summ).to_csv(RES / "within_fair_summary.csv", index=False)
    print("\nSaved", RES / "within_fair_summary.csv")


if __name__ == "__main__":
    main()
