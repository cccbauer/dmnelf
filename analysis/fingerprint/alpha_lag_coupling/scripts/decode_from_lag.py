#!/usr/bin/env python3
"""
decode_from_lag.py  —  cross-validated single-feature "best own lag" decoder
--------------------------------------------------------------------------------
The lagged-correlation/ACA analysis (lagged_coupling.py, compute_aca.py) shows *whether* and
*at what lag* alpha and DMN/CEN BOLD are coupled, but picking the best lag and reporting its
correlation on the SAME data is circular (peak-picking always inflates r). This script reports
the honest, cross-validated version: for each subject, find the best lag on run 1, then report
the correlation that same lag actually achieves on the held-out run 2 (and vice versa) — a
genuine train/test split, directly comparable to efp_meirhasson's CV ridge and test_replay.py's
online-vs-observed r as a decoder-performance number.

This is deliberately the simplest possible decoder (one feature: alpha power at whichever lag
was best in the *other* run) — a floor/reference point against which a multivariate version
(e.g. alpha at several lags, or per-band rather than just 8-12 Hz) could be compared later.

Output: results/decode_from_lag.csv — one row per (subject, region): r_run1_to_run2 (lag chosen
on run 1, tested on run 2), r_run2_to_run1 (vice versa), r_cv_mean (Fisher-z average of the two).

Usage:  python decode_from_lag.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text())
D = CFG["data"]
FEAT_DIR = Path(D["features_dir_local"]).expanduser()
TASK = D["task"]
TR = float(D["fmri"]["tr"])
DMN_IDX, CEN_IDX = int(D["fmri"]["dmn_idx"]), int(D["fmri"]["cen_idx"])

ALPHA_DIR = HERE / "results" / "residual_alpha"
LAGCORR_DIR = HERE / "results" / "lagged_coupling"
OUT = HERE / "results" / "decode_from_lag.csv"


def _lagged_corr(x: np.ndarray, y: np.ndarray, lag_tr: int) -> float:
    n = len(x)
    if lag_tr >= 0:
        xs, ys = (x[: n - lag_tr] if lag_tr else x), (y[lag_tr:] if lag_tr else y)
    else:
        xs, ys = x[-lag_tr:], y[: n + lag_tr]
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 10:
        return np.nan
    return float(np.corrcoef(xs[m], ys[m])[0, 1])


def cv_decode_subject_region(sub: str, region: str, idx: int, lagcorr: pd.DataFrame) -> dict:
    alpha_npz = ALPHA_DIR / f"{sub}_alpha.npz"
    z = np.load(alpha_npz, allow_pickle=True)
    runs = [str(r) for r in z["_runs"]]
    if len(runs) < 2:
        return {"subject": sub, "region": region, "r_run1_to_run2": np.nan,
                "r_run2_to_run1": np.nan, "r_cv_mean": np.nan, "note": "only 1 run available"}

    r1_key, r2_key = runs[0], runs[1]
    alpha = {rk: z[f"{rk}_residual_alpha"] for rk in (r1_key, r2_key)}
    bold = {}
    for rk in (r1_key, r2_key):
        run_num = rk.replace("run", "")
        feat = np.load(FEAT_DIR / f"sub-{sub}" / f"sub-{sub}_task-{TASK}_run-{run_num}_features.npz",
                       allow_pickle=True)["fmri_features"][:, idx]
        bold[rk] = feat

    sub_lc = lagcorr[(lagcorr.subject == sub) & (lagcorr.region == region)]

    def best_lag_from(train_key: str) -> int:
        # re-derive the single-run lag profile isn't cached per-run, so recompute here directly
        n = min(len(alpha[train_key]), len(bold[train_key]))
        lags = sub_lc.lag_tr.unique()
        rs = [(lag_tr, _lagged_corr(alpha[train_key][:n], bold[train_key][:n], int(lag_tr)))
              for lag_tr in lags]
        rs = [(lag, r) for lag, r in rs if np.isfinite(r)]
        if not rs:
            return 0
        return max(rs, key=lambda t: abs(t[1]))[0]

    def test_r(test_key: str, lag_tr: int) -> float:
        n = min(len(alpha[test_key]), len(bold[test_key]))
        return _lagged_corr(alpha[test_key][:n], bold[test_key][:n], lag_tr)

    lag_from_1 = best_lag_from(r1_key)
    r_1to2 = test_r(r2_key, lag_from_1)
    lag_from_2 = best_lag_from(r2_key)
    r_2to1 = test_r(r1_key, lag_from_2)

    finite = [r for r in (r_1to2, r_2to1) if np.isfinite(r)]
    r_cv = float(np.tanh(np.mean(np.arctanh(np.clip(finite, -0.999, 0.999))))) if finite else np.nan
    return {"subject": sub, "region": region, "lag_from_run1_sec": lag_from_1 * TR,
           "r_run1_to_run2": r_1to2, "lag_from_run2_sec": lag_from_2 * TR,
           "r_run2_to_run1": r_2to1, "r_cv_mean": r_cv}


def main():
    lagcorr = pd.read_csv(LAGCORR_DIR / "all_subjects_lagcorr.csv")
    subs = sorted(lagcorr.subject.unique())
    rows = []
    for sub in subs:
        for region, idx in (("DMN", DMN_IDX), ("CEN", CEN_IDX)):
            rows.append(cv_decode_subject_region(sub, region, idx, lagcorr))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    for region in ("DMN", "CEN"):
        sub_df = out[(out.region == region) & out.r_cv_mean.notna()]
        print(f"\n{region} cross-validated decoding: mean r={sub_df.r_cv_mean.mean():+.3f} "
             f"sd={sub_df.r_cv_mean.std():.3f} (n={len(sub_df)} subjects)")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
