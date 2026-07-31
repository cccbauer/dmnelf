#!/usr/bin/env python3
"""
decode_from_lag_stockwell.py  —  cross-validated "best own (band, lag)" decoder, Stockwell
------------------------------------------------------------------------------------------------
Same honest, non-circular check as decode_from_lag.py: with 10 Stockwell bands x 11 lags (220
candidates per subject/region) to search, in-sample peak |r| is even more inflated than the
single-feature alpha case (look-elsewhere effect) — so this picks the best (band, lag) on run 1,
reports what that SAME (band, lag) actually achieves on held-out run 2 (and vice versa).

Output: results/decode_from_lag_stockwell.csv

Usage:  python decode_from_lag_stockwell.py
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

STOCKWELL_DIR = HERE / "results" / "stockwell_bands"
LAGCORR_DIR = HERE / "results" / "lagged_coupling_stockwell"
OUT = HERE / "results" / "decode_from_lag_stockwell.csv"


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
    stockwell_npz = STOCKWELL_DIR / f"{sub}_stockwell.npz"
    z = np.load(stockwell_npz, allow_pickle=True)
    runs = [str(r) for r in z["_runs"]]
    if len(runs) < 2:
        return {"subject": sub, "region": region, "r_run1_to_run2": np.nan,
                "r_run2_to_run1": np.nan, "r_cv_mean": np.nan, "note": "only 1 run available"}

    r1_key, r2_key = runs[0], runs[1]
    bandpower = {rk: z[f"{rk}_bandpower"] for rk in (r1_key, r2_key)}
    bold = {}
    for rk in (r1_key, r2_key):
        run_num = rk.replace("run", "")
        feat = np.load(FEAT_DIR / f"sub-{sub}" / f"sub-{sub}_task-{TASK}_run-{run_num}_features.npz",
                       allow_pickle=True)["fmri_features"][:, idx]
        bold[rk] = feat

    sub_lc = lagcorr[(lagcorr.subject == sub) & (lagcorr.region == region)]

    def best_band_lag_from(train_key: str) -> tuple[int, int]:
        n = min(len(bandpower[train_key]), len(bold[train_key]))
        candidates = sub_lc[["band", "lag_tr"]].drop_duplicates().itertuples(index=False)
        rs = []
        for band, lag_tr in candidates:
            r = _lagged_corr(bandpower[train_key][:n, band], bold[train_key][:n], int(lag_tr))
            if np.isfinite(r):
                rs.append((band, lag_tr, r))
        if not rs:
            return 0, 0
        band, lag_tr, _ = max(rs, key=lambda t: abs(t[2]))
        return band, lag_tr

    def test_r(test_key: str, band: int, lag_tr: int) -> float:
        n = min(len(bandpower[test_key]), len(bold[test_key]))
        return _lagged_corr(bandpower[test_key][:n, band], bold[test_key][:n], lag_tr)

    band1, lag1 = best_band_lag_from(r1_key)
    r_1to2 = test_r(r2_key, band1, lag1)
    band2, lag2 = best_band_lag_from(r2_key)
    r_2to1 = test_r(r1_key, band2, lag2)

    finite = [r for r in (r_1to2, r_2to1) if np.isfinite(r)]
    r_cv = float(np.tanh(np.mean(np.arctanh(np.clip(finite, -0.999, 0.999))))) if finite else np.nan
    return {"subject": sub, "region": region, "band_from_run1": band1, "lag_from_run1_sec": lag1 * TR,
           "r_run1_to_run2": r_1to2, "band_from_run2": band2, "lag_from_run2_sec": lag2 * TR,
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
        print(f"\n{region} cross-validated decoding (Stockwell, best of 10 bands x 11 lags): "
             f"mean r={sub_df.r_cv_mean.mean():+.3f} sd={sub_df.r_cv_mean.std():.3f} "
             f"(n={len(sub_df)} subjects)")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
