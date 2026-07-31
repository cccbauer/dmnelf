#!/usr/bin/env python3
"""
lagged_coupling_stockwell.py  —  lagged correlation, Stockwell bands vs. DMN/CEN BOLD
------------------------------------------------------------------------------------------
Same lag-correlation design as lagged_coupling.py (11 lags, ±10 s, Fisher-z averaged across
runs), applied per-band to extract_stockwell_bands.py's 10 equal-energy Stockwell bands instead
of the single FOOOF residual-alpha feature — testing whether efp_meirhasson's own feature
construction shows different/stronger lagged coupling with DMN/CEN than residual alpha did.

Output: results/lagged_coupling_stockwell/<sub>_lagcorr.csv — one row per (region, band, lag_sec).

Usage:  python lagged_coupling_stockwell.py --subjects all
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text())
D, L, S = CFG["data"], CFG["lag"], CFG["stockwell"]

FEAT_DIR = Path(D["features_dir_local"]).expanduser()
STOCKWELL_DIR = HERE / "results" / "stockwell_bands"
OUT = HERE / "results" / "lagged_coupling_stockwell"
TR = float(D["fmri"]["tr"])
DMN_IDX, CEN_IDX = int(D["fmri"]["dmn_idx"]), int(D["fmri"]["cen_idx"])
TASK = D["task"]
N_BANDS = int(S["n_bands"])

LAG_LO, LAG_HI = L["lag_range_sec"]
LAG_STEP = float(L["lag_step_sec"])
LAG_SECS = np.arange(LAG_LO, LAG_HI + 1e-9, LAG_STEP)


def _lagged_corr(x: np.ndarray, y: np.ndarray, lag_tr: int) -> tuple[float, int]:
    n = len(x)
    if lag_tr >= 0:
        xs, ys = (x[: n - lag_tr] if lag_tr else x), (y[lag_tr:] if lag_tr else y)
    else:
        xs, ys = x[-lag_tr:], y[: n + lag_tr]
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 10:
        return np.nan, int(m.sum())
    return float(np.corrcoef(xs[m], ys[m])[0, 1]), int(m.sum())


def subject_lagged_coupling(sub: str) -> pd.DataFrame:
    stockwell_npz = STOCKWELL_DIR / f"{sub}_stockwell.npz"
    if not stockwell_npz.exists():
        raise FileNotFoundError(f"no Stockwell-band file for {sub}: {stockwell_npz}")
    z = np.load(stockwell_npz, allow_pickle=True)
    runs = [str(r) for r in z["_runs"]]

    rows = []
    for band in range(N_BANDS):
        for region, idx in (("DMN", DMN_IDX), ("CEN", CEN_IDX)):
            for lag_sec in LAG_SECS:
                lag_tr = int(round(lag_sec / TR))
                per_run_r = []
                n_tr_total = 0
                for run_key in runs:
                    run_num = run_key.replace("run", "")
                    feat_npz = FEAT_DIR / f"sub-{sub}" / f"sub-{sub}_task-{TASK}_run-{run_num}_features.npz"
                    if not feat_npz.exists():
                        continue
                    bandpower = z[f"{run_key}_bandpower"][:, band]
                    bold = np.load(feat_npz, allow_pickle=True)["fmri_features"][:, idx]
                    n = min(len(bandpower), len(bold))
                    r, n_tr = _lagged_corr(bandpower[:n], bold[:n], lag_tr)
                    if np.isfinite(r):
                        per_run_r.append(r); n_tr_total += n_tr
                if per_run_r:
                    z_mean = np.mean(np.arctanh(np.clip(per_run_r, -0.999, 0.999)))
                    r_mean = float(np.tanh(z_mean))
                else:
                    r_mean = np.nan
                rows.append({"subject": sub, "band": band, "region": region,
                            "lag_sec": float(lag_sec), "lag_tr": lag_tr, "r": r_mean,
                            "n_runs": len(per_run_r), "n_tr_total": n_tr_total})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True)
    a = ap.parse_args()
    subs = D["subjects"]["all"] if a.subjects == ["all"] else a.subjects
    OUT.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for sub in subs:
        try:
            df = subject_lagged_coupling(sub)
        except FileNotFoundError as exc:
            print(f"  {sub}: {exc}"); continue
        df.to_csv(OUT / f"{sub}_lagcorr.csv", index=False)
        all_rows.append(df)
        best = df.loc[df.groupby("region").r.apply(lambda s: s.abs().idxmax())]
        print(f"  {sub}: saved (best |r| per region: "
             + ", ".join(f"{row.region} band{row.band}@{row.lag_sec:g}s r={row.r:+.3f}"
                         for _, row in best.iterrows()) + ")")

    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(OUT / "all_subjects_lagcorr.csv", index=False)
        print(f"saved all_subjects_lagcorr.csv ({len(all_rows)} subjects)")


if __name__ == "__main__":
    main()
