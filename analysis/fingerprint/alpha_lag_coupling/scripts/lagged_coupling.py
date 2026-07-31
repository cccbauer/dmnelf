#!/usr/bin/env python3
"""
lagged_coupling.py  —  lagged correlation between residual alpha and DMN/CEN BOLD
------------------------------------------------------------------------------------
For each subject/run, correlates the residual-alpha time series (extract_residual_alpha.py)
against the DMN and CEN composite BOLD time series (this repo's standard DiFuMo-64 + 2
composite convention, columns 64/65 of the cached fmri_features) at 11 lags spanning
-10..+10 s in 2 s steps, following Jacob et al. 2025's lagged-correlation design.

Lag convention (matches the paper's figures): positive lag = alpha PRECEDES BOLD by that many
seconds (the canonical direction — neural activity, then a delayed hemodynamic response);
negative lag = BOLD PRECEDES alpha (noncanonical). r(lag) = corr(alpha[t], bold[t + lag_tr]).

Per-run correlations are Fisher-z averaged across a subject's runs (typically 2).

Output: results/lagged_coupling/<sub>_lagcorr.csv — one row per (region, lag_sec): r, n_tr.

Usage:  python lagged_coupling.py --subjects dmnelf010
        python lagged_coupling.py --subjects all
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text())
D, L = CFG["data"], CFG["lag"]

FEAT_DIR = Path(D["features_dir_local"]).expanduser()
ALPHA_DIR = HERE / "results" / "residual_alpha"
OUT = HERE / "results" / "lagged_coupling"
TR = float(D["fmri"]["tr"])
DMN_IDX, CEN_IDX = int(D["fmri"]["dmn_idx"]), int(D["fmri"]["cen_idx"])
TASK = D["task"]

LAG_LO, LAG_HI = L["lag_range_sec"]
LAG_STEP = float(L["lag_step_sec"])
LAG_SECS = np.arange(LAG_LO, LAG_HI + 1e-9, LAG_STEP)      # 11 lags, matching the paper


def _lagged_corr(x: np.ndarray, y: np.ndarray, lag_tr: int) -> tuple[float, int]:
    """corr(x[t], y[t + lag_tr]) over the valid, finite overlap."""
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
    alpha_npz = ALPHA_DIR / f"{sub}_alpha.npz"
    if not alpha_npz.exists():
        raise FileNotFoundError(f"no residual-alpha file for {sub}: {alpha_npz}")
    z = np.load(alpha_npz, allow_pickle=True)
    runs = [str(r) for r in z["_runs"]]

    rows = []
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
                alpha = z[f"{run_key}_residual_alpha"]
                bold = np.load(feat_npz, allow_pickle=True)["fmri_features"][:, idx]
                n = min(len(alpha), len(bold))
                r, n_tr = _lagged_corr(alpha[:n], bold[:n], lag_tr)
                if np.isfinite(r):
                    per_run_r.append(r); n_tr_total += n_tr
            if per_run_r:
                z_mean = np.mean(np.arctanh(np.clip(per_run_r, -0.999, 0.999)))
                r_mean = float(np.tanh(z_mean))
            else:
                r_mean = np.nan
            rows.append({"subject": sub, "region": region, "lag_sec": float(lag_sec),
                        "lag_tr": lag_tr, "r": r_mean, "n_runs": len(per_run_r),
                        "n_tr_total": n_tr_total})
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
        print(f"  {sub}: saved {sub}_lagcorr.csv "
             f"(DMN peak |r|={df[df.region=='DMN'].r.abs().max():.3f}, "
             f"CEN peak |r|={df[df.region=='CEN'].r.abs().max():.3f})")

    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(OUT / "all_subjects_lagcorr.csv", index=False)
        print(f"saved all_subjects_lagcorr.csv ({len(all_rows)} subjects)")


if __name__ == "__main__":
    main()
