#!/usr/bin/env python3
"""
compute_aca.py  —  Accumulated Correlation Asymmetry (Jacob et al. 2025) per subject/region
-----------------------------------------------------------------------------------------------
ACA = sum(|r| at lag >= 0) - sum(|r| at lag < 0), from lagged_coupling.py's per-lag correlations.
ACA > 0: canonical coupling (alpha precedes BOLD, weight on positive lags — the classic HRF
direction). ACA < 0: noncanonical (BOLD precedes and predicts alpha). Also reports each
region's peak-|r| lag per subject, since that's the natural per-subject "best lag" a simple
lag-tuned decoder (decode_from_lag.py) would use.

Output: results/aca_summary.csv — one row per (subject, region): aca, peak_lag_sec, peak_r.

Usage:  python compute_aca.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text())
L = CFG["lag"]
POS_LO, POS_HI = L["aca_positive_range_sec"]
NEG_LO, NEG_HI = L["aca_negative_range_sec"]

LAGCORR = HERE / "results" / "lagged_coupling" / "all_subjects_lagcorr.csv"
OUT = HERE / "results" / "aca_summary.csv"


def aca_for_subject_region(df: pd.DataFrame) -> dict:
    pos = df[(df.lag_sec >= POS_LO) & (df.lag_sec <= POS_HI)]
    neg = df[(df.lag_sec >= NEG_LO) & (df.lag_sec < NEG_HI)]
    aca = float(pos.r.abs().sum() - neg.r.abs().sum())
    peak_row = df.loc[df.r.abs().idxmax()] if df.r.notna().any() else None
    return {
        "aca": aca,
        "peak_lag_sec": float(peak_row.lag_sec) if peak_row is not None else np.nan,
        "peak_r": float(peak_row.r) if peak_row is not None else np.nan,
    }


def main():
    df = pd.read_csv(LAGCORR)
    rows = []
    for (sub, region), g in df.groupby(["subject", "region"]):
        row = {"subject": sub, "region": region}
        row.update(aca_for_subject_region(g))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["region", "subject"])
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"\nsaved {OUT}")

    for region in ("DMN", "CEN"):
        sub_df = out[out.region == region]
        print(f"\n{region}: ACA mean={sub_df.aca.mean():+.3f} sd={sub_df.aca.std():.3f} "
             f"(n={len(sub_df)}) — {'canonical-leaning' if sub_df.aca.mean() > 0 else 'noncanonical-leaning'} on average")


if __name__ == "__main__":
    main()
