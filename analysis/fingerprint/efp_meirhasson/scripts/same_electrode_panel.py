#!/usr/bin/env python3
"""
same_electrode_panel.py  (within-subject comparison, panel B)
-------------------------------------------------------------
Fair, same-electrode design comparison: at each network's GROUP-PEAK electrode
(from electrode_r_all.csv), score EFP / HRF / T-A per subject with concatenated
out-of-fold CV (no per-method electrode selection). This isolates the
"sliding-delay EFP vs fixed-HRF vs T/A design" question with all methods on the
identical electrode and estimator. Complements the nested Table 1 (which also
cross-validates the electrode selection).

Output: results/full/same_electrode_panel_{res}.csv (subject,target,electrode,EFP,HRF,TA)
"""
import argparse
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from efp_features import load_config, load_subject_features
from efp_decode import (assemble, assemble_hrf, assemble_ta, oof_r, mk_block_folds,
                        canonical_hrf)

PROJ = Path(__file__).resolve().parent.parent
RES = PROJ / "results" / "full"


def n_delays_for(cfg, res):
    e = cfg["efp"]; tr = cfg["data"]["fmri"]["tr"]
    return (int(round(e["delay_window_s"] / tr)) + 1 if res == "tr"
            else int(round(e["delay_window_s"] * e["hz4"])) + 1)


def group_peaks(res):
    """Group-mean-r peak electrode per target from electrode_r_all.csv."""
    d = pd.read_csv(RES / "electrode_r_all.csv")
    d = d[d.resolution == res]
    peaks = {}
    for t, g in d.groupby("target"):
        gm = g.groupby("electrode")["r"].agg(["mean", "count"])
        gm = gm[gm["count"] >= 10]
        if len(gm):
            peaks[t] = gm["mean"].idxmax()
    return peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="tr")
    ap.add_argument("--cache", default=str(PROJ / "results" / "features_cache"))
    args = ap.parse_args()
    cfg = load_config()
    e = cfg["efp"]; tr = cfg["data"]["fmri"]["tr"]; res = args.res
    alphas = np.logspace(np.log10(e["alpha_grid_lo"]), np.log10(e["alpha_grid_hi"]), e["alpha_grid_n"])
    hrf = canonical_hrf(tr, cfg["hrf"]["length_s"], cfg["hrf"]["delay"], cfg["hrf"]["undershoot"])
    nd = n_delays_for(cfg, res)
    peaks = group_peaks(res)

    rows = []
    for sub in cfg["data"]["subjects"]["all"]:
        try:
            runs, chs = load_subject_features(Path(args.cache), sub)
        except FileNotFoundError:
            continue
        for target in cfg["targets"]:
            el = peaks.get(target)
            if el is None or el not in chs:
                continue
            ci = chs.index(el)
            def _r(assembler):
                X, y = assembler(runs, ci, target, res, nd) if assembler is assemble \
                    else assembler(runs, ci, target, res, hrf)
                if X is None or len(y) < 10:
                    return np.nan
                return oof_r(X, y, alphas, mk_block_folds(len(y), e["cv_outer_k"], e["cv_outer_m"]))
            rows.append(dict(subject=sub, target=target, electrode=el,
                             EFP=_r(assemble), HRF=_r(assemble_hrf), TA=_r(assemble_ta)))
        print(f"  {sub}: done")
    df = pd.DataFrame(rows)
    out = RES / f"same_electrode_panel_{res}.csv"
    df.to_csv(out, index=False)
    print("saved", out)
    if len(df):
        print(df.groupby("target")[["EFP", "HRF", "TA"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
