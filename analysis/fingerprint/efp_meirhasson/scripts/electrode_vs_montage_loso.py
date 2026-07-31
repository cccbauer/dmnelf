#!/usr/bin/env python3
"""
electrode_vs_montage_loso.py  —  isolate channel count in the SAME validated LOSO pipeline
-------------------------------------------------------------------------------------------------
efp_group.py's cached results/full/efp_group_loso.csv (leak-free per-fold electrode selection,
raw/GSR targets via load_targets_run, no baseline-TR masking) and export_model.py's deployed-model
target/design (confound-regressed "clean" targets via clean_targets/cenmean_dmnelf, BASELINE_TR/
HRF_DROP-masked, fixed EPOC-12 multivariate design) differ in THREE ways at once — target cleaning,
baseline masking, and electrode selection — not just channel count. A prior ad-hoc comparison mixed
these pipelines, so it couldn't isolate whether channel count itself matters.

This script holds efp_group.py's own target/design/CV methodology fixed (assemble(), raw DiFuMo
CEN/DMN/PDA + GSR variants, no baseline masking, mk_block_folds/RidgeCV as in efp_decode.py) and
adds ONLY a second arm: instead of a single leak-free-selected electrode, pool ALL 12 EPOC-montage
channels into one multivariate design (assemble_multi, channel-major, same per-run z-scoring
convention as assemble()). Same subjects, same target, same CV — channel count is now the only
variable between the two arms.

Output: results/electrode_vs_montage_loso.csv — one row per (target, arm in
{leakfree_electrode, epoc12_montage}): loso_mean_r, sign_flip_p, n.

Usage:  python electrode_vs_montage_loso.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, zscore
from sklearn.linear_model import RidgeCV

from efp_features import load_config, load_subject_features, make_delay_design
from efp_decode import assemble, mk_block_folds
from efp_group import electrode_r_matrix, _group_peak, sign_flip_test

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPT_DIR.parent
OUT_DIR = PROJ_DIR / "results" / "full"
CACHE_DIR = PROJ_DIR / "results" / "features_cache"
EPOC12 = ["F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8"]


def assemble_multi(runs, ch_names, channels, target, res, n_delays):
    """Channel-major multivariate design across `channels`, same per-run z-score convention as
    assemble() (one z-score per channel's design block, per run) — the only difference from
    assemble() is pooling multiple channels' delay designs into one row instead of one channel."""
    bp_key = "bp_tr" if res == "tr" else "bp_hz4"
    tg_key = "tgt_tr" if res == "tr" else "tgt_hz4"
    eidx = [ch_names.index(c) for c in channels if c in ch_names]
    if len(eidx) != len(channels):
        return None, None
    Xs, ys = [], []
    for rd in runs:
        if target not in rd[tg_key]:
            return None, None
        per_ch, off = [], None
        for ci in eidx:
            Xc, off = make_delay_design(rd[bp_key][ci], n_delays)
            per_ch.append((Xc - Xc.mean(0)) / (Xc.std(0) + 1e-12))
        if per_ch[0].shape[0] == 0:
            continue
        y = zscore(rd[tg_key][target][off:off + per_ch[0].shape[0]])
        Xs.append(np.column_stack(per_ch)); ys.append(y)
    if not Xs:
        return None, None
    return np.vstack(Xs), np.concatenate(ys)


def loso_leakfree_electrode(cfg, feats, chmat, target, res, nd, alphas):
    """Verbatim electrode-selection arm of efp_group.py::loso_transfer, for one target/res."""
    cm = chmat.get((target, res), {})
    usable = [s for s in cm if cm[s]]
    if len(usable) < 4:
        return None
    r_subs = []
    for held in usable:
        trains = [s for s in usable if s != held]
        el = _group_peak({s: cm[s] for s in usable}, trains, min_n=min(10, len(trains)))
        if el is None or el not in feats[held][1]:
            continue
        Xtr, ytr = [], []
        for s in trains:
            runs, chs = feats[s]
            if el not in chs:
                continue
            X, y = assemble(runs, chs.index(el), target, res, nd)
            if X is not None:
                Xtr.append(zscore(X, axis=0)); ytr.append(y)
        runs, chs = feats[held]
        Xh, yh = assemble(runs, chs.index(el), target, res, nd)
        if not Xtr or Xh is None:
            continue
        model = RidgeCV(alphas=alphas).fit(np.vstack(Xtr), np.concatenate(ytr))
        pred = model.predict(zscore(Xh, axis=0))
        if np.std(pred) > 1e-9:
            r_subs.append(pearsonr(yh, pred)[0])
    if not r_subs:
        return None
    m, p = sign_flip_test(r_subs)
    return {"loso_mean_r": m, "sign_flip_p": p, "n": len(r_subs)}


def loso_epoc12_montage(feats, target, res, nd, alphas):
    """Same LOSO loop, but pooling all 12 EPOC channels multivariately instead of selecting one."""
    designs = {}
    for s, (runs, chs) in feats.items():
        X, y = assemble_multi(runs, chs, EPOC12, target, res, nd)
        if X is not None:
            designs[s] = (X, y)
    usable = list(designs)
    if len(usable) < 4:
        return None
    r_subs = []
    for held in usable:
        Xtr = np.vstack([designs[s][0] for s in usable if s != held])
        ytr = np.concatenate([designs[s][1] for s in usable if s != held])
        Xh, yh = designs[held]
        model = RidgeCV(alphas=alphas).fit(Xtr, ytr)
        pred = model.predict(Xh)
        if np.std(pred) > 1e-9:
            r_subs.append(pearsonr(yh, pred)[0])
    if not r_subs:
        return None
    m, p = sign_flip_test(r_subs)
    return {"loso_mean_r": m, "sign_flip_p": p, "n": len(r_subs)}


def main():
    cfg = load_config()
    e = cfg["efp"]
    alphas = np.logspace(np.log10(e["alpha_grid_lo"]), np.log10(e["alpha_grid_hi"]), e["alpha_grid_n"])
    feats = {}
    for s in cfg["data"]["subjects"]["all"]:
        try:
            feats[s] = load_subject_features(CACHE_DIR, s)
        except FileNotFoundError:
            continue
    print(f"usable subjects (any cache): {len(feats)}")

    tr = cfg["data"]["fmri"]["tr"]
    nd_tr = int(round(e["delay_window_s"] / tr)) + 1
    chmat = electrode_r_matrix(cfg, OUT_DIR, feats)

    rows = []
    for target in ("CEN", "DMN", "PDA"):
        r1 = loso_leakfree_electrode(cfg, feats, chmat, target, "tr", nd_tr, alphas)
        if r1:
            rows.append({"target": target, "arm": "leakfree_electrode", **r1})
        r2 = loso_epoc12_montage(feats, target, "tr", nd_tr, alphas)
        if r2:
            rows.append({"target": target, "arm": "epoc12_montage", **r2})
        print(f"{target}: leakfree_electrode={r1}  epoc12_montage={r2}")

    out = pd.DataFrame(rows)
    out.to_csv(PROJ_DIR / "results" / "electrode_vs_montage_loso.csv", index=False)
    print(f"\nsaved {PROJ_DIR / 'results' / 'electrode_vs_montage_loso.csv'}")


if __name__ == "__main__":
    main()
