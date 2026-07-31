#!/usr/bin/env python3
"""
frontoparietal_montage_loso.py  —  does restricting to frontal+parietal electrodes help?
------------------------------------------------------------------------------------------
electrode_vs_montage_loso.py found that a single well-chosen electrode (P8 for CEN, O1 for DMN)
beats pooling all 12 EPOC channels in the zero-shot LOSO regime — and the per-electrode ranking
(see conversation) is dominated by POSTERIOR sites (occipital/parietal/temporal), not frontal
ones, despite CEN/DMN being textbook fronto-parietal/frontal-hub networks anatomically. Frontal
EPOC channels (F7,F3,F4,F8,FC5,FC6) rank near the BOTTOM of all 12 for both targets.

This tests two frontal+parietal-restricted arms (F7,F3,FC5,FC6,F4,F8,P7,P8 — excluding the two
temporal and two occipital EPOC channels), both in the SAME validated LOSO pipeline as the
existing arms so all numbers are directly comparable:

  1. frontoparietal_best_electrode — the SAME leak-free per-fold single-electrode selection the
     deployed decoder uses (efp_group.py::_group_peak / loso_transfer), but with candidates
     restricted to the frontoparietal set — i.e. "if you only trusted frontal/parietal sites,
     what's the single best one, chosen the same rigorous way?" ("best data points.")
  2. frontoparietal_montage — pooling all 8 frontoparietal channels multivariately
     (assemble_multi, reused from electrode_vs_montage_loso.py), the same pattern as the
     existing epoc12_montage arm but restricted to this 8-channel subset.

Usage:  python frontoparietal_montage_loso.py
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, zscore
from sklearn.linear_model import RidgeCV

from efp_features import load_config, load_subject_features
from efp_decode import assemble
from efp_group import _group_peak, sign_flip_test
from electrode_vs_montage_loso import assemble_multi, PROJ_DIR, CACHE_DIR, OUT_DIR

FRONTOPARIETAL = ["F7", "F3", "FC5", "FC6", "F4", "F8", "P7", "P8"]


def loso_best_electrode_restricted(feats, chmat, target, res, nd, alphas, candidates):
    """Verbatim electrode_vs_montage_loso.py::loso_leakfree_electrode, but _group_peak only
    considers `candidates` — the leak-free-selected BEST electrode within that subset."""
    cm_full = chmat.get((target, res), {})
    cm = {s: {ch: r for ch, r in d.items() if ch in candidates} for s, d in cm_full.items()}
    usable = [s for s in cm if cm[s]]
    if len(usable) < 4:
        return None, None
    r_subs, chosen = [], []
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
            r_subs.append(pearsonr(yh, pred)[0]); chosen.append(el)
    if not r_subs:
        return None, chosen
    m, p = sign_flip_test(r_subs)
    return {"loso_mean_r": m, "sign_flip_p": p, "n": len(r_subs)}, chosen


def loso_montage(feats, channels, target, res, nd, alphas):
    designs = {}
    for s, (runs, chs) in feats.items():
        X, y = assemble_multi(runs, chs, channels, target, res, nd)
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
    print(f"usable subjects: {len(feats)}, frontoparietal channels: {FRONTOPARIETAL}")

    tr = cfg["data"]["fmri"]["tr"]
    nd_tr = int(round(e["delay_window_s"] / tr)) + 1
    # reuse the already-cached per-subject/electrode CV ranking (results/full/electrode_r_all.csv)
    # instead of recomputing electrode_r_matrix() from scratch (expensive nested CV across every
    # electrode x target x resolution x subject — this file's numbers are the same ranking used
    # to build that cache in earlier runs of electrode_vs_montage_loso.py / efp_group.py).
    ranked = pd.read_csv(OUT_DIR / "electrode_r_all.csv")
    ranked = ranked[ranked.resolution == "tr"]
    chmat = {}
    for target, g in ranked.groupby("target"):
        d = {}
        for sub, gg in g.groupby("subject"):
            d[sub] = dict(zip(gg.electrode, gg.r))
        chmat[(target, "tr")] = d

    rows = []
    for target in ("CEN", "DMN", "PDA"):
        r1, chosen = loso_best_electrode_restricted(feats, chmat, target, "tr", nd_tr, alphas, FRONTOPARIETAL)
        from collections import Counter
        print(f"{target}: frontoparietal_best_electrode={r1}  electrodes chosen per fold: {Counter(chosen)}")
        if r1:
            rows.append({"target": target, "arm": "frontoparietal_best_electrode", **r1})

        r2 = loso_montage(feats, FRONTOPARIETAL, target, "tr", nd_tr, alphas)
        print(f"{target}: frontoparietal_montage={r2}")
        if r2:
            rows.append({"target": target, "arm": "frontoparietal_montage", **r2})

    out = pd.DataFrame(rows)
    out.to_csv(PROJ_DIR / "results" / "frontoparietal_montage_loso.csv", index=False)
    print(f"\nsaved {PROJ_DIR / 'results' / 'frontoparietal_montage_loso.csv'}")


if __name__ == "__main__":
    main()
