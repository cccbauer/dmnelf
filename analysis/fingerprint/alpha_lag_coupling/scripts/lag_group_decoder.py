#!/usr/bin/env python3
"""
lag_group_decoder.py  —  own-group vs. pooled-all LOSO, split by resting-state coupling direction
------------------------------------------------------------------------------------------------------
Tests whether splitting the DMNELF training cohort by coupling direction (define_lag_groups.py's
canonical/noncanonical, from alpha_lag_coupling's resting-state ACA) and fitting a SEPARATE ridge
per group predicts CEN/DMN better for a held-out subject than the single all-subjects-pooled model
currently deployed (mindwear/model/efp_epoc_model.npz, via mindwear/export_model.py).

Reuses export_model.py's EXACT per-subject design construction (subject_designs/clean_targets —
channel-major [10 band x 11 delay] x 12 EPOC channel design, per-run z-scored, BASELINE_TR/HRF_DROP
masked) and the same RidgeCV alpha grid, so the only difference between the "own_group" and
"pooled_all" arms is which subjects get pooled into the training fit for each held-out subject —
an apples-to-apples paired comparison, not a different modeling approach.

CAVEAT (see FINDINGS.md): groups are defined from resting-state Pz-alpha coupling, evaluated here
on feedback-task decoding — this assumes trait-like consistency across states, which this
comparison itself is a test of, not an assumption it relies on being true. n=12 canonical / n=4
noncanonical is small, especially for the noncanonical LOSO arm (each held-out subject trained on
only 3 others) — treat accordingly.

Output: results/lag_group_decoder_loso.csv — one row per (subject, target, arm): r, n_train.

Usage:  python lag_group_decoder.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
EFP_SCRIPTS = HERE.parent / "efp_meirhasson" / "scripts"
MINDWEAR = HERE.parent / "mindwear"
sys.path.insert(0, str(EFP_SCRIPTS))
sys.path.insert(0, str(MINDWEAR))
from efp_features import load_config, load_subject_features  # noqa: E402
from export_model import ALPHAS, clean_targets, subject_designs, MONTAGES  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402

CACHE = EFP_SCRIPTS.parent / "results" / "features_cache"
CENMEAN_DIR = HERE.parent / "fsnr_eeg" / "results" / "cen_ceiling"
GROUPS_CSV = HERE / "results" / "lag_groups.csv"
OUT = HERE / "results" / "lag_group_decoder_loso.csv"
CHANNELS = MONTAGES["epoc12"]


def load_subject(cfg, sub):
    """Per-subject (X, y) dict keyed by target, or None if unusable — same prerequisites as
    export_model.py's main loop (cached features + clean targets + full EPOC-12 montage)."""
    try:
        runs, ch_names = load_subject_features(CACHE, sub)
    except FileNotFoundError:
        return None
    eidx = [ch_names.index(c) for c in CHANNELS if c in ch_names]
    if len(eidx) != len(CHANNELS):
        return None
    tv = clean_targets(str(CENMEAN_DIR), sub, runs)
    if not tv:
        return None
    cfg_ = cfg  # noqa: F841 (kept for clarity/future use)
    tr = cfg["data"]["fmri"]["tr"]
    n_delays = int(round(cfg["efp"]["delay_window_s"] / tr)) + 1
    out = {}
    for tgt in ["CEN", "DMN"]:
        Xs, ys = subject_designs(runs, ch_names, eidx, n_delays, tv, tgt)
        if not Xs:
            continue
        out[tgt] = (Xs, ys)
    return out or None


def fit_and_predict(train_data: dict, held_Xs, held_ys, target) -> float | None:
    """Pool train_data's per-subject (Xs, ys) lists, fit RidgeCV, predict held-out subject."""
    Xs_all, ys_all = [], []
    for d in train_data.values():
        if target not in d:
            continue
        Xs, ys = d[target]
        Xs_all += Xs; ys_all += ys
    if not Xs_all:
        return None
    X = np.vstack(Xs_all); y = np.concatenate(ys_all)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    m = RidgeCV(alphas=ALPHAS).fit((X - mu) / sd, y)
    Xh = np.vstack(held_Xs); yh = np.concatenate(held_ys)
    pred = m.predict((Xh - mu) / sd)
    if np.std(pred) < 1e-9 or np.std(yh) < 1e-9:
        return None
    return float(np.corrcoef(pred, yh)[0, 1])


def main():
    cfg = load_config()
    groups = pd.read_csv(GROUPS_CSV).set_index("subject")["group"].to_dict()

    print("loading cached per-subject designs...")
    all_data = {}
    for sub in sorted(set(groups) | {"dmnelf016"}):
        d = load_subject(cfg, sub)
        if d is not None:
            all_data[sub] = d
    print(f"  usable subjects: {len(all_data)} / {len(groups)}")

    rows = []
    for held, held_d in all_data.items():
        group = groups.get(held, "unknown")
        for target in ("CEN", "DMN"):
            if target not in held_d:
                continue
            held_Xs, held_ys = held_d[target]

            # pooled-all arm: every other usable subject regardless of group
            others_all = {s: d for s, d in all_data.items() if s != held}
            r_pooled = fit_and_predict(others_all, held_Xs, held_ys, target)
            rows.append({"subject": held, "target": target, "arm": "pooled_all", "r": r_pooled,
                        "n_train": len(others_all)})

            # own-group arm: only same-group subjects (skipped for "unknown", e.g. dmnelf016)
            if group in ("canonical", "noncanonical"):
                own_group = {s: d for s, d in all_data.items()
                            if s != held and groups.get(s) == group}
                r_group = fit_and_predict(own_group, held_Xs, held_ys, target)
                rows.append({"subject": held, "target": target, "arm": "own_group", "r": r_group,
                            "n_train": len(own_group)})

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))

    print("\n=== paired comparison (own_group - pooled_all), subjects with both arms ===")
    for target in ("CEN", "DMN"):
        piv = out[out.target == target].pivot(index="subject", columns="arm", values="r")
        piv = piv.dropna(subset=["own_group", "pooled_all"])
        if piv.empty:
            continue
        diff = piv["own_group"] - piv["pooled_all"]
        n_pos = int((diff > 0).sum()); n = len(diff)
        print(f"{target}: mean own_group r={piv['own_group'].mean():+.3f}, "
             f"mean pooled_all r={piv['pooled_all'].mean():+.3f}, "
             f"mean diff={diff.mean():+.3f} (own_group better in {n_pos}/{n} subjects)")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
