#!/usr/bin/env python
"""
diagnose_subjects.py
--------------------
Diagnostic for low-correlation subjects: is the prediction failing because of
(a) bad/flat model output, (b) degenerate ground-truth PDA, or (c) bad input
features (NaNs, flat channels, EEG dropout)?

Compares target subjects against reference (good) subjects.

Usage:
    python diagnose_subjects.py --subjects dmnelf009 dmnelf015 \
        --reference dmnelf013 dmnelf014 --config config.yaml
"""
import argparse
from pathlib import Path
import numpy as np
import yaml
from scipy.stats import pearsonr


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    d = cfg.get("data", {})
    if "features_dir_cluster" in d and "features_dir_local" in d:
        d["features_dir"] = (d["features_dir_cluster"]
                             if Path("/projects/swglab").exists()
                             else d["features_dir_local"])
    return cfg


def arr_stats(name, x):
    x = np.asarray(x, dtype=float)
    n_nan = int(np.isnan(x).sum())
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        print(f"    {name:16s} shape={x.shape} ALL NON-FINITE")
        return
    print(f"    {name:16s} shape={str(x.shape):14s} "
          f"mean={np.mean(finite):+.4f} std={np.std(finite):.4f} "
          f"min={finite.min():+.3f} max={finite.max():+.3f} nan={n_nan}")


def diagnose_prediction(pred_path):
    if not pred_path.exists():
        print(f"  [MISSING] {pred_path}")
        return
    d = np.load(pred_path, allow_pickle=True)
    keys = list(d.keys())
    print(f"  prediction file keys: {keys}")

    pda_pred = np.asarray(d["pda_predicted"], dtype=float)
    arr_stats("pda_predicted", pda_pred)

    # reconstruct pda_true the way evaluate_predictions does
    if "fmri_true" in keys:
        fmri_true = np.asarray(d["fmri_true"], dtype=float)
        arr_stats("fmri_true", fmri_true)
        try:
            dmn_idx = int(d["dmn_idx"]); cen_idx = int(d["cen_idx"])
            pda_true = (fmri_true[:, dmn_idx] - fmri_true[:, cen_idx]
                        if fmri_true.ndim == 2 and fmri_true.shape[1] > max(dmn_idx, cen_idx)
                        else None)
        except Exception:
            pda_true = None
    else:
        pda_true = None

    if pda_true is not None and np.isfinite(pda_true).all():
        arr_stats("pda_true", pda_true)
        n = min(len(pda_pred), len(pda_true))
        r, p = pearsonr(pda_pred[:n], pda_true[:n])
        # dynamic range diagnostics
        print(f"    pred dynamic range = {pda_pred.max()-pda_pred.min():.4f}  "
              f"(std {pda_pred.std():.4f})")
        print(f"    true dynamic range = {pda_true.max()-pda_true.min():.4f}  "
              f"(std {pda_true.std():.4f})")
        print(f"    pearson r = {r:+.4f} (p={p:.2e}), n={n}")


def diagnose_features(subj, feat_dir):
    """Inspect feedback-run feature files for NaNs / flat channels."""
    files = sorted(feat_dir.glob(f"sub-{subj}_task-feedback_run-*_features.npz"))
    print(f"  {len(files)} feedback feature files")
    for f in files:
        d = np.load(f, allow_pickle=True)
        keys = list(d.keys())
        run = f.name.split("run-")[1][0]
        print(f"  -- run {run}  keys={keys}")
        for k in keys:
            v = d[k]
            if isinstance(v, np.ndarray) and v.dtype.kind in "fc" and v.ndim >= 1:
                v = v.astype(float)
                n_nan = int(np.isnan(v).sum())
                # flat channels: columns with ~zero variance
                flat = ""
                if v.ndim == 2:
                    stds = np.nanstd(v, axis=0)
                    n_flat = int((stds < 1e-8).sum())
                    if n_flat:
                        flat = f" FLAT_CH={n_flat}/{v.shape[1]}"
                tag = "" if n_nan == 0 else f" NaN={n_nan}"
                if tag or flat:
                    arr_stats(f"{k}", v)
                    print(f"       ^^{tag}{flat}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--reference", nargs="+", default=[])
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    feat_dir_base = Path(cfg["data"]["features_dir"])

    for label, subs in [("TARGET (low-r)", args.subjects),
                        ("REFERENCE (good)", args.reference)]:
        for subj in subs:
            print("\n" + "=" * 78)
            print(f"{label}: {subj}")
            print("=" * 78)
            sdir = feat_dir_base / f"sub-{subj}"
            pred = sdir / "predictions" / f"sub-{subj}_task-feedback_pda_prediction.npz"
            print("\n[PREDICTION]")
            diagnose_prediction(pred)
            print("\n[INPUT FEATURES] (only channels with NaN/flat shown)")
            diagnose_features(subj, sdir)


if __name__ == "__main__":
    main()
