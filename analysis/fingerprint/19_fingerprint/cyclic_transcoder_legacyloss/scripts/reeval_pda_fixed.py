#!/usr/bin/env python
"""
reeval_pda_fixed.py
-------------------
Corrected PDA evaluation.

The committed evaluate_predictions.py reconstructs the ground-truth PDA from the
saved `fmri_true` array, but predict_pda.py saves that array with a scrambled
reshape (batch & ROI axes flattened together -> shape (window_trs, n_win*66)).
Indexing column dmn_idx/cen_idx of that array therefore returns only the FIRST
window's 50 samples, with sign DMN-CEN (opposite of the predicted CEN-DMN).

This script reconstructs the true PDA directly from the per-run feature files,
replaying the exact windowing used by make_predict_loader:
  - tasks = ["feedback"], runs sorted by filename
  - non-overlapping windows: stride = window_trs
  - row-major flatten -> window0 t0..tN, window1 t0..tN, ...
so it aligns 1:1 with the saved `pda_predicted`. The feature `pda` array is
already CEN-DMN (see dataset.py docstring), matching the prediction sign.

Outputs a corrected per-subject CSV (raw + smoothed w11).
"""
import argparse
from pathlib import Path
import numpy as np
import yaml
from scipy.stats import pearsonr, spearmanr


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    d = cfg.get("data", {})
    if "features_dir_cluster" in d and "features_dir_local" in d:
        d["features_dir"] = (d["features_dir_cluster"]
                             if Path("/projects/swglab").exists()
                             else d["features_dir_local"])
    return cfg


def moving_average(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    pl = w // 2
    pr = w - 1 - pl
    return np.convolve(np.pad(x, (pl, pr), mode="edge"), k, mode="valid")


def reconstruct_true_pda(feat_dir, subject, window_trs, task="feedback"):
    """Replay make_predict_loader windowing to build true PDA in pred order."""
    files = sorted((feat_dir / f"sub-{subject}").glob(
        f"sub-{subject}_task-{task}_run-*_features.npz"))
    segs = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        pda = np.asarray(d["pda"], dtype=float)          # (T,) CEN-DMN
        n = len(pda)
        for start in range(0, n - window_trs + 1, window_trs):  # stride=window
            segs.append(pda[start:start + window_trs])
    if not segs:
        return None
    return np.concatenate(segs)


def metrics(pred, true, smooth=1):
    n = min(len(pred), len(true))
    pred, true = pred[:n], true[:n]
    if smooth > 1:
        pred = moving_average(pred, smooth)
    r, p = pearsonr(pred, true)
    rho, _ = spearmanr(pred, true)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return dict(pearson_r=r, pearson_p=p, spearman_rho=rho, r2=r2,
                pred_std=float(np.std(pred)), true_std=float(np.std(true)),
                n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smooth", type=int, default=11)
    ap.add_argument("--out", default="results/evaluation_results_n16_FIXED.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    feat_dir = Path(cfg["data"]["features_dir"])
    window_trs = cfg["data"]["windowing"]["window_trs"]
    excluded = set(cfg["data"]["subjects"].get("exclude", []))
    subs = [s for s in cfg["data"]["subjects"]["all"] if s not in excluded]

    rows = []
    print(f"{'subject':12s} {'r_raw':>8s} {'p_raw':>9s} {'r_smooth':>9s} "
          f"{'pred_std':>9s} {'true_std':>9s} {'n':>4s}  flag")
    print("-" * 78)
    for s in sorted(subs):
        pred_path = (feat_dir / f"sub-{s}" / "predictions" /
                     f"sub-{s}_task-feedback_pda_prediction.npz")
        if not pred_path.exists():
            print(f"{s:12s}  [no prediction]")
            continue
        pred = np.asarray(np.load(pred_path, allow_pickle=True)["pda_predicted"],
                          dtype=float)
        true = reconstruct_true_pda(feat_dir, s, window_trs)
        if true is None:
            print(f"{s:12s}  [no features]")
            continue
        if len(true) != len(pred):
            flag = f"LEN_MISMATCH pred={len(pred)} true={len(true)}"
        else:
            flag = ""
        mr = metrics(pred, true, smooth=1)
        ms = metrics(pred, true, smooth=args.smooth)
        # flag near-flat predictions
        if mr["pred_std"] < 0.05 * mr["true_std"]:
            flag = (flag + " FLAT_PRED").strip()
        rows.append(dict(subject=s, pearson_r=mr["pearson_r"],
                         pearson_p=mr["pearson_p"],
                         spearman_rho=mr["spearman_rho"], r2=mr["r2"],
                         pearson_r_smooth=ms["pearson_r"],
                         pearson_p_smooth=ms["pearson_p"],
                         pred_std=mr["pred_std"], true_std=mr["true_std"],
                         n_timepoints=mr["n"]))
        print(f"{s:12s} {mr['pearson_r']:+8.4f} {mr['pearson_p']:9.2e} "
              f"{ms['pearson_r']:+9.4f} {mr['pred_std']:9.4f} "
              f"{mr['true_std']:9.4f} {mr['n']:4d}  {flag}")

    if rows:
        rs = np.array([r["pearson_r"] for r in rows])
        rss = np.array([r["pearson_r_smooth"] for r in rows])
        print("-" * 78)
        print(f"GROUP  raw   mean r = {rs.mean():+.4f} ± {rs.std():.4f}   "
              f"median = {np.median(rs):+.4f}   positive = {(rs>0).sum()}/{len(rs)}")
        print(f"GROUP  w11   mean r = {rss.mean():+.4f} ± {rss.std():.4f}   "
              f"median = {np.median(rss):+.4f}   positive = {(rss>0).sum()}/{len(rss)}")
        try:
            import pandas as pd
            outp = Path(cfg["project"]["base_dir"]) / args.out
            outp.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(outp, index=False)
            print(f"\nSaved: {outp}")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
