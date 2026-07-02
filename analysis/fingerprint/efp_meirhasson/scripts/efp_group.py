#!/usr/bin/env python3
"""
efp_group.py
------------
Group-level analysis for the EEG Finger-Print (run after the per-subject array):

  1. Aggregate per-subject CSVs -> group mean r/NMSE per target/resolution/method,
     with a sign-flip permutation p across subjects (EFP vs baselines).
  2. Group-averaged EFP fingerprint: average per-subject best-electrode [band x delay]
     matrices (by band index) per target/resolution -> heatmap.
  3. LOSO cross-subject transfer: for a common electrode (modal best per target/res),
     train ridge on N-1 subjects (band-index-aligned, per-subject z-scored), predict the
     held-out subject; report mean r + sign-flip p. (Bands are per-subject data-driven,
     so band-index alignment is an approximation - noted as a caveat.)

Outputs (results/<outdir>/):
  efp_persubject_all.csv, efp_group_summary.csv, efp_group_loso.csv,
  efp_group_fingerprint_<target>_<res>.png, group_fingerprints.npz
"""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import zscore, pearsonr
from sklearn.linear_model import RidgeCV

from efp_features import load_config, load_subject_features, make_delay_design
from efp_decode import assemble, nmse, mk_block_folds

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_DIR = SCRIPT_DIR.parent


def sign_flip_test(r_vals, n_flips=10000, seed=42):
    r = np.asarray([v for v in r_vals if np.isfinite(v)])
    if len(r) < 3:
        return np.nanmean(r_vals), 1.0
    obs = r.mean()
    rng = np.random.default_rng(seed)
    null = np.array([np.mean(rng.choice([-1, 1], size=len(r)) * r) for _ in range(n_flips)])
    p_right = (np.sum(null >= obs) + 1) / (n_flips + 1)
    p = 2 * min(p_right, (np.sum(null <= obs) + 1) / (n_flips + 1))
    return obs, min(p, 1.0)


def aggregate(out_dir):
    csvs = sorted(out_dir.glob("efp_persubject_*.csv"))
    if not csvs:
        raise SystemExit(f"No per-subject CSVs in {out_dir}")
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    df.to_csv(out_dir / "efp_persubject_all.csv", index=False)

    rows = []
    for (tgt, res, method), g in df.groupby(["target", "resolution", "method"]):
        m, p = sign_flip_test(g["mean_r"].values)
        rows.append(dict(target=tgt, resolution=res, method=method,
                         n=len(g), mean_r=m, sign_flip_p=p,
                         mean_nmse=g["mean_nmse"].mean()))
    summ = pd.DataFrame(rows)
    summ.to_csv(out_dir / "efp_group_summary.csv", index=False)
    return df, summ


def group_fingerprints(cfg, out_dir):
    """Average per-subject best-electrode EFP matrices (by band index)."""
    saved = {}
    for npz in out_dir.glob("efp_*_*.npz"):
        # filename: efp_<sub>_<target>_<res>.npz  (target may contain underscore)
        stem = npz.stem[len("efp_"):]
        parts = stem.split("_")
        sub = parts[0]; res = parts[-1]; target = "_".join(parts[1:-1])
        z = np.load(npz, allow_pickle=True)
        saved.setdefault((target, res), []).append(z["efp"])
    out = {}
    for (target, res), mats in saved.items():
        shp = min(m.shape[1] for m in mats)
        stack = np.array([m[:, :shp] for m in mats])          # (n_sub, bands, delays)
        # normalize each subject's matrix (unit max-abs) before averaging
        stack = np.array([m / (np.abs(m).max() + 1e-12) for m in stack])
        out[(target, res)] = stack.mean(axis=0)
        _plot_fingerprint(out[(target, res)], target, res, cfg,
                          out_dir / f"efp_group_fingerprint_{target}_{res}.png")
    np.savez_compressed(out_dir / "group_fingerprints.npz",
                        **{f"{t}__{r}": v for (t, r), v in out.items()})
    return out


def _plot_fingerprint(mat, target, res, cfg, path):
    n_bands, n_delays = mat.shape
    tr = cfg["data"]["fmri"]["tr"]
    step = tr if res == "tr" else 1.0 / cfg["efp"]["hz4"]
    delays = -np.arange(n_delays) * step
    fig, ax = plt.subplots(figsize=(7, 4))
    vmax = np.abs(mat).max()
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[delays[-1], delays[0], n_bands + 0.5, 0.5])
    ax.set_xlabel("Time delay (s)"); ax.set_ylabel("Frequency band (low→high)")
    ax.set_title(f"Group EFP fingerprint — {target} ({res})")
    fig.colorbar(im, ax=ax, label="norm. ridge weight")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def loso_transfer(cfg, out_dir, cache_dir, df):
    """LOSO: common electrode per target/res; train N-1, predict held-out subject."""
    e = cfg["efp"]; tr = cfg["data"]["fmri"]["tr"]
    alphas = np.logspace(np.log10(e["alpha_grid_lo"]), np.log10(e["alpha_grid_hi"]), e["alpha_grid_n"])
    subs = cfg["data"]["subjects"]["all"]
    # cache features per subject once
    feats = {}
    for s in subs:
        try:
            runs, chs = load_subject_features(cache_dir, s)
            feats[s] = (runs, chs)
        except FileNotFoundError:
            continue

    rows = []
    for res in e["resolutions"]:
        n_delays = int(round(e["delay_window_s"] / tr)) + 1 if res == "tr" \
            else int(round(e["delay_window_s"] * e["hz4"])) + 1
        for target in cfg["targets"]:
            # common electrode = modal best EFP electrode across subjects for this target/res
            sel = df[(df.target == target) & (df.resolution == res) & (df.method == "EFP")]
            if sel.empty:
                continue
            common_ch = Counter(sel["best_ch"]).most_common(1)[0][0]
            # build per-subject (X, y) for the common electrode
            data = {}
            for s, (runs, chs) in feats.items():
                if common_ch not in chs:
                    continue
                ci = chs.index(common_ch)
                X, y = assemble(runs, ci, target, res, n_delays)
                if X is not None:
                    data[s] = (zscore(X, axis=0), y)
            if len(data) < 4:
                continue
            r_subs = []
            for held in data:
                Xtr = np.vstack([data[s][0] for s in data if s != held])
                ytr = np.concatenate([data[s][1] for s in data if s != held])
                Xte, yte = data[held]
                model = RidgeCV(alphas=alphas).fit(Xtr, ytr)
                pred = model.predict(Xte)
                if np.std(pred) > 1e-9:
                    r_subs.append(pearsonr(yte, pred)[0])
            m, p = sign_flip_test(r_subs)
            rows.append(dict(target=target, resolution=res, common_ch=common_ch,
                             n=len(r_subs), loso_mean_r=m, sign_flip_p=p))
            print(f"  LOSO {target} {res} (ch {common_ch}): r={m:+.3f} p={p:.3f} n={len(r_subs)}")
    loso = pd.DataFrame(rows)
    loso.to_csv(out_dir / "efp_group_loso.csv", index=False)
    return loso


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="full")
    ap.add_argument("--cache", default=str(PROJ_DIR / "results" / "features_cache"))
    ap.add_argument("--skip_loso", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    out_dir = PROJ_DIR / "results" / args.outdir
    cache_dir = Path(args.cache)

    df, summ = aggregate(out_dir)
    print("=== Group summary (mean r, sign-flip p) ===")
    print(summ.pivot_table(index=["target", "resolution"], columns="method",
                           values="mean_r").round(3))
    group_fingerprints(cfg, out_dir)
    if not args.skip_loso:
        print("=== LOSO cross-subject transfer ===")
        loso_transfer(cfg, out_dir, cache_dir, df)
    print(f"\nSaved group outputs to {out_dir}")


if __name__ == "__main__":
    main()
