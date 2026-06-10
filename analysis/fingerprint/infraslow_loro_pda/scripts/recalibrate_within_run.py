"""
recalibrate_within_run.py
--------------------------
Does PER-RUN RE-CALIBRATION (the realistic real-time-neurofeedback setup)
recover infraslow->PDA signal where pooled cross-run LORO fails?

LORO pools 3 runs and must TRANSFER the decoder across the run-to-run
non-stationarity (which kills 007: diagnose_collapse.py). Per-run re-calibration
instead fits the decoder WITHIN the run being decoded:
  - forward split: train Ridge on the first `frac` of a run (calibration),
    decode the remaining TRs (no shuffle -> respects real-time causality).
Model is matched (RidgeCV) so the comparison is validation-scheme, not model.

For each subject we report, per run and pooled:
  - within-run re-calibration r   (forward split)
  - cross-run Ridge LORO r        (train other 3 runs, predict this one)
Cohort: mean r per scheme, n significant (circular-shift null), and a
calibration-fraction sweep. Highlights 007 (LORO-collapse) and 008 (LORO-works).

Usage: python scripts/recalibrate_within_run.py --config config.yaml --frac 0.5
Output: results/recalibrate_within_run.csv  +  results/figures/recalibrate_within_run.png
"""
import argparse, sys, warnings, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_loro import load_config, gather, moving_average  # noqa: E402
import mne  # noqa: E402
warnings.simplefilter("ignore"); mne.set_log_level("ERROR")

ALPHAS = np.logspace(-1, 4, 30)


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xte))


def recal_run(y, X, frac):
    """Forward within-run split: train first `frac`, decode the rest.
    Returns (true_test, pred_test) or None if calibration too short."""
    n = len(y); k = int(round(n * frac))
    if k < 20 or n - k < 20:
        return None
    pred = fit_predict(X[:k], y[:k], X[k:])
    return y[k:], pred


def loro_runs(runs):
    """Cross-run Ridge LORO: per held-out run (true, pred)."""
    out = []
    for i in range(len(runs)):
        tr = [runs[j] for j in range(len(runs)) if j != i]
        Xtr = np.vstack([X for _, X in tr]); ytr = np.concatenate([y for y, _ in tr])
        yte, Xte = runs[i]
        out.append((yte, fit_predict(Xtr, ytr, Xte)))
    return out


def pooled_r(per_seg, smooth=0):
    segs_t, segs_p = [], []
    for t, p in per_seg:
        if smooth > 1:
            p = moving_average(p, smooth)
        segs_t.append(t); segs_p.append(p)
    true = np.concatenate(segs_t); pred = np.concatenate(segs_p)
    if pred.std() < 1e-12:
        return np.nan, 1.0, segs_t, segs_p
    r = pearsonr(pred, true)[0]
    # circular-shift null: shift true within each segment
    rng = np.random.default_rng(0); nperm = 1000; ge = 0; obs = abs(r)
    lens = [len(t) for t in segs_t]
    for _ in range(nperm):
        sh = [];
        for t, L in zip(segs_t, lens):
            kk = int(rng.integers(5, max(6, L - 5))); sh.append(np.roll(t, kk))
        if abs(pearsonr(pred, np.concatenate(sh))[0]) >= obs:
            ge += 1
    return r, (ge + 1) / (nperm + 1), segs_t, segs_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--frac", type=float, default=0.5, help="calibration fraction")
    ap.add_argument("--sweep", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    a = ap.parse_args(); cfg = load_config(a.config)
    task = cfg["data"]["task"]; di = cfg["data"]["eeg"]["desc_infraslow"]
    subs = [s for s in cfg["data"]["subjects"]["all"]
            if s not in set(cfg["data"]["subjects"].get("exclude", []))]

    print(f"per-run re-calibration (Ridge, forward split frac={a.frac}) vs cross-run Ridge LORO")
    print(f"{'subject':11s} | {'RECAL r  per-run':28s} | recal_pooled p | LORO_pooled")
    print("-" * 86)
    rows = []; sweep_cohort = {f: [] for f in a.sweep}
    for s in subs:
        runs = gather(cfg, s, task, di)
        if len(runs) < 2:
            print(f"{s:11s} | <2 runs"); continue
        # within-run re-calibration at headline frac
        recal = [recal_run(y, X, a.frac) for y, X in runs]
        recal = [r for r in recal if r is not None]
        rec_r, rec_p, _, _ = pooled_r(recal)
        per_run_r = [pearsonr(p, t)[0] if p.std() > 1e-12 else np.nan for t, p in recal]
        # cross-run Ridge LORO
        lo = loro_runs(runs)
        lo_r, lo_p, _, _ = pooled_r(lo)
        # calibration-fraction sweep
        for f in a.sweep:
            rr = [recal_run(y, X, f) for y, X in runs]
            rr = [x for x in rr if x is not None]
            if rr:
                sweep_cohort[f].append(pooled_r(rr)[0])
        print(f"{s:11s} | {np.array2string(np.round(per_run_r,2), separator=' '):28s} "
              f"| recal={rec_r:+.3f} p={rec_p:.3f} | LORO={lo_r:+.3f}")
        rows.append(dict(subject=s, recal_r=rec_r, recal_p=rec_p,
                         loro_ridge_r=lo_r, loro_p=lo_p,
                         per_run_r=";".join(f"{x:.3f}" for x in per_run_r)))

    rec = np.array([r["recal_r"] for r in rows], float)
    lor = np.array([r["loro_ridge_r"] for r in rows], float)
    nsig_rec = sum(1 for r in rows if r["recal_p"] < 0.05 and r["recal_r"] > 0)
    nsig_lor = sum(1 for r in rows if r["loro_p"] < 0.05 and r["loro_ridge_r"] > 0)
    print("-" * 86)
    print(f"COHORT  recal mean r={np.nanmean(rec):+.3f}  (sig {nsig_rec}/{len(rows)})   "
          f"LORO mean r={np.nanmean(lor):+.3f}  (sig {nsig_lor}/{len(rows)})")
    sweep_means = {f: np.nanmean(v) for f, v in sweep_cohort.items()}
    print("  calibration-fraction sweep (cohort mean recal r): " +
          "  ".join(f"{f:.1f}:{m:+.3f}" for f, m in sweep_means.items()))

    # ---------- figure ----------
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(rows)); labels = [r["subject"].replace("dmnelf", "") for r in rows]
    ax[0].bar(x - 0.2, lor, 0.4, label="cross-run LORO (Ridge)", color="C1")
    ax[0].bar(x + 0.2, rec, 0.4, label=f"within-run recal (frac={a.frac})", color="C0")
    ax[0].axhline(0, color="k", lw=0.7); ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, rotation=90, fontsize=7)
    for xi, r in zip(x, rows):
        if r["subject"] in ("dmnelf007", "dmnelf008"):
            ax[0].axvspan(xi - 0.5, xi + 0.5, color="0.9", zorder=0)
    ax[0].legend(fontsize=9); ax[0].set_ylabel("pooled Pearson r")
    ax[0].set_title("per-subject: within-run recalibration vs cross-run LORO")

    ax[1].scatter(lor, rec, s=40, color="C0")
    for r in rows:
        if r["subject"] in ("dmnelf007", "dmnelf008"):
            ax[1].annotate(r["subject"].replace("dmnelf", ""),
                           (r["loro_ridge_r"], r["recal_r"]), fontsize=9, weight="bold")
    lim = [min(lor.min(), rec.min()) - 0.05, max(lor.max(), rec.max()) + 0.05]
    ax[1].plot(lim, lim, "k--", lw=0.8, label="y=x"); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].axhline(0, color="0.7", lw=0.6); ax[1].axvline(0, color="0.7", lw=0.6)
    ax[1].set_xlabel("cross-run LORO r"); ax[1].set_ylabel("within-run recal r")
    ax[1].legend(fontsize=9); ax[1].set_title("does recalibration beat LORO? (above y=x)")

    fr = list(sweep_means); ax[2].plot(fr, [sweep_means[f] for f in fr], "o-", color="C0")
    ax[2].axhline(np.nanmean(lor), color="C1", ls="--", label="cohort LORO mean")
    ax[2].axhline(0, color="0.7", lw=0.6)
    ax[2].set_xlabel("calibration fraction"); ax[2].set_ylabel("cohort mean recal r")
    ax[2].legend(fontsize=9); ax[2].set_title("calibration-fraction sweep")

    fig.suptitle("Per-run re-calibration vs pooled LORO (infraslow -> PDA, within-feedback)",
                 fontsize=13)
    fig.tight_layout()
    base = Path(cfg["project"]["base_dir"]) / "results"
    (base / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(base / "figures" / "recalibrate_within_run.png", dpi=130); plt.close(fig)
    with open(base / "recalibrate_within_run.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print(f"saved: {base/'recalibrate_within_run.csv'} and figures/recalibrate_within_run.png")


if __name__ == "__main__":
    main()
