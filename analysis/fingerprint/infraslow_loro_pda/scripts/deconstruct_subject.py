"""
deconstruct_subject.py
----------------------
Render every stage of the within-feedback LORO PDA decode for ONE subject, so
each matrix / timeseries / correlation can be eyeballed. Defaults to dmnelf008
(the lone positive case: infraslow r_smooth=+0.25, circ-p=0.035).

Stages visualised:
  1. INPUTS      - z-scored infraslow block-mean feature matrix [31 ch x N_TR]
                   per run (concatenated, run boundaries marked) + PDA target.
  2. LORO DECODE - held-out prediction (raw) and smoothed prediction overlaid
                   on the true PDA, per run, concatenated.
  3. CORRELATION - scatter pred-vs-true (raw + smoothed) with Pearson r;
                   per-channel univariate r(feature, PDA).
  4. MODEL       - ElasticNet coefficients per LORO fold [fold x ch] + mean,
                   and channel-x-channel feature correlation matrix (collinearity
                   sanity check).  Topomap of mean |coef| if a montage is present.

Reuses gather/block_mean/moving_average from decode_loro so the figures show
EXACTLY what the cohort decode used.

Usage:
  python scripts/deconstruct_subject.py --config config.yaml --subject dmnelf008
Cluster env: /home/cccbauer/.conda/envs/eeg_preproc/bin/python
Output: results/figures/<subject>/*.png
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_loro import load_config, gather, moving_average, loro_predict, score  # noqa: E402

import mne  # noqa: E402
warnings.simplefilter("ignore"); mne.set_log_level("ERROR")


def loro_full(runs, mcfg):
    """LORO ElasticNet returning per-run (true, pred) AND per-fold coefficients."""
    per_run, coefs, alphas, l1s = [], [], [], []
    for i in range(len(runs)):
        tr = [runs[j] for j in range(len(runs)) if j != i]
        Xtr = np.vstack([X for _, X in tr]); ytr = np.concatenate([y for y, _ in tr])
        yte, Xte = runs[i][0], runs[i][1]
        sc = StandardScaler().fit(Xtr)
        m = ElasticNetCV(l1_ratio=mcfg["l1_ratios"], n_alphas=mcfg["n_alphas"],
                         cv=mcfg["cv_inner"], max_iter=mcfg["max_iter"])
        m.fit(sc.transform(Xtr), ytr)
        per_run.append((yte, m.predict(sc.transform(Xte))))
        coefs.append(m.coef_.copy()); alphas.append(m.alpha_); l1s.append(m.l1_ratio_)
    return per_run, np.array(coefs), np.array(alphas), np.array(l1s)


def ch_names_of(cfg, subj, task, desc):
    eroot = Path(cfg["data"]["eeg_preproc_dir"]); ses = cfg["data"]["session"]
    fif = sorted((eroot/f"sub-{subj}"/ses/"eeg").glob(
        f"sub-{subj}_{ses}_task-{task}_run-*_desc-{desc}_eeg.fif"))[0]
    raw = mne.io.read_raw_fif(str(fif), preload=False, verbose=False)
    return raw.copy().pick("eeg"), [raw.ch_names[i] for i in
                                    mne.pick_types(raw.info, eeg=True, exclude=[])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--subject", default="dmnelf008")
    a = ap.parse_args(); cfg = load_config(a.config)
    s = a.subject; task = cfg["data"]["task"]; mcfg = cfg["model"]
    w = cfg["smoothing"]["window"]
    di = cfg["data"]["eeg"]["desc_infraslow"]
    outdir = Path(cfg["project"]["base_dir"]) / "results" / "figures" / s
    outdir.mkdir(parents=True, exist_ok=True)

    runs = gather(cfg, s, task, di)
    print(f"{s}: {len(runs)} feedback runs, lengths={[len(y) for y,_ in runs]}")
    raw_eeg, chs = ch_names_of(cfg, s, task, di)
    nch = len(chs)

    per_run, coefs, alphas, l1s = loro_full(runs, mcfg)
    true = np.concatenate([t for t, _ in per_run])
    pred = np.concatenate([p for _, p in per_run])
    pred_s = np.concatenate([moving_average(p, w) for _, p in per_run])
    X_all = np.vstack([X for _, X in runs])
    bounds = np.cumsum([len(y) for y, _ in runs])[:-1]

    true_s = np.concatenate([moving_average(t, w) for t, _ in per_run])  # reference only
    r_raw = pearsonr(pred, true)[0]
    r_smo = pearsonr(pred_s, true)[0]
    r_both = pearsonr(pred_s, true_s)[0]   # smooth-both = inflated, NOT the honest metric
    print(f"  r_raw={r_raw:+.3f}  r_smooth(pred only)={r_smo:+.3f}  r_smooth-both(inflated)={r_both:+.3f}")
    print(f"  alphas={np.round(alphas,4)}  l1_ratios={l1s}")

    # ---------- FIG 1: inputs + decode, aligned in time ----------
    fig, ax = plt.subplots(5, 1, figsize=(14, 13.5), sharex=True)
    im = ax[0].imshow(X_all.T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
                      interpolation="nearest")
    ax[0].set_ylabel("EEG ch"); ax[0].set_yticks(range(0, nch, 4))
    ax[0].set_yticklabels([chs[i] for i in range(0, nch, 4)], fontsize=6)
    ax[0].set_title(f"{s}  infraslow block-mean features [z]  ({nch} ch x {len(true)} TR)")
    fig.colorbar(im, ax=ax[0], fraction=0.015, pad=0.01)

    ax[1].plot(true, lw=0.8, color="k"); ax[1].set_ylabel("PDA (true)")
    ax[1].set_title("target: PDA (CEN-DMN) per TR")
    ax[2].plot(true, lw=0.7, color="k", alpha=0.5, label="true")
    ax[2].plot(pred, lw=0.8, color="C3", label=f"pred raw (r={r_raw:+.3f})")
    ax[2].legend(loc="upper right", fontsize=8); ax[2].set_ylabel("PDA")
    ax[3].plot(true, lw=0.7, color="k", alpha=0.5, label="true (raw)")
    ax[3].plot(true_s, lw=1.0, color="k", ls="--",
               label=f"true smoothed w={w} (reference only)")
    ax[3].plot(pred_s, lw=1.0, color="C0", label=f"pred smoothed w={w} (r={r_smo:+.3f})")
    ax[3].legend(loc="upper right", fontsize=8); ax[3].set_ylabel("PDA")
    ax[3].set_title(f"honest r uses pred-smoothed vs true-RAW (r={r_smo:+.3f}); "
                    f"smoothing BOTH inflates to r={r_both:+.3f}")
    # raw vs smoothed prediction overlaid (no true) - what the w=11 smoothing does
    ax[4].plot(pred, lw=0.7, color="C3", alpha=0.6, label=f"pred raw (r={r_raw:+.3f})")
    ax[4].plot(pred_s, lw=1.2, color="C0", label=f"pred smoothed w={w} (r={r_smo:+.3f})")
    ax[4].legend(loc="upper right", fontsize=8); ax[4].set_ylabel("pred PDA")
    ax[4].set_title("prediction: raw vs smoothed")
    ax[4].set_xlabel("TR (feedback runs concatenated; dashed = run boundary)")
    for axi in ax:
        for b in bounds:
            axi.axvline(b, color="0.6", ls="--", lw=0.8)
    fig.tight_layout(); f1 = outdir / "1_inputs_decode.png"
    fig.savefig(f1, dpi=130); plt.close(fig)

    # ---------- FIG 2: correlations ----------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    def fitline(axi, x, y, color):
        b, a = np.polyfit(x, y, 1)           # y = b*x + a
        xs = np.array([x.min(), x.max()])
        axi.plot(xs, b * xs + a, color=color, lw=2,
                 label=f"slope={b:+.2f}")
        axi.legend(loc="upper left", fontsize=8)

    ax[0].scatter(pred, true, s=8, alpha=0.4, color="C3")
    fitline(ax[0], pred, true, "darkred")
    ax[0].set_xlabel("pred raw"); ax[0].set_ylabel("true PDA")
    ax[0].set_title(f"raw  r={r_raw:+.3f}")
    ax[1].scatter(pred_s, true, s=8, alpha=0.4, color="C0")
    fitline(ax[1], pred_s, true, "navy")
    ax[1].set_xlabel("pred smoothed"); ax[1].set_ylabel("true PDA")
    ax[1].set_title(f"smoothed w={w}  r={r_smo:+.3f}")
    rch = np.array([pearsonr(X_all[:, c], true)[0] for c in range(nch)])
    order = np.argsort(rch)
    ax[2].barh(range(nch), rch[order], color=["C0" if v > 0 else "C3" for v in rch[order]])
    ax[2].set_yticks(range(nch)); ax[2].set_yticklabels([chs[i] for i in order], fontsize=5)
    ax[2].axvline(0, color="k", lw=0.7)
    ax[2].set_xlabel("r(channel feature, PDA)")
    ax[2].set_title("univariate channel-PDA correlation")
    fig.tight_layout(); f2 = outdir / "2_correlations.png"
    fig.savefig(f2, dpi=130); plt.close(fig)

    # ---------- FIG 3: model coefficients + collinearity ----------
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    im = ax[0].imshow(coefs, aspect="auto", cmap="RdBu_r",
                      vmin=-np.abs(coefs).max() or 1, vmax=np.abs(coefs).max() or 1)
    ax[0].set_yticks(range(len(coefs)))
    ax[0].set_yticklabels([f"fold {i+1}\n(held run {i+1})" for i in range(len(coefs))], fontsize=7)
    ax[0].set_xticks(range(0, nch, 3)); ax[0].set_xticklabels([chs[i] for i in range(0, nch, 3)],
                                                              rotation=90, fontsize=5)
    ax[0].set_title("ElasticNet coefficients per LORO fold")
    fig.colorbar(im, ax=ax[0], fraction=0.04, pad=0.02)

    mc = coefs.mean(0); mo = np.argsort(mc)
    ax[1].barh(range(nch), mc[mo], color=["C0" if v > 0 else "C3" for v in mc[mo]])
    ax[1].set_yticks(range(nch)); ax[1].set_yticklabels([chs[i] for i in mo], fontsize=5)
    ax[1].axvline(0, color="k", lw=0.7); ax[1].set_xlabel("mean coef across folds")
    nz = (coefs != 0).any(0).sum()
    ax[1].set_title(f"mean coef ({nz}/{nch} ch ever non-zero)")

    cc = np.corrcoef(X_all.T)
    im2 = ax[2].imshow(cc, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[2].set_title("feature channel-channel correlation\n(collinearity)")
    ax[2].set_xticks([]); ax[2].set_yticks([])
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.02)
    fig.tight_layout(); f3 = outdir / "3_model_weights.png"
    fig.savefig(f3, dpi=130); plt.close(fig)

    # ---------- FIG 4: topomap of mean |coef| (best effort) ----------
    try:
        raw_eeg.set_montage("easycap-M1", match_case=False, on_missing="warn")
        fig, axx = plt.subplots(1, 2, figsize=(9, 4.5))
        mne.viz.plot_topomap(np.abs(mc), raw_eeg.info, axes=axx[0], show=False, cmap="viridis")
        axx[0].set_title("mean |coef|")
        mne.viz.plot_topomap(rch, raw_eeg.info, axes=axx[1], show=False, cmap="RdBu_r")
        axx[1].set_title("univariate r(ch, PDA)")
        fig.tight_layout(); f4 = outdir / "4_topomap.png"
        fig.savefig(f4, dpi=130); plt.close(fig)
        print(f"  wrote {f4.name}")
    except Exception as e:
        print(f"  topomap skipped: {e}")

    # ---------- FIG 5: robustness checks ----------
    frontal = {"Fp1", "Fp2", "F7", "F8"}
    keep = np.array([c not in frontal for c in chs])

    def mask_cols(rs, m):
        return [(y, X[:, m]) for y, X in rs]

    def rm_global(rs):
        return [(y, X - X.mean(1, keepdims=True)) for y, X in rs]

    variants = {
        "full\n(31 ch)": runs,
        "no frontal/ocular\n(-Fp1/Fp2/F7/F8)": mask_cols(runs, keep),
        "per-TR global\nmean removed": rm_global(runs),
    }
    print("  robustness:")
    vres = {}
    for name, rs in variants.items():
        rr_, rs_, p_, _ = score(loro_predict(rs, mcfg), w)
        vres[name] = (rr_, rs_, p_)
        print(f"    {name.replace(chr(10),' '):40s} r_raw={rr_:+.3f} r_sm={rs_:+.3f} circ-p={p_:.3f}")
    runr = [pearsonr(p, t)[0] if p.std() > 0 else np.nan for t, p in per_run]
    print(f"    per-run held-out r (raw): {np.round(runr, 3)}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    names = list(vres); xpos = np.arange(len(names))
    raws = [vres[n][0] for n in names]; smos = [vres[n][1] for n in names]
    ax[0].bar(xpos - 0.2, raws, 0.4, label="r raw", color="C3")
    ax[0].bar(xpos + 0.2, smos, 0.4, label=f"r smooth w={w}", color="C0")
    for i, n in enumerate(names):
        ax[0].text(i, max(raws[i], smos[i]) + 0.005, f"p={vres[n][2]:.3f}",
                   ha="center", fontsize=8)
    ax[0].set_xticks(xpos); ax[0].set_xticklabels(names, fontsize=8)
    ax[0].axhline(0, color="k", lw=0.7); ax[0].legend(fontsize=8)
    ax[0].set_ylabel("Pearson r (pred vs true PDA)")
    ax[0].set_title("decode robustness to channel set / global mean")

    ax[1].bar(range(len(runr)), runr, color=["C0" if v > 0 else "C3" for v in runr])
    ax[1].axhline(0, color="k", lw=0.7)
    ax[1].set_xticks(range(len(runr)))
    ax[1].set_xticklabels([f"run {i+1}" for i in range(len(runr))])
    ax[1].set_ylabel("held-out r (raw)")
    ax[1].set_title("per-run held-out correlation (full model)")
    fig.tight_layout(); f5 = outdir / "5_robustness.png"
    fig.savefig(f5, dpi=130); plt.close(fig)
    print(f"  wrote {f5.name}")

    print(f"figures -> {outdir}")


if __name__ == "__main__":
    main()
