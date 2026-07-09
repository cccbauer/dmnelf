#!/usr/bin/env python3
"""
fsnr_proxy.py
-------------
Is the fMRI f-SNR a good NEUROFEEDBACK signal? We build a causal (real-time-feasible)
running f-SNR and test whether it is a better feedback target than the raw PDA currently
fed back.

Causal running signal for network r (trailing window w, no future data):
    s_pda(t)  = trailing_mean(r)(t)                      # the current NF target
    s_fsnr(t) = trailing_mean(r)(t) / trailing_std(r)(t) # running signal-to-noise (f-SNR)
    s_fsnr_db(t) = 10*log10( trailing_mean^2 / trailing_var )   # dB form for plotting

NF-suitability tests (feedback block TR 30:125 vs rest baseline 0:25):
  1. Modulation: does s_fsnr rise in feedback vs rest? (a NF signal must move with state)
  2. vs current target: is the rest->feedback discriminability d' HIGHER for s_fsnr than
     for raw s_pda? (does normalizing by noise sharpen the feedback signal?)
  3. Controllability: does feedback s_fsnr track regulation success (beta_PDA)?
  4. Reliability (run-to-run ICC) + temporal smoothness (lag-1 autocorr; jitter is bad).
  5. Learning: feedback s_fsnr by run number (within-session trend).
"""
from pathlib import Path
import numpy as np, pandas as pd, re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fsnr_fmri import BASELINE_TR, HRF_DROP, DMN_I, CEN_I, TR, task_regressor, glm_fsnr

PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data"; RES = PROJ / "results"
W = 10                      # trailing window (TR) ~ 12 s
FB0 = BASELINE_TR + HRF_DROP  # first feedback TR used (post-HRF)


def trailing(x, w=W):
    s = pd.Series(x)
    return s.rolling(w, min_periods=max(3, w // 2)).mean().values, \
           s.rolling(w, min_periods=max(3, w // 2)).std().values


def running_fsnr(r, w=W):
    """Causal running signal-to-noise and its dB form."""
    m, sd = trailing(r, w)
    s_snr = m / (sd + 1e-9)
    s_db = 10 * np.log10((m ** 2) / (sd ** 2 + 1e-12) + 1e-12)
    return s_snr, s_db, m, sd


def dprime(sig, n):
    """rest(0:BASELINE) vs feedback(FB0:n) separation of a signal timeseries."""
    rest, fb = sig[:BASELINE_TR], sig[FB0:n]
    rest, fb = rest[np.isfinite(rest)], fb[np.isfinite(fb)]
    if len(rest) < 3 or len(fb) < 3:
        return np.nan
    pooled = np.sqrt(0.5 * (rest.var() + fb.var()))
    return (fb.mean() - rest.mean()) / (pooled + 1e-12)


def sflip(x, n=10000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) < 3:
        return np.nan, np.nan
    obs = x.mean(); rng = np.random.default_rng(seed)
    null = (rng.choice([-1, 1], size=(n, len(x))) * np.abs(x)).mean(1)
    return obs, float((np.sum(null >= obs) + 1) / (n + 1))


def load_fb():
    out = []
    for f in sorted(DATA.glob("sub-*_task-feedback_run-*_features.npz")):
        m = re.match(r"sub-(\w+?)_task-feedback_run-(\d+)_features", f.stem)
        z = np.load(f, allow_pickle=True); fm = np.asarray(z["fmri_features"], float)
        out.append(dict(sub=m.group(1), run=int(m.group(2)), n=fm.shape[0],
                        DMN=fm[:, DMN_I], CEN=fm[:, CEN_I], PDA=np.asarray(z["pda"], float)))
    return out


def main():
    fb = load_fb()
    rows = []
    for r in fb:
        n = r["n"]; pda = r["PDA"]
        snr, db, m, sd = running_fsnr(pda)
        s_pda, _ = trailing(pda)
        x = task_regressor(n); _, beta, _ = glm_fsnr(pda, x)
        fbslice = slice(FB0, n)
        rows.append(dict(subject=r["sub"], run=r["run"],
                         fsnr_fb=np.nanmean(snr[fbslice]),
                         fsnr_rest=np.nanmean(snr[:BASELINE_TR]),
                         dprime_fsnr=dprime(snr, n), dprime_pda=dprime(s_pda, n),
                         betaPDA=beta,
                         smooth=pd.Series(snr[FB0:n]).autocorr(lag=1)))
    df = pd.DataFrame(rows); df.to_csv(RES / "fsnr_proxy.csv", index=False)
    print(f"{len(df)} runs, W={W} TR (~{W*TR:.0f}s) causal trailing window\n")

    # 1. modulation: feedback vs rest running f-SNR
    o, p = sflip((df.fsnr_fb - df.fsnr_rest).values)
    d = (df.fsnr_fb - df.fsnr_rest)
    print("=== 1. modulation (does running f-SNR rise in feedback vs rest?) ===")
    print(f"   Δ(fb−rest) running-fSNR = {o:+.3f}  p={p:.4f}  (>0 in {100*(d>0).mean():.0f}% runs)")

    # 2. vs current target: discriminability d' (rest vs feedback)
    print("\n=== 2. NF signal quality: rest↔feedback d'  (f-SNR vs raw PDA) ===")
    of, pf = sflip(df.dprime_fsnr.values); op, pp = sflip(df.dprime_pda.values)
    print(f"   d'(running f-SNR) = {of:+.2f}  (p={pf:.4f})")
    print(f"   d'(raw PDA)       = {op:+.2f}  (p={pp:.4f})")
    od, pd_ = sflip((df.dprime_fsnr - df.dprime_pda).values)
    print(f"   Δd' (f-SNR − PDA) = {od:+.2f}  p={pd_:.4f}  → f-SNR {'BETTER' if od>0 else 'not better'} discriminator")

    # 3. controllability: does feedback f-SNR track regulation success?
    dd = df[["fsnr_fb", "betaPDA"]].dropna()
    print(f"\n=== 3. controllability: corr(feedback f-SNR, β_PDA) = {np.corrcoef(dd.fsnr_fb, dd.betaPDA)[0,1]:+.2f} ===")

    # 4. reliability + smoothness
    g = df.groupby("subject")["fsnr_fb"]; b, w = g.mean().var(), g.var().mean()
    print(f"\n=== 4. reliability run-to-run ICC = {b/(b+w):.2f} ; temporal smoothness (lag-1 autocorr) = {df.smooth.mean():.2f} ===")

    # 5. learning across runs
    print("\n=== 5. learning: feedback running f-SNR by run number ===")
    for rn in sorted(df.run.unique()):
        v = df[df.run == rn].fsnr_fb
        print(f"   run {rn}: {v.mean():+.3f}  (n={len(v)})")

    # ---- verdict figure ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax[0].bar(["rest", "feedback"], [df.fsnr_rest.mean(), df.fsnr_fb.mean()],
              color=["#999", "#7030a0"], edgecolor='k')
    ax[0].axhline(0, color='k', lw=.6); ax[0].set_ylabel("running f-SNR")
    ax[0].set_title(f"1. Modulation (Δ p={p:.3f})", fontweight='bold', fontsize=10.5)
    ax[1].bar(["raw PDA", "f-SNR"], [op, of], color=["#1f77b4", "#7030a0"], edgecolor='k')
    ax[1].axhline(0, color='k', lw=.6); ax[1].set_ylabel("rest↔feedback d'")
    ax[1].set_title(f"2. Better NF signal? Δd' p={pd_:.3f}", fontweight='bold', fontsize=10.5)
    dd2 = df[["fsnr_fb", "betaPDA"]].dropna(); rr = np.corrcoef(dd2.fsnr_fb, dd2.betaPDA)[0, 1]
    ax[2].scatter(dd2.betaPDA, dd2.fsnr_fb, s=22, alpha=.7, color="#7030a0", edgecolor='w', lw=.4)
    ax[2].axvline(0, color='grey', ls='--', lw=.8); ax[2].set_xlabel("β_PDA (regulation)")
    ax[2].set_ylabel("feedback running f-SNR")
    ax[2].set_title(f"3. Controllability (r={rr:+.2f})", fontweight='bold', fontsize=10.5)
    for a in ax: a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(RES / "fig_fsnr_proxy.png", dpi=150)
    print("\nsaved fig_fsnr_proxy.png")


if __name__ == "__main__":
    main()
