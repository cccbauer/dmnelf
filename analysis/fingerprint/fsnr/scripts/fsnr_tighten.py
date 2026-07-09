#!/usr/bin/env python3
"""
fsnr_tighten.py
---------------
Tighten the DMN-as-noise (variability quenching) result against two confounds:
  (1) STARTUP TRANSIENT: the 25-TR "rest" baseline is the run start. Control = the SAME
      first-25 vs rest-of-run split applied to DEDICATED REST runs (no regulation). If DMN
      variance also "quenches" there, the feedback effect is a startup artifact.
  (2) RAW WHOLE-BRAIN: DiFuMo cols are z-scored, washing out absolute quench. Use the
      fMRIPrep global_signal (raw whole-brain) + personalized DMN/CEN (scale-retained) to
      test whether DMN decluttering is network-SPECIFIC vs global arousal.

Also answers: relation between DMN quench (noise down) and CEN/PDA GLM f-SNR (signal up) —
coupled or dissociable mechanisms of higher f-SNR?
"""
from pathlib import Path
import numpy as np, pandas as pd, re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fsnr_fmri import task_regressor, glm_fsnr, BASELINE_TR, HRF_DROP, DMN_I, CEN_I

PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data"; RES = PROJ / "results"
GS = dict(np.load(DATA / "global_signal.npz", allow_pickle=True))  # key "sub|task|run"


def vsplit(r):
    """(var of first-25-TR block, var of post-HRF feedback/rest block)."""
    n = len(r)
    return r[:BASELINE_TR].var(), r[BASELINE_TR + HRF_DROP:n].var()


def qdb(vr, vf):
    return 10 * np.log10(vr / vf) if (vf > 1e-12 and vr > 0) else np.nan


def sflip(x, n=10000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) < 3:
        return np.nan, np.nan
    obs = x.mean(); rng = np.random.default_rng(seed)
    null = (rng.choice([-1, 1], size=(n, len(x))) * np.abs(x)).mean(1)
    return obs, float((np.sum(null >= obs) + 1) / (n + 1))


def corr(df, a, b):
    d = df[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
    return np.corrcoef(d[a], d[b])[0, 1] if len(d) > 3 else np.nan


def load(task):
    rows = []
    for f in sorted(DATA.glob(f"sub-*_task-{task}_run-*_features.npz")):
        m = re.match(rf"sub-(\w+?)_task-{task}_run-(\d+)_features", f.stem)
        z = np.load(f, allow_pickle=True)
        rows.append((m.group(1), int(m.group(2)), np.asarray(z["fmri_features"], float),
                     np.asarray(z["pda"], float)))
    return rows


def main():
    fb = load("feedback")

    # ===== (1) STARTUP-TRANSIENT CONTROL: same 25-vs-rest split in dedicated rest runs =====
    print("=== startup control: 'quench' (first25 vs rest-of-run) in FEEDBACK vs REST runs ===")
    print("    (if REST runs also show DMN quench -> the effect is a startup artifact)\n")
    for cond, task in [("FEEDBACK", "feedback"), ("REST", "rest"), ("SHORTREST", "shortrest")]:
        runs = fb if task == "feedback" else load(task)
        qD = [qdb(*vsplit(fm[:, DMN_I])) for _, _, fm, _ in runs]
        qC = [qdb(*vsplit(fm[:, CEN_I])) for _, _, fm, _ in runs]
        # global signal quench for these runs
        qG = []
        for s, r, fm, _ in runs:
            g = GS.get(f"{s}|{task}|{r}")
            if g is not None and len(g) >= BASELINE_TR + HRF_DROP + 10:
                qG.append(qdb(*vsplit(g)))
        oD, pD = sflip(qD); oC, pC = sflip(qC); oG, pG = sflip(qG)
        print(f"  {cond:10s} (n={len(runs):2d})  DMN={oD:+.2f}dB p={pD:.3f}   "
              f"CEN={oC:+.2f}dB p={pC:.3f}   global={oG:+.2f}dB p={pG:.3f}")

    # ===== (2) per feedback run: raw quench + specificity + GLM signal =====
    rows = []
    for s, run, fm, pda in fb:
        dmn, cen = fm[:, DMN_I], fm[:, CEN_I]
        d = dict(subject=s, run=run)
        d["qDMN"] = qdb(*vsplit(dmn)); d["qCEN"] = qdb(*vsplit(cen)); d["qPDA"] = qdb(*vsplit(pda))
        g = GS.get(f"{s}|feedback|{run}")
        d["qGLOBAL"] = qdb(*vsplit(g)) if (g is not None and len(g) >= 40) else np.nan
        x = task_regressor(len(dmn))
        d["fsnrPDA"], d["betaPDA"], d["r2PDA"] = glm_fsnr(pda, x)
        d["fsnrCEN"], d["betaCEN"], d["r2CEN"] = glm_fsnr(cen, x)
        # noise term = residual variance of PDA/CEN GLM (the "N" in signal/N)
        Xd = np.column_stack([np.ones_like(x), x])
        for nm, r in [("PDA", pda), ("CEN", cen)]:
            b, *_ = np.linalg.lstsq(Xd, r, rcond=None)
            d[f"resid_{nm}"] = np.var(r - Xd @ b)
        rows.append(d)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "fsnr_tighten.csv", index=False)

    print("\n=== DMN-specificity (raw): is DMN quench > CEN / > global? (paired, dB) ===")
    for other in ["qCEN", "qGLOBAL"]:
        o, p = sflip((df["qDMN"] - df[other]).values)
        print(f"  DMN − {other[1:]:6s}: {o:+.2f} dB  p={p:.4f}  ({'DMN MORE' if o > 0 else 'not more'})")

    print("\n=== reliability (run-to-run ICC) ===")
    for c in ["qDMN", "qCEN", "qGLOBAL"]:
        dd = df[["subject", c]].replace([np.inf, -np.inf], np.nan).dropna()
        g = dd.groupby("subject")[c]; b, w = g.mean().var(), g.var().mean()
        print(f"  {c:8s} ICC={b/(b+w):.2f}")

    # ===== Q2: DMN quench (noise down) vs CEN/PDA f-SNR (signal up) =====
    print("\n=== Q2: relation DMN quench (noise↓) ↔ CEN/PDA f-SNR / signal ===")
    print(f"  corr(qDMN, fsnr PDA)  = {corr(df,'qDMN','fsnrPDA'):+.2f}   (noise-drop ~ higher f-SNR?)")
    print(f"  corr(qDMN, fsnr CEN)  = {corr(df,'qDMN','fsnrCEN'):+.2f}")
    print(f"  corr(qDMN, β_PDA)     = {corr(df,'qDMN','betaPDA'):+.2f}   (noise-drop ~ stronger regulation?)")
    print(f"  corr(qDMN, qCEN)      = {corr(df,'qDMN','qCEN'):+.2f}   (networks declutter together?)")
    print(f"  corr(qDMN, qGLOBAL)   = {corr(df,'qDMN','qGLOBAL'):+.2f}")
    # decompose f-SNR gain: signal (beta^2) vs noise (residual) -- which drives fsnr?
    print("\n  f-SNR = signal/noise: does f-SNR track signal-up (β²) or noise-down (residual)?")
    df["beta2PDA"] = df["betaPDA"] ** 2
    print(f"    corr(fsnrPDA, β²_PDA)    = {corr(df,'fsnrPDA','beta2PDA'):+.2f}  (signal channel)")
    print(f"    corr(fsnrPDA, resid_PDA) = {corr(df,'fsnrPDA','resid_PDA'):+.2f}  (noise channel; expect <0)")
    print(f"    corr(qDMN, resid_PDA)    = {corr(df,'qDMN','resid_PDA'):+.2f}  (DMN declutter ~ lower PDA noise?)")


if __name__ == "__main__":
    main()
