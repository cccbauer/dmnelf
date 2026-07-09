#!/usr/bin/env python3
"""
fsnr_fmri.py
------------
Purely-fMRI functional signal-to-noise ratio (f-SNR) proxy for DMNELF neurofeedback,
faithful to the law-of-total-variance decomposition (Laukkonen 2026; Nath 2026):

    Var(r) = Var_z(E[r|z])   +   E_z(Var(r|z))
             \___ signal ___/       \__ noise __/     ,   f-SNR = signal / noise

where the goal-relevant cause z = {rest, feedback} is the WITHIN-RUN condition:
each feedback run = 25 TR rest baseline + 100 TR feedback (drop the first HRF_DROP
feedback TRs for the ~5 s hemodynamic lag). f-SNR reported in dB (10*log10), matching
Nath's ERP-SNR. r = personalized DMN (col 64), CEN (col 65), PDA (= CEN - DMN).

Variants:
  fSNR_{PDA,CEN,DMN} : per-network law-of-total-variance ratio.
  fSNR_CENDMN        : CEN task-signal over DMN endogenous noise (the "DMN = noise"
                       reading): signal_CEN / noise_DMN.

Also records directional regulation (delta = mean_feedback - mean_rest) to TEST the
hypothesis that higher PDA / lower DMN goes with higher f-SNR (non-circular: f-SNR is a
variance ratio, delta is a mean).

Reads local npz pulled from cyclic_features (fmri_features [n_tr,66], key 'pda').
Writes results/fsnr_fmri.csv (per run) + prints validation.
"""
from pathlib import Path
import numpy as np, pandas as pd, re

PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data"
RES = PROJ / "results"; RES.mkdir(exist_ok=True)
DMN_I, CEN_I = 64, 65
BASELINE_TR = 25      # first 25 TR = rest baseline
HRF_DROP = 5          # drop first 5 feedback TRs (HRF lag ~6 s at TR 1.2)


def lotv(r, zmask):
    """Law-of-total-variance decomposition of series r under binary condition labels.
    zmask: boolean array, True=feedback, False=rest (only these TRs are used).
    Returns dict(signal, noise, fsnr, fsnr_db, mean_rest, mean_fb, delta)."""
    rest, fb = r[~zmask], r[zmask]
    n0, n1 = len(rest), len(fb); n = n0 + n1
    p0, p1 = n0 / n, n1 / n
    m0, m1 = rest.mean(), fb.mean()
    grand = p0 * m0 + p1 * m1
    signal = p0 * (m0 - grand) ** 2 + p1 * (m1 - grand) ** 2       # Var_z(E[r|z])
    noise = p0 * rest.var() + p1 * fb.var()                        # E_z(Var(r|z))
    fsnr = signal / noise if noise > 1e-12 else np.nan
    return dict(signal=signal, noise=noise, fsnr=fsnr,
                fsnr_db=10 * np.log10(fsnr) if fsnr and fsnr > 0 else np.nan,
                mean_rest=m0, mean_fb=m1, delta=m1 - m0)


def run_fsnr(fm, pda):
    n_tr = fm.shape[0]
    dmn, cen = fm[:, DMN_I], fm[:, CEN_I]
    # condition mask over the TRs we keep: rest = 0:BASELINE_TR, feedback = BASELINE_TR+HRF_DROP:end
    keep = np.zeros(n_tr, bool); zmask = np.zeros(n_tr, bool)
    keep[:BASELINE_TR] = True
    keep[BASELINE_TR + HRF_DROP:] = True
    zmask[BASELINE_TR + HRF_DROP:] = True
    kz, kk = zmask[keep], keep
    out = {}
    for nm, r in [("PDA", pda), ("CEN", cen), ("DMN", dmn)]:
        d = lotv(r[kk], kz)
        out[f"fsnr_{nm}_db"] = d["fsnr_db"]
        out[f"signal_{nm}"] = d["signal"]; out[f"noise_{nm}"] = d["noise"]
        out[f"delta_{nm}"] = d["delta"]; out[f"meanfb_{nm}"] = d["mean_fb"]
    # DMN-as-noise variant: CEN task-signal over DMN endogenous noise
    sC = out["signal_CEN"]; nD = out["noise_DMN"]
    out["fsnr_CENDMN_db"] = 10 * np.log10(sC / nD) if (nD > 1e-12 and sC > 0) else np.nan
    return out


def main():
    rows = []
    for f in sorted(DATA.glob("sub-*_task-feedback_run-*_features.npz")):
        m = re.match(r"sub-(\w+?)_task-feedback_run-(\d+)_features", f.stem)
        sub, run = m.group(1), int(m.group(2))
        z = np.load(f, allow_pickle=True)
        fm = np.asarray(z["fmri_features"], float)
        pda = np.asarray(z["pda"], float)
        if fm.shape[0] < BASELINE_TR + HRF_DROP + 10:
            continue
        rows.append(dict(subject=sub, run=run, n_tr=fm.shape[0], **run_fsnr(fm, pda)))
    df = pd.DataFrame(rows).sort_values(["subject", "run"]).reset_index(drop=True)
    df.to_csv(RES / "fsnr_fmri.csv", index=False)
    print(f"{len(df)} runs, {df.subject.nunique()} subjects -> {RES/'fsnr_fmri.csv'}\n")

    # ---- per-network group summary (dB) ----
    print("=== per-run f-SNR (dB), group mean ± sd ===")
    for c in ["fsnr_PDA_db", "fsnr_CEN_db", "fsnr_DMN_db", "fsnr_CENDMN_db"]:
        v = df[c].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  {c:16s} {v.mean():+.2f} ± {v.std():.2f} dB   (n={len(v)})")

    print("\n=== directional regulation (mean_feedback - mean_rest), group mean ===")
    for c in ["delta_PDA", "delta_CEN", "delta_DMN"]:
        print(f"  {c:10s} {df[c].mean():+.3f}   (>0 in {100*(df[c]>0).mean():.0f}% of runs)")

    # ---- reliability: run-to-run (ICC-ish via between/within variance) ----
    print("\n=== reliability (per-subject mean, and run-to-run) ===")
    for c in ["fsnr_PDA_db", "fsnr_CEN_db", "fsnr_DMN_db", "fsnr_CENDMN_db"]:
        d = df[["subject", c]].replace([np.inf, -np.inf], np.nan).dropna()
        g = d.groupby("subject")[c]
        btw = g.mean().var()                     # between-subject variance of means
        wth = g.var().mean()                     # mean within-subject variance
        icc = btw / (btw + wth) if (btw + wth) > 0 else np.nan
        print(f"  {c:16s} ICC≈{icc:.2f}  (between={btw:.2f}, within={wth:.2f})")

    # ---- non-redundancy with PDA level ----
    print("\n=== non-redundancy: corr(f-SNR, mean feedback PDA) across runs ===")
    for c in ["fsnr_PDA_db", "fsnr_CEN_db", "fsnr_CENDMN_db"]:
        d = df[[c, "meanfb_PDA"]].replace([np.inf, -np.inf], np.nan).dropna()
        r = np.corrcoef(d[c], d["meanfb_PDA"])[0, 1]
        print(f"  corr({c}, meanfb_PDA) = {r:+.2f}   (want |r|<0.8)")

    # ---- hypothesis: f-SNR vs DMN suppression / PDA elevation ----
    print("\n=== hypothesis: does f-SNR track regulation direction? (across runs) ===")
    for c in ["fsnr_PDA_db", "fsnr_CEN_db", "fsnr_CENDMN_db"]:
        d = df[[c, "delta_DMN", "delta_PDA"]].replace([np.inf, -np.inf], np.nan).dropna()
        rD = np.corrcoef(d[c], d["delta_DMN"])[0, 1]
        rP = np.corrcoef(d[c], d["delta_PDA"])[0, 1]
        print(f"  {c:16s} vs ΔDMN r={rD:+.2f} (expect <0)   vs ΔPDA r={rP:+.2f} (expect >0)")


if __name__ == "__main__":
    main()
