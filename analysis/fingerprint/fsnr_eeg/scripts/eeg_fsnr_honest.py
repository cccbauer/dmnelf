#!/usr/bin/env python3
"""
eeg_fsnr_honest.py  —  honest re-run of the band-power EEG f-SNR (manuscript §3.2/§3.3)
--------------------------------------------------------------------------------------
The published f-SNR match (eeg_fsnr_bandpower.py) has the TWO inflation modes we corrected
for the EFP:
  (1) TARGET = rd["targets"]["PDA"]  -> the motion-CONTAMINATED cached target (no confound
      regression); cleaning it roughly halved every EFP number.
  (2) CV = folds(n, k=5) over POOLED runs -> leaks within-run HRF-autocorrelated TRs.
This script fixes BOTH, to the same standard as efp_cen_clean.py:
  * clean confound-regressed targets from cenmean (CEN=run{N}, DMN=run{N}_dmn, PDA=CEN-DMN)
  * LORO (leave-one-run-out) folds
  * FEEDBACK block only (drop rest baseline + HRF), killing the rest->feedback state-step
Reports, per target (PDA/CEN/DMN) x montage (all/frontal/post):
  fitted-LORO ceiling  : raw band power   vs target   (nested single-site, LORO)
                         f-SNR running S/N vs target   (the construct's fitted ceiling)
  a-priori construct   : fixed frontal-theta / frontal-alpha / post-alpha f-SNR (NO fitting)
                         + the RAW-power counterpart, to test the "f-SNR > raw" claim honestly
Group sign-flip p. DMNELF n=17 (clean targets local). Runs locally.
"""
from pathlib import Path
import numpy as np, glob, re, sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "fsnr" / "scripts"))
from fsnr_proxy import running_fsnr

PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data"; RES = PROJ / "results"; CEN = RES / "cen_ceiling"
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
POST = ["P3", "P4", "P7", "P8", "O1", "O2", "Oz", "Pz", "POz", "PO3", "PO4"]
FRONTAL = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "FC1", "FC2", "FC5", "FC6"]
BASELINE_TR, HRF_DROP = 25, 5
RNG = np.random.default_rng(0)


def zs(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def clean_targets(sub):
    f = CEN / f"cenmean_dmnelf_{sub}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    out = {}
    for k in z.files:
        m = re.fullmatch(r"run(\d+)", k)
        if m and f"{k}_dmn" in z.files:
            r = int(m.group(1)); cen = np.asarray(z[k], float); dmn = np.asarray(z[f"{k}_dmn"], float)
            out[r] = {"CEN": cen, "DMN": dmn, "PDA": cen - dmn}
    return out


def fb_mask(n):
    m = np.zeros(n, bool); m[BASELINE_TR + HRF_DROP:] = True
    return m


def features(rd, chs, kind):
    """[n_tr, 31*5] raw band power or running f-SNR of it."""
    if kind == "raw":
        return np.column_stack([rd["bp"][b][:, ci] for b in BANDS for ci in range(len(chs))])
    return np.column_stack([running_fsnr(rd["bp"][b][:, ci])[1] for b in BANDS for ci in range(len(chs))])


def feat_names(chs):
    return [(b, chs[ci]) for b in BANDS for ci in range(len(chs))]


def loro_match(run_designs, cols):
    """Leave-one-run-out nested single-site selection over `cols`; OOF correlation."""
    rl = [(X[:, cols], y) for X, y in run_designs]
    if len(rl) < 2:
        return np.nan
    obs, pred = [], []
    for i in range(len(rl)):
        tr = [j for j in range(len(rl)) if j != i]
        Xt = np.vstack([rl[j][0] for j in tr]); yt = np.concatenate([rl[j][1] for j in tr])
        ok = np.isfinite(yt) & np.all(np.isfinite(Xt), 1)
        if ok.sum() < 30:
            continue
        c = np.array([abs(np.corrcoef(Xt[ok, j], yt[ok])[0, 1]) for j in range(Xt.shape[1])])
        j = int(np.nanargmax(c)); s = np.sign(np.corrcoef(Xt[ok, j], yt[ok])[0, 1])
        Xi, yi = rl[i][0][:, j], rl[i][1]; ok2 = np.isfinite(Xi) & np.isfinite(yi)
        if ok2.sum() > 10:
            pred.append(s * zs(Xi[ok2])); obs.append(zs(yi[ok2]))
    if not obs:
        return np.nan
    o, p = np.concatenate(obs), np.concatenate(pred)
    return float(np.corrcoef(o, p)[0, 1]) if np.std(p) > 1e-9 else np.nan


def apriori(runs, chs, tv, band, chset, kind):
    """Fixed construct (no fitting): raw or f-SNR of `band` over `chset`, per-run corr w/ target, avg.
    `tv` maps run-int -> clean target vector for one target."""
    pi = [i for i, c in enumerate(chs) if c in chset]
    if not pi:
        return np.nan
    rs = []
    for rd in runs:
        r = int(rd["run"]); n = rd["n_tr"]
        if r not in tv:
            continue
        y = zs(tv[r]); mask = fb_mask(n)
        if kind == "raw":
            x = zs(np.nanmean(np.column_stack([rd["bp"][band][:, i] for i in pi]), 1))
        else:
            x = zs(np.nanmean(np.column_stack([running_fsnr(rd["bp"][band][:, i])[1] for i in pi]), 1))
        m = mask[:len(y)] & np.isfinite(y[:len(x)]) & np.isfinite(x[:len(y)])
        if m.sum() > 20:
            rs.append(np.corrcoef(x[:len(y)][m], y[:len(x)][m])[0, 1])
    return np.nanmean(rs) if rs else np.nan


def sflip(x, n=10000):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) < 3:
        return np.nan, np.nan
    obs = x.mean(); null = (RNG.choice([-1, 1], (n, len(x))) * np.abs(x)).mean(1)
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1)


def main():
    files = sorted(glob.glob(str(DATA / "dmnelf*_bandpower.npz")))
    fitted = {(t, k, m): [] for t in ["PDA", "CEN", "DMN"] for k in ["raw", "fsnr"]
              for m in ["all", "frontal", "post"]}
    apri = {(t, k, cs): [] for t in ["PDA", "CEN", "DMN"] for k in ["raw", "fsnr"]
            for cs in ["frontal_theta", "frontal_alpha", "post_alpha"]}
    nsub = 0
    for f in files:
        sub = re.search(r"(dmnelf\w+)_bandpower", f).group(1)
        if re.search(r"dmnelf999", sub):
            continue
        tgt = clean_targets(sub)
        if not tgt:
            continue
        z = np.load(f, allow_pickle=True); chs = [str(c) for c in z["ch_names"]]
        runs = list(z["runs_data"]); names = feat_names(chs)
        fc = [j for j, (b, c) in enumerate(names) if c in FRONTAL]
        pc = [j for j, (b, c) in enumerate(names) if c in POST]
        allc = list(range(len(names)))
        for t in ["PDA", "CEN", "DMN"]:
            tv = {r: tgt[r][t] for r in tgt}          # run-int -> clean target vector for target t
            for kind in ["raw", "fsnr"]:
                rundes = []
                for rd in runs:
                    r = int(rd["run"])
                    if r not in tv:
                        continue
                    X = features(rd, chs, kind); y = np.asarray(tv[r], float)[:len(X)]
                    m = fb_mask(len(X))[:len(y)]
                    rundes.append((X[:len(y)][m], y[m]))
                if len(rundes) < 2:
                    continue
                for mname, cols in [("all", allc), ("frontal", fc), ("post", pc)]:
                    fitted[(t, kind, mname)].append(loro_match(rundes, cols))
                for cs, (band, chset) in [("frontal_theta", ("theta", FRONTAL)),
                                          ("frontal_alpha", ("alpha", FRONTAL)),
                                          ("post_alpha", ("alpha", POST))]:
                    apri[(t, kind, cs)].append(apriori(runs, chs, tv, band, chset, kind))
        nsub += 1

    print(f"Honest EEG band-power f-SNR (clean targets + LORO + feedback block) — n={nsub}\n")
    print("=== FITTED-LORO ceiling: nested single-site, leave-one-run-out ===")
    print(f"  {'target/montage':20s} {'raw':>16s}   {'f-SNR':>16s}")
    for t in ["PDA", "CEN", "DMN"]:
        for m in ["all", "frontal", "post"]:
            ro, rp = sflip(fitted[(t, "raw", m)]); fo, fp = sflip(fitted[(t, "fsnr", m)])
            print(f"  {t+'/'+m:20s} {ro:+.3f} (p{rp:.3f})   {fo:+.3f} (p{fp:.3f})")
    print("\n=== A-PRIORI construct (NO fitting): raw vs f-SNR ===")
    print(f"  {'target/construct':24s} {'raw':>16s}   {'f-SNR':>16s}")
    for t in ["PDA", "CEN", "DMN"]:
        for cs in ["frontal_theta", "frontal_alpha", "post_alpha"]:
            ro, rp = sflip(apri[(t, "raw", cs)]); fo, fp = sflip(apri[(t, "fsnr", cs)])
            print(f"  {t+'/'+cs:24s} {ro:+.3f} (p{rp:.3f})   {fo:+.3f} (p{fp:.3f})")


if __name__ == "__main__":
    main()
