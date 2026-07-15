#!/usr/bin/env python3
"""
fz_theta_clean.py  —  PI's frontal-midline-theta question, redone honestly
--------------------------------------------------------------------------
Re-run of the Brandmeyer&Delorme/Scheeringa FMθ question on CLEAN (confound-regressed)
targets (n=17 DMNELF), feedback block + full run, with TWO angles:
  DIRECT      : Fz theta, and a frontal-midline cluster theta, vs CEN/DMN/PDA
  HEMISPHERIC : frontal theta ASYMMETRY (right - left) vs CEN/DMN/PDA
Scheeringa predicts FMθ INVERSELY tracks DMN. Targets are the clean personalized mask-means
(CEN=run{N}, DMN=run{N}_dmn, PDA=CEN-DMN). EEG theta = HRF-convolved band power (cache).
"""
from pathlib import Path
import numpy as np, glob, re
from scipy import stats

DATA = Path(__file__).resolve().parents[1] / "data"
CEN = Path(__file__).resolve().parents[1] / "results" / "cen_ceiling"
QA = re.compile(r"dmnelf999")
BASELINE_TR, HRF_DROP = 25, 5
LFRONT = ["Fp1", "F3", "F7", "FC5", "FC1"]
RFRONT = ["Fp2", "F4", "F8", "FC6", "FC2"]
FMID = ["Fz", "F3", "F4", "FC1", "FC2"]


def zs(x): return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def fz(r): return np.arctanh(np.clip(r, -0.999, 0.999))


def feats(rd, chs):
    th = rd["bp"]["theta"]
    def grp(names): return th[:, [chs.index(c) for c in names if c in chs]].mean(1)
    return {"Fz_theta": th[:, chs.index("Fz")] if "Fz" in chs else None,
            "frontal_midline_theta": grp(FMID),
            "theta_asym(R-L)": grp(RFRONT) - grp(LFRONT),
            "left_frontal_theta": grp(LFRONT), "right_frontal_theta": grp(RFRONT)}


def main():
    files = sorted(f for f in glob.glob(str(DATA / "dmnelf*_bandpower.npz")) if not QA.search(f))
    pat = re.compile(r"(dmnelf\w+)_bandpower")
    subs = [(f, CEN / ("cenmean_dmnelf_" + pat.search(f).group(1) + ".npz")) for f in files]
    subs = [(f, c) for f, c in subs if c.exists()]
    print(f"n={len(subs)} DMNELF subjects (clean targets)\n")
    FEATS = ["Fz_theta", "frontal_midline_theta", "theta_asym(R-L)", "left_frontal_theta", "right_frontal_theta"]
    for window in ["fb", "full"]:
        print(f"===== {window} window  (Scheeringa: FMθ should be NEGATIVE with DMN) =====")
        print(f"  {'feature':22s} " + "  ".join(f"{t:>8s}" for t in ["CEN", "DMN", "PDA"]))
        subj = {(ft, tg): [] for ft in FEATS for tg in ["CEN", "DMN", "PDA"]}
        for bp, cm in subs:
            z = np.load(bp, allow_pickle=True); chs = [str(c) for c in z["ch_names"]]
            m = np.load(cm, allow_pickle=True)
            per = {(ft, tg): [] for ft in FEATS for tg in ["CEN", "DMN", "PDA"]}   # per-run within subject
            for rd in z["runs_data"]:
                run = int(rd["run"]); n = rd["n_tr"]
                if f"run{run}" not in m.files or f"run{run}_dmn" not in m.files:
                    continue
                cen = np.asarray(m[f"run{run}"], float); dmn = np.asarray(m[f"run{run}_dmn"], float)
                tgts = {"CEN": cen, "DMN": dmn, "PDA": cen - dmn}
                sl = slice(BASELINE_TR + HRF_DROP, n) if window == "fb" else slice(0, n)
                ft = feats(rd, chs)
                for fn in FEATS:
                    x = ft[fn]
                    if x is None: continue
                    for tg, tv in tgts.items():
                        a, b = zs(x[sl]), zs(tv[:n][sl]); ok = np.isfinite(a) & np.isfinite(b)
                        if ok.sum() > 20:
                            per[(fn, tg)].append(np.corrcoef(a[ok], b[ok])[0, 1])
            for key, v in per.items():
                if v:
                    subj[key].append(np.tanh(np.nanmean(fz(np.array(v)))))   # subject-level r
        for fn in FEATS:
            out = []
            for tg in ["CEN", "DMN", "PDA"]:
                v = np.array(subj[(fn, tg)]); v = v[np.isfinite(v)]
                if len(v) < 3:
                    out.append("  nan  "); continue
                o = v.mean(); _, p = stats.ttest_1samp(v, 0)
                out.append(f"{o:+.3f}{'*' if p < 0.05 else ' '}")
            print(f"  {fn:22s} " + "  ".join(f"{c:>8s}" for c in out))
        print()


if __name__ == "__main__":
    main()
