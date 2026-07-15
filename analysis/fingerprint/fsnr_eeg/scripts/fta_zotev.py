#!/usr/bin/env python3
"""
fta_zotev.py  —  Zotev 2025 (HBM) portable theta indices vs our clean CEN/DMN/PDA
---------------------------------------------------------------------------------
Zotev's EEG-nf target is the FRONTAL THETA ASYMMETRY  FTA = ln P(F3) - ln P(F4)  in
theta [4-7] Hz, chosen because a LEFT-minus-RIGHT difference CANCELS head-motion
artifacts (symmetric electrodes pick up equal-amplitude nod/roll artifact). He shows
(E-PPI) F3 theta -> left DLPFC (CEN), and FTA -> medial-frontal BA32 near the DMN vmPFC
node. Two more of his features we never tested in this exact form:
  FTA     = theta(F3) - theta(F4)                       [motion-robust, 2 electrodes]
  F3theta = theta(F3)                                    [Zotev's strongest single chan]
  CEN_PC  = 1st PCA comp of theta[F3,F4,Fz,P3,P4]        [network-theta composite, 5 chan]
(Our prior fz_theta_clean tested a broad 5-chan R-L average, NOT the F3-F4 pair.)
Our bandpower 'theta' is ALREADY ln-power (HRF-convolved). Targets = clean personalized
mask-means (CEN=run{N}, DMN=run{N}_dmn, PDA=CEN-DMN), feedback block + full run.
Reports subject-level r (within-subject, per-run Fisher-z averaged) for DMNELF + rtBPD
nf1 + nf2 -> tests whether a motion-robust 2-5 electrode index replicates across cohorts.
"""
from pathlib import Path
import numpy as np, glob, re
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CEN = ROOT / "results" / "cen_ceiling"
QA = re.compile(r"dmnelf999")
BASELINE_TR, HRF_DROP = 25, 5
PC_CH = ["F3", "F4", "Fz", "P3", "P4"]
FEATS = ["FTA(F3-F4)", "F3_theta", "CEN_PC"]
TGTS = ["CEN", "DMN", "PDA"]

COHORTS = [
    ("DMNELF",    sorted(f for f in glob.glob(str(DATA / "dmnelf*_bandpower.npz")) if not QA.search(f)),
     "cenmean_dmnelf_"),
    # rtBPD local clean caches carry CEN only (no _dmn key) -> CEN replication only.
    # rtBPD nf2 targets are a separate session, not extracted locally (needs cluster re-run).
    ("rtBPD nf1 (CEN only)", sorted(glob.glob(str(DATA / "rtbpd_nf1" / "rtbpd*_bandpower.npz"))), "cenmean_rtbpd_"),
]


def zs(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def fz(r):
    return np.arctanh(np.clip(r, -0.999, 0.999))


def feats(rd, chs):
    th = rd["bp"]["theta"]                      # [n_tr, n_ch], already ln-power
    idx = {c: chs.index(c) for c in PC_CH if c in chs}
    out = {}
    out["FTA(F3-F4)"] = th[:, idx["F3"]] - th[:, idx["F4"]] if {"F3", "F4"} <= idx.keys() else None
    out["F3_theta"] = th[:, idx["F3"]] if "F3" in idx else None
    if len(idx) == 5:
        M = np.column_stack([zs(th[:, idx[c]]) for c in PC_CH])   # per-run z, then PCA
        M = M - M.mean(0)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        pc = U[:, 0] * S[0]
        if Vt[0, 0] < 0:                         # fix sign: positive loading on F3
            pc = -pc
        out["CEN_PC"] = pc
    else:
        out["CEN_PC"] = None
    return out


def target_id(bp_path):
    return re.search(r"((?:dmnelf|rtbpd)\w+?)_bandpower", Path(bp_path).name).group(1)


def run_cohort(name, files, prefix):
    subs = [(f, CEN / f"{prefix}{target_id(f)}.npz") for f in files]
    subs = [(f, c) for f, c in subs if c.exists()]
    print(f"===== {name}  (n={len(subs)}) =====")
    for window in ["fb", "full"]:
        subj = {(ft, tg): [] for ft in FEATS for tg in TGTS}
        for bp, cm in subs:
            z = np.load(bp, allow_pickle=True); chs = [str(c) for c in z["ch_names"]]
            m = np.load(cm, allow_pickle=True)
            per = {(ft, tg): [] for ft in FEATS for tg in TGTS}
            for rd in z["runs_data"]:
                run = int(rd["run"]); n = rd["n_tr"]
                if f"run{run}" not in m.files:
                    continue
                cen = np.asarray(m[f"run{run}"], float)
                tv = {"CEN": cen}
                if f"run{run}_dmn" in m.files:                    # DMN/PDA only where DMN mask exists
                    dmn = np.asarray(m[f"run{run}_dmn"], float)
                    tv["DMN"] = dmn; tv["PDA"] = cen - dmn
                sl = slice(BASELINE_TR + HRF_DROP, n) if window == "fb" else slice(0, n)
                ft = feats(rd, chs)
                for fn in FEATS:
                    x = ft[fn]
                    if x is None:
                        continue
                    for tg in TGTS:
                        if tg not in tv:
                            continue
                        a, b = zs(x[sl]), zs(tv[tg][:n][sl]); ok = np.isfinite(a) & np.isfinite(b)
                        if ok.sum() > 20:
                            per[(fn, tg)].append(np.corrcoef(a[ok], b[ok])[0, 1])
            for key, v in per.items():
                if v:
                    subj[key].append(np.tanh(np.nanmean(fz(np.array(v)))))
        print(f"  --- {window} window ---")
        print(f"  {'feature':14s} " + "  ".join(f"{t:>9s}" for t in TGTS))
        for fn in FEATS:
            out = []
            for tg in TGTS:
                v = np.array(subj[(fn, tg)]); v = v[np.isfinite(v)]
                if len(v) < 3:
                    out.append("   nan   "); continue
                o = v.mean(); _, p = stats.ttest_1samp(v, 0)
                out.append(f"{o:+.3f}{'**' if p < 0.01 else '*' if p < 0.05 else ' '}")
            print(f"  {fn:14s} " + "  ".join(f"{c:>9s}" for c in out))
        print()


def main():
    print("Zotev 2025 portable theta indices vs clean CEN/DMN/PDA (within-subject r)\n")
    for name, files, prefix in COHORTS:
        run_cohort(name, files, prefix)


if __name__ == "__main__":
    main()
