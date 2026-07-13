#!/usr/bin/env python3
"""
fz_theta.py  —  the simple question: does frontal-midline theta (Fz) fluctuate with DMN/PDA?
--------------------------------------------------------------------------------------------
Reproduces the paper's construct-matching method exactly (apriori_match: per-run z-score,
Pearson r, mean across runs) and toggles the correlation WINDOW to separate two things:
  full : whole run (rest + feedback) — includes the rest->feedback state step (as in the paper)
  fb   : feedback block only (TR 30..end) — pure within-"ball-task" moment-to-moment tracking

For each electrode set (Fz alone / frontal-midline cluster / the paper's FRONTAL headset set)
and feature (raw theta power / running theta f-SNR), report r vs PDA, DMN, CEN, per window.
Discovery cohort = schizophrenia (DMNELF). Sign: PDA = CEN - DMN.
"""
from pathlib import Path
import numpy as np, glob, re, sys
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eeg_fsnr_bandpower import running_fsnr, FRONTAL

CACHE = Path(__file__).resolve().parents[1] / "data"
BASELINE_TR, HRF_DROP = 25, 5
QA = re.compile(r"dmnelf(999|1\d\d\d)")
TARGETS = ["PDA", "DMN", "CEN"]
MIDLINE = ["Fz", "FCz", "Cz", "FC1", "FC2"]


def zs(x): return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)


def feat_ts(rd, chs, chset, kind):
    idx = [chs.index(c) for c in chset if c in chs]
    if not idx: return None
    theta = rd["bp"]["theta"][:, idx]
    if kind == "raw":
        return theta.mean(1)
    return np.nanmean(np.column_stack([running_fsnr(theta[:, k])[1] for k in range(theta.shape[1])]), 1)


def subject_r(npz, chset, kind, window):
    z = np.load(npz, allow_pickle=True); chs = [str(c) for c in z["ch_names"]]
    out = {t: [] for t in TARGETS}
    for rd in z["runs_data"]:
        ef = feat_ts(rd, chs, chset, kind)
        if ef is None: continue
        n = rd["n_tr"]; sl = slice(BASELINE_TR + HRF_DROP, n) if window == "fb" else slice(0, n)
        ef = zs(ef[sl])
        for t in TARGETS:
            y = zs(np.asarray(rd["targets"][t], float)[sl])
            m = np.isfinite(ef) & np.isfinite(y)
            if m.sum() > 20:
                out[t].append(np.corrcoef(ef[m], y[m])[0, 1])
    return {t: (np.nanmean(v) if v else np.nan) for t, v in out.items()}


def group(files, chset, kind, window, label):
    rows = {t: [] for t in TARGETS}
    for f in files:
        s = subject_r(f, chset, kind, window)
        for t in TARGETS: rows[t].append(s[t])
    parts = []
    for t in TARGETS:
        a = np.array(rows[t]); a = a[np.isfinite(a)]
        _, p = stats.ttest_1samp(a, 0)
        parts.append(f"{t} r={a.mean():+.3f}{'*' if p<0.05 else ' '}(p={p:.2f})")
    print(f"  {label:44s} | " + "  ".join(parts))


def main():
    files = sorted(f for f in glob.glob(str(CACHE / "*_bandpower.npz")) if not QA.search(f))
    print(f"DMNELF (schizophrenia) discovery cohort: {len(files)} subjects with EEG")
    print("method = paper's apriori match (per-run z-score, Pearson, mean over runs). PDA=CEN-DMN.\n")
    for window, wlab in [("full", "FULL run (rest+feedback; incl. state step) [= paper]"),
                         ("fb", "FEEDBACK block only (within ball-task tracking)")]:
        print(f"===== {wlab} =====")
        group(files, ["Fz"], "raw", window, "Fz theta  RAW power")
        group(files, ["Fz"], "fsnr", window, "Fz theta  running f-SNR")
        group(files, MIDLINE, "fsnr", window, f"frontal-MIDLINE {MIDLINE} f-SNR")
        group(files, FRONTAL, "fsnr", window, f"FRONTAL headset set ({len(FRONTAL)} el) f-SNR [paper marker]")
        print()


if __name__ == "__main__":
    main()
