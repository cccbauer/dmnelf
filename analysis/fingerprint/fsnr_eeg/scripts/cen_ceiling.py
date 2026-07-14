#!/usr/bin/env python3
"""
cen_ceiling.py  (local)  —  the realistic r ceiling for EEG->CEN, by timescale
-------------------------------------------------------------------------------
Aggregate the per-subject split-half reliability of the CEN-BOLD target (from
cen_ceiling_extract.py) into: reliability + ceiling (sqrt(reliability)) by temporal
smoothing (1/3/5 TR) x window (feedback / full-run), per cohort. Validates that the
extracted CEN mask-mean matches the existing target used for EEG decoding.

Verdict: the ceiling is the MAX r any EEG feature can reach with the CEN timecourse.
"""
from pathlib import Path
import numpy as np, pandas as pd, glob, re

RES = Path(__file__).resolve().parents[1] / "results" / "cen_ceiling"
DATA = Path(__file__).resolve().parents[1] / "data"
QA = re.compile(r"dmnelf999")   # 999 has no EEG; 1001-1003 are REAL subjects


def validate():
    """Correlate extracted CEN mask-mean vs the band-power cache target['CEN'] per run."""
    rs = []
    for npz in glob.glob(str(RES / "cenmean_dmnelf_*.npz")):
        sub = re.search(r"cenmean_dmnelf_(dmnelf\w+)\.npz", npz).group(1)
        bp = DATA / f"{sub}_bandpower.npz"
        if not bp.exists():
            continue
        mm = np.load(npz, allow_pickle=True)
        z = np.load(bp, allow_pickle=True)
        tgt = {int(rd["run"]): np.asarray(rd["targets"]["CEN"], float) for rd in z["runs_data"]}
        for k in mm.files:
            if "_gsr" in k or not k.startswith("run"):
                continue
            run = int(k.replace("run", ""))
            if run in tgt:
                a, b = mm[k], tgt[run]
                n = min(len(a), len(b))
                if n > 20 and np.std(a[:n]) > 0 and np.std(b[:n]) > 0:
                    rs.append(np.corrcoef(a[:n], b[:n])[0, 1])
    return np.array(rs)


def main():
    files = [f for f in glob.glob(str(RES / "cenrel_*.csv")) if not QA.search(f)]
    if not files:
        print("no cenrel CSVs yet"); return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"{d.subject.nunique()} subjects, {len(d)} rows\n")
    v = validate()
    if len(v):
        print(f"validation: extracted CEN mask-mean vs existing target — r={np.nanmean(v):.3f} "
              f"(median {np.nanmedian(v):.3f}, n={len(v)} runs)  [want ~1]\n")
    for coh in sorted(d.cohort.unique()):
        c = d[d.cohort == coh]
        print(f"===== {coh}  (n={c.subject.nunique()}) =====")
        print(f"  {'denoise':7s} {'window':6s} {'smooth':>7s} {'reliability':>12s} {'ceiling(=√rel)':>15s}")
        for den in ["raw", "gsr"]:
            for win in ["fb", "full"]:
                for w in sorted(c.smooth_tr.unique()):
                    s = c[(c.denoise == den) & (c.window == win) & (c.smooth_tr == w)]
                    if len(s):
                        print(f"  {den:7s} {win:6s} {int(w):>5d}TR {s.reliability.mean():>12.3f} "
                              f"{s.ceiling.mean():>15.3f}")
            print()
    print("raw = mask-mean incl. shared global; gsr = global-signal removed (CEN-specific ceiling).")
    print("ceiling = max achievable r(EEG, CEN) at that timescale. r=0.8 needs a ceiling >= 0.8.")


if __name__ == "__main__":
    main()
