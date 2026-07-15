#!/usr/bin/env python3
"""
efp_cen_group.py  (local)  —  aggregate honest EFP re-scoring (CEN/DMN/PDA x electrode modes)
---------------------------------------------------------------------------------------------
Reads efp_cen_clean_<sub>.csv. Reports group LORO feedback-block r for each target x mode
(best / frontal / all), with sign-flip p. Answers: (1) DMN/PDA honest numbers, (2) does
combining electrodes (frontal / all) beat single-best?
"""
from pathlib import Path
import numpy as np, pandas as pd, glob

RES = Path(__file__).resolve().parent.parent / "results" / "cen_clean"
RNG = np.random.default_rng(0)


def sflip(a, n=10000):
    a = a[np.isfinite(a)]
    if len(a) < 3:
        return np.nan, np.nan
    obs = a.mean(); null = (RNG.choice([-1, 1], (n, len(a))) * np.abs(a)).mean(1)
    return obs, float((np.abs(null) >= abs(obs)).mean())


def main():
    files = glob.glob(str(RES / "efp_cen_clean_*.csv"))
    if not files:
        print("no CSVs yet"); return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    n = d.subject.nunique()
    print(f"EFP honest re-scoring (LORO, feedback block) — {n} subjects\n")
    modes = ["best", "frontal", "all", "epoc", "epoc_afproxy"]
    print(f"  {'target':10s} " + "  ".join(f"{m:>14s}" for m in modes))
    for tgt, ttype in [("CEN", "orig"), ("CEN", "clean"), ("DMN", "orig"), ("PDA", "orig")]:
        cells = []
        for m in modes:
            s = d[(d.target == tgt) & (d.ttype == ttype) & (d["mode"] == m)]
            o, p = sflip(s.r.values)
            cells.append(f"{o:+.3f}{'*' if p < 0.05 else ' '}(p{p:.2f})")
        print(f"  {tgt+'/'+ttype:10s} " + "  ".join(f"{c:>14s}" for c in cells))
    print("\n  best=single-electrode(nested) | frontal=multivar 11 frontal | all=multivar 31")
    print("  epoc=EPOC-X 12ch subset | epoc_afproxy=+Fp1/Fp2 as AF3/AF4")
    print("  * p<.05 sign-flip. EPOC-X deployment: how much does epoc lose vs all (no midline)?")


if __name__ == "__main__":
    main()
