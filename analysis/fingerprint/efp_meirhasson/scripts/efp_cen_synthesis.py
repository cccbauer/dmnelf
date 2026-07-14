#!/usr/bin/env python3
"""
efp_cen_synthesis.py  —  double-replication table for honest EFP re-scoring
---------------------------------------------------------------------------
Combines DMNELF (discovery) + rtBPD nf1 + rtBPD nf2 (double replication) EFP LORO
feedback-block results. Reports group r per target x electrode-mode per cohort, so we can
see whether the honest EFP CEN/DMN/PDA decoding replicates across cohort and session.
"""
from pathlib import Path
import numpy as np, pandas as pd, glob

BASE = Path(__file__).resolve().parent.parent / "results"
COHORTS = {"DMNELF": "cen_clean", "rtBPD-nf1": "cen_clean_rt_nf1", "rtBPD-nf2": "cen_clean_rt_nf2"}
RNG = np.random.default_rng(0)


def load(dirn):
    fs = glob.glob(str(BASE / dirn / "efp_cen_clean_*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None


def sflip(a, n=10000):
    a = a[np.isfinite(a)]
    if len(a) < 3:
        return np.nan, np.nan, len(a)
    obs = a.mean(); null = (RNG.choice([-1, 1], (n, len(a))) * np.abs(a)).mean(1)
    return obs, float((np.abs(null) >= abs(obs)).mean()), len(a)


def main():
    data = {c: load(d) for c, d in COHORTS.items()}
    data = {c: v for c, v in data.items() if v is not None}
    ns = {c: v.subject.nunique() for c, v in data.items()}
    print("Honest EFP re-scoring — DOUBLE REPLICATION (LORO, feedback block, orig targets)")
    print("n:", ns, "\n")
    for mode in ["best", "frontal", "all"]:
        print(f"===== mode = {mode} =====")
        print(f"  {'target':6s} " + "  ".join(f"{c:>14s}" for c in data))
        for tgt in ["CEN", "DMN", "PDA"]:
            cells = []
            for c, v in data.items():
                s = v[(v.target == tgt) & (v.ttype == "orig") & (v["mode"] == mode)]
                o, p, _ = sflip(s.r.values)
                cells.append(f"{o:+.3f}{'*' if p < 0.05 else ' '}(p{p:.2f})")
            print(f"  {tgt:6s} " + "  ".join(f"{x:>14s}" for x in cells))
        print()
    print("best = single-electrode (confound-robust honest number). PDA = CEN-DMN differential.")


if __name__ == "__main__":
    main()
