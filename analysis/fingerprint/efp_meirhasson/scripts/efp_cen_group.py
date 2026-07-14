#!/usr/bin/env python3
"""
efp_cen_group.py  (local)  —  aggregate honest EFP CEN re-scoring
-----------------------------------------------------------------
Reads efp_cen_clean_<sub>.csv (from efp_cen_clean.py) and reports group EFP CEN r per
condition (clean/orig x fb/full), sign-flip p, and the decomposition:
  state-step share  = clean/full - clean/fb
  confound share    = orig/fb   - clean/fb
Verdict vs band-power baseline (~0.08) and the frozen full-run number (~0.279).
"""
from pathlib import Path
import numpy as np, pandas as pd, glob

RES = Path(__file__).resolve().parent.parent / "results" / "cen_clean"
RNG = np.random.default_rng(0)


def sflip(a, n=10000):
    a = a[np.isfinite(a)]
    if len(a) < 3:
        return np.nan, np.nan, len(a)
    obs = a.mean(); null = (RNG.choice([-1, 1], (n, len(a))) * np.abs(a)).mean(1)
    return obs, float((np.abs(null) >= abs(obs)).mean()), len(a)


def main():
    files = glob.glob(str(RES / "efp_cen_clean_*.csv"))
    if not files:
        print("no efp_cen_clean CSVs yet"); return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"EFP CEN honest re-scoring — {d.subject.nunique()} subjects\n")
    print(f"  {'target':6s} {'window':5s} {'EFP CEN r':>10s} {'p':>7s} {'n':>4s}")
    cell = {}
    for target in ["clean", "orig"]:
        for window in ["fb", "full"]:
            s = d[(d.target == target) & (d.window == window)]
            o, p, n = sflip(s.efp_cen_r.values)
            cell[(target, window)] = o
            print(f"  {target:6s} {window:5s} {o:>+10.3f} {p:>7.3f} {n:>4d}")
    print()
    print(f"  frozen reference (orig/full, block-CV): ~0.279")
    print(f"  band-power+per-TR baseline (clean/fb):   ~0.08")
    print(f"  --- decomposition ---")
    print(f"  state-step share (clean/full - clean/fb): {cell[('clean','full')]-cell[('clean','fb')]:+.3f}")
    print(f"  confound share   (orig/fb  - clean/fb):   {cell[('orig','fb')]-cell[('clean','fb')]:+.3f}")
    print(f"\n  HONEST EFP CEN (clean/fb) = {cell[('clean','fb')]:+.3f}")
    print("  verdict: >>0.08 -> EFP representation is the CEN win; ~0.08 -> edge was state+motion.")


if __name__ == "__main__":
    main()
