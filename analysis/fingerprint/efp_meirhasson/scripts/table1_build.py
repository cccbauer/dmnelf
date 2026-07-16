#!/usr/bin/env python3
"""
table1_build.py  —  Table 1: within-subject EEG decoding (clean targets, LORO, feedback block)
----------------------------------------------------------------------------------------------
Aggregates efp_cen_clean_*.csv across the three cohorts/sessions into the manuscript's Table 1
(group mean r +/- SD, sign-flip p) for CEN / DMN / PDA x montage {all (multivariate), best}.
All rows use ttype='clean' (confound-regressed targets). Prints a markdown table.
"""
from pathlib import Path
import numpy as np, pandas as pd, glob

RES = Path(__file__).resolve().parent.parent / "results"
RNG = np.random.default_rng(0)
COHORTS = [("DMNELF (SZ)", "cen_clean"),
           ("rtBPD nf1", "cen_clean_rt_nf1_clean"),
           ("rtBPD nf2", "cen_clean_rt_nf2_clean")]
TARGETS = ["CEN", "DMN", "PDA"]
MODES = [("all", "multivariate"), ("best", "single-electrode")]


def load(sub_dir):
    fs = glob.glob(str(RES / sub_dir / "efp_cen_clean_*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def sflip(r, n=10000):
    r = np.asarray([v for v in r if np.isfinite(v)])
    if len(r) < 3:
        return np.nan, np.nan, np.nan, 0
    obs = r.mean(); null = (RNG.choice([-1, 1], (n, len(r))) * np.abs(r)).mean(1)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1)
    return obs, r.std(ddof=1), p, len(r)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def main():
    print("## Table 1 — within-subject EEG decoding of the DMN/CEN networks")
    print("*Clean (confound-regressed) targets, leave-one-run-out, feedback block. "
          "Group mean r (± SD); sign-flip p (\\* <.05, \\*\\* <.01, \\*\\*\\* <.001).*\n")
    for mode, mlabel in MODES:
        print(f"\n**{mlabel.capitalize()} decoder (`{mode}`)**\n")
        print("| Cohort (n) | CEN | DMN | PDA |")
        print("|---|---|---|---|")
        for cname, d in COHORTS:
            df = load(d)
            if df.empty:
                print(f"| {cname} | — | — | — |"); continue
            cells, n = [], 0
            for t in TARGETS:
                s = df[(df.target == t) & (df.ttype == "clean") & (df["mode"] == mode)]
                m, sd, p, n = sflip(s.r.values)
                cells.append(f"{m:+.3f}{stars(p)} (±{sd:.2f})" if np.isfinite(m) else "—")
            print(f"| {cname} (n={n}) | {cells[0]} | {cells[1]} | {cells[2]} |")
    print()


if __name__ == "__main__":
    main()
