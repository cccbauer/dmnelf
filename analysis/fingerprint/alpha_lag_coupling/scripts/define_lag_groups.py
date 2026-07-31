#!/usr/bin/env python3
"""
define_lag_groups.py  —  split subjects into coupling-direction groups from ACA
-----------------------------------------------------------------------------------
Reads compute_aca.py's results/aca_summary.csv and splits subjects by the sign of their DMN ACA
(the network most central to this project's PDA target): canonical (ACA >= 0 — BOLD lags alpha,
the classic direction) vs noncanonical (ACA < 0 — BOLD precedes alpha). A simple sign split, not
k-means/clustering — with only 16 subjects, anything fancier isn't well-motivated.

Feeds lag_group_decoder.py, which tests whether training a separate ridge per group (on
feedback-task data) actually decodes better for subjects in that group than the current
all-subjects-pooled model — this script only defines the groups, it doesn't evaluate them.

Output: results/lag_groups.csv — subject, group, aca_dmn, aca_cen.

Usage:  python define_lag_groups.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
ACA_SUMMARY = HERE / "results" / "aca_summary.csv"
OUT = HERE / "results" / "lag_groups.csv"


def main():
    df = pd.read_csv(ACA_SUMMARY)
    wide = df.pivot(index="subject", columns="region", values="aca").reset_index()
    wide = wide.rename(columns={"DMN": "aca_dmn", "CEN": "aca_cen"})
    wide["group"] = wide["aca_dmn"].apply(lambda v: "canonical" if v >= 0 else "noncanonical")
    wide = wide[["subject", "group", "aca_dmn", "aca_cen"]].sort_values(["group", "subject"])
    wide.to_csv(OUT, index=False)
    print(wide.to_string(index=False))
    print(f"\n{wide.group.value_counts().to_dict()}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
