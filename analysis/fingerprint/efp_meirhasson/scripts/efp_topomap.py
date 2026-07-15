#!/usr/bin/env python3
"""
efp_topomap.py  (local)  —  scalp topomaps of per-electrode EFP decodability
----------------------------------------------------------------------------
Reads efp_electrode_map.csv (per target x electrode: DMNELF within-cohort LOSO r, and
cross-cohort transfer r to rtBPD nf1/nf2; CLEAN targets). Renders a 3x3 grid of scalp
topomaps: rows = CEN/DMN/PDA, cols = DMNELF (LOSO) | ->rtBPD nf1 | ->rtBPD nf2.
Shows where each network's EEG fingerprint lives and whether it transfers.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import mne
mne.set_log_level("ERROR")

RES = Path(__file__).resolve().parent.parent / "results"
CSV = RES / "efp_electrode_map.csv"
COLS = [("dmnelf_loso_r", "DMNELF (LOSO)"), ("nf1_r", "→ rtBPD nf1"), ("nf2_r", "→ rtBPD nf2")]
TARGETS = ["CEN", "DMN", "PDA"]


def main():
    d = pd.read_csv(CSV)
    chs = d[d.target == "CEN"].sort_values("ch_idx")["ch_name"].tolist()
    info = mne.create_info(chs, 250.0, "eeg"); info.set_montage("standard_1020", match_case=False, on_missing="warn")
    vmax = np.nanpercentile(np.abs(d[[c for c, _ in COLS]].values), 98)
    fig, axes = plt.subplots(len(TARGETS), len(COLS), figsize=(9.5, 9.5))
    for i, tgt in enumerate(TARGETS):
        s = d[d.target == tgt].sort_values("ch_idx")
        for j, (metric, title) in enumerate(COLS):
            vals = s[metric].values
            im, _ = mne.viz.plot_topomap(vals, info, axes=axes[i, j], show=False,
                                         cmap="RdBu_r", vlim=(-vmax, vmax), contours=4, sensors=True)
            best = s.iloc[np.nanargmax(s[metric].values)]["ch_name"]
            axes[i, j].set_title(f"{tgt} {title}\nbest={best} r={np.nanmax(vals):+.2f}", fontsize=9)
    cbar = fig.colorbar(im, ax=axes, shrink=0.5, location="right"); cbar.set_label("decoding r")
    fig.suptitle("EFP per-electrode decodability (clean targets): where the fingerprint lives + transfers",
                 fontsize=11)
    out = RES / "efp_topomap.png"; fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
