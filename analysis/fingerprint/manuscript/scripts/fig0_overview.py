#!/usr/bin/env python3
"""
fig0_overview.py  —  Figure 0: study overview schematic
-------------------------------------------------------
Conceptual 4-stage flow: (1) the DMN-CEN neurofeedback target (PDA), (2) simultaneous
EEG-fMRI acquisition + paradigm (parameters from the Prisma protocols), (3) the frozen
EFP decoder fMRI->EEG, (4) portable deployment. Pure schematic (no data). Author may
replace with polished artwork; this is the draft.
Output: manuscript/figures/fig0_overview.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Ellipse

FIG = Path(__file__).resolve().parent.parent / "figures"
CEN_C, DMN_C = "#d7301f", "#2c7fb8"   # CEN red, DMN blue
INK = "#222222"
plt.rcParams.update({"font.size": 10})


def panel(ax, x, w, title, sub):
    ax.add_patch(FancyBboxPatch((x, 0.06), w, 0.82, boxstyle="round,pad=0.008,rounding_size=0.02",
                                lw=1.2, ec="#888", fc="#fafafa", transform=ax.transAxes))
    ax.text(x + w / 2, 0.925, title, ha="center", va="center", weight="bold", fontsize=11,
            transform=ax.transAxes)
    ax.text(x + w / 2, 0.10, sub, ha="center", va="bottom", fontsize=7.4, color="#555",
            transform=ax.transAxes, wrap=True)


def arrow(ax, x0, x1, y=0.48):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), transform=ax.transAxes,
                                 arrowstyle="-|>", mutation_scale=18, lw=2, color="#444"))


def main():
    fig, ax = plt.subplots(figsize=(15, 4.3)); ax.axis("off")
    W, G = 0.225, 0.033
    xs = [0.01 + i * (W + G) for i in range(4)]

    # ── 1. Target ──
    panel(ax, xs[0], W, "1 · The neurofeedback target",
          "Auditory hallucinations (schizophrenia) &\nborderline traits → DMN–CEN imbalance.\n"
          "rt-fMRI neurofeedback up-regulates PDA.")
    cx = xs[0] + W / 2
    ax.add_patch(Circle((cx - 0.035, 0.55), 0.055, fc=CEN_C, ec="none", alpha=0.75, transform=ax.transAxes))
    ax.add_patch(Circle((cx + 0.035, 0.55), 0.055, fc=DMN_C, ec="none", alpha=0.75, transform=ax.transAxes))
    ax.text(cx - 0.055, 0.55, "CEN", ha="center", va="center", color="w", weight="bold", fontsize=8, transform=ax.transAxes)
    ax.text(cx + 0.055, 0.55, "DMN", ha="center", va="center", color="w", weight="bold", fontsize=8, transform=ax.transAxes)
    ax.text(cx, 0.40, "PDA = CEN − DMN  ↑", ha="center", va="center", weight="bold", fontsize=9.5, transform=ax.transAxes)

    # ── 2. Acquisition ──
    panel(ax, xs[1], W, "2 · Simultaneous EEG–fMRI",
          "3T Prisma · 31-ch MR-EEG cap · TR 1.2 s · 2 mm · MB4.\n"
          "Feedback: 30 s rest → continuous PDA.\n"
          "DMNELF 4×125 vol · rtBPD 5×150 vol.")
    hx = xs[1] + W / 2
    ax.add_patch(Circle((hx, 0.60), 0.062, fc="#eee", ec=INK, lw=1.3, transform=ax.transAxes))     # head
    rng = np.random.default_rng(0)
    for _ in range(22):                                                                             # cap electrodes
        a = rng.uniform(0, 2 * np.pi); r = rng.uniform(0, 0.05)
        ax.add_patch(Circle((hx + r * np.cos(a), 0.60 + r * np.sin(a) * 0.9), 0.004, fc="#2c7fb8", ec="none", transform=ax.transAxes))
    # paradigm timeline
    ax.add_patch(Rectangle((hx - 0.075, 0.30), 0.03, 0.05, fc="#cccccc", ec=INK, lw=.8, transform=ax.transAxes))
    ax.add_patch(Rectangle((hx - 0.045, 0.30), 0.12, 0.05, fc="#9ecae1", ec=INK, lw=.8, transform=ax.transAxes))
    ax.text(hx - 0.06, 0.275, "rest", ha="center", va="top", fontsize=6.5, transform=ax.transAxes)
    ax.text(hx + 0.015, 0.275, "PDA feedback", ha="center", va="top", fontsize=6.5, transform=ax.transAxes)

    # ── 3. Decoder ──
    panel(ax, xs[2], W, "3 · Frozen EFP decoder",
          "Single-electrode Stockwell 10-band ×\nHRF-delay → ridge → BOLD network.\n"
          "Motion-controlled, LORO. CEN r ≈ 0.10.")
    dx = xs[2] + W / 2
    t = np.linspace(0, 1, 200)
    ax.plot(dx - 0.09 + 0.05 * t, 0.60 + 0.02 * np.sin(40 * t), color=INK, lw=0.7, transform=ax.transAxes)
    ax.add_patch(FancyBboxPatch((dx - 0.03, 0.545), 0.075, 0.11, boxstyle="round,pad=0.004",
                                fc="#e8eef5", ec=CEN_C, lw=1.2, transform=ax.transAxes))
    ax.text(dx + 0.0075, 0.60, "band ×\ndelay\nridge", ha="center", va="center", fontsize=6.8, transform=ax.transAxes)
    ax.plot(dx + 0.055 + 0.04 * t, 0.60 + 0.025 * np.sin(8 * t), color=CEN_C, lw=1.3, transform=ax.transAxes)
    ax.text(dx, 0.44, "EEG  →  predicted CEN/DMN → PDA", ha="center", va="center", fontsize=7.5, transform=ax.transAxes)

    # ── 4. Deployment ──
    panel(ax, xs[3], W, "4 · Portable deployment",
          "Frozen decoder → consumer EEG (EPOC-X)\noutside the scanner. ~92% of CEN retained.\n"
          "Personalized / 1-run calibrated.")
    px = xs[3] + W / 2
    ax.add_patch(Circle((px, 0.60), 0.05, fc="#eee", ec=INK, lw=1.3, transform=ax.transAxes))
    ax.add_patch(Ellipse((px, 0.635), 0.108, 0.055, angle=0, fill=False, ec="#41ab5d", lw=2.4, transform=ax.transAxes))  # headset band
    ax.add_patch(Circle((px - 0.052, 0.605), 0.008, fc="#41ab5d", ec="none", transform=ax.transAxes))
    ax.add_patch(Circle((px + 0.052, 0.605), 0.008, fc="#41ab5d", ec="none", transform=ax.transAxes))
    ax.text(px, 0.40, "scalable EEG neurofeedback", ha="center", va="center", fontsize=8, style="italic", transform=ax.transAxes)

    for i in range(3):
        arrow(ax, xs[i] + W + 0.002, xs[i + 1] - 0.002)
    fig.suptitle("A portable, personalized EEG decoder of a DMN–executive neurofeedback target",
                 fontsize=12.5, weight="bold", y=1.02)
    fig.savefig(FIG / "fig0_overview.png", bbox_inches="tight", dpi=150)
    plt.close(fig); print("wrote fig0")


if __name__ == "__main__":
    main()
