#!/usr/bin/env python3
"""Build a summary PPTX for the EFP (Meir-Hasson 2014) replication on DMN/CEN/PDA."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

PROJ = Path(__file__).resolve().parent.parent
RES = PROJ / "results" / "full"
NAVY = RGBColor(0x1F, 0x3A, 0x5F); GREY = RGBColor(0x55, 0x55, 0x55)
GREEN = RGBColor(0x2E, 0x7D, 0x32); RED = RGBColor(0xB0, 0x3A, 0x2E)

summ = pd.read_csv(RES / "efp_group_summary.csv")
loso = pd.read_csv(RES / "efp_group_loso.csv")

# ── Fig A: EFP vs HRF vs T/A group bars (TR resolution) ──
targets = ["DMN", "CEN", "PDA", "GSR_DMN", "GSR_CEN", "GSR_PDA"]
methods = ["EFP", "HRF", "TA"]; colors = {"EFP": "#1f77b4", "HRF": "#7fb069", "TA": "#e08e45"}
res = "tr"
fig, ax = plt.subplots(figsize=(9, 4.3))
x = np.arange(len(targets)); w = 0.26
for j, mth in enumerate(methods):
    vals = [summ[(summ.target == t) & (summ.resolution == res) & (summ.method == mth)]["mean_r"].values[0]
            for t in targets]
    ax.bar(x + (j - 1) * w, vals, w, label=mth, color=colors[mth], edgecolor="k", linewidth=0.5)
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(targets, rotation=15)
ax.set_ylabel("group mean r"); ax.set_title("Within-subject EFP vs baselines (TR resolution, n=17)")
ax.legend(title="method"); fig.tight_layout()
figA = RES / "fig_efp_vs_baselines.png"; fig.savefig(figA, dpi=140); plt.close(fig)

# ── Fig B: benchmark comparison (EFP within, EFP LOSO, HMM, DWT) ──
bench_targets = ["GSR_CEN", "PDA", "GSR_DMN"]
efp_within = [summ[(summ.target == t) & (summ.resolution == "tr") & (summ.method == "EFP")]["mean_r"].values[0]
              for t in bench_targets]
efp_loso = []
for t in bench_targets:
    row = loso[(loso.target == t) & (loso.resolution == "tr")]
    efp_loso.append(row["loso_mean_r"].values[0] if len(row) else np.nan)
# prior benchmarks from earlier projects (within/LOSO multivariate & HMM)
dwt = {"GSR_CEN": 0.17, "PDA": 0.11, "GSR_DMN": 0.12}
hmm = {"GSR_CEN": 0.115, "PDA": 0.096, "GSR_DMN": np.nan}
fig, ax = plt.subplots(figsize=(9, 4.3))
x = np.arange(len(bench_targets)); w = 0.2
series = [("EFP within-subj", efp_within, "#1f77b4"),
          ("EFP LOSO", efp_loso, "#4c9be8"),
          ("DWT+stats Ridge", [dwt[t] for t in bench_targets], "#7fb069"),
          ("HMM feedback", [hmm[t] for t in bench_targets], "#e08e45")]
for j, (lab, vals, col) in enumerate(series):
    ax.bar(x + (j - 1.5) * w, vals, w, label=lab, color=col, edgecolor="k", linewidth=0.5)
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(bench_targets)
ax.set_ylabel("mean r"); ax.set_title("EFP vs prior DMNELF methods")
ax.legend(fontsize=9); fig.tight_layout()
figB = RES / "fig_benchmark.png"; fig.savefig(figB, dpi=140); plt.close(fig)

# ── Presentation ──
prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def title(s, text, sub=None):
    tb = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(1.0)); tf = tb.text_frame; tf.word_wrap = True
    r = tf.paragraphs[0].add_run(); r.text = text; r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = NAVY
    if sub:
        p = tf.add_paragraph(); rr = p.add_run(); rr.text = sub; rr.font.size = Pt(15); rr.font.color.rgb = GREY


def bullets(s, items, top=1.5, left=0.6, width=12.1, height=5.5, size=17):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height)); tf = tb.text_frame; tf.word_wrap = True
    for i, b in enumerate(items):
        lvl = 0; txt = b; col = None; bold = False
        if isinstance(b, tuple):
            txt, lvl = b[0], b[1]
            col = b[2] if len(b) > 2 else None; bold = b[3] if len(b) > 3 else False
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.level = lvl
        r = p.add_run(); r.text = ("• " if lvl == 0 else "– ") + txt
        r.font.size = Pt(size - 2 * lvl)
        if col: r.font.color.rgb = col
        r.font.bold = bold; p.space_after = Pt(6)


def img(s, path, left, top, width=None, height=None):
    if Path(path).exists():
        kw = {}
        if width: kw["width"] = Inches(width)
        if height: kw["height"] = Inches(height)
        s.shapes.add_picture(str(path), Inches(left), Inches(top), **kw)


# 1 title
s = prs.slides.add_slide(BLANK)
tb = s.shapes.add_textbox(Inches(0.8), Inches(2.5), Inches(11.7), Inches(2.5)); tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run(); r.text = "EEG Finger-Print (EFP) for DMN / CEN / PDA"
r.font.size = Pt(36); r.font.bold = True; r.font.color.rgb = NAVY
for line, sz in [("Replicating Meir-Hasson et al. 2014 on the DMNELF simultaneous EEG–fMRI cohort", 20),
                 ("Single electrode · Stockwell time-frequency · data-driven bands · sliding-delay ridge · n=17", 15)]:
    p = tf.add_paragraph(); rr = p.add_run(); rr.text = line; rr.font.size = Pt(sz); rr.font.color.rgb = GREY

# 2 method
s = prs.slides.add_slide(BLANK); title(s, "The EFP method (what makes it different)")
bullets(s, [
    "Single electrode (data-driven best, by validation NMSE) — not multivariate.",
    "Stockwell (S-)transform time-frequency at 1 Hz (implemented from scratch, FFT-based).",
    "10 data-driven equal-energy frequency bands (not fixed delta/theta/alpha).",
    ("KEY: a [frequency × sliding time-delay] design (0..−12 s) → ridge learns a SEPARATE "
     "delay for each frequency. No assumed HRF shape.", 0, NAVY, True),
    "Double cross-validation: outer m-k-fold block CV (k=5,m=2) + inner RidgeCV for λ.",
    "Baselines on same folds: Theta/Alpha ratio (traditional) and fixed-HRF predictor.",
    "Built at both native-TR and 4 Hz-upsampled resolution to test the paper's up-sampling claim.",
], size=16)

# 3 within-subject bars
s = prs.slides.add_slide(BLANK); title(s, "Within-subject: EFP beats both baselines (EFP ≥ HRF ≥ T/A)")
img(s, figA, left=1.5, top=1.5, width=10.3)
bullets(s, [("Replicates the paper's ordering across every target; strongest for CEN, PDA, "
             "GSR_CEN, GSR_PDA (r≈0.25–0.28). GSR_DMN weakest (r≈0.11–0.17), consistent with our other pipelines.", 0)],
        top=6.4, size=13)

# 3b per-subject scatter (individual variability)
s = prs.slides.add_slide(BLANK)
title(s, "Substantial individual variability",
      "Per-subject EFP r by target (bar = group mean, band = 95% CI). Every target's CI excludes zero, "
      "but spread is wide — motivates the subject-specific electrode selection.")
img(s, RES / "paper_fig_persubject_scatter_tr.png", left=1.4, top=1.7, width=10.5)

# 4 fingerprints
s = prs.slides.add_slide(BLANK); title(s, "Interpretable fingerprints — a learned HRF per frequency")
img(s, RES / "efp_group_fingerprint_PDA_tr.png", left=0.3, top=1.4, width=6.4)
img(s, RES / "efp_group_fingerprint_GSR_CEN_tr.png", left=6.7, top=1.4, width=6.4)
bullets(s, [("Group-averaged [band × delay] EFP. Weights peak at ~−4 to −7 s delay — an "
             "HRF-plausible lag learned data-drivenly, differing across frequency bands.", 0)],
        top=5.6, size=13)

# 5 benchmark
s = prs.slides.add_slide(BLANK); title(s, "EFP vs our prior methods — a clear improvement")
img(s, figB, left=1.5, top=1.5, width=10.3)
bullets(s, [
    ("Within-subject EFP roughly doubles PDA decoding (r=0.26 vs DWT 0.11, HMM 0.10) and "
     "improves GSR_CEN (0.26 vs 0.17).", 0, GREEN, True),
    ("Even cross-subject (LOSO) EFP transfers for PDA (r=0.13, p=0.01) — a general fingerprint.", 0),
], top=6.2, size=13)

# 6 LOSO + resolution
s = prs.slides.add_slide(BLANK); title(s, "Cross-subject generalization & TR vs 4 Hz")
loso_tr = loso[loso.resolution == "tr"].set_index("target")
bullets(s, [
    ("LOSO general fingerprint (train N−1, predict held-out), TR resolution:", 0, NAVY, True),
    (f"PDA: r={loso_tr.loc['PDA','loso_mean_r']:+.3f} (p={loso_tr.loc['PDA','sign_flip_p']:.3f}, ch {loso_tr.loc['PDA','common_ch']})", 1, GREEN),
    (f"CEN: r={loso_tr.loc['CEN','loso_mean_r']:+.3f} (p={loso_tr.loc['CEN','sign_flip_p']:.3f})", 1),
    (f"DMN: r={loso_tr.loc['DMN','loso_mean_r']:+.3f} (p={loso_tr.loc['DMN','sign_flip_p']:.3f})", 1),
    ("GSR'd targets transfer weakly across subjects (p>0.05) — GSR removes shared signal.", 1, RED),
    ("TR vs 4 Hz up-sampling: comparable within-subject accuracy — confirms the paper's finding "
     "that up-sampling adds temporal detail but not correlation.", 0),
], size=16)

# 7 paper-style fingerprints (Fig 5c/7c analog)
s = prs.slides.add_slide(BLANK)
title(s, "Paper-style figures I — EFP fingerprints", "Meir-Hasson Fig 5c/7c analog: [frequency × time-delay] per target")
img(s, RES / "paper_fig_fingerprints_tr.png", left=0.6, top=1.5, width=12.1)

# 8 paper-style predictor overlays + topomaps (Fig 3d, 3e/5b analogs)
s = prs.slides.add_slide(BLANK)
title(s, "Paper-style figures II — predictor overlay & electrode topography",
      "Fig 3d (predictor vs fMRI) and Fig 3e/5b (per-electrode CV correlation r; red=better)")
img(s, RES / "paper_fig_predictor_PDA_tr.png", left=0.3, top=1.5, width=8.1)
img(s, RES / "paper_fig_r_topomap_PDA_tr.png", left=8.7, top=1.4, width=3.2)
img(s, RES / "paper_fig_predictor_CEN_tr.png", left=0.3, top=4.3, width=8.1)
img(s, RES / "paper_fig_r_topomap_CEN_tr.png", left=8.7, top=4.2, width=3.2)

# Paper Fig 2 analog — post-processing schematic
s = prs.slides.add_slide(BLANK)
title(s, "Paper-style figures III — post-processing (Fig 2 analog)",
      "EEG → S-transform → 4 Hz → data-driven bands → band-averaged TF → sliding window; fMRI ROI → 4 Hz + normalized")
img(s, RES / "paper_fig2_schematic_PDA_tr.png", left=0.5, top=1.4, width=12.3)

# Paper Fig 3 analog — prediction input & output, one slide per target
for _t in ["PDA", "CEN", "GSR_CEN"]:
    s = prs.slides.add_slide(BLANK)
    title(s, f"Paper-style figures IV — prediction I/O, {_t} (Fig 3 analog)",
          "a) ROI signal + ROI mask  b) bottom/top-25% EEG TF  c) EFP  d) predictor vs fMRI  e) per-electrode r")
    img(s, RES / f"paper_fig3_composite_{_t}_tr.png", left=0.4, top=1.4, width=12.5)

# Positive control — visual cortex (Fig 5 analog): the blind EFP pipeline
# recovers the known posterior/occipital-alpha signature on a focal V1 ROI.
s = prs.slides.add_slide(BLANK)
title(s, "Positive control — visual cortex (Fig 5 analog)",
      "Focal 6mm calcarine sphere (MNI -1,-86,13 ~ V1). Blind EFP localizes posteriorly "
      "(group-mean peak Pz; LOSO best O2) with alpha modulation — recovers occipital topography, validating the method")
img(s, RES / "paper_fig3_composite_VIS_tr.png", left=0.3, top=1.5, width=9.4)
img(s, RES / "paper_fig_group_topomap_VIS_tr.png", left=9.9, top=1.9, width=3.2)

# 9 verdict
s = prs.slides.add_slide(BLANK); title(s, "Verdict")
bullets(s, [
    ("Faithful EFP replication succeeds on DMNELF.", 0, NAVY, True),
    ("The sliding-delay single-electrode fingerprint is our best DMN/CEN/PDA decoder to date, "
     "within-subject (PDA r≈0.26, CEN r≈0.28).", 1),
    ("A general (LOSO) PDA fingerprint transfers across subjects (r≈0.13, p=0.01) — promising for "
     "EEG-only neurofeedback.", 1),
    ("Interpretable [freq × delay] fingerprints show HRF-plausible per-frequency delays — good paper figures.", 1),
    ("Caveats", 0, NAVY, True),
    ("GSR_DMN remains hard (within and across subjects).", 1),
    ("LOSO uses band-index alignment across per-subject data-driven bands (approximation); a fixed-band "
     "group model is the natural next step.", 1),
], size=16)

out = PROJ / "efp_meirhasson_results.pptx"; prs.save(str(out))
print(f"Saved {out} ({len(prs.slides._sldIdLst)} slides)")
