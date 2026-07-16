# Figure plan — comprehensive, intuitive, pipeline-walking

*Narrative (author, 2026-07): start from the full preprocessing chain (BrainVision Analyzer gradient
correction → our MNE pipeline, every step), then the EFP method rendered very visually (Meir-Hasson
2014 style), then the DMNELF fingerprint with predicted↔observed timeseries, then transfer +
calibration. All intuitive. All numbers on clean targets / LORO / feedback block.*

Build order: data-only panels first (from committed CSVs), then signal panels needing a representative
subject's EEG pulled from the cluster. Master builder: `manuscript/scripts/paper_figures_honest.py`.
Legacy figures (`fsnr_eeg/fig_*.png`, `paper_fig_*`, `eeg_bold_coupling/figures/*`) are **stale/
inflated — do NOT reuse.**

---

## Figure 1 — Simultaneous EEG–fMRI acquisition & the EEG preprocessing chain
*Purpose: show, with real traces at each stage, how analyzable EEG is recovered inside the scanner.*
- **(A) Acquisition.** Schematic: 31-ch MR-cap + BrainAmp MR (5000 Hz, Cz ref) recording during
  multiband EPI (TR 1.2 s); scanner gradient + volume triggers.
- **(B) Gradient-artifact correction** (BrainVision Analyzer, average artifact subtraction). Real
  trace: raw gradient-contaminated EEG (mV-scale artifact) → template subtraction → gradient-corrected,
  downsampled 1 kHz. Before/after trace + PSD inset.
- **(C) Ballistocardiogram (BCG) correction** (MNE, ECG-R-locked AAS, −0.2–0.6 s). Before/after trace
  with ECG channel + R-peaks marked.
- **(D) ICA artifact rejection** (Picard 29-comp, ICLabel + ECG/EOG). Show 3–4 rejected component
  topographies (eye/muscle/cardiac) + cleaned trace.
- **(E) Final signal.** Band-pass 1–40 Hz, downsample 500 Hz, bad-channel interpolation, common-average
  reference → clean 31-ch. Final trace + PSD.
- Data: one representative DMNELF subject's EEG at each stage → **pull from cluster** (raw 5 kHz +
  BrainVision export + MNE intermediates via `../mne_eeg_preprocessing/deploy_scripts/eeg_preproc.py`).

## Figure 2 — The EFP decoding method (Meir-Hasson 2014 style, very visual)
*Purpose: make the single-electrode Stockwell band×delay ridge intuitive.*
- **(A)** One electrode's cleaned EEG → **Stockwell time-frequency** transform (spectrogram).
- **(B)** Collapse to the **10 frequency bands** × time (band-power timeseries), HRF-relevant.
- **(C)** **Sliding EEG→BOLD delays** (0–~10 s HRF lag): assemble the **[10 band × 11 delay] design**
  for a TR (the Meir-Hasson panel).
- **(D)** **Ridge** maps design → BOLD network activation; render the learned **[band × delay]
  weight matrix** = "the fingerprint."
- **(E)** Predicted vs observed network timeseries (teaser into Fig 3).
- Data: EFP feature cache + a fitted model (local `results/features_cache/dmnelf001_efp.npz`;
  fingerprint weights from the ridge). Spectrogram from one subject's EEG.

## Figure 3 — The DMNELF fingerprint: within-subject decoding of the DMN–executive networks
*Purpose: the discovery result, made concrete.*
- **(A)** Personalized DMN / CEN masks (brain render) and the target **PDA = CEN − DMN**.
- **(B)** **Predicted ↔ observed BOLD timeseries** for CEN / DMN / PDA, example subject, feedback block
  (the money panel — shows the decoder tracking the network in time).
- **(C)** Within-subject decoding **bars CEN/DMN/PDA** (DMNELF, Table 1) with per-subject dots.
- **(D)** **Clean scalp topography** of decodability — centro-parietal, < 20 Hz (neural, not EMG).
- Data: `results/cen_clean/`, `efp_topomap.png`, example predicted/observed trace (cluster).

## Figure 4 — Replication, transfer & calibration across cohorts
*Purpose: generalization + the deployable path.*
- **(A)** Within-subject **replication**: CEN/DMN/PDA bars × DMNELF + rtBPD nf1 + nf2 (full Table 1).
- **(B)** **Calibration ladder**: 0-shot transfer → +1-run calibration → within-subject (CEN, DMN;
  nf1 & nf2) — shows transfer is weak but calibration recovers it.
- **(C)** Per-subject transfer/calibration scatter (optional).
- Data: `results/cen_clean*/` (3 cohorts), `results/efp_calibrate_mv.csv`.

## Figure 5 — Portable deployment on a consumer headset (Emotiv EPOC X)
*Purpose: the portability punchline.*
- **(A)** EPOC-X 14-ch montage vs our 31-ch cap (head schematic; highlight the **missing
  centro-parietal midline** and the **kept posterior ring**).
- **(B)** Decoding **retention**: EPOC-12 vs full cap, CEN/DMN/PDA bars (~92% CEN).
- **(C)** Topography: where the signal lives vs what EPOC covers.
- Data: `efp_cen_group.py` mode `epoc`, `DEPLOY_EPOC.md`, `efp_topomap.png`.

## Figure 6 — Rigor (controls) & clinical anchor
*Purpose: why the honest ~0.10 is credible, and that the target matters clinically.*
- **(A)** **Confound-cleaning**: naive (`orig`) vs clean r for CEN/DMN/PDA — the ~2–3× motion inflation
  removed.
- **(B)** **Controls fail**: linear EFP vs deep (R-EEGNet) vs frontal-theta/FTA vs f-SNR — all ≈ 0
  except EFP.
- **(C)** **Clinical anchor**: PDA regulation ↔ state calm, r × 3 cohorts.
- **(D)** State calm **increases across sessions** (rtBPD nf1→nf2 paired).
- Data: `cen_clean` (orig vs clean), `fsnr_eeg/results/eeg_fsnr_honest.txt`, `deep_eeg/`, `fta_zotev.py`,
  `fsnr_eeg/results/sliders_both.csv`.

---

## Data-readiness
| Fig | Panels buildable now (committed CSVs) | Needs cluster EEG pull |
|---|---|---|
| 1 | acquisition schematic | B–E signal stages (raw→final) |
| 2 | design-matrix + fingerprint-weights schematic | (A) spectrogram from one subject |
| 3 | (C) bars, (D) topography | (B) predicted↔observed trace; (A) mask renders |
| 4 | (A) bars, (B) ladder | — |
| 5 | (B) bars, (A) montage schematic | — |
| 6 | all (A–D) | — |

→ Build Figs 4, 5, 6 and the bar/topography panels of 2–3 immediately; pull one representative
subject's EEG (all preprocessing stages + a network timeseries) for Figs 1–3 signal panels.
