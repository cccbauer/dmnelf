# Honest EFP spine — locked numbers for the manuscript reframe

*All numbers below are on **clean confound-regressed targets**, **leave-one-run-out (or frozen
cross-cohort)**, **feedback block only**. This is the honest standard established this arc. Verified
2026-07 after retiring the inflated f-SNR/frozen-EFP numbers. Sources cited per row.*

## Why the old manuscript numbers are retired
| Old claim | Source of inflation | Honest value |
|---|---|---|
| f-SNR "calibration-free" CEN/PDA r≈0.12, replicates 0.13/0.15 | contaminated `rd["targets"]` + pooled 5-fold CV | **NULL** (PDA/frontal-theta f-SNR −0.04; raw ≥ f-SNR) — `eeg_fsnr_honest.py` |
| single-electrode transfer CEN 0.138 / PDA 0.067 | `cross_cohort_efp.py` uses `orig` targets | multivariate clean transfer CEN 0.066–0.102 |
| frozen-EFP within CEN 0.279 / PDA 0.258 | block-CV leakage | LORO clean CEN 0.10 / PDA 0.06 |

## Within-subject personalized decoding (clean, LORO, feedback) — the core, REPLICATED
| Cohort | n | CEN | DMN | PDA | source |
|---|---|---|---|---|---|
| DMNELF (schizophrenia) | 17 | **0.10** | 0.06 | 0.06 | `efp_cen_group.py` (all-electrode) |
| rtBPD nf1 (borderline traits) | 19 | **0.095** | 0.090 | — | `efp_calibrate_mv.csv` `within_loro` |
| rtBPD nf2 (retest) | 11 | **0.084** | 0.090 | — | `efp_calibrate_mv.csv` `within_loro` |

→ Personalized EEG decoding of the DMN/CEN networks **replicates across 2 clinical cohorts + 3
sessions** at r ≈ 0.08–0.10 (CEN, DMN). **PDA — the differential NF target — is the weakest readout
(~0.06); CEN is the robust one.** (rtBPD clean PDA not yet extracted; DMNELF PDA 0.06 is the estimate.)

## 0-shot cross-cohort transfer (frozen DMNELF → rtBPD, clean, multivariate)
| | CEN | DMN | source |
|---|---|---|---|
| DMNELF → rtBPD nf1 | 0.066 | 0.045 | `efp_calibrate_mv.csv` `transfer` |
| DMNELF → rtBPD nf2 | 0.102 | 0.043 | `efp_calibrate_mv.csv` `transfer` |

→ Calibration-free transfer is **weak but mostly significant** (~0.07–0.10 CEN). Not "magic"; the
honest deployability lever is calibration ↓.

## +1 calibration run (fit on one rtBPD run, apply to the rest)
| | CEN | source |
|---|---|---|
| dmnelf + cal1 → rtBPD nf1 | 0.075 | `efp_calibrate_mv.csv` `dmnelf+cal1` |
| dmnelf + cal1 → rtBPD nf2 | **0.117** | `efp_calibrate_mv.csv` `dmnelf+cal1` |

→ One calibration run beats 0-shot and approaches within-subject.

## Portable-hardware feasibility (Emotiv EPOC X, clean)
| | CEN | DMN | PDA | source |
|---|---|---|---|---|
| EPOC-12 montage | **0.095** | 0.061 | 0.051 | `efp_cen_group.py` mode `epoc` |

→ ~92% of full-cap CEN retained despite no centro-parietal midline (see `DEPLOY_EPOC.md`).

## Negative controls (all NULL honestly — strengthen the paper)
| Approach | Result | source |
|---|---|---|
| f-SNR (band-power signal/noise) | null; raw ≥ f-SNR | `eeg_fsnr_honest.py` |
| Deep learning (R-EEGNet, raw EEG) | ~0 in LOSO + LORO + transfer | `deep_eeg/` |
| Frontal-theta / FTA (Zotev, Scheeringa) | ~0 within feedback | `fsnr_eeg/fz_theta_clean.py`, `fta_zotev.py` |

## One-line honest story
A **personalized, motion-controlled EEG decoder (EFP)** indexes the DMN–executive networks at
**r ≈ 0.08–0.10 (CEN-dominant)**, **replicated across two clinical cohorts and three sessions**,
**calibratable from one run**, and **deployable on consumer EEG (~0.095)** — while calibration-free
f-SNR, deep learning, and frontal-theta do **not** improve on it. The PDA differential target itself
is decodable only weakly (~0.06); CEN is the robust portable readout.
