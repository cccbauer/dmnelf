# Interim findings — honest EEG→network decoding (EFP), CEN/DMN/PDA

*Checkpoint of the "high-r EEG index of CEN" exploration (2026-07). All numbers are leave-one-run-out
(LORO) or cross-cohort, feedback block, clean confound-regressed targets unless stated.*

## The question that started it
Aspiration: an EEG feature at **r=0.8** with CEN-BOLD. Verdict: **not reachable.** r=0.8 for
EEG↔single-network BOLD would be unprecedented (field best ~0.3–0.5). We ruled out the usual levers:
- **Target reliability** — CEN mask-mean split-half ≈ 1.0 (raw AND global-removed) → target is not the limit.
- **Timescale / downsampling** — r flat ~0.08 from 1.2 s to 12 s → limit is EEG *information*, not fast noise.
- **Sampling rate (1 kHz vs 500/250)** — Nyquist: our features are ≤45 Hz → no gain.
- **Representation (EFP Stockwell 10-band sliding-delay)** — the real lever, but honest ceiling ~0.11–0.20.

## Two methodological corrections (both inflated prior numbers)
1. **Block-CV → LORO.** Frozen EFP (CEN 0.279, PDA 0.258, DMN 0.225) used contiguous-block CV that
   leaks autocorrelated within-run TRs. Honest **LORO** drops these to ~0.11–0.20. *All frozen EFP
   numbers must be re-scored with LORO before publication.*
2. **Un-cleaned → clean targets.** The `cyclic_features` personalized CEN/DMN targets had **no confound
   regression** (motion/physiology retained). Cleaning (motion-full + WM/CSF + cosine) is essential —
   it removed inflation from both the decoding and the transfer.

## Honest numbers (clean targets, LORO, feedback block)
| Regime | CEN | DMN | PDA | Note |
|---|---|---|---|---|
| **Personalized within-subject** (multivariate, all elec) | **0.11–0.13** | 0.08–0.10 | ~0.20 (orig) | robust; replicates DMNELF + rtBPD nf1 + nf2 |
| Single-best electrode (within) | ~0.05–0.07 | ~0.04 | — | weak; no stable electrode (diffuse field) |
| **Cross-cohort transfer** (DMNELF→rtBPD, 0-shot) | 0.07–0.10 | marginal | null | *orig transfer (0.12) was ~half motion* |
| **DMNELF prior + 1 rtBPD calibration run** | nf1 0.075 / nf2 0.117 | nf2 0.066 | — | beats 0-shot; approaches within-subject |

## Validity / mechanism
- **Neural, not muscle:** restricting to **<20 Hz** (δ/θ/α/low-β, dropping 20–40 Hz EMG bands) leaves
  transfer essentially unchanged (CEN nf2 0.089 vs 0.104; DMN nf1 improves 0.036→0.046*). Muscle can't
  live <20 Hz → the fingerprint is neural. Rebuts the EMG critique.
- **Distributed, not focal-frontal:** topomaps show CEN = diffuse **centro-parietal** field (Pz/POz,
  with a weaker medial-frontal Fz/F4 plateau), DMN = temporal/posterior, PDA = frontal but non-transferring.
  The "single best electrode" is selection noise on a plateau; **multivariate is the right decoder.**
  (Earlier "CEN=FC2 frontal" was a motion artifact of the orig target; on clean it moves posterior.)
- **PI frontal-midline-theta question:** on clean targets, Fz/frontal-midline theta is **weakly
  anti-correlated with DMN** (~−0.03, Scheeringa's predicted sign; the orig +0.29 was arousal), and
  sig-negative with CEN full-run (−0.058*), but effects are tiny. **Hemispheric theta asymmetry (R−L): null.**

## What did NOT help
- Downsampling / timescale; higher sampling rate.
- Multivariate/frontal electrode modes on ORIG targets (motion-inflated; collapse on clean).
- Joint CEN+DMN decoding — plain multi-output ridge = solo (null by construction); reduced-rank(1)
  helps DMN in DMNELF only, hurts in rtBPD → no robust gain.
- PDA transfer (differential is motion-robust within-cohort but does not cross cohorts on clean).

## Bottom line for the manuscript
A **neural (<20 Hz), personalized** EEG fingerprint decodes the DMN–executive networks at ~0.11
(CEN) / ~0.10 (DMN), **replicating across disorder and session** (DMNELF + rtBPD nf1 + nf2), and is
**calibratable from a single run** (~0.08–0.12). Not calibration-free-magic; the value is **rigor +
replication + deployability**, not magnitude. Retire the r=0.8 / calibration-free-transfer framing.

## Key files
- `scripts/efp_cen_clean.py` (LORO honest re-scoring, electrode modes) · `efp_cen_group.py`
- `scripts/efp_transfer.py` (cross-cohort transfer; `EFP_MAXBAND` band cutoff) · `efp_electrode_map.py` · `efp_topomap.py`
- `scripts/efp_calibrate.py` (multivariate calibration ladder + reduced-rank joint)
- `../fsnr_eeg/scripts/cen_ceiling_extract.py` (clean CEN+DMN targets, split-half reliability) · `fz_theta_clean.py`
- results: `cen_clean*/`, `efp_transfer*.csv`, `efp_calibrate_mv.csv`, `efp_joint_rrr.csv`, `efp_topomap.png`, `efp_electrode_map.csv`
