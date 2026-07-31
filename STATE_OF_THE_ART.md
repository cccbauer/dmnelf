# State of the Art — DMNELF / rtBPD EEG-fMRI Neurofeedback

A map of every analysis sub-project in this repo: what it studies, its current status, and the key
number or conclusion if it has one. Written to answer "what's our best approach, and what's already
been tried and didn't work" without having to re-derive it from scratch.

**Cohorts:** DMNELF (schizophrenia + healthy controls, simultaneous EEG-fMRI, task `feedback`) and
rtBPD (adolescent borderline traits, sessions `nf1`/`nf2`). **Target:** PDA = CEN (central executive
network) activation − DMN (default mode network) activation, the real-time neurofeedback signal.

Status tags used below: **VALIDATED (POSITIVE)** · **NULL** · **MIXED / WEAK** · **IN PROGRESS** ·
**DEPLOYED** · **INFRASTRUCTURE** · **DORMANT / SUPERSEDED**.

---

## 1. The core method: EFP single-electrode decoding — VALIDATED (POSITIVE), DEPLOYED

**`efp_meirhasson/`** — Meir-Hasson et al. (2014) EEG FingerPrint (EFP) replication: single-electrode
Stockwell time-frequency transform → 10 data-driven equal-energy frequency bands → sliding time-delay
ridge regression, selected via leak-free nested cross-validation.

This is the best-validated decoder in the whole repo. Zero-shot, cross-subject LOSO (a frozen model
applied to a totally new subject, no personal calibration):

| target | best electrode (unrestricted) | LOSO r | best EPOC-X-available electrode | LOSO r |
|---|---|---|---|---|
| CEN | Cz | 0.121 (p<0.001) | **P8** | 0.118 (p=0.001) |
| DMN | O1 | 0.087 (p=0.019) | **O1** | 0.082 (p=0.034) |
| PDA | Pz | 0.172 (p=0.006) | O1 | 0.014 (n.s., p=0.778) |

Key finding: in this zero-shot regime, a single well-chosen electrode generalizes **better** than
pooling all 12 EPOC-X channels multivariately (12-channel montage: CEN=0.080, DMN=0.058, PDA=−0.035
n.s. — see `efp_meirhasson/scripts/electrode_vs_montage_loso.py`). Restricting further to only
frontal+parietal electrodes (excluding occipital) costs DMN accuracy (0.082→0.068) for no CEN gain,
since DMN's real signal is at an occipital site (O1), not a frontal/parietal one
(`frontoparietal_montage_loso.py`). PDA's own EPOC-restricted electrode is at chance — Pz was doing
the work and isn't on a consumer headset — so **PDA is derived downstream as CEN_pred − DMN_pred**,
not independently decoded (matches the manuscript's own recommendation).

Within-subject/personalized-calibration numbers (a different, easier regime — same subject gets
their own fit) are documented in the manuscript: CEN r≈0.08–0.10, DMN r≈0.06–0.09, PDA≈0.06;
zero-shot transfer CEN r≈0.07–0.10 improving to ≈0.12 with one calibration run; a 14-channel consumer
montage retains ≈92% of full-cap CEN accuracy.

**`mindwear/`** — the deployed live neurofeedback app (Flet operator console + PsychoPy stimulus),
running the EFP decoder in real time on an Emotiv EPOC-X. Ships **two selectable decoder variants**:
the original 12-channel montage model and the newer P8(CEN)/O1(DMN) dual-electrode model matching
the table above (PDA always derived as CEN−DMN). Recent work (this session): fixed a bug where the
feedback block silently aborted instead of erroring; fixed a "Next run" bug that redundantly
recalibrated on every button press; added a proper per-run mbNF flow (ratings after each run, then
an operator choice to recalibrate or continue); calibration reuse across separate manual restarts
for the same participant; an optional BIDS `ses-` label threaded through all saved filenames.

**`method_comparison/`** — VALIDATED (POSITIVE) for EFP. Head-to-head: single-electrode EFP vs. a
155-feature multivariate band-power model (31ch × 5 bands, ridge/elasticnet), matched nested-CV,
cross-cohort (train DMNELF → predict rtBPD). Within-subject performance is comparable, but on the
**cross-cohort double-replication scorecard EFP wins 4/5 vs. band-power's 2/5** — the multivariate
model transfers on raw/arousal-loaded targets but collapses on GSR'd (arousal-removed) and
differential (PDA) targets; the single-electrode EFP fingerprint holds. This is the project that
established EFP isn't just riding global arousal the way the multivariate alternative is.

---

## 2. Alternative EEG features / decoding approaches — mostly NULL or WEAK

A sequence of attempts to find a better (or complementary) EEG feature than EFP's single-electrode
Stockwell bands. None has beaten it; several concretely rule out approaches worth not repeating.

- **`eeg_bold_coupling/`** — MIXED. Step A: EEG-BOLD coupling exists but behaves like a **global
  arousal confound**, not a network-specific signal (univariate null). Step C: a **multivariate**
  Ridge model does decode GSR'd CEN (r=0.17, p=0.002), GSR'd DMN (r=0.12, p=0.006), and PDA (r=0.11,
  p=0.04) — a real, distributed positive signal, now also has LORO (leave-one-run-out) validation
  results (`results/multivariate_loro/`). The strongest positive multivariate finding outside EFP
  itself, but still weaker than EFP's cross-cohort robustness (see `method_comparison/`).
- **`wavelet_coupling/`** — MIXED / WEAK POSITIVE, completed. Compares wavelet/Hilbert/Morlet/PLV
  feature families; **`dwt_stats` (wavelet-derived statistics) is the strongest**, cross-subject
  GSR_CEN r=0.179 (p=0.0012), PDA r=0.155 (p=0.0058). PLV (phase) is consistently the weakest,
  especially cross-subject (near zero, n.s.) — reinforcing the phase-vs-amplitude lesson from
  `adaptive_sync_dmnelf`.
- **`alpha_lag_coupling/`** — NULL (honest). Adapts Jacob et al. (2025) noncanonical
  EEG-BOLD-coupling framework: lagged correlation between posterior alpha (or Stockwell bands) and
  DMN/CEN BOLD, ±10s. In-sample lag correlations replicate the paper's own numbers, but the
  cross-validated single-lag decoder is at chance (mean r≈0 both for residual-alpha and Stockwell
  variants). One exception worth flagging: `dmnelf008`'s DMN result is stable across CV directions
  (r=+0.33/+0.32) — an n=1 curiosity, not a pattern. Splitting subjects by resting coupling direction
  (canonical vs. noncanonical) and fitting separate decoders showed **no benefit** over pooling.
- **`adaptive_sync_dmnelf/`** (submodule) — DORMANT / SUPERSEDED. Kuramoto-oscillator +
  reinforcement-learning framework (Hall et al. 2025) modeling PLV as a PDA predictor. Own README:
  "Initial setup (April 2026)"; superseded by `spectral_power_pda`, whose README calls this repo
  archived and reports its concrete failure (PLV r<0.25).
- **`spectral_power_pda/`** (submodule) — IN PROGRESS / WEAK POSITIVE, stalled. Per-subject ridge on
  20 band-power features (5 bands × 4 metrics). Actual results
  (`training_results_summary.pkl`): only 5/12 subjects succeeded, mean r=0.140 (range 0.043–0.230) —
  well short of the README's stated "expected" r≈0.4 target (that number was a target, not an
  achieved result). No follow-through yet on the planned power+microstate combination.
- **`deep_eeg/`** — NULL (decisive). R-EEGNet-style CNN decoding CEN/DMN directly from raw EEG.
  Group LOSO CEN=−0.005, DMN=+0.002 (vs. EFP's ≈0.11/0.10); light augmentation and within-subject
  LORO are all near zero too; zero-shot transfer to rtBPD likewise near zero. Commit message calls
  it directly: "close R-EEGNet route as decisive NULL."
- **`dmn_hmm_detection/`** — MIXED / WEAK, not a clean replication. TIDE-HMM (Cooray et al. 2024,
  K=12 states) validated against *simultaneous* fMRI (something the original paper couldn't do). The
  best rest-identified "DMN state" correlates with fMRI DMN at only r=0.050 (n=24); applied
  out-of-sample to feedback runs, that correlation with DMN actually drops to −0.039, while
  incidentally tracking PDA (r=0.094) and GSR_CEN (r=0.115) better. Net: the state doesn't reliably
  track fMRI DMN out-of-sample.
- **`cyclic_transcoder/`** — struggling / in diagnosis. A previously-committed PDA evaluation was
  found buggy (scrambled/sign-flipped); the corrected evaluation gives r≈0 — the model does not
  decode PDA. Diagnosis points to a feature-design problem (per-TR mean-amplitude features lack band
  power, plus cross-subject/cross-task demand). This session added an infraslow-band variant
  (`config_infraslow.yaml`, `swap_eeg_infraslow.py`) and diagnostic tooling
  (`scripts/diagnose_subjects.py`, `scripts/reeval_pda_fixed.py`) — actively being iterated, not yet
  resolved.
- **`infraslow_pda/`** — companion analysis to the above: cohort-level and within-subject infraslow
  EEG-BOLD coupling/decoding. Result not yet reconciled/written up here; treat as in-progress.
- **`infraslow_loro_pda/`** — effectively empty. Only a compiled `.pyc` cache exists; the source
  script was never committed. Never started, or abandoned before any code landed.
- **`microstate_pda/`** (submodule) — NULL with a tiny significant effect. TESS/Custo
  microstate-projection decoder of PDA. Best strategy (personalized ridge, n=12, transfer CV):
  r=+0.030 (p=0.011) — statistically real but R²<1% of PDA variance; sign accuracy 51.3% (chance).
  `DECODER_SUMMARY.md`'s own words: "the signal is real but weak."
- **`sret_calibration/`** — IN PROGRESS, no written conclusion. Tests whether the self-referential
  encoding task (SRET) can serve as a faster EEG calibration localizer than one feedback run, in
  rtBPD. Pilot-scale results exist (r≈0.10–0.28 per subject/strategy) but no aggregate statistics or
  interpretation have been written up yet — pilot data collected, analysis incomplete.

---

## 3. fMRI-side validation and neurofeedback-effect checks — mostly NULL (by design, well-controlled)

These ask "does neurofeedback actually change resting brain organization," independent of EEG
decoding accuracy — and consistently say no, which is itself a useful, rigorously-obtained result.

- **`fox_seed/`** — validation POSITIVE + NULL. An MPFC seed cleanly recovers the canonical DMN
  posterior hub (PCC t=+4.29, precuneus t=+2.98, n=40) — the connectivity method works. But one
  mindfulness-NF session does **not** shift MPFC↔PCC coupling pre→post in either cohort (all
  n.s.); a single nominal rtBPD effect (nf1 MPFC–rDLPFC, p=.010) fails to replicate in nf2.
- **`restfc/`** — clean NULL, well-controlled. Resting-state connectivity (DiFuMo-64 within-DMN,
  within-CEN, hub edges) doesn't shift pre→post in rtBPD (flat, both sessions). DMNELF shows some
  apparent increases but they're confound-flagged (only 2 rest runs, no control condition). Notably
  **self-corrects an artifact**: an initially striking "decoherence" effect traced to a run-1
  global-signal/arousal confound (reverses sign when run 1 is dropped) — a good example of the
  project catching its own false positive before it became a claim.
- **`fsnr/`** — MIXED, precursor to `fsnr_eeg`. Pure-fMRI "functional SNR" framework (DMN=noise,
  CEN/PDA=signal). The construct is real (group up-regulates PDA, β=+0.18; DMN variance quenches
  rest→feedback, +2.2dB, p=1e-4) but the quenching is mostly **global**, not DMN-specific, and
  **f-SNR does not beat raw PDA** as an NF target (d′=0.70 vs 0.82, n.s.). Feeds its `glm_PDA_db`
  target forward into the EEG decoding phase (`fsnr_eeg/`).
- **`fsnr_eeg/`** — INFRASTRUCTURE / supporting caches for the above on the EEG side: per-subject
  aperiodic/periodic `specparam` fits and confound-regressed "CEN-ceiling" targets
  (`cenrel_*.csv`), consumed by several other projects (including `mindwear`'s clean-target training
  path). `results/pertr_fsnr/` caches are gitignored (regenerable on cluster).
- **`rtbpd_replication/`** — pre/post neurofeedback-effect replication battery in rtBPD: band power,
  PLV connectivity, EEG microstates (including a theta/fullspectrum k=4 variant), QC preprocessing
  figures, and neural-clinical (PHQ-9) correlations. The microstates arm is a documented **NULL**
  (no temporal parameter survives BH-FDR correction pre-vs-post, n=15, GEV=0.665) — status of the
  band-power/connectivity/clinical-correlation arms isn't fully reconciled in this document; check
  each subfolder's own results before citing a number from it.

---

## 4. The manuscript — IN PROGRESS, near-complete

**`manuscript/`** — full draft targeting *Biological Psychiatry: Cognitive Neuroscience and
Neuroimaging* (Archival Report). Title: *"A portable, personalized EEG decoder of a
default-mode–executive neurofeedback target, replicated across two clinical cohorts."*
`MANUSCRIPT.md` has Abstract/Intro/Methods/Results/Discussion fully drafted; remaining gaps are
explicitly factual-only ([CONFIRM] tags for symptom-scale names, medication, IRB/site details,
fMRIPrep version), not analysis gaps. `SUPPLEMENTARY.md` (MRI acquisition parameters, transcribed
from actual scanner protocols) and `FIGURES.md` (6-figure build plan + provenance, all 7 PNGs
already rendered) round it out. **The manuscript explicitly excludes `restfc/` and `fox_seed/`** as
internal negative/validation results, not paper-bound findings — consistent with §3 above.

---

## 5. Infrastructure (not research findings)

- **`mne_eeg_preprocessing/`** and **`fmri_preprocessing/`** — the two halves of the shared
  preprocessing pipeline (gradient/BCG artifact removal, ICA, band-pass/resample/re-reference for
  EEG; fMRIPrep-based preprocessing for fMRI), deployed to the Northeastern `explorer` HPC cluster.
  Still actively used to regenerate manuscript figures as of 2026-07-16.
- **`analysis/pineuro`** (submodule, third-party: `github.com/marlvan/pineuro`, Apache-2.0, currently
  v0.5.0) — "Python for Individualized Neurofeedback": DICOM watching, motion correction, ICA/GLM
  network-mask extraction, incremental GLM, connectivity feedback, task framework. Used here as a
  **dependency** for personalized network-mask extraction (`cyclic_transcoder/scripts/
  mask_extraction.py`, `rtbpd_replication/scripts/mask_extraction_rtbpd.py` import it directly;
  `microstate_pda` previously vendored an adapted copy and was migrated this session to depend on
  the real package instead — see its `analysis/fingerprint/pineuro/` legacy copy, now superseded).
- **`murfi-rt-PyProject/`** and **`rt-psychopy/`** — the actual scanner-session infrastructure: MURFI
  real-time fMRI control scripts, and the PsychoPy ball-task/experience-sampling stimulus code run
  during acquisition. Not analyzed further here; last touched 2025-10-31.

---

## 6. Data (gitignored, not otherwise documented)

`data/` (≈94GB, intentionally excluded from git — see top-level `.gitignore`) holds:

- **`data/DMNELF/derivatives/`**: `eeg_preprocessed/` (16 subjects, missing 002/003),
  `fmriprep_25.2.5_fmap/` (20 subjects), `cyclic_features/` (17 subjects) — derivative counts don't
  all match, i.e. not every subject has every derivative type finalized yet.
- **`data/rtBPD/derivatives/eeg_preprocessed/`**: 20 subjects (rtbpd002–rtbpd040, non-contiguous
  IDs), session structure `sub-rtbpdNNN/ses-nf/eeg/`.

---

*Compiled 2026-07-31. Numbers/status above reflect each project's own results files and commit
history as of that date — re-check the cited CSV/README directly before citing a number in a
paper or presentation, since several projects (cyclic_transcoder, sret_calibration,
rtbpd_replication's non-microstate arms) are explicitly flagged above as incomplete or unreconciled.*
