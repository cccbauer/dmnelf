# HANDOFF — EEG → DMN/CEN/PDA decoding (DMNELF)

Resume doc for picking this up later (with any agent). Last updated 2026-06-11.

## The goal
Decode the fMRI network state — **DMN**, **CEN**, and/or **PDA = CEN − DMN** —
from **simultaneous EEG**, so EEG can drive real-time neurofeedback without fMRI.
16 subjects (dmnelf001/004/005/006/007/008/009/010/011/012/013/014/015/1001/1002/1003),
simultaneous EEG-fMRI, `feedback` task = 4 runs × 125 TRs (TR=1.2 s), 31 EEG channels.

## Targets (exact definitions — important)
From `derivatives/cyclic_features/sub-*/sub-*_task-feedback_run-*_features.npz`,
key `fmri_features` is `[125 × 66]` (z-scored):
- cols 0–63 = **DiFuMo-64 atlas** parcels (used as the alpha positive control; nilearn
  `fetch_atlas_difumo(64)` for labels — already cached on cluster).
- col **64 = DMN**, col **65 = CEN** — these are the *personalized* neurofeedback ROIs.
- **PDA = CEN − DMN** (col65 − col64), exactly. DMN/CEN weakly correlated (r≈0.20).

## What has been TRIED and RULED OUT (do not repeat)
1. **Cyclic transcoder** (neural net, EEG→fMRI→PDA): overfits, does not generalize.
   Archived negative result. See memory `cyclic-transcoder-modeling-diagnosis`.
2. **Per-TR amplitude block-mean** EEG feature → PDA: this averages oscillations to
   ~0 and keeps only infraslow drift. NULL in every framing — within-rest coupling,
   rest→feedback transfer, within-feedback LORO, per-run recalibration. The infraslow
   preprocessing fix (recovering the <0.1 Hz band a 1 Hz high-pass had deleted) was a
   real pipeline fix but did NOT make the feature work. See `infraslow_loro_pda/`
   (branch `infraslow-loro-pda`) and memory `cyclic-transcoder-modeling-diagnosis`.
   - Single-subject "wins" (007 rest→feedback, 008 LORO) were validation-scheme
     artifacts: 007's LORO r=−0.39 was an ElasticNet collapse (3/4 folds constant);
     diagnosed as cross-run non-stationarity. Cohort always null.
3. **EEG band power (BLP) + HRF** (this project, `eeg_bold_coupling/`, branch
   `eeg-bold-coupling`): the standard EEG-fMRI feature. See current state below.

## CURRENT STATE — `eeg_bold_coupling/` (branch `eeg-bold-coupling`, pushed)
Feature = per channel × band (δ1-4/θ4-8/α8-13/β13-30/γ30-40) Hilbert-envelope power,
averaged per TR, log, **HRF-convolved** (canonical double-gamma). `bandpower.py`.

**STEP A (coupling screen + positive control + group inference) — DONE:**
- ✅ **Pipeline is sound.** RAW HRF-band-power couples to BOLD robustly cohort-wide
  (group sign-flip max-stat null: DMN best T8/δ mean r=+0.30, t=7.5, p_fwer=2e-4).
  EEG↔BOLD coupling EXISTS → alignment/HRF/artifact-correction NOT broken. (Closes the
  "is the data broken?" question = Step B.)
- ⚠️ **The coupling is a GLOBAL broadband factor**, not a network-specific code: group
  RAW coupling map is UNIFORM across all 5 bands × 31 channels. After CAR (per-band
  spatial-mean removal) or PARTIAL (regress global+trend), **nothing** survives group
  FWER for DMN/CEN/PDA (all p_fwer > 0.23). So band power, like block-mean, gives **no
  network-specific univariate decoder**.

**STEP A follow-up — IDENTIFY the global factor (`identify_global.py`) — DONE:**
The global EEG→DMN factor is the **AROUSAL/VIGILANCE axis, NOT motion**:
- eeg→DMN = +0.30 cohort (up to +0.68 in 008), t=6.5, consistent across 16 subjects.
- It tracks the **fMRI whole-brain global signal +0.45** (the established EEG-fMRI
  vigilance marker) but ~0 with mean-cortical (−0.06).
- Band profile is perfectly **flat/broadband** (δ=θ=α=β=γ≈+0.30) = aperiodic/1-f offset.
- **Not motion**: survives partialling FD unchanged (+0.30); DMN_motion ≈ 0
  (fMRIPrep cleaned the ROI). Survives partialling the cortical-mean signal (+0.34).
- **RESOLVED (job 7590358): eeg→DMN is PURE global-signal/arousal — collapses to zero
  when partialling the whole-brain global signal.** `eeg_DMN_pGS = −0.042, p=0.52` (was
  +0.30). The chain is: arousal/vigilance → fMRI whole-brain global signal, which loads
  onto the DMN ROI at **+0.70** (`DMN_globalsig`, t=15.9) AND onto broadband EEG power
  at **+0.45** (`eeg_globalsig`). eeg→DMN (+0.30) is entirely that shared pathway. There
  is **NO DMN-specific EEG component**.

### ⚠️ CRITICAL DATA FINDING for the next agent
The **DMN/CEN ROI targets are NOT global-signal-regressed** — the DMN timeseries is
**~70% the whole-brain global signal**. So "decoding DMN" via band power = decoding the
global signal = decoding arousal. Implication:
- **PDA = CEN − DMN intrinsically cancels the shared global signal** (both load on it),
  so PDA is the arousal-free, network-specific contrast — the RIGHT target if you want
  something beyond vigilance. (But note PDA was already null after the global control in
  Step A; confirm with the multivariate decoder.)
- Consider explicitly **GSR-ing DMN/CEN** before any further decoding, or work in PDA.

### Bottom line for the next agent
Band power gives a **robust EEG→DMN signal, but it is 100% arousal/vigilance** (global
signal), with **nothing DMN-specific**. Usable only if decoding arousal is acceptable.
For a network-specific (arousal-free) signal, the target must be PDA or GSR'd DMN/CEN —
and band power has not yet shown a specific signal there (univariate). The multivariate
decoder + microstates + state-classification (below) remain untested on the arousal-free
target.

## NEXT STEPS (recommended order)
1. **Decide the target framing given the arousal finding** — work in **PDA** (arousal-
   free by construction) and/or **GSR'd DMN/CEN**. Decoding raw DMN/CEN just decodes
   arousal. This reframes everything below.
2. **Multivariate decode** (the real test; univariate-null ≠ multivariate-null): within-
   subject Ridge/ElasticNet on global-removed (CAR) BLP → predict **PDA / GSR'd DMN/CEN**,
   contiguous-fold CV + circular-shift null. Does a distributed specific pattern decode
   where single channel×band doesn't?  (Also report the RAW arousal decoder as the
   practical upper bound / sanity check.)
3. **Step D — microstates / connectivity**: EEG microstates C/D ↔ DMN (`microstate_pda`
   sibling exists); or alpha connectivity instead of power.
4. **Step E — slow-state classification**: classify high-DMN vs high-CEN windows instead
   of continuous per-TR regression (may be far more robust).

## INFRA / GOTCHAS
- **Cluster**: `ssh cccbauer@explorer.northeastern.edu`. Project on cluster:
  `/projects/swglab/data/DMNELF/analysis/fingerprint/eeg_bold_coupling/`.
- **Env** (mne+sklearn+nilearn+pandas): `/home/cccbauer/.conda/envs/eeg_preproc/bin/python`.
- **ALWAYS sbatch heavy jobs** (`--partition=sharing`, see `scripts/*_job.sh`). Band-power
  extraction for 16 subjects is killed on the login node with NO output (silent death).
- **EEG fif**: `derivatives/eeg_preprocessed/sub-*/ses-dmnelf/eeg/...desc-preproc500Hz_eeg.fif`
  (1–40 Hz, clean for bands). `desc-preproc500HzISp01` = 0.01–40 Hz infraslow variant.
- **Motion/confounds**: `derivatives/fmriprep_25.2.5_fmap/sub-*/ses-dmnelf/func/
  ...desc-confounds_timeseries.tsv` — 125 rows, aligns 1:1 with npz; cols
  `framewise_displacement`, `global_signal`, `dvars`. (First FD row is NaN.)
- **Code gotchas**: use `scipy.signal.hilbert` (NOT `mne.filter.hilbert`).
  `fetch_atlas_difumo(dimension=64, resolution_mm=2)` (no `legacy_format` kwarg in this
  nilearn). Mac shell cwd resets between tool calls — use absolute paths or `cd` each time.

## METHODOLOGICAL LESSONS (hard-won — keep using)
- **Always use blocked/contiguous CV** for these autocorrelated timeseries; shuffled
  KFold leaks (gave spurious within-run r=0.5–0.75; contiguous → ~0).
- **Always use max-statistic circular-shift (within-subject) / sign-flip (group) nulls**
  for multiply-tested coupling maps; parametric p and uncorrected thresholds are wildly
  anti-conservative here.
- **Smoothing both prediction AND true inflates r** (shared autocorrelation) — smooth the
  prediction only. The remembered "+75% from smoothing" was this artifact.
- A passing null ≠ neural signal: a real global/arousal co-fluctuation passes nulls but is
  a confound. Check spatial/spectral *specificity* (uniform map = global factor).

## KEY FILES
- `eeg_bold_coupling/`: `bandpower.py`, `coupling_screen.py`, `coupling_specificity.py`,
  `coupling_group.py`, `identify_global.py`, `scripts/*_job.sh`; `results/*.csv` +
  `results/figures/*.png`; `README.md`; this `HANDOFF.md`.
- Memory: `eeg-bold-coupling-stepA`, `cyclic-transcoder-modeling-diagnosis` (full
  negative-result history), `MEMORY.md` index.
- Branches (all pushed): `eeg-bold-coupling` (current), `infraslow-loro-pda` (block-mean
  negative result + diagnostics).
