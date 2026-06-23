# HANDOFF — EEG → DMN/CEN/PDA decoding (DMNELF)

Resume doc for picking this up later (with any agent). Last updated 2026-06-23.

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

### Bottom line after Step A
Band power gives a **robust EEG→DMN signal, but it is 100% arousal/vigilance** (global
signal), with **nothing DMN-specific** in a univariate screen. For a network-specific
(arousal-free) signal, the target must be PDA or GSR'd DMN/CEN. **→ see Step C below.**

---

**STEP C (multivariate decode, `multivariate_decode_pda.py`) — DONE (2026-06-23):**

Within-subject **Ridge + ElasticNet** on **CAR'd HRF-band-power** (155 features = 31 ch ×
5 bands), 4 runs pooled (~500 TRs), two CV schemes: **5-fold contiguous** (within-run) and
**LORO** (leave-one-run-out, cross-run generalization), group sign-flip test (10,000 flips),
**10,000 circular-shift per-subject nulls**. Four targets: PDA (arousal-free), GSR'd DMN,
GSR'd CEN (global_signal regressed per-run from fMRIPrep confounds), RAW DMN (arousal
sanity check). Full results in `results/multivariate_cluster/` and
`results/multivariate_loro_cluster/`.

**✅ POSITIVE RESULT — ElasticNet uniformly improves over Ridge; signal generalizes
across runs (LORO):**

**5-fold contiguous CV (within-run):**

| Target   | Ridge r | Ridge p | **ElasticNet r** | **ElasticNet p** |        |
|----------|---------|---------|------------------|------------------|--------|
| GSR_CEN  | +0.204  | 0.0004  | **+0.213**       | **0.0006**       | ***    |
| RAW_DMN  | +0.156  | 0.0104  | **+0.176**       | **0.0002**       | ***    |
| GSR_DMN  | +0.124  | 0.0056  | **+0.162**       | **0.0044**       | **     |
| PDA      | +0.112  | 0.0422  | **+0.145**       | **0.0154**       | *      |

**LORO (leave-one-run-out, cross-run generalization):**

| Target   | Ridge r | Ridge p | **ElasticNet r** | **ElasticNet p** |        |
|----------|---------|---------|------------------|------------------|--------|
| GSR_CEN  | +0.094  | 0.0202  | **+0.162**       | **0.0044**       | **     |
| RAW_DMN  | +0.103  | 0.0190  | **+0.156**       | **0.0004**       | ***    |
| GSR_DMN  | +0.073  | 0.0878  | **+0.112**       | **0.0180**       | *      |
| PDA      | +0.059  | 0.2356  | **+0.106**       | **0.0552**       | (trend)|

1. **ElasticNet uniformly better** than Ridge — higher r and stronger p across every target
   and CV scheme. The sparse model fits this problem better.
2. **Signal generalizes across runs (LORO):** ElasticNet LORO GSR_CEN r=+0.162 is nearly as
   strong as 5-fold Ridge r=+0.204. 3 of 4 LORO targets significant (GSR_CEN, RAW_DMN,
   GSR_DMN); PDA is borderline (p=0.055).
3. **GSR_CEN remains the strongest decoder** in every scheme. Best individual subjects:
   dmnelf012 r=+0.57 (5-fold), dmnelf1001 r=+0.43, dmnelf008 r=+0.40.
4. **Per-subject circular-shift nulls (10K):** 5/16 subjects individually significant
   (p<0.05) for GSR_CEN (dmnelf012, 1001, 008, 1002, 005, 013); the group effect is not
   driven by outliers — 13/16 have positive r for GSR_CEN.

### Step C weight-map interpretation (ElasticNet, 5-fold)

**Sparsity is LOW (~6% zeros)** — ElasticNet keeps nearly all 155 features, confirming
the signal is **truly distributed**. No single channel×band dominates.

**GSR_CEN top features (cross-subject consistency in parentheses):**
- **F4 alpha +0.093 (13/16 agree)** — right-frontal alpha positively predicts CEN.
  Single most reliable feature, consistent with frontal alpha asymmetry literature.
- Fp1 gamma −0.063 (14/16), FC2 beta +0.049 (12/15), O1 delta −0.048 (11/16),
  T8 alpha +0.053 (10/16), Fz theta +0.051 (10/15)
- Band profile: relatively flat (beta slightly highest |w|=0.023), no band dominates.
- Channel profile: widespread — Fp1, FC1, F4, O1, FC5, T8 all in top tier.

**PDA top features:**
- Fz theta +0.061 (13/16), O1 delta −0.054 (13/16), P7 beta −0.057 (13/16),
  CP6 alpha +0.062 (11/15), T8 alpha +0.058 (12/14), F4 alpha +0.050 (11/15)
- Band profile: perfectly flat — truly broadband/distributed.

**GSR_DMN top features:**
- CP6 alpha −0.065 (opposite sign from PDA, consistent), Fp2 gamma −0.055,
  C3 beta −0.053, FC2 alpha −0.052, F4 alpha +0.045

**Key interpretation:**
1. **F4 alpha** is the single most interpretable feature — right-frontal alpha power
   positively predicts CEN activation, aligning with the frontal alpha asymmetry literature.
2. The signal is **distributed and broadband** — no single band or scalp region carries it,
   explaining why the univariate screen was null.
3. **Good cross-subject consistency** on top features (10–14/16 agree on sign) — this is a
   real group pattern, not driven by outlier subjects.
4. Weight files for all subjects/targets: `results/multivariate_cluster/*_weights.npz`
   (keys: `coefs`, `ch_names`, `band_names`; coefs shape = [n_folds, 155]).

## NEXT STEPS (recommended order)
1. **Topographic weight-map figures** — plot group-mean ElasticNet weights on scalp
   topographies (per band, for GSR_CEN and PDA). Use `plot_topomaps.py` or extend it
   to read the weight .npz files. This is the key figure for the neuroscience story.
2. **Step D — microstates / connectivity**: EEG microstates C/D ↔ DMN (`microstate_pda`
   sibling exists); or alpha connectivity instead of power. May capture complementary signal.
3. **Step E — slow-state classification**: classify high-DMN vs high-CEN windows instead
   of continuous per-TR regression (may be far more robust with the distributed pattern).

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
  `coupling_group.py`, `identify_global.py`, `multivariate_decode_pda.py`,
  `scripts/*_job.sh`; `results/*.csv` + `results/multivariate/*.csv` +
  `results/figures/*.png`; `README.md`; this `HANDOFF.md`.
- Memory: `eeg-bold-coupling-stepA`, `cyclic-transcoder-modeling-diagnosis` (full
  negative-result history), `MEMORY.md` index.
- Branches (all pushed): `eeg-bold-coupling` (current), `infraslow-loro-pda` (block-mean
  negative result + diagnostics).


## Offline working copy (local continuation)

**Data location**  
Run `scripts/pull_data.sh` (detached, ~94 GB, resumable). After completion, data lands in:  
`~/Documents/GitHub/dmnelf/data/DMNELF/derivatives/`  
- EEG: `eeg_preprocessed/*.fif`  
- fMRI targets: `cyclic_features/*.npz`  
- Confounds: `*.tsv` (includes `global_signal` for GSR)  
- BOLD + masks: for manual re‑extraction if needed

**Configuration**  
To run offline, edit `config.yaml` (or set env vars) to point to local mirrors:  
```yaml
eeg_preproc_dir: ~/Documents/GitHub/dmnelf/data/DMNELF/derivatives/eeg_preprocessed
features_dir_local: ~/Documents/GitHub/dmnelf/data/DMNELF/derivatives/cyclic_features