# eeg_bold_coupling

Decode DMN / CEN (and PDA = CEN−DMN) from EEG **band-limited power (BLP)** — the
standard EEG-fMRI coupling feature — after the per-TR amplitude **block-mean**
approach (`infraslow_loro_pda` / `cyclic_transcoder` lineage) was shown to be a
confirmed negative result.

## Feature
Per channel × band (δ 1–4, θ 4–8, α 8–13, β 13–30, γ 30–40 Hz): Hilbert-envelope
power averaged over the 600 samples of each 1.2 s TR, log-transformed, then
**convolved with a canonical double-gamma HRF** (~5–6 s lag). → `[N_TR × 31 × 5]`.
Targets from the `cyclic_features` npz: DMN (parcel 64), CEN (parcel 65),
PDA = CEN−DMN. Parcels 0–63 are the DiFuMo-64 atlas (alpha positive control).

## Plan (A→E)
- **A. Coupling screen + positive control** (`coupling_screen.py`,
  `coupling_specificity.py`, `coupling_group.py`) — DONE, see verdict.
- B. Pipeline/alignment audit — effectively answered by A's positive control.
- C. Target framing (DMN/CEN separately + PDA) — folded into A.
- D. Microstates / connectivity features — TODO.
- E. Slow-state classification — TODO.

## STEP A VERDICT (2026-06-10)
1. **Pipeline is sound / positive control PASSES.** RAW HRF-band-power couples to
   BOLD robustly and consistently cohort-wide (DMN best T8/δ mean r=+0.30, t=7.5,
   p_FWER=2e-4; CEN +0.18 t=7.5; PDA Pz/α −0.14). EEG↔BOLD coupling exists →
   alignment / HRF / artifact correction are NOT broken.
2. **But it's a GLOBAL broadband confound.** The group RAW map is uniform across
   all bands & channels; the mean of all band-power channels correlates with DMN
   at +0.68 (008, survives detrending), with CEN only +0.02. PDA ≈ −DMN inherits it.
3. **No specific coupling survives the global control.** After common-average-
   reference on the power (CAR) or partialling the global+trend (PARTIAL), nothing
   survives sign-flip max-stat FWER correction for DMN/CEN/PDA (all p_FWER>0.23).

So band power, like the block-mean, yields **no network-specific univariate
decoder**; the only robust EEG→DMN signal is a single global (arousal/motion-like)
factor coupling to DMN.

## Open forks
- **Identify the global factor**: correlate it with fMRI motion (framewise
  displacement) and the global fMRI signal. Arousal/vigilance → potentially usable
  DMN decoder; motion/physiology → discard.
- **Multivariate decode**: univariate-null ≠ multivariate-null — a Ridge decoder on
  global-removed (CAR) BLP may extract a distributed specific pattern.

## Run
```
python scripts/coupling_screen.py      --config config.yaml          # per-subject screen + control
python scripts/coupling_specificity.py --config config.yaml --subjects dmnelf008 dmnelf007
sbatch  scripts/group_job.sh           # group inference (coupling_group.py) on sharing
```
Cluster env: `/home/cccbauer/.conda/envs/eeg_preproc/bin/python` (mne+sklearn+nilearn).
Stats: max-statistic circular-shift (within-subject) / sign-flip (group) nulls — the
honest controls for autocorrelated, multiply-tested coupling maps.
