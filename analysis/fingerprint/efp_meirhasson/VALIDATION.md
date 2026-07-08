# EFP pipeline — validation & audit

Step-by-step adversarial audit of the EEG Finger-Print (EFP) pipeline, run before paper
write-up to bullet-proof every stage. Each section states what the step does, what was
checked, a **status** (✅ OK / ⚠️ risk / ❌ bug), the **evidence** (file:line), and the
**recommended fix**. Line numbers refer to the state at git tag `efp-preprint-v1`
(commit `178bcbe`); the current ("v1") results are frozen under `results/frozen_v1/`.

## TL;DR

- **The audit caught two real artifacts, both now fixed (v3):**
  1. 🔴 **Best-electrode selection scored on the same CV folds** inflated within-subject r
     (~+0.05–0.15). **Fix = nested CV** (select electrode on inner-train folds, score on the
     held-out outer fold; report concatenated out-of-fold Pearson r). Within-subject Table 1
     now reflects honest, de-biased values.
  2. 🔴 **Normalization asymmetry** — EFP designs were returned *raw* while the HRF and T/A
     baselines were per-run z-scored, which deflated EFP (110 heterogeneous band×delay
     features) both within-subject and cross-cohort. **Fix = per-run standardize in
     `assemble`**, so all three methods are on equal footing. EFP recovered substantially
     (e.g. CEN within 0.089→0.159; cross-cohort roughly doubled).
- **Outcome after both fixes: the EFP-centric story holds on honest numbers.** Within-subject
  EFP wins the task-relevant targets (PDA/CEN/GSR_CEN/GSR_PDA); the same-electrode Panel B has
  EFP > HRF 6/7; LOSO is significant 6/7; and the DMNELF→rtBPD **cross-cohort** transfer is
  significant for **7/7 targets in nf1 and 6/7 in nf2** — a clean double replication with no
  rtBPD data used in training or electrode selection.
- **Estimator note:** electrode ranking must use concatenated-OOF r (`oof_r`), not
  mean-of-per-fold-r (`cv_score`) — the latter is inflated on small test blocks and produced
  spurious "all-electrodes-significant" topographies. Fixed.
- Remaining medium issues (purged folds, fixed-band transfer, 4 Hz upsampling, motion
  regression) are deferred controls, not blockers; several checks came back clean.

Severity legend: 🔴 critical · 🟠 medium · 🟡 low/cosmetic · ✅ verified-OK.

---

## 1. fMRI target construction

**What it does.** DMN/CEN = mean BOLD in subject-specific CanICA masks (matched to Yeo-7),
stored as cols 64/65 of `fmri_features`; PDA = CEN − DMN; GSR_* = each residualized on the
global signal; VIS = mean BOLD in a 6 mm calcarine sphere.

- ✅ **GSR correct.** `residualize()` is OLS on `[1, global_signal]`; applied to DMN and CEN
  *before* GSR_PDA = GSR_CEN − GSR_DMN. `efp_features.py:45`, `:120`.
- ✅ **Target column indices** 64 (DMN) / 65 (CEN) match the `extract_features.py` layout
  (DiFuMo-64 then [DMN, CEN]). `efp_features.py` `load_targets_run`; `config.yaml` fmri idx.
- ✅ **Confounds consistent across cohorts.** DMNELF and rtBPD both use the same
  `cyclic_transcoder/data/extract_features.py` logic: DiFuMo via `NiftiMapsMasker(..,
  confounds=conf_mat)`; personal DMN/CEN masks via `NiftiMasker(standardize,detrend)` with
  **no** confounds. So the cross-cohort comparison is fair.
- 🟠 **Base targets have no motion regression.** The personal-mask DMN/CEN timeseries (hence
  DMN/CEN/PDA) get only standardize+detrend — motion/tissue confounds are *not* regressed
  (only DiFuMo columns get them; only GSR_* add the global signal). Motion can drive both
  EEG and BOLD → a shared confound. **Fix:** add a motion-regressed control for the base
  targets and report the delta (ties into the "global/arousal" interpretation).
- ✅ **VIS extracted comparably.** `NiftiSpheresMasker(standardize,detrend,t_r)`, no
  confounds — same footing as the personal masks. `extract_visual_sphere.py`.

## 2. EEG preprocessing

**What it does.** BVA-exported EDF → bad-channel/edge annotation → 1–40 Hz filter → BCG
(NeuroKit2 R-peaks) → downsample → ICA (Picard, 29 comp) with ICLabel + CTPS cardiac + EOG
proxy → interpolate → average reference → FIF.

- ✅ **ICA seeded** (`random_state=42`) → reproducible components.
- ✅ **Resampling consistent.** `load_eeg_run` resamples to 250 Hz (MNE anti-alias) whenever
  source ≠ 250; rtBPD 500 Hz and DMNELF `preproc500Hz` both land at 250. `efp_features.py`
  `load_eeg_run`.
- ✅ **Bad-EEG guarding.** Corrupt 50 Hz exports were detected and **excluded** (rtbpd027
  ses-nf1, rtbpd028 ses-nf1, rtbpd018 ses-nf2), pending re-export. Documented in the
  external-validation section.
- 🟡 **Session handling** is per-cohort (`session_eeg`/`session_fmri`); rtBPD new-subject EEG
  was written to a unified `ses-nf` (nf1) / `ses-nf2` (nf2). Verify run numbers are
  synchronized between EEG and fMRI at the protocol level (code pairs by run index).

## 3. EFP feature construction

**What it does.** Per electrode: S-transform (1 Hz, 1–40 Hz) → 10 equal-energy data-driven
bands → band power downsampled to the fMRI grid (TR and 4 Hz) → [band × sliding-delay]
design at decode time.

- ✅ **Stockwell transform correct** — integer-Hz power, Fourier-domain Gaussian window,
  mean removed. `stockwell.py`.
- ✅ **EEG↔fMRI downsample alignment.** `bin_average` maps 250 Hz band power to exactly
  `n_tr` (and `n_hz4`) contiguous bins with a ≥1-sample guard; matches the target length.
  `efp_features.py` `bin_average`.
- ✅ **Sliding-delay design aligned.** `make_delay_design` builds lags 0..−(n_delays−1) and
  returns offset `n_delays−1`; `assemble` trims `y[off:off+len]` so X rows and y align, and
  concatenates per-run. `efp_features.py:192`; `efp_decode.py assemble`.
- 🟠 **4 Hz target upsampling is cubic-spline** (`interp1d(kind="cubic")`,
  `efp_features.py:250`), which injects artificial autocorrelation into the target while the
  EEG side is bin-averaged — 4 Hz r may be inflated relative to TR. **Fix:** keep **TR as
  the primary** result; add a control (4 Hz-vs-TR comparison + a null-target upsampling
  check) before reporting any 4 Hz number, or drop 4 Hz from the headline.
- 🟡 **`band_hz` label bug (cosmetic).** The per-run `band_hz` is overwritten in the channel
  loop so only the **last channel's** band edges are stored (`efp_features.py:241,:254`) and
  reused for labels (`efp_decode.py:168`). Decoding is unaffected (per-channel power is
  extracted with each channel's own bands); only figure/table band-axis labels can be
  slightly wrong. **Fix:** store per-channel edges, or label as the across-channel median.

## 4. Cross-validated decoding  🔴→✅ (critical issue, FIXED in v3 — see §9)

**What it does.** Per electrode, double CV: outer contiguous m×k block folds + inner RidgeCV
for λ; metrics = out-of-fold r and NMSE. Best electrode = min CV NMSE; baselines HRF and T/A
on the same folds.

- ✅ **No standardization leakage.** In `cv_score`, `mu,sd = X[tr].mean/std` are fit on the
  **training fold only** and applied to test. `efp_decode.py:70-71`.
- 🔴 **CRITICAL — best-electrode selection bias.** `process_subject` scans ~31 electrodes,
  each scored by CV, and selects the min-NMSE electrode (`efp_decode.py:151-159`), then
  **reports that same electrode's CV r/NMSE** (`:173`). Choosing the winner of 31 on the
  very folds used to score it is selection-on-the-evaluation-metric → the reported
  within-subject r is optimistically biased (order ~+0.05–0.15, worse with more electrodes /
  fewer folds). There is no nested/held-out protection; a fresh fold set is even drawn per
  electrode (`:155`). **This inflates `efp_persubject_all.csv` / `efp_group_summary.csv` and,
  through them, the group sign-flip p-values.**
  **✅ FIXED (v3): nested CV** (`nested_cv_r` + `oof_r`) — the electrode is picked on
  inner-training folds and scored only on the held-out outer fold. Within-subject Table 1 now
  reports de-biased OOF r. A second artifact (EFP returned raw while baselines were z-scored)
  was also caught here and fixed by per-run standardizing in `assemble` — see §9.
- 🟠 **No gap between contiguous CV folds.** `mk_block_folds` (`:50`) makes adjacent
  train/test blocks with no buffer → temporal autocorrelation can leak across the boundary
  and mildly inflate CV. **Fix:** purge a few TRs (≥ HRF span) around each test block.
- 🟡 **z-score scope mismatch.** y is z-scored per-run (`assemble`) while X is z-scored
  per-fold (`cv_score`); when a fold spans two runs the scales differ slightly. Mitigated by
  contiguous folds. **Fix:** standardize consistently (e.g., per-run for both, or per-fold
  for both).
- 🟡 **`RidgeCV` has no `random_state`.** Its α search is deterministic so risk is low; set
  it for airtight reproducibility.

## 5. Group analyses

**What it does.** Aggregate per-subject r; sign-flip permutation test; group [band×delay]
fingerprints; LOSO transfer at the modal best electrode.

- ✅ **Sign-flip test is a valid test** of "mean r across subjects > 0/≠0," seeded
  (`efp_group.py:37`, `seed=42`). **Caveat:** it is computed on the electrode-selection-biased
  per-subject r's (§4), so within-subject group p-values are correspondingly optimistic —
  they will tighten/loosen after the nested-CV re-run.
- ✅ **Duplicate-CSV bug already fixed.** `aggregate` excludes its own `efp_persubject_all.csv`
  from the glob and `drop_duplicates` (`efp_group.py:51,:56`); the committed
  `efp_group_summary.csv` is clean (verified n=17 per row).
- ✅ **LOSO is leak-free** — the held-out subject is excluded from training; each subject gets
  its own z-score. `loso_transfer` (`efp_group.py:109`).
- 🟠 **LOSO band-index alignment.** Bands are per-subject data-driven, aligned by *index*
  across subjects — an approximation (subject A's band 3 ≠ subject B's band 3 in Hz). The
  modal electrode (`:132`) is also drawn from the biased per-subject selection. **Fix:** a
  **fixed-band group model** (shared Hz edges) for the transfer analyses — the robust
  alternative; expected to be more conservative but cleaner.

## 6. Positive control (visual cortex)

**What it does.** VIS = focal 6 mm calcarine sphere (MNI [−1,−86,13]); the blind EFP should
recover occipital electrodes + alpha.

- ✅ **Sound as a control.** Group-mean per-electrode map peaks posteriorly (Pz), LOSO
  electrode O2, alpha-band fingerprint. Documented that a *broad* visual-network ROI failed
  to localize (frontal/global) and the **focal** ROI is required — a key methods caveat to
  keep in the paper.
- Note: VIS inherits the same §4 within-subject selection caveat; its LOSO/cross-cohort
  numbers are the clean ones.

## 7. External validation (DMNELF → rtBPD, nf1 + nf2)  ✅ (cleanest results)

**What it does.** Train the DMNELF general fingerprint at the LOSO transfer electrode
(band-index aligned), predict each rtBPD subject; nf1 and nf2 as a double replication.

- ✅ **No leakage.** `StandardScaler` and RidgeCV are fit on **DMNELF only**
  (`cross_cohort_efp.py:113-114`); rtBPD is only ever transformed/predicted. The transfer
  electrode comes from the DMNELF LOSO CSV (`electrode_map`, `:45,:98`) — independent of
  rtBPD. This is the **cleanest** result in the paper.
- 🟠 **Band-index alignment across cohorts** (same caveat as §5) → superseded by the
  fixed-band group model when implemented.
- ✅ **nf1/nf2 isolation.** Separate `cyclic_features_nf2`, `visual_sphere_nf2`,
  `features_cache_rtbpd_nf2` dirs prevent the session-less feature filenames from colliding;
  `--tag _nf2` keeps the result CSVs separate.
- ✅ **Mask reuse across sessions** (nf1 masks applied to nf2) is subject-level/anatomical —
  acceptable; state the assumption in Methods.
- ✅ **Bad-EEG exclusions** applied per session (50 Hz corrupt files).

## 8. Reproducibility & provenance

- ✅ Seeds fixed for CanICA (`random_state=0`), EEG ICA (`42`), permutation tests (`42`/`0`).
- 🟡 `RidgeCV` unseeded (low risk, §4).
- 🟡 **Version pins are lower-bound only** (`environment*.yml`); DiFuMo/Yeo atlases are
  fetched+cached (intra-project stable). **Fix:** pin exact versions / archive the env for
  the paper.
- ✅ **Regenerable.** Committed result CSVs derive from committed scripts + cached
  per-subject features; no undocumented manual edits. Cluster stages are captured in the
  `submit_*.sh` scripts; `manuscript_stats.py` regenerates Tables 1–3 from the CSVs.
- ✅ **v1 preserved.** Tag `efp-preprint-v1` / branch `efp-results-v1` at `178bcbe`; headline
  CSVs + manuscript frozen under `results/frozen_v1/`.

---

## 9. Remediation roadmap

### Done (v3 — current)

1. ✅ **Nested CV for electrode selection** (🔴). `nested_cv_r` selects the electrode on
   inner-train folds and scores on the held-out outer fold; `oof_r` gives honest
   concatenated-OOF Pearson r. Removed the within-subject selection bias. HRF and T/A
   baselines are nested over the same candidate sets so the comparison is fair.
2. ✅ **Normalization asymmetry fixed** (🔴). `assemble` now per-run standardizes the EFP
   design (matching the HRF/T-A assemblers). Cross-cohort `build_design` relies on this.
   EFP recovered within-subject and roughly doubled cross-cohort.
3. ✅ **Approach-B transfer electrode** (🟠). LOSO uses the **group-mean-r peak over the
   training subjects only** (per fold), and cross-cohort uses the all-DMNELF group-peak
   (rtBPD untouched) — both leak-free; replaces the unstable per-subject modal argmax.
4. ✅ **Per-electrode BH-FDR topography** (🟠), "mark, don't mask" — the continuous r map is
   preserved and FDR-significant electrodes are dotted; `electrode_fdr_{target}_{res}.csv`
   written. Estimator corrected to `oof_r` (was `cv_score` → spurious all-significant).
5. ✅ **Same-electrode Panel B** (`same_electrode_panel.py`) — EFP/HRF/T-A at each network's
   group-peak electrode with matched OOF CV, isolating the design question. EFP > HRF 6/7.

### Deferred (controls, not blockers)

6. **Purged/gapped CV folds** (🟠) in `mk_block_folds`. *Effect:* small additional decrease
   in within-subject CV; negligible on transfer.
7. **Fixed-band group model** (🟠) for LOSO + cross-cohort (shared Hz band edges instead of
   per-subject index alignment). *Effect:* removes the alignment caveat.
8. **Control analyses** (🟠): (a) motion-regressed base targets; (b) 4 Hz vs TR + null-target
   upsampling check (4 Hz now demoted to **supplementary**; TR primary).
9. **Minor fixes** (🟡): per-channel `band_hz` storage; `RidgeCV(random_state=0)`; consistent
   z-scoring scope; exact version pins.

v1 stays frozen at tag `efp-preprint-v1` / `results/frozen_v1/`; v3 lives on branch
`efp-bulletproof-v2`.

## Files audited
`scripts/efp_decode.py`, `efp_features.py`, `efp_group.py`, `cross_cohort_efp.py`,
`stockwell.py`, `extract_visual_sphere.py`, `manuscript_stats.py`;
`../cyclic_transcoder/data/extract_features.py`; `../rtbpd_replication/scripts/*`.
