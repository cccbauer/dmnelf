# efp_pooled — pooled DMNELF+rtBPD EFP fingerprint

**Status:** Phases 1–3 all COMPLETE. Every model/montage/estimator/calibration arm tried is
NEGATIVE for CEN/DMN transfer beyond the shipped model. Read all result sections below — this
line of investigation is closed for now; see "Bottom line" at the end of PHASE 3.
**Branch:** `efp-pooled` in the `dmnelf` repo. **Last updated:** 2026-09-01.

Goal: rebuild the mindwear EEG→BOLD decoder on a pooled DMNELF+rtBPD cohort (28 train subjects vs
the current 19), fix four verified defects in the training script, and — for the first time —
produce an honest held-out score. Full plan: `~/.claude/plans/prancy-zooming-nautilus.md`.

Nothing under `efp_epoc/` or `efp_meirhasson/` is modified. Frozen v1 stays as the baseline.

---

## Why this exists — what the investigation found

Work had stopped at `mindlab` commit `5315f77`, a revert of a display-only EMA smoothing attempt on
Compare mode's dual-ball motion. Investigating that revert established:

1. **The revert was right; its commit's evidence was not.** The α sweep is a plateau from 0.15–0.80
   (57–61% direction agreement), not an optimum at 0.25. The claimed 53%→60% gain is ~1.4 SE on 103
   binary trials (SE ≈ 4.9 pp).
2. **No timing bug.** ±8 TR lag scan is flat within its CIs; `tr0 = n_delays−1` alignment is correct.
3. **The decoder is genuinely network-specific** (not disguised global-signal feedback). Double
   dissociation on dmnelf005 run 1: eeg_PDA→bold_PDA **+0.217** vs eeg_SUM→bold_PDA +0.072;
   eeg_SUM→bold_SUM +0.240 vs eeg_PDA→bold_SUM +0.018. Differencing costs only ~0.03
   (CEN +0.244, DMN +0.231 → PDA +0.217).
4. **The quoted r=+0.22 is IN-SAMPLE.** `dmnelf005` is in `efp19_subs.txt` with 380 training TRs,
   and is the best subject in every dmnelf analysis (within-subject EFP PDA 0.413).
   **`efp_epoc_model.npz` has never been scored out-of-sample anywhere.**
5. Its build log (`efp_epoc/export_n19_9817780.out`) reports in-sample only:
   `CEN: n=7164 feat=1320 alpha=100000 in-sample r=+0.168` / `DMN: alpha=31622.8 in-sample r=+0.246`.
   α pinned at the grid ceiling + in-sample r=0.168 on 1320 features = **underfitting**, the regime
   where more subjects actually helps.

**Expectation setting.** The dual-ball display needs r ≈ 0.588 for 70% direction agreement
(`agreement = 1 − arccos(r)/π`; +0.309 for even 60%). Validated LOSO ceiling is PDA +0.157 (n=19).
Pooling plausibly reaches +0.20–0.25. **A convincing dual-ball display is not reachable — not the goal.**

---

## Cohorts harmonize cleanly (verified, not assumed)

| | DMNELF | rtBPD |
|---|---|---|
| TR | 1.2 s | 1.2 s |
| EEG | 250 Hz, 31 ch + ECG | 500 Hz → resampled to 250 on load, same 31 ch, same order |
| Session | `ses-dmnelf` | `ses-nf1` (and `ses-nf2` for 11 subjects) |
| Runs × TRs | 4 × 125 | 5 × 150 |

Gotchas that cost time — do not rediscover them:
- rtBPD EEG sessions are **`ses-nf1` / `ses-nf2`**, and the universal file format is
  **`desc-preproc500Hz`**. A `desc-preproc250Hz` copy exists for only *some* subjects (e.g. 004,
  022 have it; 002 does not), so globbing 250Hz makes 9 subjects look like they have no feedback
  EEG. Always use the 500Hz files and resample to 250 -- which is what
  `efp_meirhasson/config_rtbpd.yaml` does (`desc: preproc500Hz`, `sfreq: 250.0`). Note some
  subjects also have a `ses-nf1-archive` directory; ignore it.
- DMNELF EEG is natively `desc-preproc250Hz` except `dmnelf016`, which is 500Hz only.
- **Two DMNELF feature caches exist and they differ.** Use the `19_fingerprint` one (19 subjects).
  The v1 `efp_meirhasson/results/features_cache` has only 17 — no dmnelf002/003.

## Feature caches — already built, no re-extraction needed

```
DMNELF (19)    /projects/swglab/data/DMNELF/analysis/fingerprint/19_fingerprint/efp_meirhasson/results/features_cache
rtBPD nf1 (19) /projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/results/features_cache_rtbpd
rtBPD nf2 (11) /projects/swglab/data/DMNELF/analysis/fingerprint/efp_meirhasson/results/features_cache_rtbpd_nf2
targets        /projects/swglab/data/DMNELF/analysis/fingerprint/efp_epoc/cen_mean_cache
```
Schemas verified identical: `runs` (object array) + `ch_names` (31); each run dict has
`run, n_tr, n_hz4, bp_tr (31,10,nTR), bp_hz4, ta_tr, ta_hz4, tgt_tr, tgt_hz4, band_hz`.

## Cohort split — LOCKED, see `cohort_split.json`. Never edit.

- **Train (28):** 19 DMNELF + 9 rtBPD nf1 (`002 003 009 010 011 015 022 026 030`)
- **Locked external test (12):** rtBPD nf1-only (`004 012 013 018 020 021 024 034 038 040`)
  + nf2-only (`027 028`). **Scored exactly once per arm.**
- **Session-transfer test (9):** nf2 sessions of the 9 training rtBPD subjects.

---

## PHASE 1 RESULT — the deployed decoder's PDA does not transfer

Scored the shipped `efp_epoc_model.npz` (DMNELF-trained, n=19) with `eval_holdout.py`.
Every rtBPD subject is genuinely held out for it. Subject-mean r ± SEM, 95% CI, sign-flip p:

| cohort | status | n | CEN | DMN | **PDA** |
|---|---|---|---|---|---|
| dmnelf | in-sample | 19 subj / 75 runs | +0.171 ± 0.020 | +0.249 ± 0.013 | +0.194 ± 0.030 |
| **rtbpd nf1** | **HELD OUT** | 19 subj / 93 runs | +0.069 ± 0.017 *(p=.0004)* | +0.052 ± 0.015 *(p=.003)* | **+0.010 ± 0.012, CI [−0.016,+0.036], p=0.43** |
| **rtbpd nf2** | **HELD OUT** | 11 subj / 51 runs | +0.093 ± 0.030 *(p=.006)* | +0.036 ± 0.020 *(p=.11)* | **+0.008 ± 0.021, CI [−0.040,+0.055], p=0.71** |

**Harness is validated:** the in-sample CEN/DMN (+0.171/+0.249) reproduce the deployed model's own
build log (`export_n19_9817780.out`: +0.168/+0.246) to within 0.003.

**The finding: PDA — the signal mindwear actually feeds back — has no out-of-sample validity.**
r = +0.010 (p=0.43) on 19 held-out subjects, replicated at +0.008 (p=0.71) on an independent second
session. Per-subject PDA on held-out rtBPD is 9 positive / 10 negative — a coin flip.

CEN and DMN individually *do* transfer, weakly but reliably (+0.069, +0.052, both p<0.005). Their
**difference** does not. This is consistent with `eeg_bold_coupling/HANDOFF.md`: the transferable
component is largely shared/global (arousal), and differencing cancels precisely the part that
transfers, leaving nothing network-specific behind.

In-sample → held-out degradation: CEN −60%, DMN −79%, **PDA −95% (to zero)**.

### What this changes

- Fitting **PDA directly** (Phase 2 defect #1) is promoted from "nice improvement" to **the central
  fix**. The current architecture — difference two independently-regularized ridges — is what
  destroys transfer. Offline, directly-fit PDA is the *best*-transferring target (LOSO n=19: PDA
  +0.157 vs CEN +0.114, DMN +0.107), so the signal exists; this build throws it away.
- **Pooling gains urgency**: a DMNELF-only model has never seen rtBPD-like variance, and the
  cross-cohort collapse is the evidence.
- Prior context: `frozen_v1` cross-cohort DMNELF→rtBPD PDA was +0.037 (n.s.). This result (+0.010)
  is consistent with it and now measured directly on the *deployed* model for the first time.
- **Do not quote r=+0.22 as the decoder's performance.** It is in-sample, on a training subject who
  is the cohort's best performer. The honest deployed number for PDA is ~0.

## PHASE 2 FIRST PASS — NEGATIVE. Pooling + directly-fit PDA did not rescue PDA transfer.

Trained two arms with all four defects fixed (PDA fitted as its own ridge, alpha grid widened to
1e8, subject-grouped `GroupKFold` alpha selection, true across-channel median band edges):

- **arm A `pooled28`** — 28 subjects (19 DMNELF + 9 rtBPD), 12,444 TRs
- **arm B `dmnelfonly19`** — same fixes, 19 DMNELF only, 7,164 TRs (isolates the effect of pooling)

### Training-side grouped CV (subject-grouped, within training cohort)

| arm | CEN | DMN | PDA | notes |
|---|---|---|---|---|
| pooled28 | **+0.075** | +0.037 | +0.041 | CEN alpha=3.9e5, inside grid |
| dmnelfonly19 | +0.042 | +0.032 | +0.043 | **CEN alpha=1e8 still saturates**; PDA picked alpha=0.01 (grid min) with in-sample +0.466 vs CV +0.043 — a flat CV surface, argmax landed on noise |

Pooling nearly doubled CEN grouped-CV (+0.042 → +0.075) and stopped the CEN alpha saturating even
against the widened 1e8 grid — so pooling *does* give the CEN model real cross-subject structure it
did not have before.

### LOCKED external test — 10 nf1-only subjects, 49 runs (each arm scored ONCE)

| model | CEN | DMN | **PDA** |
|---|---|---|---|
| shipped (n=19, PDA derived) | +0.039 *(p=.038)* | **+0.061** *(p=.016)* | −0.011 *(p=0.47)* |
| pooled28 (PDA fitted) | +0.033 *(p=.12)* | **+0.059** *(p=.006)* | −0.039 *(p=0.13)* |
| dmnelfonly19 (PDA fitted) | +0.019 *(p=.34)* | **+0.049** *(p=.026)* | +0.013 *(p=0.55)* |

(The nf2-only arm is 2 subjects / 9 runs — CIs and p are `nan`. Treat as uninformative.)

### What this means — three conclusions

1. **PDA does not transfer to new subjects under ANY variant tried.** Shipped, pooled-with-fixes,
   and DMNELF-only-with-fixes all land at ~0 (−0.039 to +0.013, every p > 0.13). Fitting PDA
   directly — which Phase 1 predicted would be the fix, and which offline LOSO supported — **did
   not work.** The problem is not the training procedure.
2. **DMN is the only target that transfers reliably**: +0.049 to +0.061, p < 0.03 in all three
   models, including the untouched shipped one. If a portable EEG neurofeedback signal is wanted,
   **DMN alone is the defensible target, not PDA.**
3. **Grouped CV overstated transfer.** Pooling's big CEN grouped-CV gain (+0.042→+0.075) did not
   appear on the locked set (+0.019 → +0.033, vs shipped's +0.039). Within-cohort grouped CV is not
   a safe proxy for new-subject transfer here — select on it only with that caveat.

This is consistent with `eeg_bold_coupling/HANDOFF.md`: the EEG-decodable component is largely
shared/global, so a contrast that cancels the shared component (CEN − DMN) cancels the decodable
part. Phase 1 showed the differencing destroys transfer; Phase 2 shows fitting the contrast directly
does not recover it, because the contrast itself is what is not decodable.

### PROTOCOL NOTE — read before running more arms

The locked set has now been spent on three arms. The remaining planned arms (`gsr` targets,
ElasticNet, `epoc_afproxy`, `cap31`) must be **selected on training-side grouped CV only**, with a
single final locked-set confirmation of the one chosen winner. Scoring every arm on the locked set
would reintroduce exactly the selection creep the lock exists to prevent.

## PHASE 2.5 — montage / ElasticNet / joint-fit arms — NEGATIVE, none earned a locked-set query

User decision after Phase 2: keep the shipped two-ridge architecture (CEN, DMN fit independently,
PDA = cen − dmn downstream) rather than keep chasing direct-PDA-fit or pooling — focus instead on
improving CEN/DMN, the two targets that DO transfer. Four single-variable-changed arms, all
`--dmnelf-only` (19 subjects, isolates each variable from the pooling question already answered in
Phase 2), trained via `train_pooled.py` on 2026-09-01 (SLURM job `9855083`, log:
`results/logs/arms2_9855083.out`). Baseline is `dmnelfonly19-with-fixes` (epoc12, ridge — the
already-locked-tested model from Phase 2: grouped-CV CEN +0.042 / DMN +0.032; locked test CEN
r=+0.019 p=.343 / DMN r=+0.049 p=.026):

| arm | montage | estimator | CEN grouped-CV | DMN grouped-CV |
|---|---|---|---|---|
| baseline (Phase 2) | epoc12 | ridge | +0.042 | +0.032 |
| C | epoc_afproxy (+Fp1/Fp2) | ridge | +0.041 | +0.030 |
| D | cap31 (full 31-ch) | ridge | +0.049 | +0.019 |
| E | epoc12 | elasticnet | +0.044 | +0.019 |
| F | epoc12 | **pls** (joint CEN+DMN, n_components grid-searched 2–80) | +0.012 (mean) | +0.012 (mean) |

New estimator added to `train_pooled.py --estimator pls`: 2-output `PLSRegression` fit jointly on
`[CEN, DMN]` (motivated by `eeg_bold_coupling/HANDOFF.md` — the EEG-decodable component is largely
shared/global, so letting the two targets share a low-rank latent structure, instead of two fully
independently-regularized ridges, seemed like it should help). Grouped-CV picked `n_components=2`;
the coefficient/intercept extraction was numerically verified against `sklearn`'s own
`.predict()` (max abs diff ~1e-16) before trusting it — this isn't a plumbing bug.

**Findings:**
- **Arm C (epoc_afproxy)**: adding Fp1/Fp2 to the portable montage does essentially nothing
  (CEN/DMN both within noise of baseline).
- **Arm D (cap31)**: the only arm with a real (non-trivial) effect, but it's a **trade-off, not a
  win** — CEN improves (+0.042→+0.049) while DMN gets *worse* (+0.032→+0.019). Not scored on the
  locked set per protocol (ambiguous candidates don't earn a confirmatory query).
- **Arm E (ElasticNet)**: flat for CEN, worse for DMN vs Ridge.
- **Arm F (joint PLS2) is the worst of the four on honest grouped-CV** despite having the
  *highest* in-sample r of any arm tried (CEN=+0.12, DMN=+0.10 in-sample vs only +0.012 grouped-CV
  mean) — a textbook overfitting signature. **The shared-latent-structure hypothesis is not
  supported**: forcing CEN and DMN to share components hurts rather than helps, at least with a
  linear low-rank (PLS) approach on this feature set.
- **None of these 4 arms was scored on the locked external test.** Per the protocol note above,
  the lock is reserved for a genuine winner, and none of these qualified.

**Net conclusion across Phase 2 + 2.5**: pooling, direct-PDA-fit, wider montages, ElasticNet, and
joint multi-task fitting have all now been tried and none improves on the already-shipped
architecture's CEN/DMN transfer. This is consistent with `eeg_bold_coupling/HANDOFF.md`: the
ceiling here looks like it's set by how much EEG-decodable signal exists at all (largely
shared/global), not by these modeling choices. The one lever not yet tried is **per-subject
calibration** (see below) — everything in Phase 2/2.5 was cohort-level modeling; nothing here
touches per-individual weight adaptation.

## PHASE 4 — Pz hypothesis REFUTED; and a possible selection bias in the published PDA LOSO

Question: does PDA fail on the portable montage because the EPOC-X lacks centro-parietal midline
coverage (Pz/POz/Cz), where `efp_cen_clean.py:29` says the CEN signal peaks? Phase 2.5 tested
`cap31` but had dropped PDA, so this was never asked.

`scripts/montage_pz_ablation.py` — method held fixed (multivariate ridge on the [10 band x 11
delay] per-channel design), channel set varied, honest nested-CV LOSO over the 19 DMNELF subjects.
Does NOT touch the locked external set. Jobs `9907733`, `9908042`, `9908150`.

**A methodological error found mid-analysis, and fixed.** The first run built targets from
`cen_mean_cache` (what the *deployed* `export_model.py` does) rather than from the research
pipeline's own `tgt_tr` targets stored in the feature cache (what the +0.157 LOSO benchmark used).
Those are different definitions and the numbers are not comparable. `--target-source {cenmean,cache}`
now selects; everything below uses `cache`.

### Montage ablation (multivariate, research-pipeline targets)

| montage | n_ch | CEN | DMN | PDA |
|---|---|---|---|---|
| epoc12 | 12 | +0.138 *(p<.001)* | +0.089 *(p=.007)* | −0.049 *(p=.138)* |
| epoc12 + **Pz** | 13 | +0.143 *(p<.001)* | +0.087 *(p=.008)* | **−0.071 *(p=.029)*** |
| epoc12 + Pz/POz/Cz | 15 | +0.145 *(p<.001)* | +0.084 *(p=.012)* | −0.034 *(p=.284)* |
| cap31 | 31 | +0.145 *(p<.001)* | +0.086 *(p=.007)* | +0.023 *(p=.497)* |

**The Pz hypothesis is refuted.** Adding Pz does not recover PDA — it makes it *significantly
negative*. Full 31-channel coverage doesn't recover it either. Missing midline coverage is not the
explanation. Note also that CEN/DMN transfer *fine* on the 12-channel montage with these targets
(+0.138/+0.089), at or above the published single-electrode LOSO figures.

### Single-electrode LOSO, same targets, NO electrode selection

| target | best 5 | at the published benchmark electrode | electrodes with r>0 |
|---|---|---|---|
| CEN | P7 +0.141, P8 +0.141, P3 +0.140, P4 +0.136, F4 +0.136 | **CP6 = +0.115** (published +0.114) | **31/31** |
| DMN | T7 +0.117, P8 +0.099, T8 +0.096, CP5 +0.092, P7 +0.091 | O1 = +0.073 (published +0.107) | 30/31 |
| PDA | POz +0.043, Fp1 +0.043, FC2 +0.035, C4 +0.031, C3 +0.026 | **Pz = +0.011** (published **+0.157**) | **9/31** |

**Harness is validated by CEN**: CP6 = +0.115 against the published +0.114 — a 0.001 match on the
same cohort, targets, and LOSO structure.

**But PDA at Pz is +0.011 here versus +0.157 published** — a ~14x discrepancy at the same electrode
under a harness that reproduces CEN exactly. PDA is positive at only 9/31 electrodes and
*significantly negative* at several (F4 −0.084, p=.0065). That is the signature of a null field in
which picking the best of 31 electrodes yields a spuriously large value.

**Hypothesis requiring confirmation:** the published PDA LOSO is inflated by best-electrode
selection performed against the same folds being scored — precisely the bias
`efp_meirhasson/VALIDATION.md:10-13` already documents for the within-subject numbers
("best-electrode selection scored on the same CV folds -> optimistically biased ~+0.05-0.15").
CEN is immune because it is uniformly positive across all 31 electrodes, so selection cannot
inflate it; PDA is not, so selection would do all the work.

**NOT yet ruled out** (do these before treating the above as established):
1. Read `efp_meirhasson/scripts/efp_group.py` / `efp_decode.py` and determine exactly how
   `common_ch` / `loso_fold_modal_ch` are chosen, and whether the choice ever sees the scored fold.
2. Alpha grid differs: theirs 0.01–1e4 (30 pts), mine 1–1e8 (25 pts).
3. Their single-electrode design may differ from my per-run z-scored construction.
4. If confirmed, the same concern applies to the **cross-cohort** PDA figures (+0.080 p=.017,
   +0.145 p=.007), which are also reported at Pz.

**Consequence if confirmed:** PDA is currently the manuscript's best-transferring target and the
headline of the preprint's generalization claim. Do not put PDA +0.157 into the RPPR or the
abstract until item 1 is resolved. CEN and DMN are unaffected and safe to report.

## Phases

- [x] **Phase 0** — folder, locked cohort split, this document.
- [x] **Phase 1** — `scripts/eval_holdout.py`: honest held-out score for the CURRENT
      `efp_epoc_model.npz`. Runs the real online path (`ReplaySource → RTFeatureExtractor →
      Decoder`), reusing `compare_engine.py:150-171` alignment/normalization. Reports per-run r for
      CEN/DMN/PDA, cohort mean ± SEM, Fisher CIs, sign-flip permutation; labels every number
      in-sample vs held-out. **Expect well below +0.22.**
- [~] **Phase 2** — `scripts/train_pooled.py` (copied from `efp_epoc/export_model.py`, not edited in
      place). Four fixes: (1) fit PDA directly as a third target, emit `pda_coef`/`pda_alpha`;
      (2) widen α grid `logspace(-2,5,15)` → `logspace(-2,8,30)` (current `cen_alpha` == grid max);
      (3) replace LOO-GCV with subject-grouped/blocked CV; (4) fix band edges (currently one
      arbitrary channel's, via a `band_hz` overwrite bug at `efp_features.py:238-241`).
      Scored arms: target `_gsr` vs `clean`; ElasticNet vs Ridge.
- [x] **Phase 2.5** — montage comparison (`epoc12`/`epoc_afproxy`/`cap31`) + ElasticNet + a new
      joint-CEN+DMN `pls` estimator, all `--dmnelf-only` to isolate each variable. NEGATIVE — see
      above. Supersedes the original Phase 3 scope (montage comparison is done; cap31 shows a
      CEN/DMN trade-off, not a win).
- [x] **Phase 3** — per-subject calibration (`efp_calibrated.py --epoc12-multi`), honest
      cross-subject LOSO. NEGATIVE for CEN/DMN (DMN gets systematically worse); positive but
      modest for PDA/GSR_PDA only. See above.
- [ ] **Phase 4** — no arm from Phase 2/2.5/3 earned a locked-set (rtBPD) confirmation; nothing
      here beat the shipped model on CEN/DMN. Nothing left to try that isn't already in "Do not
      redo" — this line of investigation is closed pending a genuinely new idea.

## Do not redo — recorded negative results

- 2-TR feature window: looked like +0.08 on dmnelf005, but Δr = −0.010 (p=0.46) across 63
  subject-runs (`mindwear/rt_features.py:69-72`).
- Cyclic transcoder (NN EEG→fMRI→PDA): mean r=0.067 across 14 subjects, does not generalize.
- HRF-convolved band-limited power: no network-specific signal survives global-signal control.
- Training on `orig` targets scores ~6× higher than `clean` but is largely arousal artifact —
  EEG→DMN r=+0.30 collapses to −0.042 (p=0.52) under global-signal control. Prefer `_gsr`.
- **Wider montage (`epoc_afproxy`, `cap31`) as a way to improve CEN+DMN**: `epoc_afproxy` does
  nothing; `cap31` trades a small CEN gain for a real DMN loss, not a net win (Phase 2.5).
- **ElasticNet instead of Ridge** for CEN/DMN: flat-to-worse vs Ridge, no benefit found (Phase 2.5).
- **Joint CEN+DMN fit via 2-output PLS** (shared low-rank latent structure): grouped-CV mean r
  +0.012, the worst of every arm tried despite the highest in-sample r — clear overfitting, not a
  fix. The "let CEN/DMN share structure" hypothesis from `eeg_bold_coupling/HANDOFF.md` does not
  translate into a working linear joint model here (Phase 2.5).
- **Per-subject calibration on the real EPOC12 montage** (`efp_calibrated.py --epoc12-multi`,
  honest cross-subject LOSO): does not improve CEN (p=0.10–0.20 at every calibration size); makes
  DMN systematically worse, more so with more calibration data. Helps PDA/GSR_PDA only — not the
  kept targets (Phase 3).

## PHASE 3 — per-subject calibration — TESTED (2026-09-01). Negative for CEN/DMN, positive for PDA.

The lever below was deliberately deferred until Phase 2/2.5 were exhausted; they are, so this got
tested. `efp_meirhasson/scripts/efp_calibrated.py` got a new `--epoc12-multi` flag (multivariate,
the REAL deployed 12-channel montage — the script's existing `--frontal`/`--frontal-multi` used a
different, generic "frontal headset" electrode set, not EPOC12's actual channels, so this had never
actually been tested on the real montage before). Ran `--mode loso --epoc12-multi`: honest
cross-subject LOSO within DMNELF (n=19) — group model trained on 18 subjects, calibrated on the
held-out subject's OWN run-1 task-design pseudo-target (boxcar, HRF-convolved — no fMRI needed at
calibration time, the deployable scenario), scored against their REAL fMRI on the remaining runs.
Results: `19_fingerprint/efp_meirhasson/results/efp_calibrated_loso_epoc12_multi{,_summary}.csv`.

Mean r, group_only vs pseudo_cal, by calibration size (n_cal = 1/2/3 runs):

| target | group_only | pseudo_cal | best p (cal beats baseline) |
|---|---|---|---|
| CEN | +0.084 / +0.094 / +0.131 | +0.071 / +0.080 / +0.124 | never (p=0.54–0.65) |
| **DMN** | +0.062 / +0.050 / +0.040 | **−0.012 / −0.048 / −0.056** | never — consistently worse, and monotonically worse with more calibration data |
| PDA | −0.048 / −0.028 / +0.015 | +0.055 / +0.017 / +0.065 | significant at n_cal=1 (p=0.008) only |
| GSR_PDA | −0.033 / −0.014 / +0.010 | +0.070 / +0.034 / +0.072 | significant at n_cal=1 (p=0.005) |

**For CEN and DMN specifically — the targets kept after Phase 2/2.5 — calibration does not help.**
CEN's blend nominally trends above group_only at every n_cal (+0.110/+0.113/+0.169 vs
+0.084/+0.094/+0.131) but never clears significance (p=0.10–0.20; n=19 is underpowered for an
effect this size). **DMN gets systematically worse with calibration, monotonically so as more
calibration data is added** — a clean, consistent negative, not noise. PDA and GSR_PDA (both
near-zero/negative for the group model on this restricted montage) do get a real, significant
one-calibration-run boost, but PDA isn't the kept target.

Earlier "evidence" cited for this lever (rtbpd002 PDA 0.268→0.283–0.289; dmnelf005 EPOC-12 PDA
0.373 under LORO) was from a *within-subject* leave-one-run-out test on hand-picked strong
performers, not this honest cross-subject LOSO — the two aren't the same claim, and the weaker,
averaged, all-subjects number above is the one to trust.

**Bottom line across every arm tried (Phase 1 → 3): pooling, direct-PDA-fit, wider montage,
ElasticNet, joint multi-task fitting, and per-subject calibration have ALL been tried, and none
improves CEN/DMN transfer beyond the currently-shipped `efp_epoc_model.npz`.** The ceiling here
looks set by how much EEG-decodable signal exists at all (see `eeg_bold_coupling/HANDOFF.md`),
not by any of these modeling/calibration choices. Meanwhile mindwear's own calibration block only
adjusts input standardization and never touches ridge weights (`calibration.py`,
`session_engine.py:588`) — consistent with this finding, not a bug to fix.

---

## Resuming on another machine

```bash
git clone git@github.com:cccbauer/dmnelf.git && cd dmnelf && git checkout efp-pooled
git clone git@github.com:cccbauer/mindlab.git      # MUST be a sibling of dmnelf/
curl -LsSf https://astral.sh/uv/install.sh | sh    # then add ~/.local/bin to PATH
cd mindlab/mindwear && uv sync                     # Python 3.12, venv at the mindlab root
```

Two untracked data files are needed locally to reproduce the baseline (kept out of git on purpose):
```bash
scp explorer:/projects/swglab/data/DMNELF/derivatives/eeg_preprocessed/sub-dmnelf005/ses-dmnelf/eeg/\
sub-dmnelf005_ses-dmnelf_task-feedback_run-01_desc-preproc250Hz_eeg.fif \
    mindlab/mindwear/testdata/dmnelf005_feedback_run-01_250Hz.fif

mkdir -p dmnelf/analysis/fingerprint/efp_epoc/cen_mean_cache
scp explorer:/projects/swglab/data/DMNELF/analysis/fingerprint/efp_epoc/cen_mean_cache/\
cenmean_dmnelf_dmnelf005.npz dmnelf/analysis/fingerprint/efp_epoc/cen_mean_cache/
```

Sanity check — must print `n=103  EEG↔BOLD PDA r=+0.217`:
```bash
cd mindlab/mindwear && uv run python compare_engine.py --subject dmnelf005 --run 1
```

## Cluster

Host alias `explorer` (passwordless ssh; prints two harmless "Loading matlab" lines on every login).
Work dir to create: `/projects/swglab/data/DMNELF/analysis/fingerprint/efp_pooled/`.
SLURM available (`partition=short`); see `efp_meirhasson/scripts/submit_*.sh` for job templates.

**Running jobs:** none. Completed: `9841912` (Phase 1 eval), `9842071` (Phase 2 train arms A/B),
`9842175` (locked-set scoring). Earlier: `9841912` (`submit_eval_holdout.sh`) — results in
`efp_pooled/results/shipped_on_{dmnelf,rtbpd,rtbpd_nf2}.csv` + `_summary.json`, mirrored into git.

**Hard-won operational fact:** the login node OOM-kills *everything*, including numpy-only scripts
(`import mne` and even plain feature loading both got `Killed`). Every compute step must go through
SLURM. Use `partition=short`, `--mem=16G` is ample for the cache-based scoring.

## Known issues to fix opportunistically

- `mindwear/test_replay.py:57-58` and all four `mindwear/scripts/*.py` point at the dead
  `fsnr_eeg/results/cen_ceiling` path (fixed only in `compare_engine.py`, by `ffb6c1e`).
  Targets now live in `efp_epoc/cen_mean_cache/`.
- `mindwear/test_replay.py` asserts nothing about r — its only failure mode is <20 usable TRs.
- `mindwear/README.md:6` cites `efp_meirhasson/DEPLOY_EPOC.md`, **which does not exist anywhere**.
  Also its "~92% retention" is for clean CEN; for clean **PDA** — the fed-back signal — it is 73%.
