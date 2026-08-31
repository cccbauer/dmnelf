# efp_pooled — pooled DMNELF+rtBPD EFP fingerprint

**Status:** Phase 0 complete (cohort split locked). Phase 1 not started.
**Branch:** `efp-pooled` in the `dmnelf` repo. **Last updated:** 2026-08-31.

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

## Phases

- [x] **Phase 0** — folder, locked cohort split, this document.
- [ ] **Phase 1** — `scripts/eval_loso.py`: honest held-out score for the CURRENT
      `efp_epoc_model.npz`. Runs the real online path (`ReplaySource → RTFeatureExtractor →
      Decoder`), reusing `compare_engine.py:150-171` alignment/normalization. Reports per-run r for
      CEN/DMN/PDA, cohort mean ± SEM, Fisher CIs, sign-flip permutation; labels every number
      in-sample vs held-out. **Expect well below +0.22.**
- [ ] **Phase 2** — `scripts/train_pooled.py` (copied from `efp_epoc/export_model.py`, not edited in
      place). Four fixes: (1) fit PDA directly as a third target, emit `pda_coef`/`pda_alpha`;
      (2) widen α grid `logspace(-2,5,15)` → `logspace(-2,8,30)` (current `cen_alpha` == grid max);
      (3) replace LOO-GCV with subject-grouped/blocked CV; (4) fix band edges (currently one
      arbitrary channel's, via a `band_hz` overwrite bug at `efp_features.py:238-241`).
      Scored arms: target `_gsr` vs `clean`; ElasticNet vs Ridge.
- [ ] **Phase 3** — montage comparison on identical 31-ch recordings: `epoc12` / `epoc_afproxy`
      (+Fp1/Fp2 for AF3/AF4) / `cap31` (has Pz/POz/TP10, the LOSO transfer electrodes EPOC X lacks).
- [ ] **Phase 4** — score all arms once on the locked 12, pick, ship to `mindwear/model/`.

## Do not redo — recorded negative results

- 2-TR feature window: looked like +0.08 on dmnelf005, but Δr = −0.010 (p=0.46) across 63
  subject-runs (`mindwear/rt_features.py:69-72`).
- Cyclic transcoder (NN EEG→fMRI→PDA): mean r=0.067 across 14 subjects, does not generalize.
- HRF-convolved band-limited power: no network-specific signal survives global-signal control.
- Training on `orig` targets scores ~6× higher than `clean` but is largely arousal artifact —
  EEG→DMN r=+0.30 collapses to −0.042 (p=0.52) under global-signal control. Prefer `_gsr`.

## Largest lever, deliberately deferred

**Per-subject calibration.** `efp_meirhasson/scripts/efp_calibrated.py` already implements the
deployable case: calibrate on run 1 against an HRF-convolved task-design boxcar pseudo-target — no
fMRI needed. Evidence: `pseudo_cal` > `group_only` (rtbpd002 PDA 0.268 → 0.283–0.289); per-subject
fit on dmnelf005's own EPOC-12 channels reaches PDA 0.373 under honest LORO. Meanwhile mindwear's
calibration block only adjusts input standardization and never touches ridge weights
(`calibration.py`, `session_engine.py:588`). Probably worth more than pooling — do it after.

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

**Running jobs:** none yet.

## Known issues to fix opportunistically

- `mindwear/test_replay.py:57-58` and all four `mindwear/scripts/*.py` point at the dead
  `fsnr_eeg/results/cen_ceiling` path (fixed only in `compare_engine.py`, by `ffb6c1e`).
  Targets now live in `efp_epoc/cen_mean_cache/`.
- `mindwear/test_replay.py` asserts nothing about r — its only failure mode is <20 usable TRs.
- `mindwear/README.md:6` cites `efp_meirhasson/DEPLOY_EPOC.md`, **which does not exist anywhere**.
  Also its "~92% retention" is for clean CEN; for clean **PDA** — the fed-back signal — it is 73%.
