# HOW TO: rtBPD nf1 rest microstate pipeline (pre vs post)

This documents the actual end-to-end run of the rtBPD microstate pipeline
(2026-07-15 through 2026-07-27), including the one ad-hoc fix required along
the way, so it can be reproduced or re-run without re-discovering any of
this.

## 0. Preflight (read-only, done before any compute)

Confirmed via `ssh`/`ls` against `/projects/swglab/data/rtBPD/rawdata_eeg/`
that only 15 of the 21 rtBPD nf1 subjects have a COMPLETE 4-run
`task-rest` EDF set (pre = runs 01/02, post = runs 03/04). Excluded, per
explicit decision (no nf2 fallback, no partial-run substitution):
`rtbpd004`, `rtbpd022`, `rtbpd026`, `rtbpd027`, `rtbpd028`, `rtbpd034`.

Also confirmed the rtBPD montage (32 ch, incl. `TP9`/`TP10`/`ECG`) and that
all 25 canonical DMN/CEN/VIS/SOM/SAL/AUD/DANT signature channel names used
by `microstate_pda/deploy_scripts/01_fit_microstates.py` exist 1:1 in this
montage — no substitutions needed.

## 1. Preprocess rest EEG (Phase 1)

```bash
scp rtbpd_replication/scripts/submit_eeg_rtbpd_rest.sh \
    cccbauer@explorer.northeastern.edu:/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/scripts/
ssh cccbauer@explorer.northeastern.edu \
    'cd /projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/scripts && sbatch submit_eeg_rtbpd_rest.sh'
```

SLURM array job (`--array=0-14`, one task per subject; `eeg_preproc_rtbpd.py`
loops all 4 rest runs internally per `--tasks rest` invocation). Per-subject
`--raw-session`: `ses-nf` for the 2 pilots (rtbpd002/003), `ses-nf1` for the
other 13; `--out-session ses-nf` for all, unifying with the already-processed
`task-feedback` derivatives.

**Result:** job `8374949`, all 15 array tasks `COMPLETED` (exit `0:0`).
Verified all 60 expected FIFs
(`sub-rtbpd{XXX}_ses-nf_task-rest_run-{01..04}_desc-preproc500Hz_eeg.fif`)
exist under `/projects/swglab/data/rtBPD/derivatives/eeg_preprocessed/`.
Note: rtbpd002/rtbpd003 rest FIFs already existed from an earlier
(pre-Phase-1) run and were skipped (`EXISTS (skip)` in the log) rather than
reprocessed — this is expected `eeg_preproc_rtbpd.py` behavior, not an error.

## 2. Fit microstate templates (Phase 2, step 01)

```bash
cd rtbpd_replication/microstates
python3 01_fit_microstates_rtbpd.py
```

This builds the cluster-side fitting script, `py_compile`-checks it, `scp`s
it + an `sbatch` wrapper to
`/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates/scripts/`,
submits, and polls `squeue` until done.

### Ad-hoc fix required: SLURM time limit too short

**First attempt (job `8375975`, `--time=04:00:00`) hit `TIMEOUT`** after
exactly 4h, with 18 of 20 k-means restarts done (log showed healthy,
converging GEV ~0.6625-0.6654 — not a hang/bug, just genuinely
compute-heavy). Root cause: pooling GFP peaks across **all 4 rest runs** x
15 subjects (473,935 total peaks) is a much larger dataset than DMNELF's
typical 2-run fit, and most restarts used the full `MAX_ITER=1000` cap
(~13 min/restart).

**Fix:** edited the `--time` directive in `01_fit_microstates_rtbpd.py`'s
generated `sbatch_lines` from `04:00:00` to `12:00:00` (the `short`
partition allows up to 24h; no other constants were changed —
`N_KMEANS_RESTARTS`/`KMEANS_MAX_ITER` stayed at the DMNELF-validated 20/1000),
then re-ran the driver script to redeploy + resubmit.

**Result:** job `8404516` `COMPLETED` in 03:55:48 (well within the 12h
budget). Outputs saved to
`/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates/microstates/`:
`templates_500Hz.npy` (7,29), `gev_500Hz.npy` (best GEV = **0.6654**),
`assignments_500Hz.json`, `channels_500Hz.json` (29 retained channels).

If re-running this step from scratch, use `--time=12:00:00` (already the
current value in the committed driver script) or more — do not go back to
4h.

## 3. Back-fit + temporal parameters (Phase 2, step 02)

```bash
cd rtbpd_replication/microstates
python3 02_temporal_params_rtbpd.py
```

Deploys/submits/monitors identically to step 01. Loads the step-01
templates + channel list, back-fits every timepoint of all 60 runs onto the
7 templates (polarity-invariant correlation), applies the majority-merge
temporal-smoothing rule (min segment = 3 samples), computes duration/
occurrence/coverage/GEV per (subject, run, microstate).

**Result:** job `8777833` `COMPLETED` in 24s (fast — vectorized, no
restarts). All 60 runs processed, 0 missing. Sanity check passed: every
single run's 7 `coverage_pct` values summed to 100.0%. Output:
`{CLUSTER_BASE}/results/temporal_params.csv` (420 rows = 60 runs x 7
microstates).

## 4. Pre-vs-post stats + figures (Phase 2, step 03)

```bash
cd rtbpd_replication/microstates
python3 03_stats_pre_vs_post_rtbpd.py
```

**Result:** job `8782188` `COMPLETED` in 21s. Paired t-test (`ttest_rel`) +
paired Cohen's d + BH-FDR (`statsmodels.stats.multitest.multipletests`,
`fdr_bh`) across 4 parameters x 7 microstates (28 tests), plus a
confirmatory MixedLM (`value ~ C(condition)`, `groups=subject`, random
intercept only) run separately per (parameter, microstate).

Note: 5 of the 28 MixedLM models (`duration_ms`/VIS, `occurrence_hz`/DANT,
`occurrence_hz`/SAL, `gev_pct`/CEN, `gev_pct`/SOM) failed to fit with
`Singular matrix` — handled gracefully (those rows have `lmm_*` columns
`None`/blank in the CSV; the t-test/Cohen's-d columns for those rows are
still valid). This is a known small-sample MixedLM convergence failure mode
(N=15 subjects x 2 conditions), not a script bug.

Outputs in `{CLUSTER_BASE}/results/`: `stats_pre_vs_post.csv`,
`templates_topomap.png` (7-panel), `pre_vs_post_params.png` (4-panel bar
chart with BH-FDR stars).

## 5. Fetch results locally

```bash
cd rtbpd_replication/microstates
python3 fetch_results_rtbpd_ms.py
```

Pulls everything under `{CLUSTER_BASE}/microstates/*` and
`{CLUSTER_BASE}/results/*` back to `rtbpd_replication/microstates/results/`
(flat layout: `results/microstates/` for step-01 outputs, `results/` root
for steps 02-03 outputs). Add `--dry-run` to preview without downloading.

## Re-running just the analysis (no re-preprocessing)

If the 15-subject preprocessed FIF set doesn't change, you can re-run
steps 2-5 independently at any point:

- **Re-fit templates only:** re-run `01_fit_microstates_rtbpd.py`
  (overwrites `templates_500Hz.npy` etc. — there is no `--overwrite` guard
  in this script, unlike `eeg_preproc_rtbpd.py`, so it always refits).
- **Re-run temporal params only** (e.g. after changing the smoothing rule
  or a bug fix in `02_temporal_params_rtbpd.py`): re-run
  `02_temporal_params_rtbpd.py` — it reads whatever templates currently
  exist in `{CLUSTER_BASE}/microstates/`, no need to redo step 01 unless the
  templates themselves need to change.
- **Re-run stats only** (e.g. a different alpha, a different figure style):
  re-run `03_stats_pre_vs_post_rtbpd.py` — it only reads
  `{CLUSTER_BASE}/results/temporal_params.csv`, so step 02 doesn't need to
  be redone either.
- After any of the above, re-run `fetch_results_rtbpd_ms.py` to pull the
  refreshed outputs locally.

## Key cluster paths

- Preprocessed rest FIFs:
  `/projects/swglab/data/rtBPD/derivatives/eeg_preprocessed/sub-{subject}/ses-nf/eeg/`
- Pipeline scripts/logs/outputs:
  `/projects/swglab/data/rtBPD/analysis/fingerprint/rtbpd_replication/microstates/{scripts,logs,microstates,results}/`
- SLURM account: `suewhit`; partition: `short` (24h cap).

## Final job IDs (for reference / log lookup)

| Step | Job ID | State | Elapsed |
|---|---|---|---|
| Phase 1 preprocessing (array, 15 tasks) | 8374949 | COMPLETED (all 15) | ~3-24 min/task |
| Template fit, attempt 1 (4h limit) | 8375975 | TIMEOUT | 4:00:15 |
| Template fit, attempt 2 (12h limit) | 8404516 | COMPLETED | 3:55:48 |
| Temporal params | 8777833 | COMPLETED | 0:24 |
| Pre-vs-post stats + figures | 8782188 | COMPLETED | 0:21 |
