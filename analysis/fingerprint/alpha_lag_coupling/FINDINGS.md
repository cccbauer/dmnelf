# alpha_lag_coupling — findings

Adapts Jacob et al. 2025 (*Biol Psychiatry CNNI*, "Noncanonical Neural-Hemodynamic Coupling by
Default and in Schizophrenia") to this project's DMN/CEN targets: residual alpha power (above the
aperiodic 1/f background) at Pz, lagged-correlated against DMN/CEN BOLD across ±10 s, summarized by
Accumulated Correlation Asymmetry (ACA). Unlike `efp_meirhasson`'s frozen sliding-delay ridge
(trained on the "feedback" task, a fixed lag structure baked into its coefficients), this uses only
resting-state data (`task-rest`) and lets the coupling lag vary freely per subject/region.

**Scope**: DMNELF only (16/17 subjects — `dmnelf016` lacks local resting EEG). rtBPD has the fMRI
side but its resting-state EEG isn't preprocessed locally beyond 3 pilot subjects (see
`config.yaml` for the exact subject list) — extending there is a preprocessing-sync step, not a
pipeline change. DMN/CEN targets use this repo's standard DiFuMo-64 + composite convention
(columns 64/65 of the cached `fmri_features`), not the paper's whole-brain/AAL3 parcellation —
directly comparable to PDA, no new atlas-fetching work.

## Pipeline

1. `scripts/extract_residual_alpha.py` — per TR (aligned to the actual 1.2 s DMNELF TR, not the
   paper's 2 s), a 2 s Hanning-window Welch PSD at Pz is fit with FOOOF (`aperiodic_mode="fixed"`,
   1-40 Hz — capped from the paper's 1-50 Hz because this repo's EEG preprocessing low-pass
   filters at 40 Hz). Residual alpha = summed oscillatory peak power with center frequency in
   8-12 Hz, i.e. alpha *above* the fit aperiodic background, not raw band power.
2. `scripts/lagged_coupling.py` — `corr(alpha[t], bold[t + lag])` for 11 lags spanning ±10 s (2 s
   nominal steps; actual lag/TR is rounded to the nearest 1.2 s TR, so realized lags land at
   multiples of 1.2 s rather than exactly on 2 s marks). Positive lag = alpha precedes BOLD
   (canonical direction). Two `task-rest` runs per subject, Fisher-z averaged.
3. `scripts/compute_aca.py` — ACA = sum(|r|, lag ≥ 0) − sum(|r|, lag < 0) per subject/region.
4. `scripts/decode_from_lag.py` — a cross-validated "decoder" reality check: pick each subject's
   best lag on run 1, report the correlation that lag *actually* achieves on held-out run 2 (and
   vice versa), Fisher-z averaged. This is deliberately the simplest possible decoder (one feature:
   alpha power at a lag chosen on independent data) — a floor reference, not a claim of a working
   real-time decoder.
5. `scripts/extract_stockwell_bands.py` / `lagged_coupling_stockwell.py` /
   `decode_from_lag_stockwell.py` — same pipeline (steps 1-2 and 4), swapping FOOOF residual alpha
   for efp_meirhasson's own Stockwell + equal-energy-band feature (see below).

## Results (16 DMNELF subjects, `task-rest` runs 1-2)

**Lagged correlations are the same order of magnitude as the paper's**: per-subject peak |r|
ranged ~0.02-0.22 (paper: individual max 0.17 to −0.25; group mean r < 0.1 — matches).

**ACA is canonical-leaning on average but highly variable across subjects** — same qualitative
picture as the paper (mixed canonical/noncanonical coupling, not a uniform direction):

| region | mean ACA | SD | range |
|---|---|---|---|
| DMN | +0.078 | 0.238 | −0.60 to +0.49 |
| CEN | +0.107 | 0.143 | −0.15 to +0.50 |

**The cross-validated single-lag decoder is at chance**:

| region | mean r (CV) | SD |
|---|---|---|
| DMN | −0.003 | 0.077 |
| CEN | −0.025 | 0.046 |

This is the important, sobering result: the in-sample lagged correlations and ACA scores describe
real structure in the data (and replicate the paper's own numbers reasonably well), but a subject's
"best lag" on one resting run does **not** reliably predict the best lag on another run — i.e.,
whatever coupling exists isn't a stable enough per-subject trait, at this single-electrode/
single-band resolution, to serve as a usable real-time decoding feature on its own. This mirrors
the broader pattern already documented elsewhere in this project (`eeg_bold_coupling`,
`cyclic_transcoder`): EEG-BOLD coupling exists but is weak, and naive single-feature/single-lag
approaches don't survive cross-validation — multivariate approaches (as in `eeg_bold_coupling`
Step C) have fared better.

## Stockwell-band variant (reusing efp_meirhasson's own feature construction)

`extract_stockwell_bands.py` swaps the feature extraction for efp_meirhasson's *own* Stockwell
S-transform + data-driven equal-energy banding (`channel_bandpower`/`bin_average`, reused directly
from `efp_meirhasson/scripts/efp_features.py`, same `n_bands=10`/`freq_min=1`/`freq_max=40` as its
config.yaml) — i.e. the exact feature this project's best-performing decoder is built on, applied
here to resting-state instead of "feedback". `lagged_coupling_stockwell.py` /
`decode_from_lag_stockwell.py` repeat the same lag/CV analysis per band.

**In-sample peak |r| is markedly higher** than the single-band residual-alpha version — up to
0.42, vs. 0.02-0.22 before — because there are now 10 bands × 11 lags (220 candidates) to
peak-pick from per subject/region, not 1 band × 11 lags.

**But the cross-validated result is, again, at chance**:

| region | mean r (CV) | SD |
|---|---|---|
| DMN | +0.005 | 0.109 |
| CEN | −0.038 | 0.094 |

This is the same honest conclusion as the residual-alpha pipeline, now demonstrated more starkly:
searching more bands/lags inflates the in-sample number without improving generalization — a
textbook multiple-comparisons effect, not a case for "Stockwell bands decode better."

**One exception worth flagging**: `dmnelf008`'s DMN result is genuinely stable across both
directions of the cross-validation (r=+0.339 and +0.324, both picking a high-frequency band at
lag=0s) — an individual case where the coupling looks like a real, reproducible trait rather than
run-specific noise. Whether this reflects something idiosyncratic about that subject/recording or
a genuine signal worth chasing would need more than n=1 to know.

## Lag-group decoder (does splitting the training cohort by coupling direction help?)

Motivated by the ACA spread above: the single deployed decoder
(`mindwear/model/efp_epoc_model.npz`) pools all 17 DMNELF subjects into one `RidgeCV` fit
(`mindwear/export_model.py`), with no accounting for the fact that some subjects show canonical
coupling and others noncanonical. `scripts/define_lag_groups.py` splits subjects by the sign of
their resting-state DMN ACA — **canonical (ACA ≥ 0): 12 subjects, noncanonical (ACA < 0): 4
subjects** (`results/lag_groups.csv` — a notably uneven split; the noncanonical LOSO arm trains on
only 3 subjects per fold, about as small as this kind of comparison can get).

`scripts/lag_group_decoder.py` reuses `export_model.py`'s exact multivariate design (channel-major,
all 12 EPOC channels, `[10 band x 11 delay]`, same `RidgeCV` alpha grid) to run a paired
leave-one-subject-out comparison: for each held-out subject, fit on (a) every other subject
regardless of group ("pooled_all," today's approach) vs. (b) only the held-out subject's own
coupling-direction group ("own_group"), predict the held-out subject both ways.

**Result: no benefit from grouping** (`results/lag_group_decoder_loso.csv`, n=16 subjects):

| target | mean r, own_group | mean r, pooled_all | mean diff | own_group wins |
|---|---|---|---|---|
| CEN | +0.002 | +0.003 | −0.001 | 8/16 |
| DMN | +0.020 | +0.029 | −0.009 | 7/16 |

Splitting by resting-state coupling direction doesn't improve (and slightly hurts, on average)
feedback-task LOSO decoding — own_group wins for barely half the subjects, a coin flip. This
directly tests the caveat flagged when the groups were defined: a subject's resting-state alpha-BOLD
coupling direction doesn't appear to be a trait stable/relevant enough to usefully split *feedback-
task* training data by. Combined with the earlier null cross-validated single-lag/single-band
decoding results, the pattern across this whole project is consistent: coupling-lag information from
resting EEG doesn't transfer into better task-state decoding, at least not via these approaches.

(Note: `pooled_all`'s LOSO r here is somewhat lower than the ~0.095-0.13 cached in
`efp_meirhasson/results/full/efp_group_loso.csv` — that number comes from a *single best electrode*
picked per target via cross-validated selection, a more optimistic modeling choice than this
script's fixed full-12-channel multivariate design, which matches what's actually deployed live in
`mindwear`. The two aren't directly comparable, but both are properly held-out numbers of the same
small-effect order of magnitude, so the "own_group vs. pooled_all" comparison itself — which uses
the identical design for both arms — is the reliable part of this result.)

`dmnelf016` (no local resting-state EEG, so no ACA/group) was intended to be included in the
`pooled_all` arm per the original plan, but also turned out to lack a usable local feedback-task
cache — excluded entirely from this analysis, not just from grouping.

## Reconciling with the "12-channel retains most of the signal" finding

A tangent that came out of comparing decoder numbers: the manuscript's 81-100% channel-retention
finding (`manuscript/MANUSCRIPT.md:359-360`, full 31-ch cap vs. 12-ch EPOC) is a **within-subject,
personalized-calibration** comparison — each subject gets their own fit. That's a different regime
from the **zero-shot, cross-subject-pooled** LOSO this whole `alpha_lag_coupling` project has been
using, where a single ridge is fit on N−1 subjects and applied unchanged to a held-out one (matching
how the *deployed* frozen model actually gets used on a brand-new EPOC-X user, before any
calibration).

To isolate channel count as the only variable in that zero-shot regime,
`efp_meirhasson/scripts/electrode_vs_montage_loso.py` holds `efp_group.py`'s own validated
methodology fixed (raw DiFuMo CEN/DMN/PDA targets via `load_targets_run`, no baseline-TR masking,
leak-free per-fold electrode selection, `mk_block_folds`/`RidgeCV` CV) and adds only a second arm —
pooling all 12 EPOC channels multivariately instead of one selected electrode:

| target | leak-free single electrode | 12-channel EPOC montage (pooled) |
|---|---|---|
| CEN | r=0.121, p<0.001 | r=0.080, p=0.015 |
| DMN | r=0.087, p=0.019 | r=0.058, p=0.022 |
| **PDA** | **r=0.172, p=0.006** | **r=−0.035, p=0.37 (n.s.)** |

In this zero-shot pooled regime, going from a well-chosen single electrode to the full 12-channel
montage **hurts**, most dramatically for PDA — the difference between a real, significant effect and
chance. This is the opposite of the within-subject finding, and not actually a contradiction: it's a
1320-feature ridge (12 ch x 10 bands x 11 delays) fit on pooled cross-subject data with modest
per-subject TR counts, vs. a ~110-feature single-electrode design — the pooled multivariate case is
simply harder to fit well without per-subject personalization to fall back on. Practically: the
*deployed* `mindwear` decoder uses the full 12-channel multivariate design zero-shot (no per-subject
refit) — this result suggests that configuration may be leaving real accuracy on the table
specifically in the un-calibrated regime, independent of everything else this project found about
lag/coupling not helping.

## Caveats / next steps

- `fooof` (v1.1, now deprecated in favor of `specparam`) had to be installed locally
  (`pip install fooof` into the `fingerprint` conda env) — not yet in `environment_fingerprint.yml`.
- Aperiodic fit range capped at 1-40 Hz (repo's EEG low-pass), vs. the paper's 1-50 Hz — a beta/
  gamma-tail difference that could affect the aperiodic slope estimate somewhat.
- Only Pz and only 8-12 Hz alpha were tested, per the paper — a natural extension would be to also
  test other posterior/frontal channels, or feed multiple lags/channels into a proper multivariate
  model (matching `eeg_bold_coupling`'s more successful multivariate design) instead of the
  single-best-lag decoder here.
- No diagnosis (SZP/UAP) labels were readily available in this repo to replicate the paper's group
  comparison (schizophrenia showing more positive/delayed ACA) — if a participants/diagnosis file
  exists elsewhere, `compute_aca.py` could be extended to group-compare directly.
