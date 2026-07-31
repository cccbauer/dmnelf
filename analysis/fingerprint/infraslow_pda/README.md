# infraslow_pda

Within-subject decoder of the neurofeedback signal **PDA = CEN − DMN** (fMRI) from
**infraslow EEG**. Sibling of [`microstate_pda`](../microstate_pda) and
[`spectral_power_pda`](../spectral_power_pda) — same target, same within-subject
framing, different EEG feature.

## Why this exists (origin)

This grew out of the [`cyclic_transcoder`](../cyclic_transcoder) project, which
tried to predict PDA via a CycleGAN-style bidirectional EEG↔fMRI transcoder
trained leave-one-subject-out. That approach **did not work**, for reasons we
traced this session:

1. The evaluation was buggy (scrambled / sign-flipped / 50-of-400 samples) and
   faked correlations of 0.5–0.66; corrected r ≈ 0 cohort-wide.
2. PDA was never in the training loss; once added, the model still showed **no
   cross-subject generalization**.
3. The EEG feature (per-TR block-mean) is meant to capture the **infraslow**
   (<0.1 Hz) band — which couples to BOLD — but the preprocessing high-passed at
   **1 Hz**, deleting exactly that band before averaging.

The recording uses a **DC-coupled** amplifier, so infraslow is in the raw data.
Re-preprocessing with a 0.01 Hz high-pass recovers it (21,698× more 0.01–0.1 Hz
power vs the 1–40 Hz files; block-mean lag-1 autocorr flips −0.56 → +0.77).

## Result so far (sub-dmnelf007, single subject)

Within-subject, train on the subject's **own rest**, predict their **own
feedback** PDA (honest significance via circular-shift permutation null, which
preserves autocorrelation — the parametric p is anti-conservative for these slow
signals):

| feature                | feedback r | circular-shift p |
|------------------------|-----------:|-----------------:|
| infraslow (all 31 ch)  |     +0.235 |           0.0002 |
| infraslow (Cz only)    |     +0.256 |            0.016 |
| baseline 1–40 Hz (ctrl)|     −0.009 |             0.53 |

Topography is centroparietal (CP2/Cz/CP6 positive). Coupling peaks at ~11–15 TR
(~13–18 s) lag — **longer than a canonical HRF**, and HRF convolution does not
improve it → genuine infraslow coupling, not a standard hemodynamic response.

**Not yet generalized to the cohort.** Validated on 007 only.

## Pipeline

1. **Infraslow preprocessing** (in `mne_eeg_preprocessing`, not here):
   ```
   eeg_preproc.py --subject sub-dmnelfNNN --sfreq 500 --infraslow
   ```
   Fits BCG/ICA on a 1 Hz copy, applies the cleaning to a 0.01 Hz copy, saves
   `*_desc-preproc500HzISp01_eeg.fif` alongside the 1–40 Hz baseline.

2. **Coupling check** (assumption-light, no model):
   ```
   python scripts/within_rest_coupling.py --subject dmnelfNNN
   ```

3. **Decode** (Ridge; lag chosen by within-rest leave-one-run-out CV):
   ```
   python scripts/ridge_decode.py --subject dmnelfNNN
   ```

4. **Refine / significance** (circular-shift null, per-channel topography, HRF):
   ```
   python scripts/refine_007.py --subject dmnelfNNN
   ```

Run with the `eeg_preproc` env on Explorer:
`/home/cccbauer/.conda/envs/eeg_preproc/bin/python` (has mne + sklearn).

## Notes / caveats

- PDA target and the 1–40 Hz baseline block-means are reused from the
  `cyclic_features` npz (shared fMRI-derived target); `features_dir` points there.
- Use the **circular-shift null**, not the parametric p, for significance.
- Per-channel topography r's are lag-selected and not multiple-comparison
  corrected.
- Only 2 rest runs/subject → noisy lag CV; consider adding shortrest.
- Use **Ridge**, not ElasticNet (ElasticNetCV collapsed to a constant here).
