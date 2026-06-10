# infraslow_loro_pda

Within-subject, **within-feedback** decoder of PDA (CEN−DMN) from infraslow EEG.
Sibling of `microstate_pda` / `spectral_power_pda` / `infraslow_pda`.

Differs from `infraslow_pda` (which trained on rest and predicted feedback — a
cross-task transfer that failed cohort-wide): here we **leave-one-run-out across
the 4 feedback runs** within each participant, removing the cross-task gap.

## Method

- **Feature**: for each TR (1.2 s), average the 600 EEG samples per channel of the
  DC-coupled **infraslow** 0.01–40 Hz signal (`desc-preproc500HzISp01`). Repeating
  across TRs gives an `[N_TR × 31]` matrix, z-scored per channel.
- **Model**: ElasticNet, penalty `α·[ρ‖w‖₁ + (1−ρ)·½‖w‖²]` (ElasticNetCV picks
  α and ρ by inner CV on the training runs).
- **CV**: leave-one-run-out over the 4 feedback runs (train on 3, test on held-out),
  per participant; held-out predictions concatenated to a full PDA timeseries.
- **Smoothing**: post-hoc centered moving average (window = 11 TRs), applied per
  run to the held-out predictions.
- **Significance**: circular-shift permutation p (shift true PDA within run; 2000
  perms) — smoothing autocorrelated signals inflates the parametric p.
- **Control**: same pipeline on the 1–40 Hz baseline (`desc-preproc500Hz`).

## Run

```
python scripts/decode_loro.py --config config.yaml
```
Cluster env: `/home/cccbauer/.conda/envs/eeg_preproc/bin/python` (mne + sklearn).
Run as a SLURM job on the `sharing` partition (ElasticNetCV LORO + perms are
heavy enough to be killed on the login node).

Output: per-subject + cohort table (raw r, smoothed r, circular-shift p, baseline
control) and `results/decode_loro.csv`.

## Context / caveats

- Depends on the infraslow preprocessing in `mne_eeg_preprocessing`
  (`eeg_preproc.py --infraslow --sfreq 500`), which preserves the <0.1 Hz band a
  legacy 1 Hz high-pass had removed.
- Smoothing both signals before correlating inflates r; here only the prediction
  is smoothed, and significance uses the circular-shift null, not the parametric p.
- PDA target reused from the `cyclic_features` npz.
- Prior cohort analyses found infraslow did NOT beat baseline for within-rest
  coupling or rest→feedback transfer; this folder tests the easier within-feedback
  LORO framing. Read raw r, smoothed r, and the circular-shift p together.
