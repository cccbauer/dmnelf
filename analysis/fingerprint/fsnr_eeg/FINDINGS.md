# A pure EEG f-SNR that matches the BOLD neurofeedback target

*EEG phase of the f-SNR project. Two streams, matched to the fMRI PDA / fMRI f-SNR.*

## Approach

- **Stream A (benchmark):** fit an EEG decoder (EFP single-electrode sliding-delay; multivariate
  band power) to the fMRI f-SNR / PDA — the supervised ceiling.
- **Stream B (the prize):** a PURE EEG f-SNR (no per-subject fitting), matched to PDA / fMRI
  f-SNR. Two flavors: (1) band power (running mean/std + variability quench); (2) oscillatory
  ÷ aperiodic (specparam 1/f). Matched HRF-aligned, leak-free, then generalized.

## Findings

1. **Within-subject:** a construct **frontal-theta EEG f-SNR** matches BOLD PDA at **r = 0.119**
   (p = 0.003) with **no fitting** — ~70% of the fitted EFP ceiling (~0.17). Frontal-alpha and
   posterior-alpha f-SNR match equally (~0.113–0.116). The specparam oscillatory÷aperiodic flavor
   is weaker (~0.05–0.10): per-TR FOOOF adds noise; simpler band power wins.

2. **The f-SNR normalization earns its keep on frontal (noisy) channels:** fitted single-site
   frontal raw 0.088 → f-SNR 0.101; posterior stays best raw (0.132). Good for a portable
   frontal headset.

3. **Generalization (the payoff):** the fixed frontal-theta f-SNR construct (zero fitting)
   - DMNELF LOSO (OOF): r = 0.116 (nothing to overfit);
   - **rtBPD cross-cohort: nf1 r = +0.126 (p = 5e-4, 79% subjects+), nf2 r = +0.147 (p = 5e-3, 91%+).**
   **Raw band power does NOT transfer** cross-cohort (r ≈ 0.00–0.03, n.s.) — the noise-normalization
   removes cohort-specific scale/gain, making the f-SNR cohort-invariant. The construct even
   matches/beats the **fitted** EFP decoder cross-cohort (EFP PDA nf1 0.067, nf2 0.153).

4. **Secondary — EEG decluttering during regulation (neural, not EMG):** non-convolved (specparam)
   band power shows beta +1.6 dB (p = 2e-4) and gamma +3.7 dB (p = 1e-4) variance drop during
   feedback; low bands flat. This is **not** an EMG artifact: the quench is **midline-strongest**
   (beta midline +1.56 vs temporal +1.06 dB), the **broadband aperiodic offset is flat** (−0.2 dB,
   n.s.; against EMG), and the **1/f exponent stabilizes** (+1.2 dB, p = 8e-4). Active novice
   noting-practice would not reduce EMG vs eyes-closed rest anyway. So: reduced high-frequency power
   variability + a more stable aperiodic 1/f slope during regulation — the stability/criticality
   face of the f-SNR framework. (The HRF-convolved cache gave a spurious +30 dB all-band onset-ramp
   artifact — corrected with the non-convolved extraction.)

## Bottom line

The pure EEG f-SNR (frontal-theta band power, zero fitting) matches the BOLD PDA the
neurofeedback trains, and — unlike raw EEG or even the fitted decoder — **transfers across
cohorts**. That is exactly the framework's claim (signal/noise is cohort-invariant), and it
yields the real prize: a **calibration-free, interpretable, portable-frontal EEG clarity index**
usable as an EEG-only proxy for the fMRI NF target on new subjects/sites. Use the fitted EFP
decoder when per-subject calibration is available (0.17 within); use the construct f-SNR when it
is not.

*Scripts: `scripts/eeg_fsnr_bandpower.py` (F1), `eeg_fsnr_specparam.py`+`eeg_fsnr_match.py` (F2),
`eeg_fsnr_generalize.py` (LOSO+cross-cohort). Figures: `results/fig_headtohead_within.png`,
`fig_crosscohort_generalize.png`. Deck: `eeg_fsnr_results.pptx`.*
