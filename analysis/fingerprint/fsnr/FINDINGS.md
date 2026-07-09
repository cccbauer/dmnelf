# An fMRI functional signal-to-noise ratio (f-SNR) for DMNELF neurofeedback

*Purely-fMRI phase, before the EEG decoder. Applying the f-SNR framework
(Laukkonen 2026, "Clear Mind"; Nath 2026) to the DMNELF CEN/DMN/PDA networks.*

## Question

f-SNR = **signal variance / noise variance** (law of total variance:
`Var(r) = Var_z(E[r|z])` signal `+ E_z(Var(r|z))` noise). The framework calls **DMN the
noise** and predicts higher f-SNR with more signal (CEN/PDA up) and less DMN noise. We
asked whether a faithful fMRI f-SNR exists in DMNELF and whether **increased PDA / reduced
DMN = higher f-SNR** — and whether it is a good neurofeedback signal.

## Method

Every feedback run = **25 TR rest + 100 TR feedback** (drop 5 HRF-lag TRs), giving a
within-run controlled cause `z = {rest, feedback}`. f-SNR computed per network (PDA, CEN,
DMN) as signal/noise in **dB**; a GLM/pseudo-target version uses the HRF-convolved
rest→feedback boxcar as `E[r|z]` (= `R²/(1−R²)`). n = 17 subjects, 67 feedback runs.

## Findings

1. **The construct is real and behaves sensibly.** The group regulates correctly
   (β_PDA +0.18, up in 78% of runs; CEN +0.14 up; DMN −0.04 down). PDA is the cleanest
   contrast (f-SNR −14.5 dB > CEN > DMN). Per-subject reliability is usable (4-run
   average ≈ 0.69–0.75).

2. **f-SNR tracks the *signal* channel, not DMN mean-suppression.** Higher f-SNR goes with
   stronger PDA/CEN regulation (glm f-SNR vs β_PDA r = +0.58) but **not** with DMN mean
   drop (β_DMN r ≈ 0). "Increased PDA → higher f-SNR" holds; "reduced DMN → higher f-SNR"
   does **not**, as a mean effect.

3. **DMN-as-noise holds in the *variance* domain — variability quenching.** DMN endogenous
   variance significantly **quenches** rest→feedback (+2.2 dB, p = 1e-4, 75% of runs),
   more than CEN (+1.25). The mean-based test missed this because the noise is a variance,
   not a mean.

4. **…but the decluttering is mostly *global*, not DMN-specific, and is dissociable from
   the signal.** With a raw whole-brain reference (fMRIPrep global_signal), **global
   quenches most (+3.1 dB) > DMN (+2.2) > CEN (+1.25)**; DMN − CEN +0.96 dB (p = .05),
   DMN − global −0.9 dB (n.s.). A **startup-transient control passes** (dedicated long-rest
   runs show no quench, DMN −0.35 dB n.s.), so the feedback quench is real, not an onset
   artifact. Crucially, the amount of DMN quench does **not** predict f-SNR or regulation
   success (r ≈ −0.02 to +0.19) — **noise-reduction and signal-amplification are
   independent axes.**

5. **f-SNR is a valid NF marker but not a better target than raw PDA.** A causal
   real-time running f-SNR is well-modulated (fb > rest +1.07, p = 1e-4, 84% of runs),
   smooth (lag-1 autocorr 0.95), reliable (ICC 0.51), and controllable (tracks β_PDA,
   r = 0.77). **But it does not beat the raw PDA already fed back**: rest↔feedback
   discriminability d′ = 0.70 (f-SNR) vs 0.82 (PDA), Δ = −0.12 (n.s.). Normalizing by
   noise does not sharpen the signal — consistent with the noise channel being global and
   dissociated.

## Bottom line

DMNELF neurofeedback engages **both** f-SNR channels — signal-up (CEN/PDA regulation) and
noise-down (global + DMN variance quenching) — but they are **dissociable**, and the
usable, state-separating, individually-reliable information lives in the **signal
channel**. For neurofeedback, **raw PDA remains the better control signal**; f-SNR is a
clean *interpretive* index of "clarity," not an improvement. The signal-channel target
(`glm_PDA_db` / PDA) is what we carry into the **EEG decoding phase**; the global
decluttering axis is a separate, arousal-like signal EEG may read but that does not serve
the NF goal.

*Figures: `results/fig_fsnr_vs_pda_dmn.png`, `fig_dmn_quench.png`, `fig_fsnr_tighten.png`,
`fig_fsnr_timeseries_group.png`, `fig_fsnr_proxy.png`. Deck: `fsnr_results.pptx`.*
