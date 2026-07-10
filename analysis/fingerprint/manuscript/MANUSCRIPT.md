# A calibration-free EEG functional signal-to-noise ratio tracks a DMN–CEN neurofeedback target and transfers transdiagnostically

*Target journal: Biological Psychiatry: Cognitive Neuroscience and Neuroimaging (Archival Report — 4000-word body, 250-word structured abstract, IMRD).*

> Draft in progress. Methods subsections that the shared documents enable are written below;
> sections marked **[PLACEHOLDER — doc]** await the forthcoming materials
> (EEG acquisition + BrainVision Analyzer artifact correction; full MURFI/scanner setup).

---

## Abstract (Background/Methods/Results/Conclusions — 250 words)

**Background:** Dysregulated default-mode (DMN)–central-executive (CEN) network interaction is a
transdiagnostic feature of psychopathology, and real-time fMRI neurofeedback targeting their
differential activation (PDA = CEN − DMN) is a candidate intervention — but fMRI is not scalable.
Whether a portable, calibration-free EEG marker can index this neurofeedback target across
disorders is unknown.

**Methods:** In two independent clinical real-time-fMRI-neurofeedback cohorts with simultaneous
EEG–fMRI — adults with schizophrenia (n=17) and adolescents with elevated borderline personality
traits (n=19; n=11 at retest) — we framed the neurofeedback signal as a functional
signal-to-noise ratio (f-SNR). We tested whether an EEG-derived f-SNR tracks the BOLD PDA,
comparing a fitted single-electrode decoder against a construct EEG f-SNR requiring no per-subject
fitting, with within-subject, leave-one-subject-out, and cross-cohort evaluation.

**Results:** A frontal-theta EEG band-power f-SNR, with no fitting, matched BOLD PDA within-subject
(r≈0.12) and transferred from schizophrenia to borderline-trait adolescents (r=0.13 and 0.15
across sessions; p<0.005), whereas raw EEG power did not — noise-normalization removes
cohort-specific gain. During regulation, EEG showed reduced high-frequency power variability and a
more stable aperiodic 1/f slope, not attributable to EMG (topographic and aperiodic controls).

**Conclusions:** A calibration-free, portable EEG index of DMN–CEN clarity generalizes
transdiagnostically and developmentally, supporting f-SNR as a deployable neurofeedback biomarker.

---

## 1. Introduction

**[TO DRAFT]** — DMN–CEN dysregulation is transdiagnostic (schizophrenia: aberrant salience,
DMN hyper/dysconnectivity; BPD/emotion-dysregulation: self-referential DMN); rt-fMRI mbNF of
PDA = CEN−DMN is a promising, personalized, circuit-based intervention (Zhang 2023; Bauer 2020)
but fMRI is not deployable. The f-SNR framework (Laukkonen 2026; Nath 2026) casts "clarity" as
signal/noise and posits it is transdiagnostically reduced. Gap: can a portable, *calibration-free*
EEG f-SNR index the fMRI-NF target and generalize across disorders? Aims: (i) build a faithful
fMRI f-SNR from CEN/DMN/PDA; (ii) derive a pure EEG f-SNR; (iii) test cross-modal match and
cross-cohort/transdiagnostic transfer.

## 2. Methods and Materials

### 2.1 Participants

Two independent clinical cohorts underwent personalized real-time-fMRI mindfulness-based
neurofeedback (mbNF) with simultaneous EEG–fMRI.
- **Cohort 1 (schizophrenia; "DMNELF"):** n=17 adults with a diagnosis of schizophrenia.
  **[clinical detail/symptom measures — CONFIRM: PANSS or other; medication.]**
- **Cohort 2 (borderline traits; "rtBPD"):** adolescents with *elevated borderline personality
  traits* (dimensional; no categorical diagnosis) assessed with **[BPD trait scale — CONFIRM name,
  e.g. BPFSC/PAI-BOR]**. Analyzed at two neurofeedback sessions (nf1: n=19; nf2: n=11).
All participants provided informed consent (assent + guardian consent for minors); procedures were
IRB-approved. **[site IRBs.]**

*Behavioural self-report (both cohorts).* After each feedback run, participants in **both** cohorts
rated four 1–9 sliders: **state calm** ("how calm are you feeling"; the affective/clinical outcome),
**mindfulness engagement** (mental "noting" in rtBPD / "describing" in DMNELF; harmonised),
**attention to the feedback** ("ball-check"), and **task difficulty**. Coverage: rtBPD 197 runs
(25 participants, nf1+nf2); DMNELF 47 rated runs (16 participants). State calm is the primary
behavioural outcome for validating the neurofeedback target. **[trait-level measures — schizophrenia
symptom scale (e.g. PANSS) and rtBPD borderline-trait scale — CONFIRM for trait biomarker analyses.]**

### 2.2 Mindfulness-based neurofeedback paradigm and the DMN–CEN target

Both cohorts completed the identical mbNF paradigm (Zhang et al., 2023; Bauer et al., 2020) using
the MURFI real-time functional imaging system (Hinds et al., 2011; Bauer et al., 2022) with
PsychoPy stimulus delivery over a TCP/IP link to the scanner. Each participant's DMN and CEN were
**personalized** (Section 2.3) and used as networks-of-interest. During feedback, MURFI fit an
incremental general linear model to each incoming volume and computed each network's activation in
standard deviations from a rolling baseline; the fed-back signal was the **prefrontal–default
differential activation, PDA = activation(CEN) − activation(DMN)** (Bauer et al., 2019; Chen
et al., 2013). A white dot moved upward when DMN activation fell below CEN activation (the target
state) and downward otherwise; participants up-regulated PDA by practicing **"mental noting"**, a
Vipassana attention technique taught before scanning. Each feedback run began with a **30 s (25-TR)
rest baseline** followed by continuous feedback. **PDA is therefore both the therapeutic target and
the fMRI signal decoded here.**

*Real-time operationalization.* Real-time computation used MURFI (multivariate and univariate
real-time functional imaging; https://github.com/gablab/murfi2), which processed neurofeedback
only — not network generation or offline preprocessing. Siemens online motion-corrected volumes
(MoCo series, realigned to each run's first volume) were streamed to MURFI over Ethernet; the
personalized DMN and CEN (frontoparietal network, FPN) masks were registered to the first volume
via FLIRT (falling back to Yeo-template masks if a participant's personalized masks were
unavailable). For each masked voxel, an incremental GLM (Gentleman's algorithm) was updated on
every incoming volume, with nuisance regressors for the six relative-displacement realignment
parameters (from the Siemens MoCo DICOM headers) and linear drift; per-voxel activation at time t
was the GLM residual (measured minus nuisance-predicted signal), z-scored to the mean and standard
deviation of the residuals over the **first 25 volumes (30 s baseline)**. Mask-level activation
used variance-based voxel-efficiency weighting (weights inversely proportional to each voxel's
30 s-baseline variance), which converges more closely with offline GLMs than a simple mean or
median (Hinds et al., 2011; Bauer et al., 2022). Feedback latency was < 1 TR (1.2 s); the intrinsic
hemodynamic delay (~6–8 s) is not resolvable by faster sampling or processing.

### 2.3 Personalized DMN/CEN localization

Personalized DMN and CEN masks were derived from resting-state fMRI by independent component
analysis (Melodic ICA), matching components to canonical DMN/CEN atlas maps, thresholding to the
upper 10% of voxel loadings, and binarizing (Zhang et al., 2023).
- **DMNELF (schizophrenia):** masks derived from a **single short resting-state run** acquired in
  the same session as feedback.
- **rtBPD (adolescents):** masks derived in a **separate localizer session** from **long
  resting-state runs acquired in both phase-encoding directions (2× AP/PA)**, during which the
  mindfulness ("mental noting") training was delivered; the feedback paradigm was identical to
  DMNELF.

### 2.4 MRI acquisition

Imaging used a Siemens MAGNETOM Prisma 3T scanner with a 64-channel head/neck coil. A T1-weighted
MPRAGE was acquired (1.0 mm isotropic, 176 sagittal slices, TR 2530 ms, TE 1.92 ms, TI 1400 ms,
flip angle 7°, GRAPPA 3). Functional runs used a T2*-weighted multiband gradient-echo EPI sequence
(TR 1200 ms, TE 30 ms, 2.0 mm isotropic voxels, 72 slices, multiband/slice-acceleration factor 4,
in-plane GRAPPA 2, phase-encode A≫P). **Feedback runs comprised 150 volumes (3 min)** each
(DMNELF: 4 runs; rtBPD: 5 runs); resting-state runs comprised 250 volumes. Spin-echo EPI
field maps with reversed phase-encode (AP/PA) were acquired for susceptibility-distortion
correction. Exact per-sequence parameters are provided in **Supplementary Table S1** (from the
scanner protocol printouts). **[CONFIRM DMNELF vs rtBPD protocol identity; any per-cohort
differences.]**

### 2.5 Simultaneous EEG acquisition and preprocessing

*EEG acquisition.* Continuous EEG was recorded simultaneously with fMRI using a 32-channel
MR-compatible BrainAmp MR amplifier and MR EEG cap (Brain Products GmbH, Gilching, Germany), with
electrodes positioned according to the international 10–20 system (31 scalp channels plus one ECG
channel). Data were sampled at 5000 Hz (0.5 µV/bit) with online hardware band-pass filtering
between 0.1 Hz (10 s time constant) and 250 Hz. All channels were referenced online to Cz (series
resistance 10 kΩ per channel; 20 kΩ for ECG); electrode-to-skin impedances were checked before each
session and kept below 25 kΩ where possible. Acquisition and gradient/volume triggers were
synchronized to the scanner for artifact correction.

*EEG preprocessing.* MRI gradient artifacts were removed in BrainVision Analyzer (v1.26) by average
artifact subtraction (AAS): a gradient-artifact template, formed by averaging artifact epochs
time-locked to the MRI volume trigger, was subtracted from the raw signal; gradient-corrected data
were then downsampled to 1 kHz and exported. All subsequent steps used MNE-Python: (1) ECG R-peak
detection (NeuroKit2); (2) automatic bad-channel detection (variance and high-frequency-noise
z-scores); (3) annotation of scanner ramp-up edges; (4) FIR band-pass filtering 1–40 Hz (no
separate notch, as the 40 Hz low-pass attenuates line noise); (5) **ballistocardiogram (BCG)
correction** by average-artifact-template subtraction time-locked to the ECG R-peaks (−0.2 to
0.6 s); (6) downsampling to 500 Hz; (7) independent component analysis (29 components, Picard) with
automatic artifact-component rejection via ICLabel together with cardiac (`find_bads_ecg`) and EOG
correlation; (8) spherical-spline interpolation of bad channels; and (9) re-referencing to the
**common average**. This yielded 500 Hz, average-referenced, 31-channel data for feature extraction.

*Per-TR EEG features.* Three feature sets were computed at the fMRI TR grid: (i) Hilbert band power
in δ/θ/α/β/γ per channel, HRF-convolved (band-power decoder); (ii) single-electrode Stockwell
time-frequency EEG-fingerprint features (EFP; Meir-Hasson et al., 2014); and (iii) sliding-window
specparam periodic/aperiodic (1/f) features (Donoghue et al., 2020).

### 2.6 fMRI processing and network timeseries

Functional data were preprocessed with fMRIPrep and denoised (**[confounds/GSR]**). Per-TR network
timeseries were extracted for the personalized DMN and CEN, PDA = CEN − DMN, and DiFuMo-64 parcels;
global-signal-regressed (GSR) variants were computed by residualizing on the fMRIPrep global
signal.

### 2.7 fMRI functional signal-to-noise ratio (f-SNR)

Following the law of total variance (Laukkonen 2026; Nath 2026), for each feedback run the within-
run condition z ∈ {rest (first 25 TR — the paradigm's own real-time baseline period, §2.2),
feedback (remaining volumes, first 5 dropped for HRF lag)}
gives signal = Var_z(E[r|z]) and noise = E_z(Var(r|z)); f-SNR = signal/noise (dB), computed for
PDA, CEN and DMN, plus a GLM/pseudo-target formulation (HRF-convolved rest→feedback boxcar) and a
causal running f-SNR for the real-time-feasible analysis.

### 2.8 EEG f-SNR and cross-modal matching

A pure EEG f-SNR was computed from EEG alone in two flavors — band-power running signal-to-noise,
and oscillatory÷aperiodic (specparam) — and matched (HRF-aligned, leak-free nested single-site
selection) to the fMRI PDA and fMRI f-SNR, within-subject, leave-one-subject-out, and cross-cohort
(DMNELF→rtBPD). A supervised single-electrode/multivariate decoder provided the fitted ceiling.

### 2.9 Statistics

Group inference used non-parametric sign-flip permutation tests on subject-level correlations;
electrode topographies used Benjamini–Hochberg FDR. **[preregistration: not preregistered
(secondary analysis of neurofeedback data).]** Data/code availability: **[repo statement].**

## 3. Results
**[TO DRAFT from committed analyses — fMRI f-SNR; EEG f-SNR within/LOSO/cross-cohort; EMG control.]**

*Clinical anchor — transdiagnostic (both cohorts).* Regulation of the DMN–CEN target related to
self-reported state calm in the **same direction in both clinical cohorts**: runs with more PDA
regulation were rated calmer (schizophrenia r=+0.21; borderline-trait r=+0.20), more mindfully
engaged (r=+0.42 / +0.11), and less difficult (r=−0.30 / −0.19). The four sliders showed a coherent,
cross-cohort-consistent structure (e.g. calm↔difficulty −0.45/−0.38; calm↔mindfulness +0.55/+0.16),
supporting construct validity. Between-subject, calmer participants tended to regulate the target
more (schizophrenia r=+0.37, n=13; borderline-trait r=+0.29–0.59 depending on the PDA metric,
n=18–25), a consistently positive but sample-limited association. The EEG frontal-theta f-SNR
tracked the BOLD PDA (r≈0.12–0.15) but did **not** by itself predict per-run calm (r≈0), bounding
the EEG marker as a scalable proxy for the (clinically-relevant) target rather than a direct outcome
predictor.

## 4. Discussion
**[TO DRAFT — transdiagnostic calibration-free marker; signal vs noise channel; limitations incl.
DMNELF short-rest = mask localizer (mild circularity); no healthy controls (no deficit claim);
sample sizes; EEG high-γ EMG caveat addressed by controls.]**

## References (to compile)
- Zhang J, et al. Reducing DMN connectivity with mbNF: a pilot in adolescents with affective
  disorder history. *Mol Psychiatry* 2023;28:2540–2548.
- Bauer CCC, et al. Real-time fMRI neurofeedback reduces auditory hallucinations… DMN. *Psychiatry
  Res* 2020;284:112770. (schizophrenia mbNF)
- Bauer CCC, et al. REMind: MURFI real-time functional imaging. 2022.
- Bauer CCC, et al. From state-to-trait meditation: reconfiguration of CEN and DMN. *eNeuro* 2019.
- Hinds O, et al. Computing moment-to-moment BOLD activation for real-time neurofeedback.
  *NeuroImage* 2011;54:361–368.
- Chen AC, et al. Causal interactions between fronto-parietal CEN and DMN. *PNAS* 2013.
- Meir-Hasson Y, et al. An EEG finger-print of fMRI deep-regional activation. *NeuroImage* 2014.
- Donoghue T, et al. Parameterizing neural power spectra into periodic and aperiodic components.
  *Nat Neurosci* 2020 (specparam/FOOOF).
- Laukkonen R. Clear Mind (f-SNR framework). 2026. — Nath et al. Meditation depth enhances f-SNR. 2026.
- Hamilton JP, et al. Depressive rumination, the DMN… *Biol Psychiatry* 2015. — Whitfield-Gabrieli
  & Ford. DMN in psychopathology. *Annu Rev Clin Psychol* 2012.

## Outstanding items / documents still needed
- ✅ **EEG acquisition + preprocessing** — written (§2.5) from the BrainAmp/BrainVision draft +
  verified against the deployed MNE pipeline (`eeg_preproc.py`): common-average reference, BCG
  correction present, 1 kHz→500 Hz, 1–40 Hz FIR, ICA/ICLabel. *(Confirm the deployed
  `mne_eeg_preprocessing` version matches this snapshot; confirm trigger/clock sync hardware.)*
- ✅ **MURFI real-time operationalization** — written (§2.2): incremental GLM (Gentleman),
  relative-displacement + drift nuisance, 25-volume/30 s baseline z-scoring, voxel-efficiency
  weighting, FLIRT mask registration w/ Yeo fallback, <1 TR latency. The 30 s baseline validates
  the f-SNR rest window (§2.7).
- ✅ **rtBPD state outcome** — per-run `slider_calm` (1–9) harvested from feedback event TSVs;
  PDA↔calm validated (§3). Still confirm **Cohort 1 schizophrenia symptom scale + meds** and the
  **rtBPD borderline-trait scale** name for trait-level biomarker analyses → §2.1.
- Confirm DMNELF vs rtBPD **fMRI protocol identity** (the shared printouts are the REMIND/rtBPD
  protocol) → Section 2.4 + Supplementary Table S1.
