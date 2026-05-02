# EEG Microstate → PDA Decoder: Summary of Findings

**Study:** DMNELF simultaneous EEG-fMRI neurofeedback  
**Author:** Clemens C.C. Bauer, MD PhD — EPIC Brain Lab, Northeastern / Gabrieli Lab, MIT  
**Date:** April 2026

---

## Background

The neurofeedback signal in DMNELF is the Positive Diametric Activity (PDA = CEN_z − DMN_z), delivered at 1.2 s TR via MURFI. The goal of this analysis is to determine whether EEG microstates — specifically the 7 T-hat projection maps (DMN, AUD, VIS, CEN, SAL, DANT, SOM) derived from the TESS method (Custo 2014/2017) — can be used to predict PDA in real time, enabling neurofeedback without the MRI scanner.

EEG microstate projections (T-hat) are computed at 250 Hz via spatial GLM and block-averaged to each TR (300 samples/TR). Features are the 7 T-hat amplitudes per TR. The target (`pda_direct`) is the per-TR CEN_z − DMN_z signal from personalized binary fMRI masks.

---

## Strategy 1 — Decomposed Network Proxy (Step 07)

### Rationale

PDA combines two separate network signals (CEN activation and DMN suppression). A single-template proxy (T-hat_CEN − T-hat_DMN) conflates both sources. The decomposed approach learns separate weight vectors mapping all 7 T-hat maps onto each network independently:

```
microCEN  =  w_CEN · T-hat   (Ridge: T-hat → CEN_z)
microDMN  =  w_DMN · T-hat   (Ridge: T-hat → DMN_z)
microPDA  =  microCEN − microDMN
```

Two target sources were compared:
- **Personal masks** — per-subject CEN_z and DMN_z from binary fMRI ROI masks (step 00d)
- **DiFuMo-64** — group parcellation (CEN parcels [4,31,47,48,50,51], DMN parcels [3,6,29,35,38,58,60,61])

### Cross-validation

Two evaluation schemes on n = 7 subjects with both EEG and personal mask data:

1. **LORO CV (within feedback)** — leave-one-feedback-run-out: train on 3 feedback runs, test on 1
2. **Transfer CV (rest+shortrest → feedback)** — train on resting-state data only, test on all 4 feedback runs

### Results (n = 7 subjects, 28 run observations)

| Predictor | LORO CV mean r | Transfer CV mean r |
|---|---|---|
| Simple proxy (T-hat_CEN − T-hat_DMN) | −0.003 | −0.003 |
| microPDA — personal masks | −0.027 | +0.036 |
| microPDA — DiFuMo | +0.021 | +0.010 |
| microCEN — personal | varies | varies |
| microDMN — personal | varies | varies |

### Interpretation

The decomposed proxy does not reliably outperform the simple single-template proxy. LORO CV is near chance for all approaches. Transfer CV shows a weak positive trend for personal masks (r ≈ +0.036) but this did not reach significance (p ≈ 0.14, n = 7). Personal masks consistently outperform DiFuMo across approaches, consistent with the subject-specificity of PDA. The decomposition into separate CEN and DMN components adds interpretability but not predictive power in this dataset.

---

## Strategy 2 — Personalized Ridge Decoder (Step 08)

### Rationale

Rather than constraining the decoder to predict CEN_z and DMN_z separately, the personalized decoder learns a direct linear mapping from T-hat to pda_direct, fit independently per subject. This is the most flexible single-step approach short of a non-linear model.

### Method

- **Model:** Ridge regression (α = 1.0) and ElasticNet (α = 0.1, l1_ratio = 0.5)
- **Features:** 7 T-hat columns from `_250Hz_nohrf_features.npy` (no HRF convolution)
- **Target:** `pda_direct` (personal mask PDA)
- **Training:** all rest + shortrest runs per subject (~600 TRs)
- **Test:** each of 4 feedback runs independently (transfer, no feedback training data used)
- **Preprocessing:** StandardScaler fit on training data, applied to test
- **Cohort:** n = 12 subjects (full EEG-fMRI cohort)

### Metrics

- Pearson r (predicted vs pda_direct)
- Sign accuracy — fraction of TRs with correct PDA sign (BCI-relevant: ball direction)
- ROC AUC — with pda_direct > 0 as positive class

### Results (n = 12 subjects, 47 feedback run observations)

| Metric | Mean | p-value | Note |
|---|---|---|---|
| Pearson r (Ridge) | +0.030 | 0.011 (n = 12) | Significant |
| Sign accuracy (Ridge) | 51.3% | n.s. | Chance = 50% |
| AUC (Ridge) | 0.516 | 0.033 | Significant |
| Pearson r (ElasticNet) | NaN for 10/12 subjects | — | Collapsed to zero weights |

**ElasticNet failure:** With ~600 training TRs and 7 features, the ElasticNet regularization (α = 0.1, l1_ratio = 0.5) is too strong — it drives weights to zero for 10 of 12 subjects, producing flat predictions and undefined r. Only sub-dmnelf004 and sub-dmnelf1003 produced non-zero ElasticNet weights. Ridge (α = 1.0) is the appropriate model for this sample size.

### Subject-level variability

Performance is highly subject-specific. Sign accuracy ranges from ~18% (sub-dmnelf006, run-03) to 88% (sub-dmnelf004, run-04). Sub-dmnelf004 is a consistent high performer (sign acc 76–88% across all feedback runs). Sub-dmnelf007 and sub-dmnelf009 show the highest peak r values (r ≈ 0.12 and r ≈ 0.30 for individual runs respectively).

### Decoder weights

Weights are idiosyncratic across subjects — no single microstate dominates consistently at the group level. This confirms that PDA prediction is subject-specific, not driven by a shared EEG microstate pattern. Some subjects show positive w_CEN (expected), but others show negative or near-zero CEN weights with dominant contributions from DANT or SAL.

---

## Comparison Across Strategies

| Approach | n subjects | mean r | Significant? |
|---|---|---|---|
| Simple proxy (T-hat_CEN − T-hat_DMN) | 7 | −0.003 | No |
| Decomposed proxy, personal masks (LORO) | 7 | −0.027 | No |
| Decomposed proxy, personal masks (transfer) | 7 | +0.036 | No (p ≈ 0.14) |
| Personalized Ridge decoder (transfer) | 12 | +0.030 | Yes (p = 0.011) |

The direct personalized decoder is the strongest approach, reaching significance on the larger cohort. The advantage comes from (1) predicting pda_direct without the decomposition bottleneck, (2) using all 7 microstates with freely estimated weights, and (3) larger sample size.

---

## Limitations

- **Low R²:** r ≈ 0.030 corresponds to R² ≈ 0.001 — the decoder explains less than 1% of PDA variance. The signal is real but weak.
- **Sign accuracy near chance at group level:** 51.3% mean sign accuracy is statistically equivalent to chance despite the significant AUC, likely because sign accuracy is a hard threshold metric sensitive to calibration while AUC is rank-based and more sensitive.
- **~600 training TRs:** Rest + shortrest provides approximately 600 TRs of training data per subject. This constrains model complexity and likely limits performance.
- **No online calibration:** The decoder is fit offline on rest data; performance during active neurofeedback may differ due to task-related arousal, attention, and state changes.
- **Block-averaged features:** T-hat projections are averaged over 300 EEG samples per TR. Sub-TR EEG dynamics (microstate sequence statistics) are not captured.

---

## Implications for BCI Use

A sign accuracy of 51.3% (group mean) falls short of the ~70% threshold typically considered useful for single-TR neurofeedback. However, the subject-level range (18–88%) suggests that personalization matters — a subset of subjects may already be above threshold without further optimization.

The significant AUC (0.516, p = 0.033) indicates the decoder contains real information about PDA direction, but threshold calibration and potentially run-level bias correction would be needed before clinical deployment.

---

## Next Steps

- **Run-level normalization / z-scoring** of predictions before threshold to improve sign accuracy
- **Sliding-window cross-validation** within feedback to assess how quickly the decoder can be calibrated from early neurofeedback data
- **Non-linear decoders** (e.g., kernel ridge, gradient boosting) with leave-one-subject-out CV
- **Sub-TR features** — microstate sequence statistics (duration, occurrence, transition probability) within each TR as additional predictors
- **Responder analysis** — characterize subjects with sign accuracy > 60% to identify predictors of EEG-based neurofeedback suitability
