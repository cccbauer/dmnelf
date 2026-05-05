# Cyclic Transcoder: Extraction, Training & Evaluation Summary

**Date:** May 5, 2026  
**Project:** Default Mode Network Electrical Fingerprint (DMNELF) — Cyclic Transcoder  
**Location:** `/Users/cccbauer/Documents/GitHub/dmnelf/analysis/fingerprint/cyclic_transcoder`

---

## Overview

This document summarizes the complete pipeline for extracting features, training models, generating predictions, and evaluating the cyclic transcoder on feedback runs.

**Pipeline Steps:**
1. ✅ Extract features (DiFuMo-64 + personal masks + EEG)
2. ✅ Train LOOCV models (EEG→fMRI transcoder)
3. ✅ Generate predictions on feedback runs (14 subjects)
4. ✅ Evaluate prediction accuracy
5. ✅ Generate performance report

---

## Architecture

**Cyclic Transcoder Model:**
- **Input:** EEG (31 channels, 500 Hz)
- **Output:** Predicted fMRI (66 DiFuMo parcels)
- **Derived Metric:** PDA = CEN activity - DMN activity
- **Validation:** Pearson correlation (pred vs true PDA)

**Data Pipeline:**
```
fMRI (rest)  →  ICA  →  Personal DMN/CEN masks
EEG (rest)   →  DiFuMo-64 features  →  Block averaging
                                ↓
                      LOOCV Training (13 leave-one-out models)
                                ↓
                    Apply to feedback EEG
                                ↓
                    Predict PDA on feedback
```

---

## Workflow & Commands

### Step 1: Feature Extraction

```bash
cd /Users/cccbauer/Documents/GitHub/dmnelf/analysis/fingerprint/cyclic_transcoder
bash scripts/deploy_and_run.sh --extract
```

**Output:** DiFuMo-64 fMRI timeseries + EEG block means for all subjects  
**Duration:** ~2 hours  
**Storage:** `/projects/swglab/data/DMNELF/derivatives/cyclic_features/sub-*/`

---

### Step 2: Model Training

```bash
bash scripts/deploy_and_run.sh --train
```

**Method:** Leave-one-out cross-validation (LOOCV)
- 14 models trained (one per subject)
- Each model: 13 subjects in training, 1 held out for testing
- GPU-accelerated (Tesla V100) on Explorer cluster

**Duration:** 12-24 hours per model  
**Parallel Jobs:** 14 concurrent  
**Storage:** `/projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder/checkpoints/`

---

### Step 3: Generate Predictions

```bash
ssh cccbauer@explorer.northeastern.edu \
  'cd /projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder && \
   sbatch predict_job.sh'
```

**Method:** Apply each subject's LOOCV model to their feedback run EEG
- Task: Feedback (neurofeedback session)
- Input: EEG only
- Output: Predicted PDA + predicted fMRI

**Duration:** ~2 hours total (14 jobs)  
**Output Format:** `.npz` files with:
- `pda_predicted`: Predicted PDA timeseries
- `fmri_predicted`: Predicted fMRI parcel timeseries (66, T)
- `fmri_true`: Ground truth fMRI (for reference)

**Storage:** `/projects/swglab/data/DMNELF/derivatives/cyclic_features/sub-*/predictions/`

---

### Step 4: Evaluation

**Method 1: On Explorer (Fast)**
```bash
bash evaluate_on_explorer.sh --plot --download
```

**Method 2: Full Workflow (with Summary)**
```bash
bash full_evaluation_workflow.sh --plot
```

**Metrics Computed:**
- **Pearson correlation** (r): How well pred matches true
- **Spearman correlation** (ρ): Rank-order correlation
- **R²:** Variance explained
- **RMSE:** Root mean squared error
- **MAE:** Mean absolute error
- **ROC AUC:** Binary classification performance (PDA > median)

---

## Results Summary

**Location:** `results/`

### Performance Statistics

| Metric | Mean | Std Dev | Min | Max |
|---|---|---|---|---|
| **Pearson r** | 0.067 | 0.330 | -0.507 | 0.532 |
| **R²** | -0.140 | 0.390 | -1.137 | 0.193 |
| **RMSE** | 0.39 | 0.22 | 0.13 | 0.88 |
| **Spearman ρ** | 0.099 | 0.336 | -0.625 | 0.604 |

**Interpretation:**
- Mean r = 0.067 indicates **weak average prediction accuracy**
- High variability (std = 0.33) suggests **strong subject-specific differences**
- Some subjects show good predictions (r > 0.4), others negative
- Possible causes: data quality issues, subject heterogeneity, model limitations

### Performance Tiers

**Excellent (r ≥ 0.4):**
- dmnelf005: r = 0.532 ✓

**Good (r ≥ 0.2):**
- dmnelf011: r = 0.342
- dmnelf012: r = 0.318
- dmnelf009: r = 0.253
- dmnelf1002: r = 0.275

**Poor (r < 0.0):**
- dmnelf010: r = -0.507
- dmnelf1003: r = -0.451
- dmnelf004: r = -0.068

### Files Generated

```
results/
├── evaluation_results.csv           # Per-subject metrics (CSV)
├── summary_correlations.png         # Bar chart of all correlations
└── evaluation_plots/
    ├── 01_metrics_distribution.png  # Histograms of metrics
    └── 02_subject_correlations.png  # Per-subject ranking
```

---

## Scripts Created

### Evaluation Pipeline

| Script | Purpose |
|---|---|
| `evaluate_predictions.py` | Compute metrics on .npz files |
| `evaluate_on_explorer.sh` | Run evaluation on cluster, download results |
| `summarize_results.py` | Generate text report + plots from CSV |
| `full_evaluation_workflow.sh` | End-to-end: evaluate + summarize + visualize |
| `download_predictions.sh` | Download prediction files to local |
| `evaluate_locally.sh` | Download and evaluate locally |

### Quick Commands

```bash
# View results
open results/

# Re-run full pipeline (on Explorer)
bash full_evaluation_workflow.sh --plot

# Regenerate summary report
python summarize_results.py --visualize

# Download specific results
bash evaluate_on_explorer.sh --download
```

---

## Next Steps

### Immediate Investigation

1. **Failure Analysis:** Why do dmnelf010, dmnelf1003, dmnelf004 have negative correlations?
   - Check data quality (missing timepoints, preprocessing issues)
   - Verify EEG-fMRI sync on feedback runs
   - Check for subject motion/artifacts

2. **Hyperparameter Tuning:** Current r=0.067 suggests room for improvement
   - Try different window sizes
   - Adjust learning rates
   - Experiment with model architecture

3. **Subject-Specific Analysis:** dmnelf005 (r=0.532) works well—why?
   - Compare EEG quality, fMRI SNR
   - Analyze network strength (DMN/CEN)
   - Check for idiosyncratic traits

### Extended Work

- [ ] Cross-subject generalization (train on subset, test on held-out subjects)
- [ ] Subject-specific models (within-subject cross-validation)
- [ ] Temporal dynamics (sliding window analysis)
- [ ] Comparison with baseline models (null, linear regression)
- [ ] Publication-quality figures and statistical tests

---

## Cluster Resources

**Explorer HPC:**
- Login: `ssh cccbauer@explorer.northeastern.edu`
- Conda env: `fingerprint`
- Base dir: `/projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder`
- Data dir: `/projects/swglab/data/DMNELF/derivatives/cyclic_features/`
- Job monitoring: `squeue -u cccbauer`
- Logs: `/projects/swglab/data/DMNELF/analysis/fingerprint/cyclic_transcoder/logs/`

**Local:**
- Project: `/Users/cccbauer/Documents/GitHub/dmnelf/analysis/fingerprint/cyclic_transcoder`
- Results: `results/`
- Config: `config.yaml`

---

## References

### Code
- Feature extraction: `data/extract_features.py`
- Training: `train.py`
- Prediction: `predict_pda.py`
- Models: `models/cyclic_transcoder.py`

### Data
- Config: `config.yaml`
- Subject list: `config.yaml` → `data.subjects.all`
- How-to guide: `HOWTO_new_subject.md`

---

**Summary:** Successfully completed cyclic transcoder pipeline from feature extraction through model training, prediction generation, and comprehensive evaluation. Results show subject-specific heterogeneity in EEG↔fMRI prediction accuracy, with best performer achieving r=0.53 and worst achieving r=-0.51. Next phase involves failure analysis and model refinement.
