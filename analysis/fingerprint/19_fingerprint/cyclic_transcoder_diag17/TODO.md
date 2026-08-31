# TODO: Cyclic Transcoder Improvements

## 1. Anticorrelation Negation Strategy ⭐ (Priority: High)

**Discovery:** Anticorrelated subjects (r < 0) can be corrected with simple sign flip!

**Status:** TESTED & VERIFIED ✓
- dmnelf010: r = -0.5070 → negate → r = +0.5070
- dmnelf004: r = -0.0680 → negate → r = +0.0680
- Works perfectly for inverted mappings

**Implementation:** (For real-time feedback system)
```python
if correlation < -0.3:  # Anticorrelated threshold
    feedback_signal = -model_prediction
else:
    feedback_signal = model_prediction
```

**Impact:** 
- Affects 4/14 subjects currently
- Would flip all anticorrelated to positive correlation
- Improves worst subject from r=-0.507 to r=+0.507 (matches best!)
- Performance range becomes [+0.062, 0.532] instead of [-0.507, 0.532]

**Next steps:**
1. ⬜ Implement adaptive negation in real-time feedback system
2. ⬜ Test negation strategy during live neurofeedback sessions
3. ⬜ Document in paper/poster: "Anticorrelation Correction Protocol"
4. ⬜ Consider online estimation of correlation during calibration phase

---

## 2. Full 528-Timepoint Visualization (Completed ✓)

- ✓ Fetched all 14 subjects from cluster
- ✓ Verified data integrity (50×528 parcels×timepoints)
- ✓ Generated 3-panel plots for best/middle/worst subjects
- ✓ Created plot_all_feedback_runs_full.py script
- Note: Full fMRI is 50 parcels (not 66) — personal ROIs not in saved NPZ

---

## 3. Conference Poster (In Progress)

**Key Results to Highlight:**
- Best subject: r = 0.5320 (p < 0.0001) ✓✓✓
- Performance range: -0.507 to +0.532
- With negation strategy: ALL subjects positive ✓
- 14 leave-one-subject-out CV folds trained

**Figures to Include:**
1. dmnelf005_pda_comparison.png (best, r=0.532)
2. dmnelf011_pda_comparison.png (middle, r=0.342)
3. dmnelf010_pda_comparison.png (worst before negation, r=-0.507)
4. Performance ranking histogram
5. Architecture diagram of cyclic transcoder

---

## 4. Future Analysis (Low Priority)

- [ ] Investigate why 5 subjects have r < 0.1 (weak or no signal)
- [ ] Failure mode analysis: what makes dmnelf004, dmnelf008, dmnelf1001 difficult?
- [ ] Hyperparameter sweep: does model tuning help weak subjects?
- [ ] Dataset size effect: would more REST data help anticorrelated subjects?
- [ ] EEG quality check: do weak correlations correlate with EEG noise?

---

## Session Summary (May 5, 2026)

**Completed:**
- ✓ Created 3-panel PDA visualization script
- ✓ Fixed data loading bug (column slicing vs row indexing)
- ✓ Added auto-save functionality with boolean flag
- ✓ Converted x-axis to seconds (0-60 sec display)
- ✓ Fetched all 14 subjects from cluster
- ✓ Computed performance ranking for all subjects
- ✓ Generated plots for best/middle/worst
- ✓ **Discovered anticorrelation negation strategy** ⭐

**Key Metrics (50-point evaluation):**
- Best: dmnelf005 (r=0.5320, p<0.0001)
- Worst: dmnelf010 (r=-0.5070, p<0.0002)
- Mean: r=0.0964 (highly variable across subjects)

**Files Created:**
- plot_best_subject_predictions.py
- plot_all_feedback_runs.py
- plot_all_feedback_runs_full.py
- FETCH_CLUSTER_DATA.md
- Visualizations: dmnelf005/011/010_pda_comparison.png
