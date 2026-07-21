#!/usr/bin/env python3
"""
calibration.py  —  per-session calibration for the EPOC NF decoder
------------------------------------------------------------------
One short calibration run (rest + light task) per subject/session removes headset- and
montage-specific gain that the frozen (DMNELF-trained) decoder cannot know about. We collect the
online design vectors during calibration and store per-feature mean/std used to z-score inputs at
run time (§6 of DEPLOY_EPOC.md). Also holds the rest-baseline of the fed-back PDA.
"""
from pathlib import Path
import numpy as np


class Calibrator:
    def __init__(self, n_features):
        self._X = []; self.mean = None; self.std = None
        self.pda_baseline_mean = 0.0; self.pda_baseline_sd = 1.0

    def add_design(self, design):
        self._X.append(np.asarray(design, float))

    def fit(self):
        if len(self._X) < 10:
            raise RuntimeError(f"calibration too short ({len(self._X)} TRs); need ≥10.")
        X = np.array(self._X)
        self.mean = X.mean(0)
        s = X.std(0)
        # Robustness: with few calibration TRs some per-feature stds are ~0 (a feature that barely
        # varied over the short window), which would explode the z-score of that feature at run
        # time. Floor each std at a fraction of the median std so no feature dominates.
        floor = 0.1 * float(np.median(s[s > 0])) if np.any(s > 0) else 1.0
        self.std = np.maximum(s, floor) + 1e-9
        return self

    def set_pda_baseline(self, pda_values):
        v = np.asarray(pda_values, float); v = v[np.isfinite(v)]
        if v.size:
            self.pda_baseline_mean = float(v.mean()); self.pda_baseline_sd = float(v.std() + 1e-9)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std,
                 pda_baseline_mean=self.pda_baseline_mean, pda_baseline_sd=self.pda_baseline_sd)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        c = cls(len(z["mean"])); c.mean = z["mean"]; c.std = z["std"]
        c.pda_baseline_mean = float(z["pda_baseline_mean"]); c.pda_baseline_sd = float(z["pda_baseline_sd"])
        return c
