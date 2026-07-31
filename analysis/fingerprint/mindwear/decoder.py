#!/usr/bin/env python3
"""
decoder.py  —  frozen EFP decoder: design vector -> CEN, DMN, PDA
-----------------------------------------------------------------
Applies the frozen CEN/DMN ridges to the online design. The design must be standardized to the
units the ridge expects (~per-run z-scored features). Standardization source, in order:
  1. a per-session Calibrator (mean/std from a calibration run) if provided, else
  2. a running (exponential) per-feature z-estimate that self-normalizes after a short warmup.
Then the model's pooled feat_mean/std (≈0/1) are applied and the ridge dotted. Returns raw
CEN(t), DMN(t) and PDA = CEN − DMN; feedback baselining is done downstream (run_nf).

Montage models (e.g. efp_epoc_model.npz) share ONE design across both targets (both ridges see
the same [1320] montage features). Dual single-electrode models (efp_epoc_dual_model.npz) instead
concatenate CEN's own 110-feature electrode block with DMN's own 110-feature block into one [220]
design (see rt_features.py) — each target must dot only its own slice, not the whole vector.
"""
import numpy as np


class RunningStats:
    """Exponential per-feature mean/variance for online standardization (warmup fallback)."""

    def __init__(self, n, halflife_tr=60):
        self.mean = np.zeros(n); self.var = np.ones(n); self.n = 0
        self.a = 1 - 0.5 ** (1.0 / max(halflife_tr, 1))

    def update(self, x):
        self.n += 1
        a = max(self.a, 1.0 / self.n)                        # fast average early on
        d = x - self.mean
        self.mean += a * d
        self.var = (1 - a) * (self.var + a * d * d)

    def z(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-9)


class Decoder:
    def __init__(self, model, calibration=None, halflife_tr=60):
        self.dual = "cen_channel" in model
        self.cen_coef = np.asarray(model["cen_coef"], float); self.cen_b = float(model["cen_intercept"])
        self.dmn_coef = np.asarray(model["dmn_coef"], float); self.dmn_b = float(model["dmn_intercept"])
        self.cen_fm = np.asarray(model["cen_feat_mean"], float); self.cen_fs = np.asarray(model["cen_feat_std"], float)
        self.dmn_fm = np.asarray(model["dmn_feat_mean"], float); self.dmn_fs = np.asarray(model["dmn_feat_std"], float)
        if self.dual:
            cen_off, dmn_off = int(model["cen_offset"]), int(model["dmn_offset"])
            self.cen_slice = slice(cen_off, cen_off + self.cen_coef.size)
            self.dmn_slice = slice(dmn_off, dmn_off + self.dmn_coef.size)
            self.design_size = int(model["total_features"])
        else:
            self.cen_slice = self.dmn_slice = slice(None)     # both targets share the whole design
            self.design_size = self.cen_coef.size
        self.calib = calibration
        self.run = None if calibration is not None else RunningStats(self.design_size, halflife_tr)

    def _standardize(self, design):
        if self.calib is not None:
            return (design - self.calib.mean) / (self.calib.std + 1e-9)
        self.run.update(design)
        return self.run.z(design)

    def predict(self, design):
        """design -> (cen, dmn, pda). Returns None during running-z warmup."""
        z = self._standardize(design)
        # warmup: RunningStats variance is unreliable from only a couple of samples regardless of
        # feature count, so this floors at 10 TRs rather than shrinking to ~1 for the much smaller
        # dual-electrode design (110 features -> size//100 would be 1, effectively no warmup).
        if self.calib is None and self.run.n < max(10, self.design_size // 100):
            return None                                      # brief warmup before stats stabilize
        z_cen, z_dmn = z[self.cen_slice], z[self.dmn_slice]
        cen = float(((z_cen - self.cen_fm) / self.cen_fs) @ self.cen_coef + self.cen_b)
        dmn = float(((z_dmn - self.dmn_fm) / self.dmn_fs) @ self.dmn_coef + self.dmn_b)
        return cen, dmn, cen - dmn
