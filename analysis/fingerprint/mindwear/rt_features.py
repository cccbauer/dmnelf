#!/usr/bin/env python3
"""
rt_features.py  —  online per-TR EFP feature extraction (matches the frozen model layout)
-----------------------------------------------------------------------------------------
Accumulates streaming EEG into TR-length windows and, per TR, produces the [10 band x 11 delay]
sliding-delay design over the EPOC-12 montage — the exact 1320-feature vector the frozen ridge
expects (channel-major, delay-major, band-minor).

Per TR window: common-average re-reference -> Stockwell power (1-40 Hz, at the source sample rate;
absolute scale is normalized out by calibration z-scoring) -> average into the model's FIXED band
edges -> one 10-band vector per channel. A ring buffer of the last n_delays TR vectors forms the
delay design. Emits a design only once n_delays TRs of history are available.
"""
from pathlib import Path
import sys
import numpy as np

# Stockwell transform is vendored into the package (mindwear/stockwell.py) so the live decoder is
# self-contained; fall back to the research-repo copy if the vendored one is somehow missing.
try:
    from stockwell import stockwell_power
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "efp_meirhasson" / "scripts"))
    from stockwell import stockwell_power


class RTFeatureExtractor:
    def __init__(self, model, source_channels, bad_channels=None):
        self.chans = list(model["channels"])                 # EPOC12 order the model expects
        self.n_bands = int(model["n_bands"]); self.n_delays = int(model["n_delays"])
        self.tr = float(model["tr"]); self.fmin = int(model["fmin"]); self.fmax = int(model["fmax"])
        self.band_edges = np.asarray(model["band_edges_hz"])  # [n_bands, 2] Hz
        # map model channels -> indices into the incoming sample vector
        self.pick = [source_channels.index(c) for c in self.chans if c in source_channels]
        if len(self.pick) != len(self.chans):
            missing = [c for c in self.chans if c not in source_channels]
            raise ValueError(f"source missing EPOC channels: {missing}")
        # bad channels (flat / noisy contacts, flagged at calibration): dropped from the common-
        # average reference and given the good-channel-mean band power (neutral), so a dead felt
        # sensor degrades gracefully instead of corrupting the CAR + that channel's ridge weights.
        self.set_bad_channels(bad_channels)
        # Per-TR band power is estimated over a CAUSAL rolling window of `window_tr` TRs. A 2-TR
        # window looked like a big win on dmnelf005 (PDA 0.17->0.25) but a cross-cohort sweep over
        # 63 DMNELF subject-runs showed it does NOT generalize (PDA Δr=-0.010, p=0.46; only ~half
        # of runs improved) — 1 TR is as good or slightly better on average. So default to 1 TR
        # (identical to the original per-TR window); overridable via model["window_tr"].
        self.window_tr = int(model["window_tr"]) if "window_tr" in model else 1
        self.sfreq = None; self._raw = []                    # rolling raw samples (last window_tr TRs)
        self._count = 0                                      # samples since the last TR boundary
        self._ring = []                                      # list of [n_ch, n_bands] per past TR

    def set_bad_channels(self, bad_channels):
        """Mark model channels (by name) as bad; they are excluded from CAR + neutral-filled."""
        bad = set(bad_channels or [])
        self.bad_mask = np.array([c in bad for c in self.chans], bool)
        self.bad_channels = [c for c in self.chans if c in bad]
        if self.bad_mask.all():
            raise ValueError("all EPOC channels flagged bad — cannot decode")

    def set_sfreq(self, sfreq):
        self.sfreq = float(sfreq); self._n_tr = int(round(self.sfreq * self.tr))

    def _bandpower(self, win):
        """win: [n_ch, n_samp] (already channel-picked). -> [n_ch, n_bands] band power for this TR."""
        good = ~self.bad_mask
        win = win - win[good].mean(axis=0, keepdims=True)     # common-average over GOOD channels
        bp = np.empty((win.shape[0], self.n_bands))
        for ci in range(win.shape[0]):
            if self.bad_mask[ci]:
                continue                                     # filled below with the good-channel mean
            freqs, power = stockwell_power(win[ci], self.sfreq, self.fmin, self.fmax)
            for bi, (lo, hi) in enumerate(self.band_edges):
                m = (freqs >= lo) & (freqs <= hi)
                bp[ci, bi] = power[m].mean() if m.any() else 0.0
        if self.bad_mask.any():
            bp[self.bad_mask] = bp[good].mean(axis=0)         # neutral fill for dead channels
        return bp

    def push(self, sample):
        """Add one multichannel sample. Returns a design vector [1320] on TR boundaries (else None)."""
        if self.sfreq is None:
            raise RuntimeError("call set_sfreq() before push()")
        self._raw.append(np.asarray(sample, float)[self.pick])
        self._count += 1
        if self._count < self._n_tr:
            return None
        self._count = 0
        win_len = self.window_tr * self._n_tr                # causal rolling window (window_tr TRs)
        if len(self._raw) > win_len:
            self._raw = self._raw[-win_len:]
        win = np.array(self._raw).T                          # [n_ch, up to win_len]
        bp = self._bandpower(win)                            # [n_ch, n_bands]
        self._ring.append(bp)
        if len(self._ring) > self.n_delays:
            self._ring.pop(0)
        if len(self._ring) < self.n_delays:
            return None
        # design row: per channel [lag0 bands, lag1 bands, ...]; lag d = TR (t-d)
        rows = []
        for ci in range(len(self.chans)):
            for d in range(self.n_delays):
                rows.append(self._ring[-1 - d][ci])          # newest first = lag 0
        return np.concatenate(rows).astype(np.float64)       # [n_ch * n_delays * n_bands] = 1320
