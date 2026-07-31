#!/usr/bin/env python3
"""
rt_features.py  —  online per-TR EFP feature extraction (matches the frozen model layout)
-----------------------------------------------------------------------------------------
Accumulates streaming EEG into TR-length windows and, per TR, produces the sliding-delay design
the frozen ridge expects.

Two model shapes are supported, detected from the model npz:
  - montage models ("channels" key, e.g. efp_epoc_model.npz): one shared [10 band x 11 delay]
    design over a multi-channel montage (channel-major, delay-major, band-minor), with a
    common-average reference across the montage.
  - dual single-electrode models ("cen_channel"/"dmn_channel" keys, e.g.
    efp_epoc_dual_model.npz): each target reads its OWN electrode with its OWN frozen band
    edges, concatenated target-major into one design vector. No common-average reference is
    applied here — the validated offline pipeline (efp_meirhasson) never CARs single-electrode
    Stockwell input, so live single-electrode features must match that (CAR across two distant,
    unrelated electrodes like P8/O1 would also not be a meaningful reference anyway).

Per TR window (both shapes): per-channel demean (removes each channel's own persistent offset;
see below) -> Stockwell power (1-40 Hz, at the source sample rate; absolute scale is normalized
out by calibration z-scoring) -> average into the model's FIXED band edges -> one 10-band vector
per channel. A ring buffer of the last n_delays TR vectors forms the delay design. Emits a design
only once n_delays TRs of history are available.
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
        # dual single-electrode model (efp_epoc_dual_model.npz): CEN's own electrode block
        # concatenated with DMN's own electrode block, each with its own frozen band edges, no
        # CAR. Montage model (efp_epoc_model.npz / efp_cap31_model.npz): one shared channel list
        # + one shared band-edge set + a common-average reference across the montage.
        self.dual = "cen_channel" in model
        self.n_bands = int(model["n_bands"]); self.n_delays = int(model["n_delays"])
        self.tr = float(model["tr"]); self.fmin = int(model["fmin"]); self.fmax = int(model["fmax"])
        if self.dual:
            self.chans = [str(model["cen_channel"]), str(model["dmn_channel"])]
            self.band_edges = np.stack([np.asarray(model["cen_band_edges_hz"]),
                                        np.asarray(model["dmn_band_edges_hz"])])   # [2, n_bands, 2] Hz
            self.car = False
        else:
            self.chans = list(model["channels"])              # EPOC12/CAP31 order the model expects
            edges = np.asarray(model["band_edges_hz"])         # [n_bands, 2] Hz, shared across channels
            self.band_edges = np.broadcast_to(edges, (len(self.chans),) + edges.shape).copy()
            self.car = True
        # map model channels -> indices into the incoming sample vector
        self.pick = [source_channels.index(c) for c in self.chans if c in source_channels]
        if len(self.pick) != len(self.chans):
            missing = [c for c in self.chans if c not in source_channels]
            raise ValueError(f"source missing model channels: {missing}")
        # bad channels (flat / noisy contacts, flagged at calibration): dropped from the common-
        # average reference and given the good-channel-mean band power (neutral), so a dead felt
        # sensor degrades gracefully instead of corrupting the CAR + that channel's ridge weights.
        # Not applicable to dual models — each channel belongs to a DIFFERENT target, so there is
        # no neutral cross-channel fallback; a bad electrode there is a hard failure.
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
        if self.dual and bad & set(self.chans):
            raise ValueError(f"dual single-electrode model has no neutral fallback for a bad "
                             f"channel (each channel is a different target): {sorted(bad)}")
        self.bad_mask = np.array([c in bad for c in self.chans], bool)
        self.bad_channels = [c for c in self.chans if c in bad]
        if self.bad_mask.all():
            raise ValueError("all model channels flagged bad — cannot decode")

    def set_sfreq(self, sfreq):
        self.sfreq = float(sfreq); self._n_tr = int(round(self.sfreq * self.tr))

    def _bandpower(self, win):
        """win: [n_ch, n_samp] (already channel-picked). -> [n_ch, n_bands] band power for this TR."""
        good = ~self.bad_mask
        # per-channel demean FIRST: EmotivPRO's LSL "EEG" stream is raw/unreferenced, not filtered —
        # each channel sits at its own large, persistent offset (confirmed live: ~4200-4500 "µV",
        # vs. real scalp EEG hovering near 0), differing enough between channels (up to ~250 units)
        # that the cross-channel CAR below can't remove it. Only zeroes each window's own DC (0 Hz)
        # component, so genuine 1 Hz+ oscillatory content (the model's fmin) is unaffected.
        win = win - win.mean(axis=1, keepdims=True)
        if self.car:
            win = win - win[good].mean(axis=0, keepdims=True)  # common-average over GOOD channels
        bp = np.empty((win.shape[0], self.n_bands))
        for ci in range(win.shape[0]):
            if self.bad_mask[ci]:
                continue                                     # filled below with the good-channel mean
            freqs, power = stockwell_power(win[ci], self.sfreq, self.fmin, self.fmax)
            for bi, (lo, hi) in enumerate(self.band_edges[ci]):
                m = (freqs >= lo) & (freqs <= hi)
                bp[ci, bi] = power[m].mean() if m.any() else 0.0
        if self.bad_mask.any():
            bp[self.bad_mask] = bp[good].mean(axis=0)         # neutral fill for dead channels
        return bp

    def push(self, sample):
        """Add one multichannel sample. Returns a design vector [n_ch*n_delays*n_bands] on TR
        boundaries (else None) — 1320 for the 12-channel montage, 220 for the dual model
        (channel-major i.e. target-major: CEN's 110 features, then DMN's 110)."""
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
