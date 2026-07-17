#!/usr/bin/env python3
"""
eeg_activation_communicator.py  —  drop-in replacement for MurfiActivationCommunicator
--------------------------------------------------------------------------------------
The scanner ball-task reads network activation from MURFI via:
    communicator.update()
    communicator.get_roi_activation('cen'|'dmn', frame)   # nan until the volume is ready

This class exposes the SAME interface but is backed by the live EEG decoder (source ->
rt_features -> frozen ridge), so the existing paradigm runs on the EPOC X with a one-line swap.
A background thread ingests EEG continuously and produces one (cen, dmn) per TR; get_roi_activation
returns that TR's value once ready, else NaN — mirroring MURFI's per-volume availability.

CEN/DMN are the decoder's z-scored network estimates (the ball task forms PDA = |CEN − DMN| and
direction from which is higher, exactly as with MURFI).
"""
import threading
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rt_features import RTFeatureExtractor
from decoder import Decoder
from calibration import Calibrator


class EEGActivationCommunicator:
    def __init__(self, source, roi_names, tr, model_path=None, calib_path=None):
        """source: an un-opened EEGSource; roi_names: e.g. ['cen','dmn']."""
        self.source = source
        self.roi_names = [r.lower() for r in roi_names]
        self.tr = float(tr)
        self.model = np.load(model_path or (HERE / "model" / "efp_epoc_model.npz"), allow_pickle=True)
        self.calib = Calibrator.load(calib_path) if calib_path else None
        self._tr = []                 # list of dict{roi: value} per completed TR
        self._lock = threading.Lock()
        self._stop = False
        self._thread = None
        self.started = False

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start(self):
        self.source.open()
        self.feat = RTFeatureExtractor(self.model, self.source.channels)
        self.feat.set_sfreq(self.source.sfreq)
        self.decoder = Decoder(self.model, calibration=self.calib)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.started = True
        return self

    def _run(self):
        for _t, s in self.source.samples():
            if self._stop:
                break
            design = self.feat.push(s)
            if design is None:
                continue
            out = self.decoder.predict(design)
            if out is None:           # running-z warmup
                continue
            cen, dmn, _pda = out
            with self._lock:
                self._tr.append({"cen": float(cen), "dmn": float(dmn)})

    def stop(self):
        self._stop = True
        try:
            self.source.close()
        except Exception:
            pass

    # ── MURFI-compatible interface ────────────────────────────────────────
    def update(self):
        """No-op: the background thread advances state. Present for API compatibility."""
        return

    def get_roi_activation(self, roi_name, frame):
        """Return the ROI activation for TR index `frame`, or NaN if not yet available."""
        with self._lock:
            if 0 <= frame < len(self._tr):
                return self._tr[frame].get(roi_name.lower(), np.nan)
        return np.nan

    @property
    def n_ready(self):
        with self._lock:
            return len(self._tr)
