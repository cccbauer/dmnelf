#!/usr/bin/env python3
"""
sources.py  —  pluggable EEG acquisition sources for the EPOC-X NF system
-------------------------------------------------------------------------
Uniform EEGSource interface so the rest of the pipeline is agnostic to where samples come
from. Three backends:
  CortexSource  — live EPOC X via the Emotiv Cortex API (cortex.py)
  LSLSource     — EmotivPRO's LSL outlet (pip install pylsl)
  ReplaySource  — a recorded .fif / .npz streamed at real-time rate (offline testing, no HW)

Each yields (t, sample) where sample is a float ndarray over `self.channels` in microvolts.
"""
import time
from abc import ABC, abstractmethod
import numpy as np

from cortex import EPOC_CHANNELS, CortexClient


class EEGSource(ABC):
    channels = None      # list[str]
    sfreq = None         # float, Hz

    @abstractmethod
    def open(self):
        ...

    @abstractmethod
    def samples(self):
        """Generator of (t: float, sample: np.ndarray[n_channels])."""
        ...

    def close(self):
        pass

    def __enter__(self):
        self.open(); return self

    def __exit__(self, *exc):
        self.close()


class CortexSource(EEGSource):
    def __init__(self, client_id, client_secret, license_id=None, headset_id=None):
        self.client = CortexClient(client_id, client_secret, license_id)
        self.headset_id = headset_id

    def open(self):
        self.client.connect()
        self.channels, self.sfreq = self.client.open_eeg_stream(self.headset_id)
        return self

    def samples(self):
        for frame in self.client.stream():
            yield frame["t"], np.array([frame["eeg"].get(c, np.nan) for c in self.channels], float)

    def close(self):
        self.client.close()


class LSLSource(EEGSource):
    """EmotivPRO -> LSL outlet. Enable the LSL outlet in EmotivPRO first."""

    def __init__(self, name_hint="EmotivDataStream-EEG"):
        self.name_hint = name_hint
        self._inlet = None
        self._pick = None

    def open(self):
        from pylsl import resolve_streams, StreamInlet
        streams = [s for s in resolve_streams() if s.type() == "EEG"]
        if not streams:
            raise RuntimeError("No LSL EEG stream. Enable the LSL outlet in EmotivPRO.")
        st = next((s for s in streams if self.name_hint in s.name()), streams[0])
        self._inlet = StreamInlet(st, max_chunklen=1)
        info = self._inlet.info(); self.sfreq = float(info.nominal_srate())
        # read channel labels from the stream description
        labels, ch = [], info.desc().child("channels").child("channel")
        for _ in range(info.channel_count()):
            labels.append(ch.child_value("label")); ch = ch.next_sibling()
        self._pick = [i for i, l in enumerate(labels) if l in EPOC_CHANNELS]
        self.channels = [labels[i] for i in self._pick]
        return self

    def samples(self):
        while True:
            sample, ts = self._inlet.pull_sample()
            if sample is not None:
                yield ts, np.array([sample[i] for i in self._pick], float)

    def close(self):
        self._inlet = None


class EmokitSource(EEGSource):
    """License-free raw EEG straight from the USB dongle via emokit (openyou/emokit).

    Bypasses Cortex/EmotivPRO entirely. Built for EPOC / EPOC+ — EPOC X support is not guaranteed
    (newer dongle); if the headset does not enumerate, use CyKit or the Cortex path instead.
    Needs:  pip install emokit hidapi   (+ `brew install hidapi` on macOS).  128 Hz, 14 channels.
    """

    def __init__(self, serial_number=None):
        self.serial_number = serial_number
        self._h = None

    def open(self):
        from emokit.emotiv import Emotiv
        self._h = Emotiv(display_output=False, verbose=False, serial_number=self.serial_number)
        try:
            self._h.__enter__()
        except Exception:
            pass                                   # some builds start the reader in __init__
        self.sfreq = 128.0
        # emokit exposes all 14 EPOC electrodes; keep those the model needs, in cap order
        self.channels = [c for c in EPOC_CHANNELS]
        return self

    def samples(self):
        while True:
            p = self._h.dequeue()
            if p is None:
                time.sleep(0.001); continue
            try:
                vals = [float(p.sensors[c]["value"]) for c in self.channels]
            except (KeyError, TypeError):
                continue
            yield time.time(), np.array(vals, float)

    def close(self):
        if self._h is not None:
            try:
                self._h.__exit__(None, None, None)
            except Exception:
                try:
                    self._h.close()
                except Exception:
                    pass


class ReplaySource(EEGSource):
    """Stream a recorded EEG file at real-time rate (or faster) for offline testing.

    path : .fif (MNE Raw) or .npz with {'data'[n_ch,n_samp], 'ch_names', 'sfreq'}.
    Restricts to EPOC channels present. speed>1 replays faster than real time; speed=0 = as-fast.
    """

    def __init__(self, path, speed=1.0):
        self.path = str(path); self.speed = speed
        self._data = None; self._t = None

    def open(self):
        if self.path.endswith(".fif"):
            import mne
            raw = mne.io.read_raw_fif(self.path, preload=True, verbose="ERROR")
            names = raw.ch_names; sf = raw.info["sfreq"]; data = raw.get_data() * 1e6   # V->uV
        else:
            z = np.load(self.path, allow_pickle=True)
            names = [str(c) for c in z["ch_names"]]; sf = float(z["sfreq"]); data = z["data"]
        pick = [i for i, n in enumerate(names) if n in EPOC_CHANNELS]
        if not pick:
            raise RuntimeError(f"No EPOC channels in {self.path} (have {names[:6]}…).")
        self.channels = [names[i] for i in pick]; self.sfreq = sf
        self._data = data[pick]                       # [n_epoc_ch, n_samp]
        return self

    def samples(self):
        dt = 1.0 / self.sfreq
        n = self._data.shape[1]; t0 = time.time()
        for k in range(n):
            if self.speed > 0:
                target = t0 + k * dt / self.speed
                sleep = target - time.time()
                if sleep > 0:
                    time.sleep(sleep)
            yield k * dt, self._data[:, k]

    def close(self):
        self._data = None
