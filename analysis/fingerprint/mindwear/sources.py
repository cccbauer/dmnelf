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
import contextlib
import queue
import re
import threading
import time
from abc import ABC, abstractmethod
import numpy as np

from cortex import EPOC_CHANNELS, CortexClient

# EmotivPRO's LSL Outlet publishes one stream per data type when enabled (Motion, Performance
# Metrics, Band Power, Contact/EEG Quality, ...) alongside the main EEG stream. Exact name()/type()
# strings vary a bit across EmotivPRO versions, so streams are classified by keyword match against
# alnum tokens of "<name> <type>" (not raw substring — avoids false hits like "eq" inside
# "frequency") rather than hardcoded exact names. LSLSource logs what it actually classified each
# stream as, so a first live run is self-documenting.
AUX_STREAM_KEYWORDS = {
    "motion": ("motion", "mot"),
    "metrics": ("performance", "metric", "met"),
    "bandpower": ("band", "pow"),
    "quality": ("quality", "contact", "eq", "cq", "device", "dev"),
}


def _tokens(*parts):
    return re.findall(r"[a-z0-9]+", " ".join(p or "" for p in parts).lower())


def _lsl_channel_labels(info):
    """Channel labels from an LSL StreamInfo's <channels><channel><label> XML, one per channel."""
    labels, ch = [], info.desc().child("channels").child("channel")
    for _ in range(info.channel_count()):
        labels.append(ch.child_value("label"))
        ch = ch.next_sibling()
    return labels


def dump_lsl_streams(on_status=None):
    """Report every stream pylsl currently sees on the network — name, type, channel count/format,
    rate, and channel labels — regardless of whether LSLSource's aux-stream classifier recognizes
    it. ``on_status``, if given, gets one call per line (so a GUI can log it); otherwise prints.
    Use this whenever "what's actually coming over LSL" is in question — e.g. a raw/unreferenced
    stream (values sitting at a large constant offset instead of near 0 µV) looks identical to our
    classifier as a properly-processed one; this shows exactly what's on the wire, unfiltered."""
    from pylsl import resolve_streams
    emit = on_status or print
    streams = resolve_streams()
    if not streams:
        emit("no LSL streams found — is EmotivPRO's LSL outlet enabled?")
        return
    emit(f"{len(streams)} LSL stream(s) found:")
    for s in streams:
        labels = []
        with contextlib.suppress(Exception):
            labels = _lsl_channel_labels(s)
        emit(f"  - name={s.name()!r} type={s.type()!r} channels={s.channel_count()} "
            f"srate={s.nominal_srate():g} format={s.channel_format()}"
            + (f"  labels={labels}" if labels else ""))


class _AuxInlet:
    """Background-thread reader for one auxiliary LSL stream (motion/metrics/bandpower/quality).

    These run at their own, lower, often irregular rates alongside EEG, so each gets its own
    inlet + thread + queue rather than being interleaved into the main (EEG-cadence) sample loop.
    pull_sample() returns plain Python floats regardless of the stream's underlying channel format
    (cf_float32 vs cf_double64), so no per-format branching is needed here.
    """

    def __init__(self, stream_info):
        from pylsl import StreamInlet
        self.name = stream_info.name()
        self.channels = _lsl_channel_labels(stream_info)
        self._inlet = StreamInlet(stream_info, max_chunklen=1)
        self._q: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                sample, ts = self._inlet.pull_sample(timeout=1.0)
            except Exception:
                break
            if sample is not None:
                self._q.put((ts, np.asarray(sample, float)))

    def drain(self):
        """Pop everything queued so far, non-blocking: [(t, values), ...]."""
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        with contextlib.suppress(Exception):
            self._inlet.close_stream()


class EEGSource(ABC):
    channels = None      # list[str]
    sfreq = None         # float, Hz
    aux_channels: dict = {}   # {category: [channel names]} for sources with aux LSL streams

    @abstractmethod
    def open(self):
        ...

    @abstractmethod
    def samples(self):
        """Generator of (t: float, sample: np.ndarray[n_channels])."""
        ...

    def flush(self):
        """Discard any samples the source has buffered while nothing was consuming it (e.g. the
        engine paused at calib_review/ready). Live sources keep producing in real time regardless
        of whether we're pulling — without this, resuming would rapid-fire through the backlog
        instead of picking up at "now", making timed phases (rest/feedback) finish far too fast.
        No-op by default; sources that can actually buffer override it."""
        pass

    def drain_aux(self, category):
        """Everything queued so far for an aux stream category, non-blocking. No-op by default —
        only sources that actually found aux LSL streams (currently LSLSource) override this."""
        return []

    def flush_aux(self):
        """Discard queued aux samples — same reasoning as flush(), for the aux streams."""
        pass

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

    def flush(self):
        self.client.flush()

    def close(self):
        self.client.close()


class LSLSource(EEGSource):
    """EmotivPRO -> LSL outlet. Enable the LSL outlet in EmotivPRO first.

    Also picks up whichever auxiliary EmotivPRO streams are present alongside EEG — Motion,
    Performance Metrics, Band Power, Contact/EEG Quality — each on its own background thread (see
    _AuxInlet), since they run at their own, lower rates. `samples()` still yields only EEG,
    unchanged, so the feature-extractor/decoder pipeline is unaffected; the aux streams are read
    via `drain_aux(category)` / `aux_channels`.
    """

    def __init__(self, name_hint="EmotivDataStream-EEG", on_status=None):
        self.name_hint = name_hint
        self._on_status = on_status      # optional callable(str) to report what was found
        self._inlet = None
        self._pick = None
        self.aux_channels = {}
        self._aux: dict = {}

    def _status(self, msg):
        if self._on_status:
            self._on_status(msg)

    def open(self):
        from pylsl import resolve_streams, StreamInlet
        streams = resolve_streams()
        eeg_idx = [i for i, s in enumerate(streams) if s.type().lower() == "eeg"]
        if not eeg_idx:
            raise RuntimeError("No LSL EEG stream. Enable the LSL outlet in EmotivPRO.")
        eeg_i = next((i for i in eeg_idx if self.name_hint in streams[i].name()), eeg_idx[0])
        st = streams[eeg_i]
        self._inlet = StreamInlet(st, max_chunklen=1)
        info = self._inlet.info(); self.sfreq = float(info.nominal_srate())
        labels = _lsl_channel_labels(info)
        self._pick = [i for i, l in enumerate(labels) if l in EPOC_CHANNELS]
        self.channels = [labels[i] for i in self._pick]
        self._status(f"LSL EEG stream: '{st.name()}' ({len(self.channels)} ch @ {self.sfreq:g} Hz)")

        # opportunistically pick up whichever aux streams EmotivPRO is also publishing
        claimed = {eeg_i}
        for category, keywords in AUX_STREAM_KEYWORDS.items():
            toks_by_i = {i: _tokens(s.name(), s.type()) for i, s in enumerate(streams) if i not in claimed}
            match_i = next((i for i, toks in toks_by_i.items()
                           if any(kw in tok for tok in toks for kw in keywords)), None)
            if match_i is None:
                self._status(f"LSL aux stream not found: {category} (skipping)")
                continue
            claimed.add(match_i)
            aux = _AuxInlet(streams[match_i])
            self._aux[category] = aux
            self.aux_channels[category] = aux.channels
            self._status(f"LSL aux stream: {category} <- '{streams[match_i].name()}' "
                         f"({len(aux.channels)} ch)")
        return self

    def samples(self):
        while True:
            sample, ts = self._inlet.pull_sample()
            if sample is not None:
                yield ts, np.array([sample[i] for i in self._pick], float)

    def drain_aux(self, category):
        aux = self._aux.get(category)
        return aux.drain() if aux else []

    def flush_aux(self):
        for aux in self._aux.values():
            aux.drain()

    def flush(self):
        if self._inlet is not None:
            self._inlet.flush()

    def close(self):
        self._inlet = None
        for aux in self._aux.values():
            aux.close()
        self._aux = {}


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
        self._k = 0            # current playback position (persists across a paused engine)
        self._t0 = None        # wall-clock pacing reference

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
        self._k = 0; self._t0 = None
        return self

    def samples(self):
        dt = 1.0 / self.sfreq
        n = self._data.shape[1]
        if self._t0 is None:
            self._t0 = time.time()
        while self._k < n:
            k = self._k
            if self.speed > 0:
                target = self._t0 + k * dt / self.speed
                sleep = target - time.time()
                if sleep > 0:
                    time.sleep(sleep)
            self._k += 1
            yield k * dt, self._data[:, k]

    def flush(self):
        """Re-sync pacing to now, so a real-time (speed>0) replay doesn't rapid-fire through
        "backlog" that only exists because nothing was consuming it during a pause."""
        if self._t0 is not None and self.speed > 0:
            dt = 1.0 / self.sfreq
            self._t0 = time.time() - self._k * dt / self.speed

    def close(self):
        self._data = None
