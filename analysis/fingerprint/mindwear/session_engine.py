#!/usr/bin/env python3
"""
session_engine.py  —  headless real-time NF engine the GUI (and stimulus) drive
-------------------------------------------------------------------------------
Refactors the ``run_nf.py`` orchestration loop into a start/stop ``SessionEngine`` class so the
Flet operator console, the PsychoPy stimulus, and the command-line runner all share one engine.

The engine runs the ``source -> rt_features -> decoder`` loop on a background thread and advances a
small state machine of phases:

    connect  -> (calibrate) -> rest -> feedback -> done

Per TR it emits a :class:`TRUpdate` to a user callback (thread-safe to marshal onto the GUI/main
thread) and appends a row to a per-run CSV. The latest fed-back signal is also exposed via
:meth:`latest` / :attr:`pda_z` so a MURFI-style stimulus can poll it exactly like
``get_roi_activation`` — the engine is display-agnostic.

Nothing here imports Flet or PsychoPy; it is pure numpy + the existing mindwear modules, so it runs
headless (ReplaySource) with no hardware or display and is unit-testable.
"""
from __future__ import annotations

import contextlib
import csv
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "model" / "efp_epoc_model.npz"

# phases, in order. "calib_review": paused after calibration, awaiting operator confirm/retry.
# "ready": paused after review is confirmed, awaiting the participant (spacebar in the stimulus).
PHASES = ("connect", "calibrate", "calib_review", "ready", "rest", "feedback", "done")


def _score_calibration(cal_obj) -> dict:
    """Lightweight QA summary of a just-fitted Calibrator, for the operator's review step.

    Not a pass/fail gate — the operator decides whether to retry from what's shown. Flags
    features that barely varied during calibration (raw std before the robustness floor), which
    can mean the subject held very still (fine) or a channel wasn't really contributing (worth
    a re-check).
    """
    X = np.array(cal_obj._X)                    # [n_tr, n_features], pre-floor
    raw_std = X.std(0)
    n_flat = int(np.sum(raw_std < 1e-6))
    return {
        "n_tr": int(X.shape[0]),
        "n_features": int(raw_std.size),
        "n_flat_features": n_flat,
        "pct_flat": 100.0 * n_flat / max(raw_std.size, 1),
    }


@dataclass
class TRUpdate:
    """One decoded TR handed to the GUI/stimulus."""

    tr: int                 # index within the current phase
    phase: str              # one of PHASES
    cen: float
    dmn: float
    pda: float
    pda_z: float            # feedback signal vs rest baseline (NaN outside feedback)
    t: float                # wall-clock seconds since engine start


@dataclass
class EngineConfig:
    """Everything the engine needs for one run. Populated from the GUI StudyConfig or CLI args."""

    subject: str = "P000"
    run: int = 1
    model_path: str = str(DEFAULT_MODEL)

    # source
    source: str = "replay"                      # replay | cortex | lsl | emokit
    replay_path: Optional[str] = None
    replay_speed: float = 1.0                    # 0 = as-fast-as-possible (tests)
    credentials_path: Optional[str] = None       # yaml with cortex client_id/secret/...

    # phases (seconds)
    do_calibrate: bool = True
    calib_path: Optional[str] = None             # load an existing calibration instead
    calib_sec: float = 60.0
    rest_sec: float = 30.0
    feedback_sec: float = 300.0

    # bad channels to drop (flat/noisy contacts flagged at calibration)
    bad_channels: Optional[list] = None

    # logging
    log_dir: str = str(HERE / "logs")

    def resolved_calib_save(self) -> Path:
        return HERE / "model" / f"calib_{self.subject}.npz"


def make_source(cfg: "EngineConfig"):
    """Construct (but do not open) the EEGSource for *cfg*. Shared by the engine and the GUI probe."""
    import sys

    import yaml

    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from sources import CortexSource, EmokitSource, LSLSource, ReplaySource

    c: dict = {}
    cred = Path(cfg.credentials_path) if cfg.credentials_path else (HERE / "credentials.yaml")
    if cred.exists():
        c = yaml.safe_load(cred.read_text()) or {}

    s = cfg.source
    if s == "cortex":
        return CortexSource(c.get("client_id"), c.get("client_secret"),
                            c.get("license_id"), c.get("headset_id"))
    if s == "lsl":
        return LSLSource()
    if s == "emokit":
        return EmokitSource(c.get("emokit_serial"))
    if not cfg.replay_path:
        raise ValueError("replay source requires replay_path")
    p = Path(cfg.replay_path)
    if not p.is_absolute() and not p.exists():
        # resolve relative paths robustly regardless of CWD: try repo-anchored forms
        for base in (Path.cwd(), HERE, HERE.parent):
            cand = base / cfg.replay_path
            if cand.exists():
                p = cand
                break
    if not p.exists():
        raise FileNotFoundError(f"replay file not found: {cfg.replay_path} (cwd={Path.cwd()})")
    return ReplaySource(str(p), speed=cfg.replay_speed)


# RMS bands (µV) shared by the contact check: below FLAT_UV ≈ disconnected, above EXTREME_UV ≈
# railing/noisy. Channels outside this band are dropped from the decode (common-average + ridge).
FLAT_UV, EXTREME_UV = 0.5, 200.0


def score_contact(rms_by_channel: dict) -> list:
    """Channel names whose RMS falls outside the good contact band (see FLAT_UV/EXTREME_UV)."""
    return [ch for ch, r in rms_by_channel.items() if r < FLAT_UV or r > EXTREME_UV]


def stream_contact(cfg: "EngineConfig", on_window, stop_event: threading.Event,
                    window_sec: float = 0.25) -> dict:
    """Open the source and continuously score contact quality, live — the EmotivPRO-style preview.

    Every ``window_sec`` seconds, calls ``on_window(channels, sfreq, X)`` with the raw window
    ``X`` [n_samp, n_ch] in µV (so the caller can both score RMS and render a live scrolling
    trace), until ``stop_event`` is set or the source errors. Returns ``{"error"}`` on failure —
    the source is always closed on the way out. Does not touch the decoder.
    """
    out: dict = {"error": None}
    src = None
    try:
        src = make_source(cfg).open()
        n = max(1, int(round(src.sfreq * window_sec)))
        buf = []
        for _t, sample in src.samples():
            if stop_event.is_set():
                break
            buf.append(np.asarray(sample, float))
            if len(buf) >= n:
                on_window(list(src.channels), float(src.sfreq), np.array(buf))
                buf = []
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if src is not None:
            try:
                src.close()
            except Exception:
                pass
    return out


class SessionEngine:
    """Drive one NF run on a background thread, emitting per-TR updates.

    Parameters
    ----------
    config : EngineConfig
    on_update : callable(TRUpdate) -> None, optional
        Invoked once per decoded TR (from the worker thread — marshal to the GUI thread yourself).
    on_phase : callable(str) -> None, optional
        Invoked when the phase changes (``connect``/``calibrate``/``rest``/``feedback``/``done``).
    on_status : callable(str) -> None, optional
        Human-readable progress/error strings for the operator log.
    """

    def __init__(
        self,
        config: EngineConfig,
        on_update: Optional[Callable[[TRUpdate], None]] = None,
        on_phase: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.cfg = config
        self._on_update = on_update
        self._on_phase = on_phase
        self._on_status = on_status

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self.phase: str = "connect"
        self.channels: list[str] = []
        self.sfreq: Optional[float] = None
        self.tr: Optional[float] = None
        self.error: Optional[str] = None
        self.log_path: Optional[Path] = None
        # True only if feedback ran its full course (not stopped early by the operator or an
        # error) — distinct from phase == "done", which is also reached on an early stop.
        self.completed: bool = False

        # latest fed-back state (thread-safe poll for a stimulus)
        self._latest: Optional[TRUpdate] = None
        self._history: list[TRUpdate] = []
        self.baseline_mean: float = 0.0
        self.baseline_sd: float = 1.0

        # calibration review gate: the worker pauses in phase "calib_review" after fitting the
        # calibrator and waits here for the operator to call confirm_calibration() (proceed to
        # rest/feedback) or retry_calibration() (redo the calibration phase from scratch).
        self._await_confirm = threading.Event()
        self._retry_calibration = False
        self.calib_summary: Optional[dict] = None

        # participant-ready gate: the worker pauses in phase "ready" (after the calibration review
        # is confirmed) until the stimulus calls participant_ready() (spacebar pressed).
        self._await_ready = threading.Event()

    # ── public control ───────────────────────────────────────────────────
    def start(self) -> "SessionEngine":
        if self._thread and self._thread.is_alive():
            raise RuntimeError("engine already running")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="SessionEngine", daemon=True)
        self._thread.start()
        return self

    def stop(self, join: bool = True, timeout: float = 5.0) -> None:
        self._stop.set()
        self._await_confirm.set()          # unblock if paused awaiting calibration review
        self._await_ready.set()            # unblock if paused awaiting the participant
        if join and self._thread:
            self._thread.join(timeout)

    def confirm_calibration(self) -> None:
        """Operator accepted the calibration — proceed to the participant-ready gate."""
        self._retry_calibration = False
        self._await_confirm.set()

    def retry_calibration(self) -> None:
        """Operator wants to redo calibration from scratch."""
        self._retry_calibration = True
        self._await_confirm.set()

    def participant_ready(self) -> None:
        """Participant pressed spacebar at the instructions screen — proceed to rest/feedback."""
        self._await_ready.set()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def latest(self) -> Optional[TRUpdate]:
        with self._lock:
            return self._latest

    @property
    def pda_z(self) -> float:
        u = self.latest()
        return float(u.pda_z) if (u and np.isfinite(u.pda_z)) else 0.0

    def get_roi_activation(self, roi: str) -> float:
        """MURFI-style poll: latest z-scored CEN or DMN (NaN if not ready)."""
        u = self.latest()
        if u is None:
            return float("nan")
        return {"cen": u.cen, "dmn": u.dmn, "pda": u.pda}.get(roi.lower(), float("nan"))

    def history(self) -> list[TRUpdate]:
        with self._lock:
            return list(self._history)

    # ── internals ────────────────────────────────────────────────────────
    def _status(self, msg: str) -> None:
        if self._on_status:
            self._on_status(msg)

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        if self._on_phase:
            self._on_phase(phase)

    def _make_source(self):
        return make_source(self.cfg)

    def _run(self) -> None:
        import sys

        sys.path.insert(0, str(HERE))
        from calibration import Calibrator
        from decoder import Decoder
        from rt_features import RTFeatureExtractor

        t0 = time.time()
        src = None
        writer = None
        fh = None
        try:
            model = np.load(self.cfg.model_path, allow_pickle=True)
            self.tr = float(model["tr"])

            self._set_phase("connect")
            self._status(f"opening source: {self.cfg.source}")
            src = self._make_source().open()
            self.channels = list(src.channels)
            self.sfreq = float(src.sfreq)
            self._status(f"connected — {len(self.channels)} ch @ {self.sfreq:g} Hz, TR {self.tr:g}s")

            feat = RTFeatureExtractor(model, src.channels, bad_channels=self.cfg.bad_channels)
            feat.set_sfreq(src.sfreq)
            if feat.bad_channels:
                self._status(f"dropping {len(feat.bad_channels)} bad channel(s): {', '.join(feat.bad_channels)}")

            # CSV log
            Path(self.cfg.log_dir).mkdir(parents=True, exist_ok=True)
            stamp = int(t0)
            self.log_path = Path(self.cfg.log_dir) / f"nf_{self.cfg.subject}_run-{self.cfg.run:02d}_{stamp}.csv"
            fh = open(self.log_path, "w", newline="")
            writer = csv.writer(fh)
            writer.writerow(["tr", "phase", "cen", "dmn", "pda", "pda_z", "t"])

            # phase setup
            n_features = int(np.asarray(model["cen_coef"]).shape[0])
            calib = Calibrator.load(self.cfg.calib_path) if self.cfg.calib_path else None
            do_cal = self.cfg.do_calibrate and calib is None
            decoder = None if do_cal else Decoder(model, calibration=calib)
            cal_obj = Calibrator(n_features) if do_cal else None

            n_cal = round(self.cfg.calib_sec / self.tr)
            n_rest = round(self.cfg.rest_sec / self.tr)
            n_fb = round(self.cfg.feedback_sec / self.tr)

            if do_cal:
                self._set_phase("calibrate")
                self._status(f"calibrating — hold still ({self.cfg.calib_sec:.0f}s)")
            else:
                self._set_phase("rest")
                self._status(f"rest baseline ({self.cfg.rest_sec:.0f}s)")

            rest_pda: list[float] = []
            k = 0
            for _t, sample in src.samples():
                if self._stop.is_set():
                    self._status("stopped by operator")
                    break
                design = feat.push(sample)
                if design is None:
                    continue

                # ---- calibrate ----
                if self.phase == "calibrate":
                    cal_obj.add_design(design)
                    k += 1
                    if k >= n_cal:
                        cal_obj.fit()
                        cal_obj.save(self.cfg.resolved_calib_save())
                        decoder = Decoder(model, calibration=cal_obj)
                        self.calib_summary = _score_calibration(cal_obj)
                        self._status(f"calibration done ({k * self.tr:.0f}s) -> "
                                    f"{self.cfg.resolved_calib_save().name} — awaiting review")
                        self._set_phase("calib_review")
                        self._await_confirm.clear()
                        self._await_confirm.wait()
                        with contextlib.suppress(Exception):
                            src.flush()   # discard backlog the source kept producing while paused
                        if self._stop.is_set():
                            self._status("stopped by operator")
                            break
                        if self._retry_calibration:
                            self._retry_calibration = False
                            self._status("repeating calibration")
                            cal_obj = Calibrator(n_features)
                            decoder = None
                            k = 0
                            self._set_phase("calibrate")
                            continue
                        self._status("calibration accepted — waiting for the participant")
                        self._set_phase("ready")
                        self._await_ready.clear()
                        self._await_ready.wait()
                        with contextlib.suppress(Exception):
                            src.flush()   # discard backlog the source kept producing while paused
                        if self._stop.is_set():
                            self._status("stopped by operator")
                            break
                        self._status(f"rest baseline ({self.cfg.rest_sec:.0f}s)")
                        self._set_phase("rest")
                        k = 0
                    continue

                out = decoder.predict(design)
                if out is None:                     # running-z warmup (no calibration path)
                    continue
                cen, dmn, pda = out

                # ---- rest baseline ----
                if self.phase == "rest":
                    rest_pda.append(pda)
                    k += 1
                    self._emit(TRUpdate(k, "rest", cen, dmn, pda, float("nan"), time.time() - t0), writer)
                    if k >= n_rest:
                        v = np.asarray(rest_pda, float)
                        v = v[np.isfinite(v)]
                        self.baseline_mean = float(v.mean()) if v.size else 0.0
                        self.baseline_sd = float(v.std() + 1e-9) if v.size else 1.0
                        if cal_obj is not None:
                            cal_obj.set_pda_baseline(rest_pda)
                            cal_obj.save(self.cfg.resolved_calib_save())
                        self._status(f"baseline PDA mean={self.baseline_mean:.3f} sd={self.baseline_sd:.3f} -> feedback")
                        self._set_phase("feedback")
                        k = 0
                    continue

                # ---- feedback ----
                pda_z = (pda - self.baseline_mean) / self.baseline_sd
                k += 1
                self._emit(TRUpdate(k, "feedback", cen, dmn, pda, pda_z, time.time() - t0), writer)
                if k >= n_fb:
                    self._status(f"feedback complete ({self.cfg.feedback_sec:.0f}s)")
                    self.completed = True
                    break

            self._set_phase("done")
            self._status(f"session done — log: {self.log_path.name if self.log_path else '(none)'}")
        except Exception as exc:  # surface to the operator rather than dying silently
            self.error = f"{type(exc).__name__}: {exc}"
            self._status(f"ERROR: {self.error}")
            self._set_phase("done")
        finally:
            if fh is not None:
                fh.flush()
                fh.close()
            if src is not None:
                try:
                    src.close()
                except Exception:
                    pass

    def _emit(self, u: TRUpdate, writer) -> None:
        with self._lock:
            self._latest = u
            self._history.append(u)
        if writer is not None:
            z = "" if not np.isfinite(u.pda_z) else f"{u.pda_z:.4f}"
            writer.writerow([u.tr, u.phase, f"{u.cen:.4f}", f"{u.dmn:.4f}", f"{u.pda:.4f}", z, f"{u.t:.2f}"])
        if self._on_update:
            self._on_update(u)


# ── tiny CLI so the engine is runnable standalone (parity with run_nf) ────────
def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run one NF session via SessionEngine (headless).")
    ap.add_argument("--source", default="replay", choices=["replay", "cortex", "lsl", "emokit"])
    ap.add_argument("--replay", default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--subject", default="P000")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--calib-sec", type=float, default=60.0)
    ap.add_argument("--rest-sec", type=float, default=30.0)
    ap.add_argument("--feedback-sec", type=float, default=300.0)
    a = ap.parse_args()

    cfg = EngineConfig(
        subject=a.subject, run=a.run, source=a.source, replay_path=a.replay, replay_speed=a.speed,
        do_calibrate=not a.no_calibrate, calib_path=a.calib, calib_sec=a.calib_sec,
        rest_sec=a.rest_sec, feedback_sec=a.feedback_sec,
    )
    eng = SessionEngine(
        cfg,
        on_phase=lambda p: print(f"[phase] {p}"),
        on_status=lambda s: print(f"[status] {s}"),
        on_update=lambda u: print(f"  TR {u.tr:3d} {u.phase:8s} cen={u.cen:+.3f} dmn={u.dmn:+.3f} "
                                  f"pda={u.pda:+.3f} z={u.pda_z:+.2f}" if np.isfinite(u.pda_z)
                                  else f"  TR {u.tr:3d} {u.phase:8s} cen={u.cen:+.3f} dmn={u.dmn:+.3f} pda={u.pda:+.3f}"),
    )
    eng.start()
    while eng.is_running():
        time.sleep(0.1)
    if eng.error:
        raise SystemExit(f"engine error: {eng.error}")


if __name__ == "__main__":
    _cli()
