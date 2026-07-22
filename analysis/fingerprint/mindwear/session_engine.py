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
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "model" / "efp_epoc_model.npz"
# Per-subject data lives under data/<subject>/ — one folder per participant. Each block writes
# BIDS-named artifacts, sub-<subject>_task-<stage>_run-<NN>_desc-<...>.{csv,npz} (see bids_stem).
DATA_DIR = HERE / "data"


def subject_dir(subject: str, data_dir: Path | str = DATA_DIR) -> Path:
    return Path(data_dir) / subject


def bids_stem(subject: str, task: str, run: int) -> str:
    """BIDS-style filename stem for one recording: ``sub-<label>_task-<task>_run-<NN>``.

    ``subject`` is the participant label configured in the study (basename + zero-padded index,
    e.g. ``dmnelf001``); the ``sub-`` prefix is added here. ``task`` is the block stage
    (calibration | transferpre | feedback | transferpost). ``run`` is the feedback-run index for
    feedback blocks (1..n_runs) and 1 for the single calibration/transfer blocks.
    """
    return f"sub-{subject}_task-{task}_run-{run:02d}"


def subject_artifacts(subject: str, data_dir: Path | str = DATA_DIR) -> list[str]:
    """Names of any already-saved recording files for this subject (empty if none).

    Used by the GUI to warn before overwriting: one protocol session always writes the same BIDS
    filenames (there is no session/run entity spanning the protocol), so re-running a subject
    overwrites its artifacts in place."""
    d = subject_dir(subject, data_dir)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_file() and not p.name.startswith("."))


# phases. "calib_review": paused after calibration, awaiting operator confirm/retry. "ready":
# paused at the start of each task block, awaiting the participant (spacebar). "transfer": a task
# block with static targets (no feedback). "question": R-mbNF end-of-run up/down report.
PHASES = ("connect", "calibrate", "calib_review", "ready", "rest",
          "feedback", "transfer", "question", "done")


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
    stage: str = ""         # block stage: calibration|transferpre|feedback|transferpost
    run: int = 0            # feedback-run index within the protocol (0 outside feedback)
    pda_sign: int = 1       # R-mbNF randomized ball-direction sign for this block (+1/-1)


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

    # protocol: ordered block list (from models.build_blocks). None -> a legacy single
    # calibrate/rest/feedback flow synthesized from the calib_sec/rest_sec/feedback_sec fields.
    blocks: Optional[list] = None
    protocol_type: str = "mbNF"

    # logging — blank means data/<subject>/ (resolved in __post_init__)
    log_dir: Optional[str] = None

    def __post_init__(self):
        if not self.log_dir:
            self.log_dir = str(subject_dir(self.subject))

    def resolved_calib_save(self) -> Path:
        return Path(self.log_dir) / f"{bids_stem(self.subject, 'calibration', 1)}_desc-calib.npz"


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
        self.log_paths: list[Path] = []
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

        # participant-ready gate: the worker pauses in phase "ready" at the start of each task
        # block until the stimulus calls participant_ready() (spacebar pressed).
        self._await_ready = threading.Event()

        # R-mbNF direction-report gate: after a randomized feedback run the worker pauses in phase
        # "question" until the stimulus calls answer_direction("up"/"down").
        self._await_question = threading.Event()
        self._direction_answer: Optional[str] = None

        # current-block state (polled by the stimulus)
        self.block_index: int = 0
        self.n_blocks: int = 0
        self.block_stage: str = ""      # calibration|transferpre|feedback|transferpost
        self.block_run: int = 0         # feedback-run index (0 outside feedback)
        self.pda_sign: int = 1          # randomized ball-direction sign for the current block
        self.feedback_active: bool = False   # True only during a (moving-ball) feedback block
        self.direction_reports: list[dict] = []   # per randomized run: {run, pda_sign, answer, ...}

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
        self._await_question.set()         # unblock if paused awaiting a direction report
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

    def answer_direction(self, direction: str) -> None:
        """R-mbNF: participant reported whether noting drove the ball 'up' or 'down'."""
        self._direction_answer = direction
        self._await_question.set()

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

    def _blocks(self) -> list:
        """Protocol block list, or a synthesized legacy calibrate->feedback flow for old configs."""
        if self.cfg.blocks:
            return list(self.cfg.blocks)
        blocks = []
        if self.cfg.do_calibrate and not self.cfg.calib_path:
            blocks.append({"kind": "calibration", "stage": "calibration", "rest_sec": self.cfg.calib_sec})
        blocks.append({"kind": "feedback", "stage": "feedback", "run": 1, "n_runs": 1,
                       "randomize": False, "rest_sec": self.cfg.rest_sec, "task_sec": self.cfg.feedback_sec})
        return blocks

    def _run(self) -> None:
        import sys

        sys.path.insert(0, str(HERE))
        from calibration import Calibrator
        from decoder import Decoder
        from rt_features import RTFeatureExtractor

        self._t0 = time.time()
        src = None
        self._writer = None
        try:
            self._model = np.load(self.cfg.model_path, allow_pickle=True)
            self.tr = float(self._model["tr"])

            self._set_phase("connect")
            self._status(f"opening source: {self.cfg.source}")
            src = self._make_source().open()
            self._src = src
            self.channels = list(src.channels)
            self.sfreq = float(src.sfreq)
            self._status(f"connected — {len(self.channels)} ch @ {self.sfreq:g} Hz, TR {self.tr:g}s")

            self._feat = RTFeatureExtractor(self._model, src.channels, bad_channels=self.cfg.bad_channels)
            self._feat.set_sfreq(src.sfreq)
            if self._feat.bad_channels:
                self._status(f"dropping {len(self._feat.bad_channels)} bad channel(s): "
                             f"{', '.join(self._feat.bad_channels)}")

            # one BIDS CSV per task block (calibration writes only its .npz); each row still carries
            # stage/run/pda_sign columns for offline analysis.
            Path(self.cfg.log_dir).mkdir(parents=True, exist_ok=True)
            self.log_paths = []

            self._gen = src.samples()
            self._n_features = int(np.asarray(self._model["cen_coef"]).shape[0])
            self._decoder = (Decoder(self._model, calibration=Calibrator.load(self.cfg.calib_path))
                             if self.cfg.calib_path else None)

            blocks = self._blocks()
            self.n_blocks = len(blocks)
            for bi, block in enumerate(blocks):
                if self._stop.is_set():
                    break
                self.block_index = bi
                self.block_stage = block.get("stage", "")
                if block["kind"] == "calibration":
                    if not self._do_calibration_block(block):
                        break                       # stopped during calibration
                else:
                    if not self._run_task_block_logged(block):
                        break                       # stopped during a task block
            else:
                self.completed = True                # ran every block without an early stop

            self._set_phase("done")
            self._status(f"session done — {len(self.log_paths)} recording(s) in {self.cfg.log_dir}")
        except Exception as exc:  # surface to the operator rather than dying silently
            self.error = f"{type(exc).__name__}: {exc}"
            self._status(f"ERROR: {self.error}")
            self._set_phase("done")
        finally:
            if src is not None:
                try:
                    src.close()
                except Exception:
                    pass

    def _next_design(self):
        """Pull streaming samples until the feature extractor emits a design, or the stream/stop
        ends. Shared across blocks so the delay ring + running-z carry over continuously."""
        for _t, sample in self._gen:
            if self._stop.is_set():
                return None
            d = self._feat.push(sample)
            if d is not None:
                return d
        return None

    def _flush_after_gate(self):
        """Drop source backlog accumulated while paused at a gate (keeps timed phases honest)."""
        with contextlib.suppress(Exception):
            self._src.flush()

    def _do_calibration_block(self, block) -> bool:
        """Collect calibration designs, fit, then pause for operator review (retry/accept).
        Returns False if the operator stopped the session."""
        from calibration import Calibrator
        from decoder import Decoder

        n_cal = round(float(block.get("rest_sec", self.cfg.calib_sec)) / self.tr)
        while True:
            cal_obj = Calibrator(self._n_features)
            self._set_phase("calibrate")
            self._status(f"calibrating — hold still ({n_cal * self.tr:.0f}s)")
            got = 0
            while got < n_cal:
                design = self._next_design()
                if design is None:
                    return False
                cal_obj.add_design(design)
                got += 1
            cal_obj.fit()
            cal_obj.save(self.cfg.resolved_calib_save())
            self._decoder = Decoder(self._model, calibration=cal_obj)
            self._cal_obj = cal_obj
            self.calib_summary = _score_calibration(cal_obj)
            self._status(f"calibration done ({got * self.tr:.0f}s) -> "
                         f"{self.cfg.resolved_calib_save().name} — awaiting review")
            self._set_phase("calib_review")
            self._await_confirm.clear()
            self._await_confirm.wait()
            self._flush_after_gate()
            if self._stop.is_set():
                return False
            if self._retry_calibration:
                self._retry_calibration = False
                self._status("repeating calibration")
                continue
            return True

    def _run_task_block_logged(self, block) -> bool:
        """Open this block's own BIDS decoder CSV, run the block, then close the file.

        One file per task block: ``sub-<subject>_task-<stage>_run-<NN>_desc-decoder.csv`` where
        ``stage`` is the block's task and ``run`` is the feedback-run index (1 for transfer blocks).
        """
        stage = block.get("stage", "feedback")
        run_idx = int(block.get("run", 1)) or 1
        path = Path(self.cfg.log_dir) / f"{bids_stem(self.cfg.subject, stage, run_idx)}_desc-decoder.csv"
        self.log_path = path
        self.log_paths.append(path)
        # start each run fresh: the finished block is already persisted to its own CSV, so drop the
        # in-memory history (otherwise it grows all session and the live plot slows to a stall).
        with self._lock:
            self._history.clear()
        fh = open(path, "w", newline="")
        self._writer = csv.writer(fh)
        self._writer.writerow(["tr", "phase", "stage", "run", "pda_sign", "cen", "dmn", "pda", "pda_z", "t"])
        try:
            return self._do_task_block(block)
        finally:
            self._writer = None
            fh.flush()
            fh.close()

    def _do_task_block(self, block) -> bool:
        """One task block: participant-ready gate -> rest baseline -> task (feedback or transfer)
        -> optional R-mbNF direction question. Returns False if stopped."""
        stage = block.get("stage", "feedback")
        kind = block["kind"]
        randomize = bool(block.get("randomize", False))
        self.feedback_active = kind == "feedback"
        self.block_run = int(block.get("run", 0))
        self.pda_sign = random.choice([-1, 1]) if randomize else 1

        if self._decoder is None:            # no calibration ran (e.g. legacy running-z path)
            from decoder import Decoder
            self._decoder = Decoder(self._model, calibration=None)

        # ---- participant-ready gate (spacebar) ----
        label = {"transferpre": "transfer (pre)", "transferpost": "transfer (post)"}.get(
            stage, f"feedback run {self.block_run}")
        self._status(f"{label}: waiting for the participant")
        self._set_phase("ready")
        self._await_ready.clear()
        self._await_ready.wait()
        self._flush_after_gate()
        if self._stop.is_set():
            return False

        # ---- rest baseline ----
        n_rest = round(float(block.get("rest_sec", 30.0)) / self.tr)
        n_task = round(float(block.get("task_sec", 150.0)) / self.tr)
        self._status(f"{label}: rest baseline ({n_rest * self.tr:.0f}s)")
        self._set_phase("rest")
        rest_pda: list[float] = []
        k = 0
        while k < n_rest:
            design = self._next_design()
            if design is None:
                return False
            out = self._decoder.predict(design)
            if out is None:
                continue                     # running-z warmup
            cen, dmn, pda = out
            rest_pda.append(pda)
            k += 1
            self._emit(TRUpdate(k, "rest", cen, dmn, pda, float("nan"), time.time() - self._t0,
                                stage=stage, run=self.block_run, pda_sign=self.pda_sign))
        v = np.asarray(rest_pda, float); v = v[np.isfinite(v)]
        self.baseline_mean = float(v.mean()) if v.size else 0.0
        self.baseline_sd = float(v.std() + 1e-9) if v.size else 1.0

        # ---- task (feedback: moving ball / transfer: static targets) ----
        self._status(f"{label}: {n_task * self.tr:.0f}s")
        self._set_phase("feedback" if self.feedback_active else "transfer")
        task_pda: list[float] = []
        k = 0
        while k < n_task:
            design = self._next_design()
            if design is None:
                return False
            out = self._decoder.predict(design)
            if out is None:
                continue
            cen, dmn, pda = out
            pda_z = (pda - self.baseline_mean) / self.baseline_sd
            task_pda.append(pda)
            k += 1
            self._emit(TRUpdate(k, self.phase, cen, dmn, pda, pda_z, time.time() - self._t0,
                                stage=stage, run=self.block_run, pda_sign=self.pda_sign))

        # ---- R-mbNF: end-of-run up/down report ----
        if randomize:
            self._set_phase("question")
            self._status(f"{label}: awaiting direction report")
            self._direction_answer = None
            self._await_question.clear()
            self._await_question.wait()
            self._flush_after_gate()
            if self._stop.is_set():
                return False
            # ball moved up when sign(mean task PDA) * pda_sign > 0
            tp = np.asarray(task_pda, float); tp = tp[np.isfinite(tp)]
            mean_pda = float(tp.mean()) if tp.size else 0.0
            true_dir = "up" if (np.sign(mean_pda) * self.pda_sign) >= 0 else "down"
            correct = (self._direction_answer == true_dir)
            report = {"run": self.block_run, "pda_sign": self.pda_sign, "mean_task_pda": mean_pda,
                      "true_direction": true_dir, "answer": self._direction_answer, "correct": correct}
            self.direction_reports.append(report)
            self._save_direction_report(report, stage)
            self._status(f"run {self.block_run}: reported {self._direction_answer}, "
                         f"ball went {true_dir} ({'correct' if correct else 'incorrect'})")
        return True

    def _save_direction_report(self, report: dict, stage: str = "feedback") -> None:
        """Append one R-mbNF direction report to this block's BIDS directions CSV (accuracy audit)."""
        stem = bids_stem(self.cfg.subject, stage, int(report["run"]) or 1)
        path = Path(self.cfg.log_dir) / f"{stem}_desc-directions.csv"
        new = not path.exists()
        with contextlib.suppress(Exception):
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["run", "pda_sign", "mean_task_pda", "true_direction", "answer", "correct"])
                w.writerow([report["run"], report["pda_sign"], f"{report['mean_task_pda']:.4f}",
                            report["true_direction"], report["answer"], report["correct"]])

    def _emit(self, u: TRUpdate) -> None:
        with self._lock:
            self._latest = u
            self._history.append(u)
        if getattr(self, "_writer", None) is not None:
            z = "" if not np.isfinite(u.pda_z) else f"{u.pda_z:.4f}"
            self._writer.writerow([u.tr, u.phase, u.stage, u.run, u.pda_sign,
                                   f"{u.cen:.4f}", f"{u.dmn:.4f}", f"{u.pda:.4f}", z, f"{u.t:.2f}"])
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
