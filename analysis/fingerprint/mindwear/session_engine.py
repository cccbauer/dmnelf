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
# BIDS-named artifacts, sub-<subject>[_ses-<session>]_task-<stage>_run-<NN>_desc-<...>.{csv,npz}
# (see bids_stem) — ses- is the operator-entered visit/day label, omitted when blank.
DATA_DIR = HERE / "data"


def subject_dir(subject: str, data_dir: Path | str = DATA_DIR) -> Path:
    return Path(data_dir) / subject


def bids_stem(subject: str, task: str, run: int, session: str = "") -> str:
    """BIDS-style filename stem for one recording:
    ``sub-<label>[_ses-<label>]_task-<task>_run-<NN>``.

    ``subject`` is the participant label configured in the study (basename + zero-padded index,
    e.g. ``dmnelf001``); the ``sub-`` prefix is added here. ``session`` is the operator-entered
    BIDS session label (e.g. a visit/day — "01", "pre") distinguishing repeat visits for the same
    participant; omitted from the stem entirely when blank (legacy studies with no session
    concept). ``task`` is the block stage (calibration | transferpre | feedback | transferpost).
    ``run`` is the feedback-run index for feedback blocks (1..n_runs) and 1 for the single
    calibration/transfer blocks.
    """
    ses = f"_ses-{session}" if session else ""
    return f"sub-{subject}{ses}_task-{task}_run-{run:02d}"


def subject_artifacts(subject: str, data_dir: Path | str = DATA_DIR) -> list[str]:
    """Names of any already-saved recording files for this subject, across every ses-/run- it has
    (empty if none). Used by the GUI to warn before overwriting a filename it's about to reuse
    (see session_runner._start) — files under a DIFFERENT ses-/run- than the one about to be
    written are simply left alone, not actually in conflict."""
    d = subject_dir(subject, data_dir)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_file() and not p.name.startswith("."))


def existing_calibration_path(subject: str, session: str = "",
                              data_dir: Path | str = DATA_DIR) -> Optional[Path]:
    """This subject's saved calibration (.npz) for this session, if one exists — calibration is a
    per-subject-per-session singleton (always saved to run=1, see
    EngineConfig.resolved_calib_save), so its mere existence means a prior run already calibrated
    this subject IN THIS SESSION. Used by the GUI to skip the calibration block and reuse it on a
    later manual run, instead of recalibrating every time the operator starts a fresh run for the
    same participant — scoped to ``session`` so a genuinely new visit/day (a different BIDS
    session label) still calibrates fresh, e.g. because electrode placement changed."""
    p = (subject_dir(subject, data_dir)
        / f"{bids_stem(subject, 'calibration', 1, session)}_desc-calib.npz")
    return p if p.exists() else None


# Ball-jump adaptive difficulty, ported verbatim from the original scanner ball-task
# (rt-psychopy/ball_task/run_ball_task.py:131-222): too many hits in a run means the task was too
# easy (less sensitive next time); too few combined hits means it was too hard (more sensitive).
DEFAULT_SCALE_FACTOR = 10.0
MIN_HITS, MAX_HITS = 3, 5


def adaptive_scale_factor(participant: str, run: int, session: str = "",
                          data_dir: Path | str = DATA_DIR) -> float:
    """Ball-jump scale factor for operator run ``run``, adapted from run ``run - 1``'s saved hit
    counts (``desc-ball.csv``, session_runner's session-level run number — one PsychoPy stimulus
    window/file per run, matching the original's one-script-execution-per-run design). Falls back
    to ``DEFAULT_SCALE_FACTOR`` for the first run or if the previous run's file is missing/unusable
    — mirroring the original's ``except: ... default_scale_factor`` fallback."""
    if run <= 1:
        return DEFAULT_SCALE_FACTOR
    ses = f"_ses-{session}" if session else ""
    prev = (subject_dir(participant, data_dir)
           / f"sub-{participant}{ses}_task-nf_run-{run - 1:02d}_desc-ball.csv")
    try:
        rows = list(csv.DictReader(open(prev)))
        if not rows:
            return DEFAULT_SCALE_FACTOR
        cen_hits = max(int(r["cen_hits"]) for r in rows)
        dmn_hits = max(int(r["dmn_hits"]) for r in rows)
        last_scale = float(rows[0]["scale_factor"])
    except Exception:
        return DEFAULT_SCALE_FACTOR
    if cen_hits > MAX_HITS or dmn_hits > MAX_HITS:
        return last_scale * 0.75
    if cen_hits + dmn_hits < MIN_HITS:
        return last_scale * 1.25
    return last_scale


# phases. "calib_review": paused after calibration, awaiting operator confirm/retry. "ready":
# paused at the start of each task block, awaiting the participant (spacebar). "transfer": a task
# block with static targets (no feedback). "question": R-mbNF end-of-run up/down report.
PHASES = ("connect", "calibrate", "calib_review", "ready", "rest",
          "feedback", "transfer", "question", "done")


def _score_calibration(cal_obj, labels: list | None = None, model=None) -> dict:
    """Lightweight QA summary of a just-fitted Calibrator, for the operator's review step.

    Not a pass/fail gate — the operator decides whether to retry from what's shown. Flags
    features that barely varied during calibration (raw std before the robustness floor), which
    can mean the subject held very still (fine) or a channel wasn't really contributing (worth
    a re-check).

    If ``labels`` (the cue active per collected TR — "rest" plus whichever task cues the
    calibration used: "self"/"flanker" for the induction design, "noting" for the rest-vs-noting
    design) and ``model`` are given, also replays the just-fit calibration through the frozen
    decoder to report per-condition PDA means and every pairwise separation (Cohen's d) between
    conditions that actually occurred — did the calibration's task(s) actually move the decoded
    signal apart? Pure readout: doesn't touch Calibrator/Decoder's own math. The per-TR cen/dmn/pda
    + cue used for that readout is also returned (``out["per_tr"]``) so the caller can persist it
    (see `SessionEngine._save_calibration_decoder_csv`) — otherwise this number only ever exists
    for the life of the live calib-review dialog and can't be checked after the fact.
    """
    X = np.array(cal_obj._X)                    # [n_tr, n_features], pre-floor
    raw_std = X.std(0)
    n_flat = int(np.sum(raw_std < 1e-6))
    out = {
        "n_tr": int(X.shape[0]),
        "n_features": int(raw_std.size),
        "n_flat_features": n_flat,
        "pct_flat": 100.0 * n_flat / max(raw_std.size, 1),
    }
    if labels and model is not None and len(labels) == X.shape[0]:
        from decoder import Decoder
        dec = Decoder(model, calibration=cal_obj)
        decoded = np.array([dec.predict(x) for x in X])    # [n_tr, 3]: cen, dmn, pda
        cen, dmn, pda = decoded[:, 0], decoded[:, 1], decoded[:, 2]
        lab = np.asarray(labels)
        conditions = [c for c in dict.fromkeys(labels) if np.any(lab == c)]   # first-seen order

        def _sep(a: str, b: str):
            va, vb = pda[lab == a], pda[lab == b]
            if va.size < 2 or vb.size < 2:
                return None
            pooled_sd = float(np.sqrt(0.5 * (va.var() + vb.var())) + 1e-9)
            return float((va.mean() - vb.mean()) / pooled_sd)

        out["pda_means"] = {c: float(pda[lab == c].mean()) for c in conditions}
        for i, a in enumerate(conditions):
            for b in conditions[i + 1:]:
                out[f"separation_{a}_vs_{b}"] = _sep(a, b)
        out["per_tr"] = {"cue": labels, "cen": cen.tolist(), "dmn": dmn.tolist(), "pda": pda.tolist()}
    return out


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
    session: str = ""                            # BIDS ses- label (visit/day); "" = omitted
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
        return (Path(self.log_dir)
               / f"{bids_stem(self.subject, 'calibration', 1, self.session)}_desc-calib.npz")


def make_source(cfg: "EngineConfig", on_status: Optional[Callable[[str], None]] = None):
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
        return LSLSource(on_status=on_status)
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
                    window_sec: float = 0.25, on_status: Optional[Callable[[str], None]] = None) -> dict:
    """Open the source and continuously score contact quality, live — the EmotivPRO-style preview.

    Every ``window_sec`` seconds, calls ``on_window(channels, sfreq, X)`` with the raw window
    ``X`` [n_samp, n_ch] in µV (so the caller can both score RMS and render a live scrolling
    trace), until ``stop_event`` is set or the source errors. Returns ``{"error"}`` on failure —
    the source is always closed on the way out. Does not touch the decoder. ``on_status``, if
    given, surfaces the same LSL stream-discovery messages a live session logs (see
    ``LSLSource.open()``) — this is the natural place to see "what's actually coming over LSL"
    before ever starting a real run.
    """
    out: dict = {"error": None}
    src = None
    try:
        if cfg.source == "lsl":
            with contextlib.suppress(Exception):
                from sources import dump_lsl_streams
                dump_lsl_streams(on_status=on_status)
        src = make_source(cfg, on_status=on_status).open()
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
        # per-block raw EEG + aux-stream capture (opened by _open_raw_and_aux_writers, used by
        # both task blocks and calibration; drained in _next_design). None/empty between blocks.
        self._raw_writer = None
        self._raw_fh = None
        self._raw_buffer: list = []
        self._aux_fhs: dict = {}
        self._aux_writers: dict = {}
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
        # "question" until the stimulus calls answer_direction("up"/"down"/"not_sure").
        self._await_question = threading.Event()
        self._direction_answer: Optional[str] = None

        # mbNF (non-randomized) per-run gates, mirroring the R-mbNF question gate above but for
        # the "normal" feedback protocol: after each feedback run (except the last) the worker
        # pauses in phase "ratings" for the participant's ratings (stimulus calls
        # ratings_submitted()), then in phase "run_choice" for the OPERATOR to decide whether to
        # redo calibration before the next run (recalibrate_next_run()) or go straight to it
        # (continue_next_run()).
        self._await_ratings = threading.Event()
        self._await_run_choice = threading.Event()
        self._run_choice: str = "continue"
        # the protocol's own calibration block dict, captured once in _run() as blocks are first
        # iterated — reused verbatim if the operator chooses to recalibrate mid-protocol.
        self._calibration_block: Optional[dict] = None
        # set by _do_task_block when the operator chooses "recalibrate" at the run_choice gate;
        # handled by _run's top-level loop AFTER the current feedback block's own writers are
        # closed (see _do_task_block's comment on why it can't call _do_calibration_block itself).
        self._recalibrate_pending: bool = False

        # current-block state (polled by the stimulus)
        self.block_index: int = 0
        self.n_blocks: int = 0
        self.block_stage: str = ""      # calibration|transferpre|feedback|transferpost
        self.block_run: int = 0         # feedback-run index (0 outside feedback)
        self.pda_sign: int = 1          # randomized ball-direction sign for the current block
        self.feedback_active: bool = False   # True only during a (moving-ball) feedback block
        self.direction_reports: list[dict] = []   # per randomized run: {run, pda_sign, answer, ...}

        # calibration induction state (polled by the stimulus during phase == "calibrate"):
        # calib_cue is "rest" | "self" | "flanker"; calib_word/calib_arrows/calib_question hold
        # whichever of the current cue's stimulus fields apply (blank otherwise). calib_cue_sec is
        # the current cue's planned duration, for the stimulus's countdown ring — reset whenever
        # calib_cue changes, since it now recurs across cycles rather than being a single constant.
        self.calib_cue: str = "rest"
        self.calib_cue_sec: float = 0.0
        self.calib_word: str = ""
        self.calib_arrows: str = ""
        self.calib_question: str = ""
        self._trial_answer: Optional[str] = None
        self._trial_log: list[dict] = []

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
        self._await_ratings.set()          # unblock if paused awaiting per-run ratings
        self._await_run_choice.set()       # unblock if paused awaiting the operator's choice
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
        """R-mbNF: participant reported 'up', 'down', or 'not_sure' (an abstention — scored as
        neither correct nor incorrect, see _do_task_block)."""
        self._direction_answer = direction
        self._await_question.set()

    def answer_trial(self, response: str) -> None:
        """Calibration self/flanker block: participant answered the current trial ('left'/'right',
        or 'yes'/'no' for the self block). Non-blocking — logged for a compliance check only; the
        calibration's wall-clock trial schedule advances regardless of whether this was called."""
        self._trial_answer = response

    def ratings_submitted(self) -> None:
        """mbNF: participant finished the post-run ratings questions."""
        self._await_ratings.set()

    def continue_next_run(self) -> None:
        """Operator: go straight to the next feedback run, reusing the current calibration."""
        self._run_choice = "continue"
        self._await_run_choice.set()

    def recalibrate_next_run(self) -> None:
        """Operator: redo calibration before the next feedback run."""
        self._run_choice = "recalibrate"
        self._await_run_choice.set()

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
        return make_source(self.cfg, on_status=self._status)

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
            if self.cfg.source == "lsl":
                # what's actually on the wire, unfiltered by our aux-stream classifier — surfaces a
                # raw/unreferenced stream (values sitting at a large offset instead of near 0 µV)
                # before it ever reaches calibration or a real run, not just via the CLI.
                with contextlib.suppress(Exception):
                    from sources import dump_lsl_streams
                    dump_lsl_streams(on_status=self._status)
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
            # design length the online decoder actually emits per TR — NOT cen_coef's own size,
            # which for a dual single-electrode model (efp_epoc_dual_model.npz) is only that
            # target's own 110-feature block, half of the real 220-feature combined design
            # RTFeatureExtractor.push() produces (see decoder.py's design_size).
            self._n_features = (int(self._model["total_features"]) if "total_features" in self._model
                               else int(np.asarray(self._model["cen_coef"]).shape[0]))
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
                    self._calibration_block = block   # reused verbatim on a mid-protocol recalibrate
                    if not self._do_calibration_block(block):
                        break                       # stopped during calibration
                else:
                    if not self._run_task_block_logged(block):
                        break                       # stopped during a task block
                if self._recalibrate_pending:
                    self._recalibrate_pending = False
                    if not self._do_calibration_block(self._calibration_block):
                        break                       # stopped during the mid-protocol recalibration
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
        ends. Shared across blocks so the delay ring + running-z carry over continuously.

        Every raw sample also gets logged verbatim (CSV row + buffered for the block's .fif) and
        used to drain whatever aux LSL streams (motion/metrics/bandpower/quality) are available —
        both no-ops outside an active task block, since those writers are only open then."""
        for t, sample in self._gen:
            if self._stop.is_set():
                return None
            if self._raw_writer is not None:
                self._raw_writer.writerow([f"{t:.4f}"] + [f"{v:.4f}" for v in sample])
                self._raw_buffer.append(np.asarray(sample, float))
            self._drain_aux()
            d = self._feat.push(sample)
            if d is not None:
                return d
        return None

    def _drain_aux(self) -> None:
        for category, writer in self._aux_writers.items():
            for t, values in self._src.drain_aux(category):
                writer.writerow([f"{t:.4f}"] + [f"{v:.4f}" for v in values])

    def _flush_after_gate(self):
        """Drop source backlog accumulated while paused at a gate (keeps timed phases honest)."""
        with contextlib.suppress(Exception):
            self._src.flush()
        with contextlib.suppress(Exception):
            self._src.flush_aux()

    def _do_calibration_block(self, block) -> bool:
        """Collect calibration designs, fit, then pause for operator review (retry/accept).
        Returns False if the operator stopped the session.

        With ``cycles > 0``, cycles through one of two designs (``block["type"]``), so the
        collected data — and the decoded PDA the QA readout is computed from — actually spans
        both poles the frozen decoder needs to track, not just quiet rest:
          - "induction" (default): rest -> flanker -> rest -> self, an active executive-control /
            self-referential contrast (see `_run_cue_block`).
          - "noting": rest -> noting, replicating the rest-baseline-then-mental-noting design the
            frozen ridge was actually trained on in the scanner — no discrete trials, just a
            sustained instruction, so it reuses `_run_cue_block` with ``deck=None`` like "rest".
        ``cycles <= 0`` (the legacy/default calibration-block dict, which has no ``cycles`` key)
        falls back to a single flat rest window, identical to the pre-induction behavior."""
        from calibration import Calibrator
        from decoder import Decoder
        from flanker import trial_deck
        from sret_words import word_deck

        cal_type = block.get("type", "induction")
        rest_sec = float(block.get("rest_sec", self.cfg.calib_sec))
        n_cycles = int(block.get("cycles", 0))
        self_sec = float(block.get("self_sec", 30.0))
        flanker_sec = float(block.get("flanker_sec", 45.0))
        noting_sec = float(block.get("noting_sec", 60.0))
        stage = block.get("stage", "calibration")
        stem = bids_stem(self.cfg.subject, stage, 1, self.cfg.session)

        while True:
            cal_obj = Calibrator(self._n_features)
            labels: list[str] = []
            self._trial_log = []
            self._set_phase("calibrate")
            # one deck per calibration attempt, shared across every cycle — not recreated per
            # cue-block — so words/trials don't start repeating until the *whole* bank (240 words;
            # 4 flanker trial types) has been exhausted across the whole session, not per block.
            # (unused for "noting", which has no discrete trials — harmless to build regardless.)
            self_deck, flanker_deck = word_deck(), trial_deck()

            # raw EEG + aux LSL streams, same as task blocks (_run_task_block_logged) — retried
            # attempts get their own fresh files, matching cal_obj being recreated each retry too.
            self._open_raw_and_aux_writers(stem)
            try:
                if n_cycles <= 0:
                    ok = self._run_cue_block("rest", rest_sec, cal_obj, labels, None)
                elif cal_type == "noting":
                    ok = True
                    for _ in range(n_cycles):
                        ok = (self._run_cue_block("rest", rest_sec, cal_obj, labels, None)
                              and self._run_cue_block("noting", noting_sec, cal_obj, labels, None))
                        if not ok:
                            break
                else:                                       # "induction"
                    ok = True
                    for _ in range(n_cycles):
                        ok = (self._run_cue_block("rest", rest_sec, cal_obj, labels, None)
                              and self._run_cue_block("flanker", flanker_sec, cal_obj, labels, flanker_deck)
                              and self._run_cue_block("rest", rest_sec, cal_obj, labels, None)
                              and self._run_cue_block("self", self_sec, cal_obj, labels, self_deck))
                        if not ok:
                            break
            finally:
                self._close_raw_and_aux_writers(stem)
            if not ok:
                return False

            cal_obj.fit()
            cal_obj.save(self.cfg.resolved_calib_save())
            self._decoder = Decoder(self._model, calibration=cal_obj)
            self._cal_obj = cal_obj
            self.calib_summary = _score_calibration(cal_obj, labels, self._model)
            self._save_induction_log(stem)
            self._save_calibration_decoder_csv(stem)
            got_sec = len(cal_obj._X) * self.tr
            self._status(f"calibration done ({got_sec:.0f}s) -> "
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

    def _run_cue_block(self, cue: str, sec: float, cal_obj, labels: list, deck) -> bool:
        """Collect ``round(sec / self.tr)`` TRs of calibration data under one cue ("rest"/"self"/
        "flanker"), advancing a self-paced trial schedule from ``deck`` (None for "rest") and
        labeling every collected TR with the active cue (for the post-fit separation QA).
        ``deck`` is shared across cycles by the caller (`_do_calibration_block`) — not recreated
        here — so the word/trial sequence doesn't repeat until the whole deck is exhausted across
        the *whole* calibration, not reset every time this cue comes back around. TR-counted (like
        every other block in the engine — rest/task in `_do_task_block`) rather than
        wall-clock-timed, so it behaves identically at any replay speed and on live hardware.
        Returns False if the stream ended or the session was stopped mid-block."""
        self.calib_cue = cue
        self.calib_word = self.calib_arrows = self.calib_question = ""
        n_tr = max(1, round(sec / self.tr))
        self.calib_cue_sec = n_tr * self.tr
        self._status(f"calibrating — {cue} ({n_tr * self.tr:.0f}s)")

        question = {"self": "Does this word describe you?",
                   "flanker": "Which way does the CENTER arrow point?"}.get(cue, "")
        trial_tr = max(1, round({"self": 2.5, "flanker": 2.0}.get(cue, sec) / self.tr))
        pending: Optional[dict] = None
        self._trial_answer = None

        got = 0
        trial_k = 0          # TRs remaining before the next trial should start
        while got < n_tr:
            design = self._next_design()
            if design is None:
                return False
            cal_obj.add_design(design)
            labels.append(cue)
            got += 1
            if deck is not None and trial_k <= 0:
                if pending is not None:
                    self._trial_log.append({**pending, "response": self._trial_answer})
                self._trial_answer = None
                stim, target = next(deck)
                if cue == "self":
                    self.calib_word = stim
                else:
                    self.calib_arrows = stim
                self.calib_question = question
                pending = {"cue": cue, "stimulus": stim,
                          ("valence" if cue == "self" else "correct"): target}
                trial_k = trial_tr
            trial_k -= 1
        if pending is not None:
            self._trial_log.append({**pending, "response": self._trial_answer})
        return True

    def _save_induction_log(self, stem: str) -> None:
        """Compliance log for the self/flanker calibration trials — word/valence or arrows/correct
        direction, plus whatever response answer_trial() collected. Not consumed by the
        calibration math (which only uses labels/designs), purely for reviewing engagement."""
        if not self._trial_log:
            return
        path = Path(self.cfg.log_dir) / f"{stem}_desc-induction.csv"
        with contextlib.suppress(Exception):
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["cue", "stimulus", "target", "response"])
                for row in self._trial_log:
                    target = row.get("valence", row.get("correct", ""))
                    w.writerow([row["cue"], row["stimulus"], target, row.get("response") or ""])
            self.log_paths.append(path)

    def _save_calibration_decoder_csv(self, stem: str) -> None:
        """Per-TR cen/dmn/pda + cue label for the just-fit calibration block (from
        `_score_calibration`'s decoder replay), so the self-vs-flanker separation can be
        recomputed/audited offline instead of only existing for the life of the calib-review
        dialog. Mirrors task blocks' desc-decoder.csv, one row per calibration TR."""
        per_tr = (self.calib_summary or {}).get("per_tr")
        if not per_tr:
            return
        path = Path(self.cfg.log_dir) / f"{stem}_desc-decoder.csv"
        with contextlib.suppress(Exception):
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["tr", "cue", "cen", "dmn", "pda"])
                for i, cue in enumerate(per_tr["cue"]):
                    w.writerow([i + 1, cue, f"{per_tr['cen'][i]:.4f}", f"{per_tr['dmn'][i]:.4f}",
                               f"{per_tr['pda'][i]:.4f}"])
            self.log_paths.append(path)

    def _run_task_block_logged(self, block) -> bool:
        """Open this block's own BIDS decoder CSV (+ raw EEG + whatever aux LSL streams the
        source found), run the block, then close everything.

        One set of files per task block, all sharing the stem
        ``sub-<subject>_task-<stage>_run-<NN>_desc-<...>``: ``decoder.csv`` (existing per-TR
        cen/dmn/pda), ``eeg.csv`` + ``eeg.fif`` (raw EEG, per-sample), and one CSV per aux category
        present (``motion``/``metrics``/``bandpower``/``quality``) — empty on sources that don't
        expose aux streams (Cortex, replay, emokit). ``stage`` is the block's task and ``run`` is
        the feedback-run index (1 for transfer blocks).
        """
        stage = block.get("stage", "feedback")
        run_idx = int(block.get("run", 1)) or 1
        stem = bids_stem(self.cfg.subject, stage, run_idx, self.cfg.session)
        logdir = Path(self.cfg.log_dir)

        path = logdir / f"{stem}_desc-decoder.csv"
        self.log_path = path
        self.log_paths.append(path)
        # start each run fresh: the finished block is already persisted to its own CSV, so drop the
        # in-memory history (otherwise it grows all session and the live plot slows to a stall).
        with self._lock:
            self._history.clear()
        fh = open(path, "w", newline="")
        self._writer = csv.writer(fh)
        self._writer.writerow(["tr", "phase", "stage", "run", "pda_sign", "cen", "dmn", "pda", "pda_z", "t"])

        self._open_raw_and_aux_writers(stem)
        try:
            return self._do_task_block(block)
        finally:
            self._writer = None
            fh.flush()
            fh.close()
            self._close_raw_and_aux_writers(stem)

    def _open_raw_and_aux_writers(self, stem: str) -> None:
        """Open this block's raw-EEG CSV + whatever aux LSL streams the source found — shared by
        task blocks (`_run_task_block_logged`) and calibration (`_do_calibration_block`)."""
        logdir = Path(self.cfg.log_dir)
        raw_path = logdir / f"{stem}_desc-eeg.csv"
        self._raw_fh = open(raw_path, "w", newline="")
        self._raw_writer = csv.writer(self._raw_fh)
        self._raw_writer.writerow(["t"] + list(self.channels))
        self._raw_buffer = []
        self.log_paths.append(raw_path)

        self._aux_fhs = {}
        self._aux_writers = {}
        for category, chans in getattr(self._src, "aux_channels", {}).items():
            apath = logdir / f"{stem}_desc-{category}.csv"
            afh = open(apath, "w", newline="")
            self._aux_writers[category] = csv.writer(afh)
            self._aux_writers[category].writerow(["t"] + list(chans))
            self._aux_fhs[category] = afh
            self.log_paths.append(apath)

    def _close_raw_and_aux_writers(self, stem: str) -> None:
        self._raw_writer = None
        self._raw_fh.flush()
        self._raw_fh.close()
        self._save_raw_fif(Path(self.cfg.log_dir) / f"{stem}_desc-eeg.fif")
        self._raw_buffer = []
        for afh in self._aux_fhs.values():
            afh.flush()
            afh.close()
        self._aux_fhs = {}
        self._aux_writers = {}

    def _save_raw_fif(self, path: Path) -> None:
        """Best-effort MNE .fif export of this block's buffered raw EEG, alongside the per-sample
        CSV (µV -> V for MNE's convention). Reports rather than raises on failure — the CSV is
        already the reliable copy, this is a convenience for MNE-based reprocessing."""
        if not self._raw_buffer:
            return
        try:
            import mne
            data = np.array(self._raw_buffer, dtype=float).T * 1e-6      # [ch, samp], V
            info = mne.create_info(list(self.channels), self.sfreq, ch_types="eeg")
            raw = mne.io.RawArray(data, info, verbose="ERROR")
            raw.save(str(path), overwrite=True, verbose="ERROR")
            self.log_paths.append(path)
        except Exception as exc:
            self._status(f"raw EEG .fif export failed for {path.name}: {exc}")

    def _do_task_block(self, block) -> bool:
        """One task block: participant-ready gate -> rest baseline -> task (feedback or transfer)
        -> optional R-mbNF direction question, or (mbNF feedback) ratings + an operator choice of
        whether to recalibrate before the next run. Returns False if stopped."""
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
            # "not_sure" is an abstention, not a wrong guess — correct=None excludes it from
            # accuracy (see RunResult.correct / SubjectResult.reports in results.py).
            correct = None if self._direction_answer == "not_sure" else (self._direction_answer == true_dir)
            report = {"run": self.block_run, "pda_sign": self.pda_sign, "mean_task_pda": mean_pda,
                      "true_direction": true_dir, "answer": self._direction_answer, "correct": correct}
            self.direction_reports.append(report)
            self._save_direction_report(report, stage)
            outcome = "not sure" if correct is None else ("correct" if correct else "incorrect")
            self._status(f"run {self.block_run}: reported {self._direction_answer}, "
                         f"ball went {true_dir} ({outcome})")

        # ---- mbNF (non-randomized) feedback: post-run ratings, then (if another run follows)
        # the operator's choice of whether to recalibrate first ----
        elif kind == "feedback":
            self._set_phase("ratings")
            self._status(f"{label}: ratings")
            self._await_ratings.clear()
            self._await_ratings.wait()
            self._flush_after_gate()
            if self._stop.is_set():
                return False
            if not block.get("is_last_run", True):
                self._set_phase("run_choice")
                self._status(f"{label}: awaiting operator — continue or recalibrate?")
                self._run_choice = "continue"
                self._await_run_choice.clear()
                self._await_run_choice.wait()
                self._flush_after_gate()
                if self._stop.is_set():
                    return False
                if self._run_choice == "recalibrate":
                    if self._calibration_block is None:
                        # a restarted session that reused a saved calibration (calib_path) never
                        # had its own calibration block to reuse the parameters from — can't
                        # recalibrate without them.
                        self._status("can't recalibrate: this session started from a saved "
                                     "calibration with no calibration block to repeat")
                    else:
                        # Deferred to the top-level block loop (_run), not called here: this
                        # method runs INSIDE _run_task_block_logged's open/close bracket for this
                        # feedback block's own raw/aux writers — calling _do_calibration_block
                        # directly here would nest a second open/close on the same shared
                        # self._raw_writer/_raw_fh slot and close the wrong (already-stale) handle
                        # once back in the outer block's own finally.
                        self._recalibrate_pending = True
        return True

    def _save_direction_report(self, report: dict, stage: str = "feedback") -> None:
        """Append one R-mbNF direction report to this block's BIDS directions CSV (accuracy audit)."""
        stem = bids_stem(self.cfg.subject, stage, int(report["run"]) or 1, self.cfg.session)
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
