"""Participant stimulus renderers driven by a running SessionEngine (main-thread / PsychoPy).

One decoder feeds two displays: the operator's live plot (via the engine's ``on_update`` callback)
and the participant stimulus here (by polling ``engine.latest()`` / ``engine.get_roi_activation``).
That keeps a single source + decoder — no double-opening the stream — which matters for Cortex.

Two modes, both faithful to the existing standalone displays:
  * ``ball``  — the scanner ball task: white ball between a top CEN (yellow) and bottom DMN (blue)
                circle, rising when CEN > DMN, hits reset the ball + shrink the circle. Physics is
                imported verbatim from :mod:`eeg_balltask`.
  * ``bars``  — the thermometer bars from :mod:`feedback_psychopy`, driven by PDA z.

These run on the main thread (submitted through the dispatcher) and loop until the engine finishes
or ``stop_event`` is set.
"""
from __future__ import annotations

import contextlib
import csv
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# ball physics + constants, verbatim from the standalone task
from eeg_balltask import (  # noqa: E402
    INTERNAL_SCALER,  # noqa: F401  (used indirectly by calculate_ball_position)
    MIN_RADIUS,
    PDA_OUTLIER_THRESHOLD,
    POSITIONS,
    ROI_COLORS,
    ROI_NAMES,
    calculate_ball_position,
    further_than_circles,
)

# ── persistent stimulus window ───────────────────────────────────────────────
# pyglet/PsychoPy on macOS crashes (segfault) if a visual.Window is closed and a new one is opened
# in the same process. So we create ONE window on first use and reuse it across every run/comparison
# (blanking it between runs instead of closing). All renderers below acquire it via get_window();
# it is only really closed at app shutdown via close_stimulus_window(). Must run on the main thread.
_WIN = None
_WIN_SIZE = (1280, 800)


def get_window():
    """Return the shared PsychoPy window, creating it once. Main thread only."""
    global _WIN
    from psychopy import visual
    if _WIN is None:
        _WIN = visual.Window(size=_WIN_SIZE, fullscr=False, screen=0, color=[-1, -1, -1],
                             units="norm", allowGUI=False, waitBlanking=True)
    return _WIN


def bring_to_front(win) -> None:
    """Raise + focus the stimulus window (pyglet backend) — used at the participant-ready screen
    so the run doesn't silently start behind the operator console."""
    wh = getattr(win, "winHandle", None)
    if wh is not None and hasattr(wh, "activate"):
        with contextlib.suppress(Exception):
            wh.activate()


def blank_window(win) -> None:
    """Clear the shared window to black (used instead of closing between runs)."""
    try:
        win.flip()
    except Exception:
        pass


def close_stimulus_window() -> None:
    """Actually close the shared window (call once, at app shutdown, on the main thread)."""
    global _WIN
    if _WIN is not None:
        try:
            _WIN.close()
        except Exception:
            pass
        _WIN = None


class BallPanel:
    """One ball-task panel (circles + ball + state) at horizontal offset *cx* within a window.

    Encapsulates the scanner physics so several panels can share one PsychoPy window (used by the
    fMRI-vs-EPOC dual comparison). ``on_tr`` updates direction/hits on each new TR; ``animate``
    advances the ball every display frame; ``draw`` renders it.
    """

    def __init__(self, win, cx: float, aspect, tr_to_frame: float, scale_factor: float = 10.0,
                 title: str | None = None):
        from psychopy import visual

        self.cx = cx
        self.tr_to_frame = tr_to_frame
        self.scale = scale_factor
        self.circles = {}
        for i, roi in enumerate(ROI_NAMES):
            c = visual.Circle(win, pos=(cx, POSITIONS[i] / 3.0), radius=0.15, fillColor=None,
                              lineColor=ROI_COLORS[roi], lineWidth=3)
            c.size *= aspect
            self.circles[roi] = c
        self.ball = visual.Circle(win, pos=(cx, 0), radius=0.03, fillColor="white",
                                  lineColor="white", lineWidth=3)
        self.ball.size *= aspect
        self.title = visual.TextStim(win, text=title or "", pos=(cx, 0.62), height=0.05, color="white")
        self.hits = {"cen": 0, "dmn": 0}
        self.direction = 0
        self.activity = 0.0
        self.outlier = False
        self.last_tr = -1

    def on_tr(self, cen: float, dmn: float, tr: int) -> None:
        if tr == self.last_tr or not (np.isfinite(cen) and np.isfinite(dmn)):
            return
        self.last_tr = tr
        roi_vals = [cen, dmn]
        self.outlier = np.nanmax(np.abs(roi_vals)) > PDA_OUTLIER_THRESHOLD
        hi = int(np.nanargmax(roi_vals))
        if np.nanmean(roi_vals) != 0:
            self.activity = abs(np.nanmax(roi_vals) - np.nanmin(roi_vals)) / 10.0
            self.direction = POSITIONS[hi]
        for i, roi in enumerate(ROI_NAMES):
            if further_than_circles(i, self.circles[roi].pos[1], self.ball.pos[1]):
                self.hits[roi] += 1
                self.ball.pos = (self.cx, 0)
                self.circles[roi].radius = max(self.circles[roi].radius * 0.9, MIN_RADIUS)

    def animate(self) -> None:
        paused = any(further_than_circles(i, self.circles[r].pos[1], self.ball.pos[1])
                     for i, r in enumerate(ROI_NAMES))
        if not paused and self.direction != 0:
            self.ball.pos = calculate_ball_position(self.direction, self.activity, self.ball.pos[0],
                                                    self.ball.pos[1], self.outlier, self.scale, self.tr_to_frame)

    def draw(self) -> None:
        for c in self.circles.values():
            c.draw()
        self.ball.draw()
        self.title.draw()


def run_dual_ball(engine, stop_event: threading.Event, scale_factor: float = 10.0) -> None:
    """Two ball panels in one window: left = fMRI(BOLD), right = EPOC(EEG decoder), synced per TR.

    *engine* is a :class:`mindwear.compare_engine.ComparisonEngine` exposing ``latest()`` →
    CompareUpdate, ``is_running()``, ``tr``, ``corr_pda``, ``subject``, ``run``.
    """
    from psychopy import core, event, visual

    win = get_window()
    try:
        frame_dur = 1.0 / (win.getActualFrameRate() or 60.0)
        tr_to_frame = float(engine.tr or 1.2) / frame_dur
        aspect = [win.size[1] / win.size[0], 1]
        left = BallPanel(win, cx=-0.5, aspect=aspect, tr_to_frame=tr_to_frame,
                         scale_factor=scale_factor, title="fMRI  (BOLD)")
        right = BallPanel(win, cx=0.5, aspect=aspect, tr_to_frame=tr_to_frame,
                          scale_factor=scale_factor, title="EPOC  (EEG decoder)")
        divider = visual.Line(win, start=(0, -0.8), end=(0, 0.8), lineColor=[0.25, 0.25, 0.25], lineWidth=2)
        info = visual.TextStim(win, text="", pos=(0, -0.9), height=0.045, color="white")

        while engine.is_running() and not stop_event.is_set() and engine.latest() is None:
            info.text = "preparing comparison…"
            info.draw()
            win.flip()
            if event.getKeys(keyList=["escape"]):
                return

        while engine.is_running() and not stop_event.is_set():
            u = engine.latest()
            if u is not None:
                left.on_tr(u.bold_cen, u.bold_dmn, u.tr)
                right.on_tr(u.eeg_cen, u.eeg_dmn, u.tr)
            left.animate()
            right.animate()
            divider.draw()
            left.draw()
            right.draw()
            r = engine.corr_pda
            info.text = (f"{engine.subject}  run {engine.run}    TR {u.tr if u else 0}"
                         f"    EEG↔BOLD PDA r = {r:+.2f}" if r == r else f"{engine.subject} run {engine.run}")
            info.draw()
            win.flip()
            if event.getKeys(keyList=["escape"]):
                break

        info.text = "Done — press Esc / close to return"
        left.draw()
        right.draw()
        info.draw()
        win.flip()
        for _ in range(180):
            if stop_event.is_set() or event.getKeys(keyList=["escape"]):
                break
            core.wait(0.01)
    finally:
        blank_window(win)          # reuse the window next run; do NOT close (segfaults on reopen)


def run_stimulus(engine, mode: str, feedback_cfg: dict, stop_event: threading.Event,
                 log_dir: Path | None = None, participant: str = "P000", run: int = 1,
                 session: str = "") -> None:
    """Entry point (main thread). Dispatch to the requested stimulus renderer."""
    if mode == "ball":
        _run_ball(engine, feedback_cfg, stop_event, log_dir, participant, run, session)
    elif mode == "bars":
        _run_bars(engine, feedback_cfg, stop_event)


# ── bars ─────────────────────────────────────────────────────────────────────
def _run_bars(engine, feedback_cfg: dict, stop_event: threading.Event) -> None:
    """Thermometer bars (red, rising with PDA z) + fixed blue target — on the shared window."""
    from psychopy import event, visual

    target_z = float(feedback_cfg.get("target_z", 1.0))
    full = 3.0                                  # full-scale z
    bar_h = 1.6
    win = get_window()
    try:
        bars = [visual.Rect(win, width=0.18, height=0.001, pos=(x, -0.8), fillColor=(1, -0.5, -0.5),
                            lineColor=None, anchor="bottom") for x in (-0.28, 0.28)]
        ty = -0.8 + bar_h * (target_z / full)
        target = visual.Line(win, start=(-0.5, ty), end=(0.5, ty), lineColor=(-0.5, -0.5, 1), lineWidth=4)
        msg = visual.TextStim(win, text="", pos=(0, 0.7), height=0.08, color="white")

        while engine.is_running() and not stop_event.is_set():
            if engine.phase == "feedback":
                frac = max(0.0, min(1.0, (engine.pda_z + full) / (2 * full)))
                h = max(0.001, bar_h * frac)
                for b in bars:
                    b.height = h
                    b.draw()
                target.draw()
                msg.text = "Raise the bars (mental noting)"
            else:
                msg.text = "Rest — relax, look at the cross"
            msg.draw()
            win.flip()
            if event.getKeys(keyList=["escape"]):
                break
    finally:
        blank_window(win)


# ── ball ─────────────────────────────────────────────────────────────────────
def _run_ball(engine, feedback_cfg: dict, stop_event: threading.Event,
              log_dir: Path | None, participant: str, run: int, session: str = "") -> None:
    from psychopy import core, event, visual

    scale_factor = float(feedback_cfg.get("scale_factor", 10.0))
    tr = float(engine.tr or 1.2)
    calib_sec = float(getattr(engine.cfg, "calib_sec", 60.0))

    win = get_window()
    try:
        frame_dur = 1.0 / (win.getActualFrameRate() or 60.0)
        tr_to_frame = tr / frame_dur
        aspect = [win.size[1] / win.size[0], 1]

        circles = {}
        for i, roi in enumerate(ROI_NAMES):
            c = visual.Circle(win, pos=(0, POSITIONS[i] / 3.0), radius=0.15, fillColor=None,
                              lineColor=ROI_COLORS[roi], lineWidth=3)
            c.size *= aspect
            circles[roi] = c
        ball = visual.Circle(win, pos=(0, 0), radius=0.03, fillColor="white", lineColor="white", lineWidth=3)
        ball.size *= aspect
        plus = visual.TextStim(win, text="+", height=0.3, color="white")
        relax = visual.TextStim(win, text="Relax", pos=(0, -0.2), height=0.07, color="white")
        msg = visual.TextStim(win, text="", pos=(0, 0.3), height=0.06, wrapWidth=1.4, color="white")

        # calibration progress: a ring that fills clockwise over calib_sec seconds
        calib_track = visual.Circle(win, pos=(0, -0.15), radius=0.18, fillColor=None,
                                    lineColor=[0.4, 0.4, 0.4], lineWidth=3)
        calib_track.size *= aspect
        calib_fill = visual.Pie(win, pos=(0, -0.15), radius=0.18, start=90, end=90, edges=64,
                                fillColor=[0.25, 0.55, 1.0], lineColor=None)
        calib_fill.size *= aspect
        calib_pct = visual.TextStim(win, text="", pos=(0, -0.15), height=0.05, color="white")
        last_phase = None
        calib_start = None

        # self/flanker induction-block stimuli (calibration cycles rest -> flanker -> rest -> self,
        # see SessionEngine._run_cue_block) — reuses `msg` above for the question text.
        calib_stim = visual.TextStim(win, text="", pos=(0, 0), height=0.14, color="white")
        calib_hint = visual.TextStim(win, text="", pos=(0, -0.35), height=0.045, color=[0.6, 0.6, 0.6])
        last_calib_cue = None
        calib_cue_start = None

        # optional ball-state CSV (hits / positions), alongside the engine's decoder CSV
        writer = None
        fh = None
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            # session-level behavioural log — the ball window spans every block, so one file per
            # session (task-nf); the stage column tags each row. run = the operator's session Run #.
            ses = f"_ses-{session}" if session else ""
            fh = open(Path(log_dir) / f"sub-{participant}{ses}_task-nf_run-{run:02d}_desc-ball.csv",
                     "w", newline="")
            writer = csv.writer(fh)
            writer.writerow(["tr", "stage", "cen", "dmn", "cen_hits", "dmn_hits", "outlier",
                             "ball_y", "top_y", "bottom_y", "scale_factor"])

        # direction-question stimuli (R-mbNF end-of-run report). Left = UP, Right = DOWN, Space = NOT SURE.
        q_prompt = visual.TextStim(win, text="", pos=(0, 0.4), height=0.06, wrapWidth=1.5, color="white")
        q_up = visual.TextStim(win, text="◄  UP", pos=(-0.4, 0), height=0.09, color="white")
        q_down = visual.TextStim(win, text="DOWN  ►", pos=(0.4, 0), height=0.09, color="white")
        q_notsure = visual.TextStim(win, text="NOT SURE", pos=(0, -0.15), height=0.07,
                                    color=[0.75, 0.75, 0.75])
        q_hint = visual.TextStim(win, text="left = UP     right = DOWN     space = NOT SURE",
                                 pos=(0, -0.35), height=0.045, color=[0.6, 0.6, 0.6])
        randomized = getattr(engine.cfg, "protocol_type", "mbNF") == "R-mbNF"

        hits = {"cen": 0, "dmn": 0}
        direction = 0
        activity = 0.0
        outlier = False
        last_tr = -1
        was_escaped = False

        def escaped() -> bool:
            nonlocal was_escaped
            if event.getKeys(keyList=["escape"]):
                was_escaped = True
                return True
            return False

        def reset_ball():
            ball.pos = (0, 0)
            for i, roi in enumerate(ROI_NAMES):
                circles[roi].pos = (0, POSITIONS[i] / 3.0)
                circles[roi].radius = 0.15
                circles[roi].size = 1.0
                circles[roi].size *= aspect

        while engine.is_running() and not stop_event.is_set():
            phase = engine.phase
            u = engine.latest()

            if phase == "rest":
                if last_phase != "rest":
                    reset_ball()            # each block starts from a centered ball + full circles
                plus.draw()
                relax.draw()
                last_phase = phase
                win.flip()
                if escaped():
                    break
                continue

            # ---- R-mbNF direction question ----
            if phase == "question":
                q_prompt.text = "Did your noting practice drive the ball UP or DOWN?"
                q_prompt.draw(); q_up.draw(); q_down.draw(); q_notsure.draw(); q_hint.draw()
                last_phase = phase
                win.flip()
                keys = event.getKeys(keyList=["left", "right", "space", "escape"])
                if "escape" in keys:
                    was_escaped = True
                    break
                if "left" in keys:
                    engine.answer_direction("up")
                elif "right" in keys:
                    engine.answer_direction("down")
                elif "space" in keys:
                    engine.answer_direction("not_sure")
                continue

            # ---- non-task phases (connect / calibrate / review / ready) ----
            if phase not in ("feedback", "transfer"):
                if phase == "calibrate":
                    cue = engine.calib_cue
                    if cue != last_calib_cue:
                        last_calib_cue = cue
                        calib_cue_start = time.time()
                    cue_sec = float(getattr(engine, "calib_cue_sec", 0.0)) or calib_sec
                    elapsed = time.time() - (calib_cue_start or time.time())
                    remaining = max(0.0, cue_sec - elapsed)
                    frac = 1.0 - remaining / cue_sec if cue_sec > 0 else 1.0
                    calib_fill.end = 90 + 360 * frac
                    calib_pct.text = f"{int(round(remaining))}s"

                    keys = event.getKeys(keyList=["left", "right"]) if cue in ("self", "flanker") else []
                    if cue == "self":
                        msg.text = engine.calib_question or "Does this word describe you?"
                        calib_stim.text = engine.calib_word
                        calib_hint.text = "left = YES     right = NO"
                        msg.draw(); calib_stim.draw(); calib_hint.draw()
                        if "left" in keys:
                            engine.answer_trial("yes")
                        elif "right" in keys:
                            engine.answer_trial("no")
                    elif cue == "flanker":
                        msg.text = engine.calib_question or "Which way does the CENTER arrow point?"
                        calib_stim.text = engine.calib_arrows
                        calib_hint.text = "left / right = direction of the CENTER arrow"
                        msg.draw(); calib_stim.draw(); calib_hint.draw()
                        if "left" in keys:
                            engine.answer_trial("left")
                        elif "right" in keys:
                            engine.answer_trial("right")
                    elif cue == "noting":
                        calib_track.draw(); calib_fill.draw(); calib_pct.draw()
                        msg.text = "Practice mental noting\n(the same technique as feedback)."
                        msg.draw()
                    else:                                       # "rest"
                        calib_track.draw(); calib_fill.draw(); calib_pct.draw()
                        msg.text = "Calibrating…\nPlease hold still."
                        msg.draw()
                elif phase == "connect":
                    msg.text = "Getting ready…\nHold still."
                    msg.draw()
                elif phase == "calib_review":
                    msg.text = "Calibration complete — reviewing…"
                    msg.draw()
                elif phase == "ratings":
                    # mbNF post-run ratings — blocking (own event loop), so this runs exactly
                    # once per visit; ratings_submitted() then lets the engine move on to the
                    # operator's continue/recalibrate choice (or straight to "done" on the last
                    # run, see SessionEngine._do_task_block).
                    if last_phase != "ratings":
                        _run_ratings(win, log_dir, participant, engine.block_run, session)
                        engine.ratings_submitted()
                elif phase == "run_choice":
                    msg.text = "Please wait — the operator is setting up the next run…"
                    msg.draw()
                elif phase == "ready":
                    if last_phase != "ready":
                        bring_to_front(win)
                    if not engine.feedback_active:            # a transfer block
                        msg.text = ("Practice mental noting.\nThe targets will appear but the ball "
                                    "will not move.\n\nPress SPACEBAR to begin.")
                    elif randomized:                          # R-mbNF feedback (don't reveal mapping)
                        msg.text = ("Practice mental noting to move the ball.\n\n"
                                    "Press SPACEBAR to begin.")
                    else:                                     # mbNF feedback
                        msg.text = ("The white ball moves toward the CEN circle when your brain "
                                    "state favors that network, and toward DMN otherwise.\n\n"
                                    "Press SPACEBAR to begin.")
                    msg.draw()
                    if event.getKeys(keyList=["space"]):
                        engine.participant_ready()
                last_phase = phase
                win.flip()
                if phase == "done":
                    break
                continue

            # ---- transfer block: static targets, ball frozen at center, no hits ----
            if phase == "transfer":
                reset_ball()
                for c in circles.values():
                    c.draw()
                ball.draw()
                last_phase = phase
                win.flip()
                if escaped():
                    break
                continue

            # ---- feedback block: moving ball (randomized sign applies in R-mbNF) ----
            if u is not None and u.tr != last_tr and np.isfinite(u.cen) and np.isfinite(u.dmn):
                last_tr = u.tr
                cen, dmn = float(u.cen), float(u.dmn)
                roi_vals = [cen, dmn]
                outlier = np.nanmax(np.abs(roi_vals)) > PDA_OUTLIER_THRESHOLD
                hi = int(np.nanargmax(roi_vals))
                if np.nanmean(roi_vals) != 0:
                    activity = abs(np.nanmax(roi_vals) - np.nanmin(roi_vals)) / 10.0
                    direction = POSITIONS[hi] * int(getattr(engine, "pda_sign", 1))   # R-mbNF flip
                for i, roi in enumerate(ROI_NAMES):
                    if further_than_circles(i, circles[roi].pos[1], ball.pos[1]):
                        hits[roi] += 1
                        ball.pos = (0, 0)
                        circles[roi].radius = max(circles[roi].radius * 0.9, MIN_RADIUS)
                if writer is not None:
                    writer.writerow([u.tr, u.stage, f"{cen:.4f}", f"{dmn:.4f}", hits["cen"], hits["dmn"],
                                     outlier, f"{ball.pos[1]:.4f}", f"{circles['cen'].pos[1]:.4f}",
                                     f"{circles['dmn'].pos[1]:.4f}", scale_factor])

            paused = any(further_than_circles(i, circles[r].pos[1], ball.pos[1])
                         for i, r in enumerate(ROI_NAMES))
            if not paused and direction != 0:
                ball.pos = calculate_ball_position(direction, activity, ball.pos[0], ball.pos[1],
                                                   outlier, scale_factor, tr_to_frame)
            for c in circles.values():
                c.draw()
            ball.draw()
            last_phase = phase
            win.flip()
            if escaped():
                break

        if fh is not None:
            fh.flush()
            fh.close()

        # mbNF ratings now happen per-run (phase == "ratings", above), not once at session end;
        # R-mbNF's measure is the per-run up/down reports instead of ratings at all.
        msg.text = "Thank you!"
        msg.draw()
        win.flip()
        core.wait(1.5)
    finally:
        blank_window(win)          # reuse the window next run; do NOT close (segfaults on reopen)


RATING_QUESTIONS = [
    "How often were you using the mental noting practice?",
    "How often did you check the position of the ball?",
    "How difficult was it to apply mental noting?",
    "How calm do you feel right now?",
]


def _run_ratings(win, log_dir: Path | None, participant: str, run: int, session: str = "") -> None:
    """Post-run 1-5 keypress ratings (direct number keys — no button box needed)."""
    from psychopy import event, visual

    intro = visual.TextStim(win, text="A few quick questions.\nPress a number key 1-5 to answer.\n\n"
                                      "Press any key to continue.", pos=(0, 0), height=0.06,
                            wrapWidth=1.4, color="white")
    intro.draw()
    win.flip()
    event.clearEvents()
    event.waitKeys()

    rows = []
    for q_text in RATING_QUESTIONS:
        q = visual.TextStim(win, text=q_text, pos=(0, 0.15), height=0.06, wrapWidth=1.5, color="white")
        scale = visual.TextStim(win, text="1        2        3        4        5",
                                pos=(0, -0.1), height=0.09, color="white")
        hint = visual.TextStim(win, text="Press a number key 1-5", pos=(0, -0.3), height=0.045,
                               color=[0.6, 0.6, 0.6])
        event.clearEvents()
        response = None
        while response is None:
            q.draw()
            scale.draw()
            hint.draw()
            win.flip()
            for k in event.getKeys(keyList=["1", "2", "3", "4", "5", "escape"]):
                if k == "escape":
                    response = np.nan
                else:
                    response = int(k)
                break
        rows.append((q_text, response))

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ses = f"_ses-{session}" if session else ""
        with open(Path(log_dir) / f"ratings_{participant}{ses}_run-{run:02d}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["question", "response"])
            w.writerows(rows)
