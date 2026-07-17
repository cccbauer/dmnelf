#!/usr/bin/env python3
"""
eeg_balltask.py  —  EEG port of the DMN ball-task neurofeedback (PsychoPy)
--------------------------------------------------------------------------
Faithful adaptation of the scanner ball-task (rt-network_feedback.py): the SAME display and
physics — a white ball between a top CEN circle (yellow) and a bottom DMN circle (blue), rising
when CEN > DMN, resetting on "hits" (which shrink the target circle) — but network activation
comes from the EPOC-X EFP decoder instead of MURFI (via EEGActivationCommunicator, a drop-in for
MurfiActivationCommunicator). Keeps the 30 s +/Relax baseline, the post-run sliders, and the
per-volume CSV log; drops MRI-only machinery (scanner trigger -> keypress; no run-chaining/BIDS).

  python eeg_balltask.py --participant rtbpd001 --run 1 --feedback Feedback --source cortex
  python eeg_balltask.py --participant test --source replay \
        --replay testdata/dmnelf005_feedback_run-01_250Hz.fif --run-sec 60 --baseline-sec 6
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# button box (scanner) + keyboard fallbacks
LEFT_KEYS = ["3", "left"]; RIGHT_KEYS = ["1", "right"]; ENTER_KEYS = ["4", "return"]
START_KEYS = ["t", "+", "5", "space"]

ROI_NAMES = ["cen", "dmn"]
ROI_COLORS = {"cen": "yellow", "dmn": "blue"}     # CEN top yellow, DMN bottom blue (as in the MRI task)
POSITIONS = [1, -1]                                # cen -> up (+1), dmn -> down (-1)
PDA_OUTLIER_THRESHOLD = 2.0
DEFAULT_SCALE_FACTOR = 10
INTERNAL_SCALER = 10
MIN_RADIUS = 0.03


def _creds():
    import yaml
    f = HERE / "credentials.yaml"
    return yaml.safe_load(f.read_text()) if f.exists() else {}


def make_source(a):
    from sources import CortexSource, LSLSource, ReplaySource, EmokitSource
    if a.source == "emokit":
        return EmokitSource(_creds().get("emokit_serial"))
    if a.source == "cortex":
        c = _creds()
        return CortexSource(c.get("client_id"), c.get("client_secret"), c.get("license_id"),
                            c.get("headset_id"))
    if a.source == "lsl":
        return LSLSource()
    return ReplaySource(a.replay, speed=a.speed)


# ── ball physics (verbatim from the scanner task) ────────────────────────────
def further_than_circles(position, circle_center, ball_center):
    return ball_center > circle_center if position == 0 else ball_center < circle_center


def calculate_ball_position(direction, activity, bx, by, outlier, scale, tr_to_frame):
    cursor = np.dot(direction, activity)
    if not outlier:
        by = by + (np.real(cursor) * (scale / INTERNAL_SCALER) / tr_to_frame)
        bx = bx + (np.imag(cursor) * (scale / INTERNAL_SCALER) / tr_to_frame)
    return (bx, by)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", default="test"); ap.add_argument("--run", default="1")
    ap.add_argument("--feedback", default="Feedback", choices=["Feedback", "No Feedback"])
    ap.add_argument("--source", default="cortex", choices=["cortex", "lsl", "replay", "emokit"])
    ap.add_argument("--replay", default=None); ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--scale-factor", type=float, default=DEFAULT_SCALE_FACTOR)
    ap.add_argument("--baseline-sec", type=float, default=30.0)
    ap.add_argument("--run-sec", type=float, default=150.0)
    ap.add_argument("--windowed", action="store_true")
    a = ap.parse_args()

    from psychopy import visual, core, event
    from eeg_activation_communicator import EEGActivationCommunicator

    # data files
    ddir = HERE / "data" / a.participant; ddir.mkdir(parents=True, exist_ok=True)
    tag = "Feedback" if a.feedback == "Feedback" else "No_Feedback"
    stem = ddir / f"{a.participant}_DMN_{tag}_{a.run}"
    with open(f"{stem}_roi_outputs.csv", "w", newline="") as f:
        csv.writer(f).writerow(["volume", "scale_factor", "time", "time_plus_1.2", "cen", "dmn",
                                "stage", "cen_cumulative_hits", "dmn_cumulative_hits", "pda_outlier",
                                "ball_y_position", "top_circle_y_position", "bottom_circle_y_position"])

    # window + stimuli
    win = visual.Window(size=(1080, 1080), fullscr=not a.windowed, screen=0, color=[-1, -1, -1],
                        units="norm", allowGUI=False)
    frameDur = 1.0 / (win.getActualFrameRate() or 60.0)
    tr = 1.2; tr_to_frame = tr / frameDur
    scale = [win.size[1] / win.size[0], 1]                       # aspect correction
    circles = {}
    for i, roi in enumerate(ROI_NAMES):
        c = visual.Circle(win, pos=(0, POSITIONS[i] / 3.0), radius=0.15, fillColor=None,
                          lineColor=ROI_COLORS[roi], lineWidth=3)
        c.size *= scale; circles[roi] = c
    ball = visual.Circle(win, pos=(0, 0), radius=0.03, fillColor="white", lineColor="white", lineWidth=3)
    ball.size *= scale
    plus = visual.TextStim(win, text="+", height=0.3, color="white")
    relax = visual.TextStim(win, text="Relax", pos=(0, -0.2), height=0.07, color="white")
    msg = visual.TextStim(win, text="", pos=(0, 0), height=0.06, wrapWidth=1.4, color="white")

    def wait_keys(keys):
        event.clearEvents()
        while not event.getKeys(keyList=keys):
            if event.getKeys(keyList=["escape"]):
                win.close(); core.quit()
            core.wait(0.005)

    def run_slider(question, left, right):
        q = visual.TextStim(win, text=question, pos=(0, 0.25), height=0.06, wrapWidth=1.4, color="white")
        vas = visual.Slider(win, size=(0.85, 0.1), ticks=(1, 9), labels=(left, right),
                            granularity=1, color="white", fillColor="white")
        vas.markerPos = 5; event.clearEvents()
        while True:
            vas.draw(); q.draw(); win.flip()
            for k in event.getKeys(keyList=LEFT_KEYS + RIGHT_KEYS + ENTER_KEYS):
                if k in LEFT_KEYS: vas.markerPos = max(1, vas.markerPos - 1)
                elif k in RIGHT_KEYS: vas.markerPos = min(9, vas.markerPos + 1)
                elif k in ENTER_KEYS:
                    return vas.markerPos
            core.wait(0.005)

    # EEG communicator (drop-in for MURFI) — start streaming the decoder
    comm = EEGActivationCommunicator(make_source(a), ROI_NAMES, tr,
                                     calib_path=a.calib).start()

    # instructions
    fbnote = ("The white ball moves UP toward the yellow circle when you are in the mindful "
              "(Noting) state.\n\nPress any button to begin.") if a.feedback == "Feedback" else \
             ("Keep practicing Mental Noting. The ball will not move this run.\n\n"
              "Press any button to begin.")
    msg.text = fbnote; msg.draw(); win.flip(); wait_keys(START_KEYS)

    # ── 30 s baseline (+/Relax); still log decoder output ──
    plus.setAutoDraw(True); relax.setAutoDraw(True); win.flip()
    frame = 0; t0 = time.time()
    while time.time() - t0 < a.baseline_sec:
        comm.update()
        cen = comm.get_roi_activation("cen", frame); dmn = comm.get_roi_activation("dmn", frame)
        if not (np.isnan(cen) or np.isnan(dmn)):
            with open(f"{stem}_roi_outputs.csv", "a", newline="") as f:
                csv.writer(f).writerow([frame, a.scale_factor, time.time() - t0, time.time() - t0 + tr,
                                        cen, dmn, "baseline", 0, 0, False, np.nan, np.nan, np.nan])
            frame += 1
        if event.getKeys(keyList=["escape"]):
            win.close(); core.quit()
        win.flip()
    plus.setAutoDraw(False); relax.setAutoDraw(False)

    # ── feedback ──
    hits = {"cen": 0, "dmn": 0}; direction = 0; activity = 0.0; outlier = False
    for c in circles.values(): c.draw()
    ball.draw(); win.flip()
    t0 = time.time(); run_stop = 0.0
    while time.time() - t0 < a.run_sec:
        run_stop = time.time() - t0
        comm.update()
        cen = comm.get_roi_activation("cen", frame); dmn = comm.get_roi_activation("dmn", frame)
        if not (np.isnan(cen) or np.isnan(dmn)):
            roi_vals = [cen, dmn]
            outlier = np.nanmax(np.abs(roi_vals)) > PDA_OUTLIER_THRESHOLD
            hi = int(np.nanargmax(roi_vals))
            if np.nanmean(roi_vals) != 0:
                activity = abs(np.nanmax(roi_vals) - np.nanmin(roi_vals)) / 10.0
                direction = POSITIONS[hi]
            for i, roi in enumerate(ROI_NAMES):
                if further_than_circles(i, circles[roi].pos[1], ball.pos[1]):
                    hits[roi] += 1; ball.pos = (0, 0)
                    circles[roi].radius = max(circles[roi].radius * 0.9, MIN_RADIUS)
            with open(f"{stem}_roi_outputs.csv", "a", newline="") as f:
                csv.writer(f).writerow([frame, a.scale_factor, time.time() - t0, time.time() - t0 + tr,
                                        cen, dmn, "feedback", hits["cen"], hits["dmn"], outlier,
                                        ball.pos[1], circles["cen"].pos[1], circles["dmn"].pos[1]])
            frame += 1
        # advance ball every display frame (unless past a circle center)
        paused = any(further_than_circles(i, circles[r].pos[1], ball.pos[1]) for i, r in enumerate(ROI_NAMES))
        if not paused and direction != 0:
            ball.pos = calculate_ball_position(direction, activity, ball.pos[0], ball.pos[1],
                                               outlier, a.scale_factor, tr_to_frame)
        if a.feedback == "Feedback":
            for c in circles.values(): c.draw()
            ball.draw()
        win.flip()
        if event.getKeys(keyList=["escape"]):
            break

    comm.stop()

    # ── sliders (only if >= 60 s of feedback, as in the scanner task) ──
    qfile = f"{stem}_slider_questions.csv"
    with open(qfile, "w", newline="") as f:
        csv.writer(f).writerow(["id", "run", "feedback_on", "question_text", "response"])
    qs = [("How often were you using the mental noting practice?", "Never", "Always"),
          ("How often did you check the position of the ball?", "Never", "All the time"),
          ("How difficult was it to apply mental noting?", "Not at all", "Very Difficult"),
          ("How calm do you feel right now?", "Not at all", "Very calm")]
    if run_stop >= 60:
        msg.text = ("A few quick questions.\nLeft/Right buttons move the slider, top button to enter.\n"
                    "Press any button to continue."); msg.draw(); win.flip()
        wait_keys(LEFT_KEYS + RIGHT_KEYS + ENTER_KEYS)
        for q, l, r in qs:
            resp = run_slider(q, l, r)
            with open(qfile, "a", newline="") as f:
                csv.writer(f).writerow([a.participant, a.run, a.feedback, q, resp])
    else:
        for q, _l, _r in qs:
            with open(qfile, "a", newline="") as f:
                csv.writer(f).writerow([a.participant, a.run, a.feedback, q, np.nan])

    msg.text = "Thank you!"; msg.draw(); win.flip(); core.wait(3)
    win.close(); core.quit()


if __name__ == "__main__":
    main()
