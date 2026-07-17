#!/usr/bin/env python3
"""
run_nf.py  —  EPOC X neurofeedback orchestrator
-----------------------------------------------
Ties source -> rt_features -> decoder -> PsychoPy feedback at TR cadence, matching the scanner
paradigm: optional 1-run calibration, then a 30 s rest baseline, then continuous PDA feedback.
Logs cen/dmn/pda per TR to a run CSV.

  # full session on the live headset with PsychoPy feedback
  python run_nf.py --source cortex --subject P001 --calibrate --feedback psychopy
  # reuse a saved calibration
  python run_nf.py --source cortex --subject P001 --calib model/calib_P001.npz
  # headless dry-run on recorded EEG (no hardware / no display)
  python run_nf.py --source replay --replay testdata/dmnelf005_feedback_run-01_250Hz.fif \
                   --feedback none --calibrate --rest-sec 6 --feedback-sec 20 --speed 0
"""
import argparse
import csv
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rt_features import RTFeatureExtractor
from decoder import Decoder
from calibration import Calibrator
from feedback_psychopy import make_feedback


def make_source(a):
    from sources import CortexSource, LSLSource, ReplaySource, EmokitSource
    import yaml
    cf = HERE / "credentials.yaml"
    c = yaml.safe_load(cf.read_text()) if cf.exists() else {}
    if a.source == "emokit":
        return EmokitSource(c.get("emokit_serial"))
    if a.source == "cortex":
        return CortexSource(c.get("client_id"), c.get("client_secret"), c.get("license_id"),
                            c.get("headset_id"))
    if a.source == "lsl":
        return LSLSource()
    return ReplaySource(a.replay, speed=a.speed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="cortex", choices=["cortex", "lsl", "replay", "emokit"])
    ap.add_argument("--replay", default=None); ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--subject", default="P000")
    ap.add_argument("--feedback", default="psychopy", choices=["psychopy", "none"])
    ap.add_argument("--calibrate", action="store_true", help="run a calibration phase first")
    ap.add_argument("--calib", default=None, help="load an existing calibration .npz")
    ap.add_argument("--calib-sec", type=float, default=60.0)
    ap.add_argument("--rest-sec", type=float, default=30.0)
    ap.add_argument("--feedback-sec", type=float, default=300.0)
    ap.add_argument("--target-z", type=float, default=1.0)
    ap.add_argument("--model", default=str(HERE / "model" / "efp_epoc_model.npz"))
    a = ap.parse_args()

    model = np.load(a.model, allow_pickle=True); tr = float(model["tr"])
    src = make_source(a).open()
    feat = RTFeatureExtractor(model, src.channels); feat.set_sfreq(src.sfreq)
    print(f"[run_nf] source={a.source} sfreq={src.sfreq}Hz ch={len(src.channels)} tr={tr}s")

    calib = Calibrator.load(a.calib) if a.calib else None
    decoder = None if a.calibrate else Decoder(model, calibration=calib)
    fb = make_feedback(a.feedback, target_z=a.target_z) if a.feedback == "psychopy" else make_feedback("none")

    n_cal = round(a.calib_sec / tr); n_rest = round(a.rest_sec / tr); n_fb = round(a.feedback_sec / tr)
    logdir = HERE / "logs"; logdir.mkdir(exist_ok=True)
    ts = int(time.time()); logpath = logdir / f"nf_{a.subject}_{ts}.csv"
    log = csv.writer(open(logpath, "w", newline="")); log.writerow(["tr", "phase", "cen", "dmn", "pda", "pda_z"])

    phase = "calibrate" if a.calibrate else "rest"
    cal_obj = Calibrator(int(model["cen_coef"].shape[0])) if a.calibrate else None
    rest_pda = []; base_m, base_s = 0.0, 1.0; k = 0
    if phase == "calibrate":
        fb.message("Calibrating — sit still")
    else:
        fb.rest()

    try:
        for _t, s in src.samples():
            d = feat.push(s)
            if d is None:
                continue
            if phase == "calibrate":
                cal_obj.add_design(d); k += 1
                if k >= n_cal:
                    cal_obj.fit(); cal_obj.save(HERE / "model" / f"calib_{a.subject}.npz")
                    decoder = Decoder(model, calibration=cal_obj)
                    print(f"  calibration done ({k} TRs) -> model/calib_{a.subject}.npz")
                    phase = "rest"; k = 0; fb.rest()
                continue
            out = decoder.predict(d)
            if out is None:            # running-z warmup
                continue
            cen, dmn, pda = out
            if phase == "rest":
                rest_pda.append(pda); k += 1
                log.writerow([k, phase, f"{cen:.4f}", f"{dmn:.4f}", f"{pda:.4f}", ""])
                if k >= n_rest:
                    v = np.array(rest_pda); base_m, base_s = float(v.mean()), float(v.std() + 1e-9)
                    print(f"  rest baseline: PDA mean={base_m:.3f} sd={base_s:.3f} -> feedback")
                    phase = "feedback"; k = 0
                continue
            # feedback
            pda_z = (pda - base_m) / base_s
            fb.feedback(pda_z); k += 1
            log.writerow([k, phase, f"{cen:.4f}", f"{dmn:.4f}", f"{pda:.4f}", f"{pda_z:.4f}"])
            if k >= n_fb:
                break
    finally:
        fb.close(); src.close()
    print(f"[run_nf] done. log -> {logpath}")


if __name__ == "__main__":
    main()
