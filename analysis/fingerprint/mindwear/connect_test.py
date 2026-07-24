#!/usr/bin/env python3
"""
connect_test.py  —  verify EEG is flowing from the EPOC X (Phase-1 deliverable)
-------------------------------------------------------------------------------
Opens an acquisition source, prints headset / stream info, then streams a few seconds of
live EEG and reports per-channel RMS + a flat-line (poor-contact) check. Run this FIRST
against the physical headset.

  # live EPOC X via Cortex (fill credentials.yaml first)
  python connect_test.py --source cortex --seconds 8
  # EmotivPRO LSL outlet
  python connect_test.py --source lsl
  # no hardware — replay a recorded file
  python connect_test.py --source replay --replay path/to/eeg.fif
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sources import CortexSource, LSLSource, ReplaySource, EmokitSource, dump_lsl_streams


def load_credentials():
    import yaml
    f = HERE / "credentials.yaml"
    return yaml.safe_load(f.read_text()) if f.exists() else {}


def make_source(args):
    if args.source == "emokit":                    # license-free, straight from the USB dongle
        return EmokitSource(load_credentials().get("emokit_serial"))
    if args.source == "cortex":
        c = load_credentials()
        if not c.get("client_id"):
            raise SystemExit("Missing credentials.yaml — copy credentials.example.yaml and fill in "
                             "your Cortex client_id / client_secret.")
        return CortexSource(c.get("client_id"), c.get("client_secret"),
                            c.get("license_id"), c.get("headset_id"))
    if args.source == "lsl":
        return LSLSource()
    if args.source == "replay":
        if not args.replay:
            raise SystemExit("--replay PATH required for --source replay")
        return ReplaySource(args.replay, speed=args.speed)
    raise SystemExit(f"unknown source {args.source}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="cortex", choices=["cortex", "lsl", "replay", "emokit"])
    ap.add_argument("--replay", default=None); ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--dump-lsl-streams", action="store_true",
                    help="print every LSL stream pylsl sees (name/type/channels/rate/labels) and exit "
                         "— independent of source selection, useful to check exactly what EmotivPRO "
                         "is publishing before trusting any decoded signal from it")
    a = ap.parse_args()

    if a.dump_lsl_streams:
        dump_lsl_streams()
        return

    print(f"[connect_test] opening source: {a.source} …", flush=True)
    src = make_source(a)
    try:
        src.open()
    except Exception as e:
        print(f"  CONNECTION FAILED: {e}"); raise SystemExit(1)

    print(f"  ✓ connected.  sfreq = {src.sfreq} Hz   channels ({len(src.channels)}): "
          f"{', '.join(src.channels)}", flush=True)
    hid = getattr(getattr(src, "client", None), "headset_id", None)
    if hid:
        print(f"  headset: {hid}")

    print(f"[connect_test] streaming {a.seconds:g} s …", flush=True)
    buf = []; t0 = time.time(); n = 0
    for _t, s in src.samples():
        buf.append(s); n += 1
        if time.time() - t0 >= a.seconds:
            break
    src.close()

    if not buf:
        print("  NO SAMPLES received."); raise SystemExit(1)
    X = np.array(buf)                                   # [n_samp, n_ch]
    dur = time.time() - t0
    mean = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    rms = np.sqrt(np.nanmean(X ** 2, axis=0))
    flat = sd < 0.5                                     # ~flat-line / no contact (µV of variation)
    # a large mean relative to the variation around it is NOT what calibrated µV EEG looks like
    # (real scalp EEG hovers near 0 µV) — usually means a raw/unreferenced stream, wrong units, or
    # an ADC offset that never got removed upstream.
    offset = np.abs(mean) > 20 * np.maximum(sd, 1e-9)
    print(f"  ✓ received {n} samples in {dur:.1f} s  (~{n/max(dur,1e-9):.0f} Hz)")
    print(f"  {'chan':>5}  {'mean µV':>9}  {'sd µV':>7}  {'RMS µV':>8}  contact")
    for c, m, s, r, fl, off in zip(src.channels, mean, sd, rms, flat, offset):
        note = "⚠ poor" if fl else ("⚠ offset?" if off else "ok")
        print(f"  {c:>5}  {m:9.1f}  {s:7.1f}  {r:8.1f}  {note}")
    if flat.any():
        print(f"  ⚠ {flat.sum()} channel(s) look flat — re-wet the felt sensors / reseat the cap.")
    if offset.any():
        print(f"  ⚠ {offset.sum()} channel(s) have a mean >>20x their variation — that's NOT what "
             "calibrated µV EEG looks like (should hover near 0). Check whether this LSL stream is "
             "raw/unreferenced rather than the processed EEG output, e.g. via --dump-lsl-streams.")
    if not flat.any() and not offset.any():
        print("  all channels have signal ✓  — ready for the decoder pipeline.")


if __name__ == "__main__":
    main()
