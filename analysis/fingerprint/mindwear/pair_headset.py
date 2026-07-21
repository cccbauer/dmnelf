#!/usr/bin/env python3
"""
pair_headset.py  —  live EPOC X ↔ dongle pairing / data-flow monitor
--------------------------------------------------------------------
The EPOC X and its USB receiver are factory-paired, so raw (emokit-style) access just needs the
headset powered ON in *dongle mode* and transmitting. This tool enumerates the Emotiv receiver,
opens every HID interface, and polls for data — telling you the moment EEG reports start arriving
and on which interface (that's the one the reader must use). Run it, then turn the headset on.

  python pair_headset.py --seconds 30

Uses the `hid` package (pip install hid) + `brew install hidapi`. No Emotiv license needed.
"""
import argparse
import time
import hid

VID, PID = 0x1234, 0xed02        # Emotiv "Brain Computer Interface USB Receiver/Dongle"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    a = ap.parse_args()

    devs = hid.enumerate(VID, PID) or [d for d in hid.enumerate() if d.get("vendor_id") == VID]
    if not devs:
        raise SystemExit("No Emotiv receiver found. Plug in the USB dongle and re-run.")
    serial = devs[0].get("serial_number")
    print(f"receiver found: serial={serial!r}  interfaces={len(devs)}")
    is_2020 = str(serial).startswith("UD2020")
    print(f"  dongle generation: {'UD2020 (EPOC X era)' if is_2020 else serial}")

    handles = []
    for d in devs:
        try:
            h = hid.Device(path=d["path"]); h.nonblocking = True
            handles.append((d["path"], d.get("usage_page"), h))
        except Exception as e:
            print(f"  could not open {d['path']}: {e}")
    if not handles:
        raise SystemExit("Could not open any interface (permissions? close EmotivPRO/Launcher first).")

    print(f"\n>>> Now POWER ON the EPOC X headset (dongle mode). Watching {a.seconds:g}s …\n")
    counts = {p: 0 for p, _u, _h in handles}
    first = {}
    t0 = time.time(); last_report = 0
    while time.time() - t0 < a.seconds:
        for path, usage, h in handles:
            try:
                r = h.read(32)
            except Exception:
                r = []
            if r:
                counts[path] += 1
                if path not in first:
                    first[path] = bytes(r[:16])
                    print(f"  ✓ DATA on interface usage_page={usage}: first bytes {first[path].hex()}")
        el = time.time() - t0
        if el - last_report >= 2:
            last_report = el
            live = {u: counts[p] for p, u, _h in handles}
            print(f"  [{el:4.0f}s] report counts by usage_page: {live}", flush=True)
        time.sleep(0.002)

    for _p, _u, h in handles:
        h.close()
    print()
    active = [(p, counts[p]) for p in counts if counts[p] > 0]
    if active:
        best = max(active, key=lambda x: x[1])
        rate = best[1] / a.seconds
        print(f"PAIRED ✓  headset is transmitting: {best[1]} reports (~{rate:.0f}/s) — "
              f"EPOC X streams at 128 Hz, so ~128/s means a solid link.")
        print("  Next: decode it with  python connect_test.py --source emokit")
    else:
        print("NOT PAIRED ✗  no data arrived. Checklist:")
        print("  • Headset powered ON (LED on) and charged.")
        print("  • Headset in DONGLE mode, not Bluetooth (EPOC X supports both).")
        print("  • Close EmotivPRO / Emotiv Launcher (they grab the device exclusively).")
        print("  • If it never paired, do a one-time pair in EmotivPRO, then retry here.")


if __name__ == "__main__":
    main()
