#!/usr/bin/env python3
"""
probe_feature_report.py  —  check the EPOC X HID feature-report channel for a crypto handshake
-------------------------------------------------------------------------------------------------
decode_probe.py ruled out all 11 known emokit/CyKit static serial-derived AES keys on this EPOC X
(counter-score ~0.25 = noise for all). One remaining theory: newer firmware exchanges a per-session
key/nonce over the HID **feature report** channel (usage=4 in the streaming interface's report
descriptor, 17 bytes) rather than deriving it purely from the serial number.

This probe reads that feature report repeatedly (and across a headset power-cycle) to check whether
it's a dynamic nonce (would change per session) or static device metadata (would not).

  python probe_feature_report.py
"""
import time
import hid

VID, PID = 0x1234, 0xed02


def main():
    devs = hid.enumerate(VID, PID) or [d for d in hid.enumerate() if d.get("vendor_id") == VID]
    if not devs:
        raise SystemExit("No Emotiv receiver. Plug in the dongle.")
    streaming = [d for d in devs if d.get("usage_page") == 65535]
    if not streaming:
        raise SystemExit("No usage_page=65535 interface found (that's the one with the feature report).")
    d = streaming[0]
    h = hid.Device(path=d["path"])

    print(f"interface {d['path']}  serial={d.get('serial_number')!r}")
    print("reading feature report (report_id=0, 17 bytes) 5x over 5s …")
    reads = []
    for i in range(5):
        fr = h.get_feature_report(0, 17)
        reads.append(fr)
        print(f"  [{i}] {fr.hex()}")
        time.sleep(1)
    h.close()

    if len(set(reads)) == 1:
        print("\n RESULT: feature report is STATIC across reads -> device metadata/capabilities, "
              "not a per-session nonce or key. No handshake exposed on this channel.")
        print("  (decodes plausibly as: packet size=32, sample rate=128 Hz, etc. — not crypto material.)")
    else:
        print("\n RESULT: feature report CHANGED between reads -> could be a session nonce. "
              "Worth correlating with the AES-ECB decrypt (e.g. as part of the key or IV).")


if __name__ == "__main__":
    main()
