#!/usr/bin/env python3
"""
decode_probe.py  —  find the EPOC X AES key + verify EEG decoding (emokit crypto, no license)
---------------------------------------------------------------------------------------------
The dongle streams AES-ECB-encrypted 32-byte reports; the key is derived from the serial's last
4 chars. emokit only special-cases UD2016 (EPOC+); for a UD2020 (EPOC X) serial we must TRY each
key variant and verify against the packet **counter** (byte 0 cycles 0..127) — a correct key makes
consecutive counters increment by 1, a wrong key gives noise. Prints the winning key model and a
few decoded channel values. Run with the headset ON (see pair_headset.py).

  python decode_probe.py --n 400
"""
import argparse
import time
import hid
from Crypto.Cipher import AES

VID, PID = 0x1234, 0xed02


def cykit_keys(serial):
    """All 7 CyKit key models (bytes) for a 16-char serial. sn = ords of the serial."""
    sn = [ord(c) for c in serial]
    a, b, c, d = sn[-1], sn[-2], sn[-3], sn[-4]        # last 4 chars
    models = {
        "1 Epoc::Premium":   [a, 0, b, 72, a, 0, b, 84, c, 16, d, 66, c, 0, d, 80],
        "2 Epoc::Consumer":  [a, 0, b, 84, c, 16, d, 66, a, 0, b, 72, c, 0, d, 80],
        "3 Insight::Premium":[b, 0, a, 68, b, 0, a, 12, d, 0, c, 21, d, 0, c, 88],
        "4 Insight::Consumer":[a, 0, b, 21, c, 0, d, 12, c, 0, b, 68, a, 0, b, 88],
        "5 Epoc+::Premium":  [b, a, b, a, c, d, c, d, d, c, d, c, a, b, a, b],
        "6 Epoc+::Consumer": [a, b, b, c, c, c, b, d, a, d, b, b, d, d, b, a],
        "7 EPOC+::14bit":    [a, 0, b, 21, c, 0, d, 12, c, 0, b, 68, a, 0, b, 88],
    }
    return {name: bytes(k) for name, k in models.items()}


def open_eeg_interface():
    devs = hid.enumerate(VID, PID) or [d for d in hid.enumerate() if d.get("vendor_id") == VID]
    if not devs:
        raise SystemExit("No Emotiv receiver. Plug in the dongle.")
    serial = devs[0]["serial_number"]
    # the EEG interface is the one that actually streams (usage_page 65535 in testing)
    for d in sorted(devs, key=lambda x: -(x.get("usage_page") or 0)):
        h = hid.Device(path=d["path"]); h.nonblocking = True
        t0 = time.time()
        while time.time() - t0 < 2:
            if h.read(32):
                return h, serial
        h.close()
    raise SystemExit("No data on any interface — is the headset ON (dongle mode)?")


def as_key(s):
    return ("".join(s) if isinstance(s, list) else s).encode("latin-1")[:16]


def decrypt(cipher, data):
    return cipher.decrypt(bytes(data[:16])) + cipher.decrypt(bytes(data[16:32]))


def counter_score(counters):
    """Fraction of consecutive steps that look like a valid 0..127 counter (+1 mod 128)."""
    good = 0
    for a, b in zip(counters, counters[1:]):
        if a < 128 and ((b - a) % 128 == 1 or b >= 128):     # +1, or a battery/sync frame
            good += 1
    return good / max(1, len(counters) - 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=400); a = ap.parse_args()
    h, serial = open_eeg_interface()
    print(f"reading {a.n} reports from serial {serial!r} …")
    reports = []
    while len(reports) < a.n:
        r = h.read(32)
        if r and len(r) >= 32:
            reports.append(r)
        else:
            time.sleep(0.001)

    candidates = cykit_keys(serial)
    print(f"\n  {'key model':22s}  counter-score")
    best = None
    for name, key in candidates.items():
        cipher = AES.new(key, AES.MODE_ECB)
        counters = [decrypt(cipher, r)[0] for r in reports]
        sc = counter_score(counters)
        print(f"  {name:22s}  {sc:5.2f}   (counters {counters[:8]})")
        if best is None or sc > best[1]:
            best = (name, sc, key)

    print(f"\nBEST: {best[0]}  score={best[1]:.2f}")
    if best[1] < 0.8:
        print("  ⚠ no key model validated (<0.80). EPOC X may use a new scheme — "
              "we may need CyKit or a serial-specific key. Share this output.")
        return
    print("  ✓ key verified. Sample decoded channels (CyKit convertEPOC_PLUS, ~µV):")
    cipher = AES.new(best[2], AES.MODE_ECB)

    def conv(v1, v2):
        return round((v1 * 0.128205128205129 + 4201.02564096001) + ((v2 - 128) * 32.82051289), 1)

    for r in reports[10:13]:
        d = [int(b) for b in decrypt(cipher, r)]
        chans = [conv(d[i], d[i + 1]) for i in range(2, 16, 2)] + \
                [conv(d[i], d[i + 1]) for i in range(18, 32, 2)]
        print(f"    counter={d[0]:3d}  ch[0:6]={chans[:6]}")
    h.close()
    print(f"\n  -> EPOC X key model: {best[0]}")


if __name__ == "__main__":
    main()
