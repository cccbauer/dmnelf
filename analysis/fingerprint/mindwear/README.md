# MindWear — portable EEG neurofeedback

Real-time EEG neurofeedback on the Emotiv EPOC X (or the 32-channel research cap), running the
frozen DMNELF **EFP** decoder to produce the **PDA = CEN − DMN** signal and drive PsychoPy
feedback. Deploys the decoder validated in
[`../efp_meirhasson/DEPLOY_EPOC.md`](../efp_meirhasson/DEPLOY_EPOC.md) (EPOC-12 montage retains
~92% of clean-CEN decoding).

## Install

Requires [conda](https://conda-forge.org/download/) (Miniconda / Miniforge / Anaconda).

```bash
cd analysis/fingerprint/mindwear
./install.sh                 # creates the `mindwear` conda env from environment.yml + verifies
conda activate mindwear
python launch_gui.py
```

`./install.sh --force` recreates an existing env; `--name NAME` uses a different env name. To set
up manually instead of the script:

```bash
conda env create -f environment.yml   # from analysis/fingerprint/mindwear
conda activate mindwear
python launch_gui.py
```

First launch opens an empty **Study Manager** — click **New Study** and pick a **Source** (LSL for a
live EmotivPRO outlet, Replay for a recorded `.fif`, or Cortex with a raw-EEG license) and a
**Decoder montage** (EPOC-X 12-ch or research-cap 32-ch). Studies are saved under
`~/.mindwear/studies/`.

## Status — all phases built & validated offline
- **Phase 1 — connection** ✅ `cortex.py` · `sources.py` · `connect_test.py` (smoke-tested)
- **Phase 2 — frozen decoder** ✅ `export_model.py` → `model/efp_epoc_model.npz` (CEN+DMN ridge, EPOC-12)
- **Phase 3 — real-time engine** ✅ `rt_features.py` · `decoder.py` · `calibration.py` — end-to-end
  replay validates online CEN↔BOLD r=+0.27, DMN +0.23 (`test_replay.py`)
- **Phase 4 — feedback + orchestrator** ✅ `feedback_psychopy.py` (MURFI-style red bars + blue target)
  · `run_nf.py` (calibrate → 30 s rest → PDA feedback → CSV log)

Remaining before a live session: run against the **physical EPOC X** (Cortex/LSL) and confirm the
**PsychoPy** display on the presentation machine. The decoder/orchestrator are hardware- and
display-agnostic and pass headless on recorded EEG.

## Operator app (Flet GUI) — recommended
A desktop operator console (`mindwear/gui/`, built with **Flet**, mirroring the pineuro rt-fMRI GUI)
wraps the whole pipeline: study/protocol management, a source + **contact-quality check**,
calibration, and a **live CEN / DMN / PDA plot** (CEN red · DMN blue · PDA green) during feedback,
with per-run CSV logging. It drives the headless `session_engine.SessionEngine`; the participant
PsychoPy stimulus opens on the main thread via a dispatcher (the macOS-safe Flet+PsychoPy pattern).

```bash
conda activate mindwear                 # env with flet, flet-charts, numpy, scipy, mne, psychopy
python mindwear/launch_gui.py           # from analysis/fingerprint/
#   or:  python -m mindwear.gui.app
```
Flow: **Study Manager → Study Editor** (Source / Decoder / Session / Feedback tabs) **→ Session
Runner** (contact-quality preview → Start calibration → calibration review → participant ready
screen → live feedback → ratings). Point a study's Source at **Replay** (a recorded `.fif`) to run
the whole console end-to-end with **no headset or license**; switch to **LSL** or **Cortex** for a
live headset — nothing else changes.

The headless engine is also runnable directly (parity with the console):
```bash
python mindwear/session_engine.py --source replay \
       --replay mindwear/testdata/dmnelf005_feedback_run-01_250Hz.fif --speed 0 \
       --subject P001 --calib-sec 12 --rest-sec 6 --feedback-sec 12
```

## Run a session (standalone scripts)
Two feedback front-ends, both driven by the same decoder:

**Ball task (matches the scanner paradigm)** — `eeg_balltask.py` is a faithful EEG port of the MRI
`rt-network_feedback.py`: white ball between a top **CEN (yellow)** and bottom **DMN (blue)** circle,
rising when CEN > DMN, hits reset the ball + shrink the circle, 30 s +/Relax baseline, post-run
sliders, per-volume CSV. MURFI is swapped for `EEGActivationCommunicator` (drop-in: same
`update()` / `get_roi_activation('cen'|'dmn', frame)` interface, backed by our decoder).
```bash
python eeg_balltask.py --participant rtbpd001 --run 1 --feedback Feedback --source cortex
# offline dry-run of the pipeline (needs PsychoPy for the window):
python eeg_balltask.py --participant test --source replay --replay testdata/…_250Hz.fif \
                       --run-sec 60 --baseline-sec 6 --windowed
```
`--scale-factor` tunes ball speed (default 10; tune in a pilot since EEG activation units differ
from MURFI's).

**Simple bars** — `run_nf.py` + `feedback_psychopy.py` (thermometer bars) for a minimal display:
```bash
python run_nf.py --source cortex --subject P001 --calibrate --feedback psychopy
```

## Setup
```bash
pip install websocket-client pyyaml numpy scipy    # core + Cortex
pip install pylsl        # optional: EmotivPRO LSL path
pip install psychopy     # feedback display (Phase 4)
```
1. Install **EmotivPRO / Emotiv Launcher** and pair the EPOC X (+ USB dongle). Raw EEG needs a
   raw-data license / EmotivPRO subscription.
2. Create an app at https://www.emotiv.com/developer → copy `credentials.example.yaml` to
   `credentials.yaml` and fill in `client_id` / `client_secret`.
3. Start EmotivPRO (the Cortex service listens on `wss://localhost:6868`).

## Connect first (Phase 1)
```bash
python connect_test.py --source cortex --seconds 8     # live EPOC X (approve app on first run)
python connect_test.py --source lsl                    # EmotivPRO LSL outlet
python connect_test.py --source replay --replay eeg.fif --speed 0   # no hardware, recorded EEG
```
Prints sample rate, channel list, per-channel RMS and a poor-contact (flat-line) check — re-wet the
felt sensors on any channel flagged ⚠ before recording.

## License-free (emokit/CyKit) — status: BLOCKED on this EPOC X
`EmokitSource` + `pair_headset.py` + `decode_probe.py` read the dongle directly (no Cortex license).
Verified on this unit (serial `UD202007080050E1`): the dongle pairs and streams **AES-encrypted**
32-byte reports at ~128 Hz. Three independent angles were tried and all failed to recover the key
(`decode_probe.py`, `probe_feature_report.py`):
1. **7 documented emokit/CyKit serial-derived key models** (Epoc/Insight × Premium/Consumer/14-bit),
   standard "last 4 serial chars" slicing — counter-validation ~0.25 (noise) for all.
2. **Full 32-byte position scan** — same 7 keys, checking every decrypted byte offset (not just 0)
   in case the counter simply landed elsewhere — best score 0.28 (still noise).
3. **Brute-force sweep, 4,368 keys** — the 7 templates × all 24 orderings of a 4-character window ×
   all 26 possible 4-char windows of the 16-char serial (covers the case where "last 4 chars" is the
   wrong slicing convention for modern 16-char serials vs. the 6-char example serials the algorithm
   was designed around) — zero candidates scored above 0.5.
4. **HID feature-report channel** (`probe_feature_report.py`) — the streaming interface exposes a
   17-byte feature report; read repeatedly across time it is **static** (matches device metadata:
   packet size 32, sample rate 128), not a per-session nonce/handshake.

Conclusion: the key is not any static function of the serial number reachable from these angles.
The EPOC X firmware hardened its encryption (key most likely tied to Cortex-side authorization, or
exchanged over a lower USB layer not visible via HID feature reports), so the older-EPOC bypass does
**not** work here. These tools still work for original EPOC / EPOC+.
→ For this EPOC X, use the **Cortex API** path (`--source cortex`) with an EmotivPRO / raw-EEG license.
→ Remaining unlicensed option, not yet attempted: USB packet capture (Wireshark + USBPcap) of an
  EmotivPRO session to look for a key-exchange at the control-transfer level.

**Running these probes (macOS, `mindwear` conda env):**
```bash
conda activate mindwear
pip install hid pycryptodome   # brew install hidapi if not already present
DYLD_LIBRARY_PATH=/opt/homebrew/lib python decode_probe.py --n 400
DYLD_LIBRARY_PATH=/opt/homebrew/lib python probe_feature_report.py
```
`DYLD_LIBRARY_PATH` is required — Homebrew's `libhidapi.dylib` lives in `/opt/homebrew/lib`, which
isn't on the default dylib search path. Also note: the current PyPI `hid` package (1.0.9) uses the
`hid.Device(path=...)` / `.nonblocking` API, not the older `hid.device()` / `.open_path()` /
`.set_nonblocking()` API some emokit-era snippets online still show.

## Acquisition paths
- **Cortex API** (`cortex.py`, `CortexSource`): the standard raw-EEG route. Handshake =
  requestAccess → authorize → queryHeadsets → controlDevice(connect) → createSession → subscribe(eeg).
- **LSL** (`LSLSource`): enable the LSL outlet in EmotivPRO; simplest if you already use EmotivPRO.
- **Replay** (`ReplaySource`): stream a recorded `.fif`/`.npz` at real-time rate for offline dev/test.

EPOC X streams at 128/256 SPS; the pipeline resamples to 250 Hz to match the frozen EFP model.
