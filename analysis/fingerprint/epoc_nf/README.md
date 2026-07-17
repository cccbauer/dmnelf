# EPOC X neurofeedback system

Real-time EEG neurofeedback on the Emotiv EPOC X, running the frozen DMNELF **EFP** decoder to
produce the **PDA = CEN − DMN** signal and drive PsychoPy feedback. Deploys the decoder validated
in [`../efp_meirhasson/DEPLOY_EPOC.md`](../efp_meirhasson/DEPLOY_EPOC.md) (EPOC-12 montage retains
~92% of clean-CEN decoding).

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

## Run a session
```bash
python run_nf.py --source cortex --subject P001 --calibrate --feedback psychopy   # live
python run_nf.py --source replay --replay testdata/…_250Hz.fif --feedback none \
                 --calibrate --speed 0                                            # dry-run
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

## Acquisition paths
- **Cortex API** (`cortex.py`, `CortexSource`): the standard raw-EEG route. Handshake =
  requestAccess → authorize → queryHeadsets → controlDevice(connect) → createSession → subscribe(eeg).
- **LSL** (`LSLSource`): enable the LSL outlet in EmotivPRO; simplest if you already use EmotivPRO.
- **Replay** (`ReplaySource`): stream a recorded `.fif`/`.npz` at real-time rate for offline dev/test.

EPOC X streams at 128/256 SPS; the pipeline resamples to 250 Hz to match the frozen EFP model.
