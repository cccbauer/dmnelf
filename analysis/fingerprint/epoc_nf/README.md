# EPOC X neurofeedback system

Real-time EEG neurofeedback on the Emotiv EPOC X, running the frozen DMNELF **EFP** decoder to
produce the **PDA = CEN − DMN** signal and drive PsychoPy feedback. Deploys the decoder validated
in [`../efp_meirhasson/DEPLOY_EPOC.md`](../efp_meirhasson/DEPLOY_EPOC.md) (EPOC-12 montage retains
~92% of clean-CEN decoding).

## Status
- **Phase 1 — connection layer** ✅ `cortex.py` · `sources.py` · `connect_test.py`
- Phase 2 — frozen decoder export (`export_model.py` → `model/efp_epoc_model.npz`)
- Phase 3 — real-time engine (`rt_features.py`, `decoder.py`, `calibration.py`)
- Phase 4 — PsychoPy feedback (`feedback_psychopy.py`) + `run_nf.py`

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
