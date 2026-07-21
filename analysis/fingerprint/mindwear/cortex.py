#!/usr/bin/env python3
"""
cortex.py  —  Emotiv Cortex API client (EPOC X raw-EEG streaming)
-----------------------------------------------------------------
Minimal, dependency-light client for the Cortex v2 JSON-RPC WebSocket API. Establishes the
auth handshake and streams the 14-channel raw EEG needed by the EFP neurofeedback decoder.

Flow (Cortex v2):
  requestAccess -> authorize (clientId/secret -> cortexToken) -> queryHeadsets ->
  controlDevice("connect") -> createSession("open") -> subscribe(["eeg"]) -> recv EEG frames.

Raw EEG is LICENSE-GATED on Emotiv: the app must have raw-EEG access approved (EmotivPRO /
a raw-data license) and, on first run, the user must APPROVE this app in the Emotiv Launcher.

Requires: pip install websocket-client   (env with SSL).  Cortex service must be running
(EmotivPRO / Emotiv Launcher) and the headset paired.
"""
import contextlib
import json
import ssl
import time

try:
    import websocket  # websocket-client
except ImportError:                       # pragma: no cover
    websocket = None

CORTEX_URL = "wss://localhost:6868"

# EPOC X 14-channel montage, in cap order (matches the EFP EPOC-12 subset + AF3/AF4).
EPOC_CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                 "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]


class CortexError(RuntimeError):
    pass


class CortexClient:
    """Synchronous Cortex client. Use as a context manager or call connect()/close()."""

    def __init__(self, client_id, client_secret, license_id=None, url=CORTEX_URL, timeout=10):
        if websocket is None:
            raise CortexError("websocket-client not installed:  pip install websocket-client")
        if not client_id or not client_secret:
            raise CortexError("Cortex client_id / client_secret required (see credentials.yaml).")
        self.client_id = client_id
        self.client_secret = client_secret
        self.license_id = license_id
        self.url = url
        self.timeout = timeout
        self.ws = None
        self.token = None
        self.session_id = None
        self.headset_id = None
        self.eeg_cols = None      # column labels from subscribe response
        self.sfreq = None
        self._id = 0

    # ── low-level JSON-RPC ────────────────────────────────────────────────
    def connect(self):
        self.ws = websocket.create_connection(
            self.url, sslopt={"cert_reqs": ssl.CERT_NONE}, timeout=self.timeout)
        return self

    def close(self):
        if self.ws is not None:
            try:
                self.ws.close()
            finally:
                self.ws = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    def _call(self, method, params=None, want_id=True):
        """Send a request and return the matching 'result', skipping warnings/stream frames."""
        self._id += 1
        rid = self._id
        self.ws.send(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method,
                                 "params": params or {}}))
        t0 = time.time()
        while time.time() - t0 < 30:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == rid:
                if "error" in msg:
                    raise CortexError(f"{method}: {msg['error'].get('message', msg['error'])}")
                return msg.get("result")
            # else: warning / other-id / stream frame arriving early — ignore during setup
        raise CortexError(f"{method}: timed out waiting for response")

    # ── handshake ─────────────────────────────────────────────────────────
    def request_access(self):
        r = self._call("requestAccess", {"clientId": self.client_id,
                                         "clientSecret": self.client_secret})
        if not r.get("accessGranted", False):
            raise CortexError("Access not granted — APPROVE this app in the Emotiv Launcher, "
                              f"then re-run.  ({r.get('message', '')})")
        return r

    def authorize(self):
        p = {"clientId": self.client_id, "clientSecret": self.client_secret, "debit": 1}
        if self.license_id:
            p["license"] = self.license_id
        self.token = self._call("authorize", p)["cortexToken"]
        return self.token

    def query_headsets(self):
        return self._call("queryHeadsets", {})

    def connect_headset(self, headset_id=None):
        """Pick the first headset (or the given id); connect it if not already."""
        hs = self.query_headsets()
        if not hs:
            raise CortexError("No headset found. Turn on the EPOC X + USB dongle and pair it "
                              "in the Emotiv Launcher.")
        chosen = next((h for h in hs if h["id"] == headset_id), hs[0])
        self.headset_id = chosen["id"]
        if chosen.get("status") != "connected":
            self._call("controlDevice", {"command": "connect", "headset": self.headset_id})
            time.sleep(2)   # allow the device to come up
        return chosen

    def create_session(self):
        r = self._call("createSession", {"cortexToken": self.token,
                                         "headset": self.headset_id, "status": "open"})
        self.session_id = r["sessionId"]
        return self.session_id

    def subscribe_eeg(self):
        r = self._call("subscribe", {"cortexToken": self.token, "session": self.session_id,
                                     "streams": ["eeg"]})
        ok = r.get("success", [])
        if not ok:
            fail = r.get("failure", [{}])[0]
            raise CortexError(f"EEG subscribe failed: {fail.get('message', fail)} "
                              "(raw-EEG license required).")
        info = ok[0]["eeg"]
        self.eeg_cols = info["cols"]                       # e.g. COUNTER,INTERPOLATED,AF3,...,MARKERS
        self.sfreq = float(info.get("sampleRate", 128))
        return self.eeg_cols, self.sfreq

    def open_eeg_stream(self, headset_id=None):
        """Full handshake -> ready to stream. Returns (channels, sfreq)."""
        self.request_access(); self.authorize()
        self.connect_headset(headset_id); self.create_session()
        cols, sf = self.subscribe_eeg()
        chans = [c for c in cols if c in EPOC_CHANNELS]
        return chans, sf

    # ── streaming ─────────────────────────────────────────────────────────
    def stream(self):
        """Yield {'t': time, 'eeg': {chan: microvolts}} per sample after open_eeg_stream()."""
        if self.eeg_cols is None:
            raise CortexError("Call open_eeg_stream() before stream().")
        idx = {c: i for i, c in enumerate(self.eeg_cols)}
        chan_idx = {c: idx[c] for c in EPOC_CHANNELS if c in idx}
        while True:
            msg = json.loads(self.ws.recv())
            if "eeg" in msg:
                row = msg["eeg"]
                yield {"t": msg.get("time", time.time()),
                       "eeg": {c: row[i] for c, i in chan_idx.items()}}

    def flush(self, max_seconds: float = 2.0) -> None:
        """Best-effort: discard whatever's already queued on the socket (call after a long pause
        so a resumed stream() doesn't rapid-fire through minutes of backlog)."""
        if self.ws is None:
            return
        orig_timeout = self.ws.gettimeout()
        try:
            self.ws.settimeout(0.05)
            t0 = time.time()
            while time.time() - t0 < max_seconds:
                try:
                    self.ws.recv()
                except Exception:
                    break
        finally:
            with contextlib.suppress(Exception):
                self.ws.settimeout(orig_timeout)
