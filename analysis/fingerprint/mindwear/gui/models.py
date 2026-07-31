"""Study/protocol data models for the MindWear GUI.

A *study* bundles everything needed to run sessions for one protocol: which EEG source to
use, the frozen decoder model, the phase timing, and the feedback display. Persisted as one
YAML file per study under ``~/.mindwear/studies/`` (see :mod:`config_manager`), with a
``_metadata`` block for the manager cards — the same layout pineuro uses.

:meth:`StudyConfig.to_engine_config` lowers a study + participant/run into the
:class:`mindwear.session_engine.EngineConfig` the headless engine consumes, so the GUI and the
engine never disagree about defaults.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

# Named montage presets — each ships a decoder trained on that electrode set.
# "custom" means the operator has typed an explicit model_path of their own.
MONTAGE_PRESETS: dict[str, dict[str, str]] = {
    "epoc12": {"label": "EPOC-X — 12 channel", "model_path": str(MODEL_DIR / "efp_epoc_model.npz")},
    "epoc_dual": {"label": "EPOC-X — dual electrode (P8 CEN / O1 DMN, experimental)",
                  "model_path": str(MODEL_DIR / "efp_epoc_dual_model.npz")},
    "cap31": {"label": "Research cap — 32 channel", "model_path": str(MODEL_DIR / "efp_cap31_model.npz")},
}

# Supported EEG headsets. Each maps to a default acquisition source + decoder montage.
HEADSET_PRESETS: dict[str, dict[str, str]] = {
    "epocx":    {"label": "Emotiv EPOC X (14-ch)",   "source": "lsl", "montage": "epoc12"},
    "epocflex": {"label": "Emotiv EPOC Flex (32-ch)", "source": "lsl", "montage": "cap31"},
    "bp32":     {"label": "Brain Products (32-ch)",   "source": "lsl", "montage": "cap31"},
}

# Protocol templates. mbNF = mindfulness-based NF (veridical feedback); R-mbNF adds per-run
# randomization of the PDA->ball direction + an end-of-run "up/down" report (accuracy self-test).
PROTOCOL_TYPES = ("mbNF", "R-mbNF")

# default protocol timing (seconds). Transfer blocks show static targets (no feedback); the
# feedback block repeats n_runs times.
DEFAULT_PROTOCOL: dict[str, Any] = {
    "type": "mbNF",
    # calibration cycles rest -> flanker -> rest -> self ("induction", default) or rest -> noting
    # ("noting" — replicates the rest-baseline/mental-noting design the frozen ridge was actually
    # trained on), `cycles` times, so more than quiet rest recurs throughout — deliberately drives
    # the decoded PDA to both poles for a better-conditioned calibration fit. cycles=0 falls back
    # to a single flat rest_sec window (legacy behavior), regardless of type.
    "calibration": {"type": "induction", "rest_sec": 15.0, "cycles": 3,
                    "self_sec": 30.0, "flanker_sec": 45.0, "noting_sec": 60.0},
    "transfer_pre":  {"enabled": True, "rest_sec": 30.0, "task_sec": 150.0},
    "feedback":      {"rest_sec": 30.0, "task_sec": 150.0, "n_runs": 1},
    "transfer_post": {"enabled": True, "rest_sec": 30.0, "task_sec": 150.0},
}


def build_blocks(protocol: dict, start_run: int = 1, first_run: bool = True) -> list[dict]:
    """Expand a protocol dict into the ordered block list the engine runs. Contact/QC is a GUI
    pre-flight step, not an engine block, so it is not included here.

    Block kinds: 'calibration' (rest/self/flanker or rest/noting cycles — see
    ``calibration["type"]`` — fit z-score stats + a PDA-separation QA readout), 'transfer' (static
    targets, silent record), 'feedback' (veridical ball). ``randomize`` on feedback blocks flips
    the PDA->direction sign per run and triggers the end-of-run up/down question (R-mbNF).

    ``first_run=False`` (an operator manually starting another session for a participant who
    already has a saved calibration) omits calibration AND the transfer blocks — those are
    once-per-participant onboarding, not per-attempt — leaving only feedback blocks, numbered
    from ``start_run`` so their BIDS filenames don't collide with the earlier attempt's."""
    p = {**DEFAULT_PROTOCOL, **(protocol or {})}
    randomize = p.get("type") == "R-mbNF"
    blocks: list[dict] = []
    if first_run:
        cal = p["calibration"]
        blocks.append({"kind": "calibration", "stage": "calibration",
                       "type": cal.get("type", "induction"),
                       "rest_sec": float(cal["rest_sec"]), "cycles": int(cal.get("cycles", 0)),
                       "self_sec": float(cal.get("self_sec", 30.0)),
                       "flanker_sec": float(cal.get("flanker_sec", 45.0)),
                       "noting_sec": float(cal.get("noting_sec", 60.0))})
        if p["transfer_pre"].get("enabled", True):
            tp = p["transfer_pre"]
            blocks.append({"kind": "transfer", "stage": "transferpre",
                           "rest_sec": float(tp["rest_sec"]), "task_sec": float(tp["task_sec"])})
    fb = p["feedback"]
    n_runs = int(fb.get("n_runs", 1))
    for i in range(max(1, n_runs)):
        blocks.append({"kind": "feedback", "stage": "feedback", "run": start_run + i, "n_runs": n_runs,
                       "randomize": randomize, "rest_sec": float(fb["rest_sec"]),
                       "task_sec": float(fb["task_sec"]), "is_last_run": i == max(1, n_runs) - 1})
    if first_run and p["transfer_post"].get("enabled", True):
        tp = p["transfer_post"]
        blocks.append({"kind": "transfer", "stage": "transferpost",
                       "rest_sec": float(tp["rest_sec"]), "task_sec": float(tp["task_sec"])})
    return blocks


# default session_config blocks (mirror EngineConfig defaults)
DEFAULT_SESSION_CONFIG: dict[str, Any] = {
    "headset": "epocx",               # epocx | epocflex | bp32 — see HEADSET_PRESETS
    "subjects": {
        "basename": "sub",            # subject-ID prefix -> sub001, sub002, ... (BIDS sub-<label>)
        "n_subjects": 1,              # planned enrollment
    },
    "source": {
        "type": "replay",             # replay | cortex | lsl | emokit
        "replay_path": "",
        "replay_speed": 1.0,
        "credentials_path": "",       # blank -> mindwear/credentials.yaml
    },
    "decoder": {
        "montage": "epoc12",          # epoc12 | cap31 | custom — see MONTAGE_PRESETS
        "model_path": "",             # blank -> preset's model_path (epoc12 -> efp_epoc_model.npz)
    },
    "protocol": dict(DEFAULT_PROTOCOL),
    # legacy single-run fields — kept for the older direct-run path + backward compat with
    # pre-protocol study YAMLs. The protocol above is the source of truth for new studies.
    "session": {
        "calibrate": True,
        "calib_sec": 60.0,
        "rest_sec": 30.0,
        "feedback_sec": 300.0,
        "n_runs": 4,
    },
    "feedback": {
        "mode": "ball",               # ball | bars | none
        "scale_factor": 10.0,
        "target_z": 1.0,
    },
}


@dataclass
class StudyMetadata:
    name: str
    description: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    modified: str = field(default_factory=lambda: datetime.now().isoformat())

    def update_modified(self) -> None:
        self.modified = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description,
                "created": self.created, "modified": self.modified}

    @classmethod
    def from_dict(cls, d: dict) -> "StudyMetadata":
        now = datetime.now().isoformat()
        return cls(name=d.get("name", ""), description=d.get("description", ""),
                   created=d.get("created", now), modified=d.get("modified", now))


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively fill *over* onto a copy of *base* (so new default keys appear on old files)."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class StudyConfig:
    metadata: StudyMetadata
    session_config: dict = field(default_factory=lambda: copy.deepcopy(DEFAULT_SESSION_CONFIG))
    file_path: Path | None = None

    def to_dict(self) -> dict:
        return {"_metadata": self.metadata.to_dict(), **self.session_config}

    @classmethod
    def from_dict(cls, data: dict, file_path: Path | None = None) -> "StudyConfig":
        meta = StudyMetadata.from_dict(data.get("_metadata", {}))
        sc = {k: v for k, v in data.items() if k != "_metadata"}
        return cls(metadata=meta, session_config=_deep_merge(DEFAULT_SESSION_CONFIG, sc), file_path=file_path)

    @classmethod
    def new(cls, name: str = "New Study") -> "StudyConfig":
        return cls(metadata=StudyMetadata(name=name), session_config=copy.deepcopy(DEFAULT_SESSION_CONFIG))

    # convenient typed getters -------------------------------------------------
    @property
    def source(self) -> dict:
        return self.session_config["source"]

    @property
    def decoder(self) -> dict:
        return self.session_config["decoder"]

    @property
    def session(self) -> dict:
        return self.session_config["session"]

    @property
    def feedback(self) -> dict:
        return self.session_config["feedback"]

    @property
    def headset(self) -> str:
        return self.session_config.get("headset", "epocx")

    @property
    def subjects(self) -> dict:
        return self.session_config.setdefault("subjects", {"basename": "sub", "n_subjects": 1})

    def subject_id(self, n: int) -> str:
        """Zero-padded subject id for the nth participant, e.g. basename 'dmnelf' -> 'dmnelf001'.
        Padded to 3 digits (001-based) so ids sort correctly and match BIDS sub-<label> naming."""
        base = (self.subjects.get("basename") or "sub").strip()
        return f"{base}{n:03d}"

    @property
    def protocol(self) -> dict:
        return self.session_config.setdefault("protocol", dict(DEFAULT_PROTOCOL))

    def blocks(self, start_run: int = 1, first_run: bool = True) -> list[dict]:
        """The ordered engine block list for this study's protocol."""
        return build_blocks(self.protocol, start_run=start_run, first_run=first_run)

    def to_engine_config(self, subject: str, run: int, calib_path: str | None = None,
                         bids_session: str = ""):
        """Lower into a runnable EngineConfig (imported lazily to keep models flet-free).

        ``calib_path``: an already-saved calibration for this subject (see
        session_engine.existing_calibration_path). When given, the block list skips
        calibration/transfer entirely (see build_blocks) and the engine loads this calibration
        instead of fitting a new one — a manual second/third/... run for a participant who's
        already calibrated shouldn't redo it.

        ``bids_session``: the operator-entered BIDS ses- label (a visit/day), threaded into
        EngineConfig.session — named to avoid colliding with ``self.session``, this study's own
        (unrelated) legacy session-timing dict, below."""
        import sys

        HERE = Path(__file__).resolve().parent.parent
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        from session_engine import DEFAULT_MODEL, EngineConfig

        src, dec, sess = self.source, self.decoder, self.session
        montage = dec.get("montage", "epoc12")
        preset_path = MONTAGE_PRESETS.get(montage, {}).get("model_path")
        first_run = calib_path is None
        return EngineConfig(
            subject=subject,
            run=run,
            session=bids_session,
            model_path=dec.get("model_path") or preset_path or str(DEFAULT_MODEL),
            source=src.get("type", "replay"),
            replay_path=src.get("replay_path") or None,
            replay_speed=float(src.get("replay_speed", 1.0)),
            credentials_path=src.get("credentials_path") or None,
            do_calibrate=first_run,
            calib_path=calib_path,
            calib_sec=float(sess.get("calib_sec", 60.0)),
            rest_sec=float(sess.get("rest_sec", 30.0)),
            feedback_sec=float(sess.get("feedback_sec", 300.0)),
            blocks=self.blocks(start_run=run, first_run=first_run),
            protocol_type=self.protocol.get("type", "mbNF"),
        )
