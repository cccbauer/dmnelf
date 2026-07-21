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
    "cap31": {"label": "Research cap — 32 channel", "model_path": str(MODEL_DIR / "efp_cap31_model.npz")},
}

# default session_config blocks (mirror EngineConfig defaults)
DEFAULT_SESSION_CONFIG: dict[str, Any] = {
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

    def to_engine_config(self, subject: str, run: int):
        """Lower into a runnable EngineConfig (imported lazily to keep models flet-free)."""
        import sys

        HERE = Path(__file__).resolve().parent.parent
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        from session_engine import DEFAULT_MODEL, EngineConfig

        src, dec, sess = self.source, self.decoder, self.session
        montage = dec.get("montage", "epoc12")
        preset_path = MONTAGE_PRESETS.get(montage, {}).get("model_path")
        return EngineConfig(
            subject=subject,
            run=run,
            model_path=dec.get("model_path") or preset_path or str(DEFAULT_MODEL),
            source=src.get("type", "replay"),
            replay_path=src.get("replay_path") or None,
            replay_speed=float(src.get("replay_speed", 1.0)),
            credentials_path=src.get("credentials_path") or None,
            do_calibrate=bool(sess.get("calibrate", True)),
            calib_sec=float(sess.get("calib_sec", 60.0)),
            rest_sec=float(sess.get("rest_sec", 30.0)),
            feedback_sec=float(sess.get("feedback_sec", 300.0)),
        )
