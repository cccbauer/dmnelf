"""Load/save MindWear studies as YAML under ``~/.mindwear/studies/``."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from .models import StudyConfig, StudyMetadata

STUDIES_DIR = Path.home() / ".mindwear" / "studies"


def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("_")
    return s or "study"


class ConfigManager:
    def __init__(self, studies_dir: Path | None = None):
        self.dir = Path(studies_dir) if studies_dir else STUDIES_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str) -> Path:
        return self.dir / f"{_slug(name)}.yaml"

    def list_studies(self) -> list[StudyMetadata]:
        out = []
        for p in sorted(self.dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(p.read_text()) or {}
                out.append(StudyMetadata.from_dict(data.get("_metadata", {"name": p.stem})))
            except Exception:
                continue
        out.sort(key=lambda m: m.modified, reverse=True)
        return out

    def load(self, name: str) -> StudyConfig:
        p = self.path_for(name)
        data = yaml.safe_load(p.read_text()) or {}
        return StudyConfig.from_dict(data, file_path=p)

    def save(self, study: StudyConfig) -> Path:
        study.metadata.update_modified()
        p = self.path_for(study.metadata.name)
        p.write_text(yaml.safe_dump(study.to_dict(), sort_keys=False))
        study.file_path = p
        return p

    def delete(self, name: str) -> None:
        self.path_for(name).unlink(missing_ok=True)

    def duplicate(self, name: str) -> StudyConfig:
        study = self.load(name)
        study.metadata.name = f"{name} copy"
        self.save(study)
        return study

    def exists(self, name: str) -> bool:
        return self.path_for(name).exists()
