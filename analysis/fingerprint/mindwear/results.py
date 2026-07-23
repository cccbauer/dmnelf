"""Read-back analysis of saved MindWear sessions — R-mbNF results (flet-free, reusable by CLI/GUI).

Parses a subject's data folder into per-run regulation + direction-report accuracy. Works with the
BIDS names (``sub-<label>_task-<stage>_run-<NN>_desc-{decoder,directions}.csv``) and the legacy
``nf_*.csv`` / ``directions_*.csv`` — both carry the same ``stage``/``run``/``phase`` columns, so
parsing keys off the CSV content, not the filename.
"""
from __future__ import annotations

import csv
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from session_engine import DATA_DIR, subject_dir  # noqa: E402


def _f(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _mean(xs) -> float:
    xs = [x for x in xs if x == x]              # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")


@dataclass
class RunResult:
    """One task block (a transfer block or one feedback run)."""

    stage: str                      # transferpre | feedback | transferpost
    run: int                        # feedback-run index (0 for transfer)
    n_tr: int = 0
    mean_pda: float = float("nan")          # mean PDA over the task phase
    mean_pda_z: float = float("nan")        # mean feedback-phase z vs that block's rest baseline
    # R-mbNF direction report (feedback blocks only)
    pda_sign: int | None = None
    answer: str | None = None
    true_direction: str | None = None
    correct: bool | None = None

    @property
    def is_feedback(self) -> bool:
        return self.stage == "feedback"

    @property
    def label(self) -> str:
        return {"transferpre": "transfer (pre)",
                "transferpost": "transfer (post)"}.get(self.stage, f"feedback run {self.run}")


@dataclass
class SubjectResult:
    subject: str
    runs: list[RunResult] = field(default_factory=list)

    @property
    def feedback_runs(self) -> list[RunResult]:
        return [r for r in self.runs if r.is_feedback]

    @property
    def reports(self) -> list[RunResult]:
        return [r for r in self.feedback_runs if r.correct is not None]

    @property
    def n_reports(self) -> int:
        return len(self.reports)

    @property
    def n_correct(self) -> int:
        return sum(1 for r in self.reports if r.correct)

    @property
    def accuracy(self) -> float | None:
        return self.n_correct / self.n_reports if self.n_reports else None

    @property
    def mean_feedback_pda_z(self) -> float:
        return _mean([r.mean_pda_z for r in self.feedback_runs])

    @property
    def has_data(self) -> bool:
        return bool(self.runs)


# ── discovery ──────────────────────────────────────────────────────────────
def subjects_for(basename: str, data_dir: Path | str = DATA_DIR) -> list[str]:
    """Subject ids (data-folder names) that belong to a study with the given subject *basename*.

    Empty basename returns every subject folder."""
    root = Path(data_dir)
    if not root.exists():
        return []
    base = (basename or "").strip()
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (not base or p.name.startswith(base)))


# ── per-file parsing ─────────────────────────────────────────────────────────
def _decoder_files(d: Path) -> list[Path]:
    return sorted(set(d.glob("*_desc-decoder.csv")) | set(d.glob("nf_*.csv")))


def _directions_files(d: Path) -> list[Path]:
    return sorted(set(d.glob("*_desc-directions.csv")) | set(d.glob("directions_*.csv")))


def _parse_decoder(path: Path) -> RunResult | None:
    stage, run = "", 0
    task_pda: list[float] = []
    fb_z: list[float] = []
    n = 0
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if not stage:
                    stage = (row.get("stage") or "").strip()
                    run = int(_f(row.get("run"))) if row.get("run") not in (None, "") else 0
                phase = (row.get("phase") or "").strip()
                if phase in ("feedback", "transfer"):
                    task_pda.append(_f(row.get("pda")))
                    n += 1
                if phase == "feedback":
                    fb_z.append(_f(row.get("pda_z")))
    except Exception:
        return None
    return RunResult(stage=stage or "feedback", run=run, n_tr=n,
                     mean_pda=_mean(task_pda), mean_pda_z=_mean(fb_z))


def _parse_directions(path: Path) -> list[RunResult]:
    out: list[RunResult] = []
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                raw_correct = str(row.get("correct", "")).strip().lower()
                # blank -> "not_sure" abstention (see session_engine._do_task_block); excluded
                # from accuracy by RunResult.correct is None, not scored as incorrect.
                correct = None if raw_correct == "" else raw_correct in ("true", "1", "yes")
                out.append(RunResult(
                    stage="feedback", run=int(_f(row.get("run")) or 0),
                    mean_pda=_f(row.get("mean_task_pda")),
                    pda_sign=int(_f(row.get("pda_sign")) or 1),
                    answer=(row.get("answer") or None),
                    true_direction=(row.get("true_direction") or None),
                    correct=correct))
    except Exception:
        pass
    return out


# ── subject-level assembly ────────────────────────────────────────────────────
_STAGE_ORDER = {"transferpre": 0, "feedback": 1, "transferpost": 2}


def subject_result(subject: str, data_dir: Path | str = DATA_DIR) -> SubjectResult:
    d = subject_dir(subject, data_dir)
    res = SubjectResult(subject=subject)
    if not d.exists():
        return res

    runs = [r for r in (_parse_decoder(p) for p in _decoder_files(d)) if r is not None]
    # fold the direction reports into the matching feedback runs (by run index)
    reports: dict[int, RunResult] = {}
    for path in _directions_files(d):
        for rep in _parse_directions(path):
            reports[rep.run] = rep
    for r in runs:
        if r.is_feedback and r.run in reports:
            rep = reports.pop(r.run)
            r.pda_sign, r.answer, r.true_direction, r.correct = (
                rep.pda_sign, rep.answer, rep.true_direction, rep.correct)
    # direction reports with no decoder file (e.g. decoder CSV missing) still count
    runs.extend(reports.values())

    runs.sort(key=lambda r: (_STAGE_ORDER.get(r.stage, 1), r.run))
    res.runs = runs
    return res


def study_results(basename: str, data_dir: Path | str = DATA_DIR) -> list[SubjectResult]:
    return [subject_result(s, data_dir) for s in subjects_for(basename, data_dir)]


# ── management (destructive — the GUI confirms first) ─────────────────────────
def subject_files(subject: str, data_dir: Path | str = DATA_DIR) -> list[str]:
    """Names of the saved files under this subject's folder (empty if none)."""
    d = subject_dir(subject, data_dir)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_file() and not p.name.startswith("."))


def delete_subject(subject: str, data_dir: Path | str = DATA_DIR) -> bool:
    """Permanently remove a subject's data folder. Returns True if something was deleted."""
    d = subject_dir(subject, data_dir)
    if not d.exists():
        return False
    shutil.rmtree(d)
    return True


def rename_subject(old: str, new: str, data_dir: Path | str = DATA_DIR) -> bool:
    """Rename a subject: move its folder and rewrite the old id embedded in each BIDS filename
    (``sub-<old>_...`` -> ``sub-<new>_...``). Refuses if *new* is blank, unchanged, or already
    exists. Returns True on success."""
    new = (new or "").strip()
    old_d = subject_dir(old, data_dir)
    new_d = subject_dir(new, data_dir)
    if not new or new == old or not old_d.exists() or new_d.exists():
        return False
    for p in list(old_d.iterdir()):
        if p.is_file() and old in p.name:
            p.rename(old_d / p.name.replace(old, new))
    old_d.rename(new_d)
    return True
