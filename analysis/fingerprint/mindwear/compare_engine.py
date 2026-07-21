#!/usr/bin/env python3
"""
compare_engine.py  —  side-by-side fMRI(BOLD) vs EPOC(EEG-decoder) replay of the same run
-----------------------------------------------------------------------------------------
Drives a two-panel ball-task comparison: the SAME recorded DMNELF/rtBPD run produces two network
tracks in lockstep —

  * BOLD  : the observed scanner CEN/DMN timeseries (the ground truth the decoder was validated
            against; loaded from fsnr_eeg/results/cen_ceiling/cenmean_*_<subject>.npz).
  * EPOC  : CEN/DMN from the frozen EFP decoder run over the recorded EEG (the portable-headset
            estimate), via the SAME real-time path used live (ReplaySource -> RTFeatureExtractor
            -> Decoder).

Both tracks are precomputed, aligned by TR (the decoder's k-th estimate is run-TR k + n_delays-1),
normalized to comparable amplitude, and then streamed at real-time TR cadence. Per TR it emits a
:class:`CompareUpdate` (both tracks) and exposes :meth:`latest` for the dual-ball stimulus to poll.
"""
from __future__ import annotations

import glob
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
FP = HERE.parent                       # analysis/fingerprint
CEN_CEILING = FP / "fsnr_eeg" / "results" / "cen_ceiling"
DEFAULT_MODEL = HERE / "model" / "efp_epoc_model.npz"


@dataclass
class CompareUpdate:
    tr: int                 # aligned run-TR index
    bold_cen: float
    bold_dmn: float
    eeg_cen: float
    eeg_dmn: float

    @property
    def bold_pda(self) -> float:
        return self.bold_cen - self.bold_dmn

    @property
    def eeg_pda(self) -> float:
        return self.eeg_cen - self.eeg_dmn


def find_bold_npz(subject: str) -> Optional[Path]:
    """Locate the observed-BOLD npz for *subject* (cenmean_<cohort>_<subject>.npz)."""
    hits = glob.glob(str(CEN_CEILING / f"cenmean_*_{subject}.npz"))
    return Path(hits[0]) if hits else None


class ComparisonEngine:
    def __init__(self, subject: str, run: int, replay_path: str, model_path: Optional[str] = None,
                 on_update: Optional[Callable[[CompareUpdate], None]] = None,
                 on_status: Optional[Callable[[str], None]] = None,
                 speed: float = 1.0):
        self.subject = subject
        self.run = int(run)
        self.replay_path = replay_path
        self.model_path = model_path or str(DEFAULT_MODEL)
        self._on_update = on_update
        self._on_status = on_status
        self.speed = speed

        self.tr: float = 1.2
        self.n: int = 0
        self.corr_pda: float = float("nan")   # EEG-PDA vs BOLD-PDA over the run (Pearson r)
        self.error: Optional[str] = None

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[CompareUpdate] = None
        self._series: list[CompareUpdate] = []

    # ── control ──────────────────────────────────────────────────────────
    def start(self) -> "ComparisonEngine":
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ComparisonEngine", daemon=True)
        self._thread.start()
        return self

    def stop(self, join: bool = False, timeout: float = 3.0) -> None:
        self._stop.set()
        if join and self._thread:
            self._thread.join(timeout)

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def latest(self) -> Optional[CompareUpdate]:
        with self._lock:
            return self._latest

    @property
    def phase(self) -> str:
        return "feedback" if self.is_running() else "done"

    def _status(self, msg: str) -> None:
        if self._on_status:
            self._on_status(msg)

    # ── precompute both aligned, normalized tracks ───────────────────────
    def _prepare(self) -> None:
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        from decoder import Decoder
        from rt_features import RTFeatureExtractor
        from sources import ReplaySource

        # observed BOLD
        bold_path = find_bold_npz(self.subject)
        if bold_path is None:
            raise FileNotFoundError(f"no observed-BOLD file for '{self.subject}' in {CEN_CEILING}")
        z = np.load(bold_path, allow_pickle=True)
        rk, dk = f"run{self.run}", f"run{self.run}_dmn"
        if rk not in z or dk not in z:
            raise KeyError(f"{bold_path.name} lacks {rk}/{dk} (has {list(z.keys())[:6]}…)")
        bold_cen = np.asarray(z[rk], float)
        bold_dmn = np.asarray(z[dk], float)
        self._status(f"loaded BOLD {bold_path.name} [{rk}] ({bold_cen.size} TR)")

        # EPOC decoder over the recorded EEG (fast; whole run)
        model = np.load(self.model_path, allow_pickle=True)
        self.tr = float(model["tr"])
        n_delays = int(model["n_delays"])
        src = ReplaySource(self.replay_path, speed=0).open()
        feat = RTFeatureExtractor(model, src.channels)
        feat.set_sfreq(src.sfreq)
        dec = Decoder(model)                                  # running-z (validated path)
        ec, ed = [], []
        for _t, s in src.samples():
            d = feat.push(s)
            if d is None:
                continue
            out = dec.predict(d)
            if out is None:
                ec.append(np.nan); ed.append(np.nan)
            else:
                ec.append(out[0]); ed.append(out[1])
        src.close()
        eeg_cen = np.asarray(ec, float)
        eeg_dmn = np.asarray(ed, float)
        tr0 = n_delays - 1                                    # decode k -> run-TR k+tr0

        # align on run-TR index: decode k covers run-TR tr0..; clip to BOLD length
        k = np.arange(eeg_cen.size)
        run_tr = k + tr0
        m = run_tr < bold_cen.size
        run_tr, kk = run_tr[m], k[m]
        b_cen, b_dmn = bold_cen[run_tr], bold_dmn[run_tr]
        e_cen, e_dmn = eeg_cen[kk], eeg_dmn[kk]
        good = np.isfinite(e_cen) & np.isfinite(e_dmn) & np.isfinite(b_cen) & np.isfinite(b_dmn)
        run_tr, b_cen, b_dmn, e_cen, e_dmn = run_tr[good], b_cen[good], b_dmn[good], e_cen[good], e_dmn[good]

        # normalize each track to comparable amplitude (common center+scale per track keeps CEN vs
        # DMN relationship intact, so "which network is higher" is preserved)
        def norm(a, c):
            allv = np.concatenate([a, c])
            mu, sd = np.nanmean(allv), np.nanstd(allv) + 1e-9
            return (a - mu) / sd, (c - mu) / sd
        b_cen, b_dmn = norm(b_cen, b_dmn)
        e_cen, e_dmn = norm(e_cen, e_dmn)

        # headline: EEG-PDA vs BOLD-PDA correlation over the run
        b_pda, e_pda = b_cen - b_dmn, e_cen - e_dmn
        if b_pda.size >= 3:
            self.corr_pda = float(np.corrcoef(e_pda, b_pda)[0, 1])

        self._aligned = [CompareUpdate(int(t), float(bc), float(bd), float(ec2), float(ed2))
                         for t, bc, bd, ec2, ed2 in zip(run_tr, b_cen, b_dmn, e_cen, e_dmn)]
        self.n = len(self._aligned)
        self._status(f"prepared {self.n} aligned TRs — EEG↔BOLD PDA r={self.corr_pda:+.2f}")

    # ── stream at real-time cadence ──────────────────────────────────────
    def _run(self) -> None:
        try:
            self._status("preparing comparison (decoding EEG + loading BOLD)…")
            self._prepare()
            dt = self.tr / self.speed if self.speed > 0 else 0.0
            for u in self._aligned:
                if self._stop.is_set():
                    break
                with self._lock:
                    self._latest = u
                    self._series.append(u)
                if self._on_update:
                    self._on_update(u)
                if dt > 0:
                    self._stop.wait(dt)
            self._status(f"comparison complete ({self.n} TRs, PDA r={self.corr_pda:+.2f})")
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._status(f"ERROR: {self.error}")
        finally:
            self._stop.set()


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Precompute + print the fMRI↔EPOC comparison for a run.")
    ap.add_argument("--subject", default="dmnelf005")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--replay", default=str(HERE / "testdata" / "dmnelf005_feedback_run-01_250Hz.fif"))
    a = ap.parse_args()
    eng = ComparisonEngine(a.subject, a.run, a.replay, speed=0, on_status=lambda s: print(f"[status] {s}"))
    eng._prepare()
    print(f"n={eng.n}  EEG↔BOLD PDA r={eng.corr_pda:+.3f}")
    for u in eng._aligned[:5]:
        print(f"  TR {u.tr:3d}  BOLD(cen={u.bold_cen:+.2f} dmn={u.bold_dmn:+.2f})  "
              f"EPOC(cen={u.eeg_cen:+.2f} dmn={u.eeg_dmn:+.2f})")


if __name__ == "__main__":
    _cli()
