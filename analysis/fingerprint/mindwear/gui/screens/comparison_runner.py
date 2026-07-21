"""Comparison runner — fMRI (BOLD) vs EPOC (EEG decoder) on the same replayed run.

Drives a :class:`mindwear.compare_engine.ComparisonEngine`: launches the two-panel ball window
(left BOLD, right EEG) on the main thread via the dispatcher, and shows a live **EEG-PDA vs
BOLD-PDA** overlay plot with the running correlation. UI updates are marshalled onto Flet's loop
via ``page.run_thread`` (same as the session runner) so the plot animates in real time.
"""
from __future__ import annotations

import contextlib
import sys
import threading
from pathlib import Path

import flet as ft
import flet_charts as fch

from .. import styles as st
from ..models import StudyConfig

HERE = Path(__file__).resolve().parent.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from compare_engine import ComparisonEngine, find_bold_npz  # noqa: E402

EEG_COLOR = st.PDA                       # green (matches EPOC PDA elsewhere)
BOLD_COLOR = ft.Colors.PURPLE_400


class _OverlayPlot:
    """Two-line PDA overlay: EEG-PDA (green) vs BOLD-PDA (purple), full run history."""

    def __init__(self, height: float = 340):
        self.height = height
        self.x: list[int] = []
        self.eeg: list[float] = []
        self.bold: list[float] = []
        self.chart: fch.LineChart | None = None

    def add(self, tr: int, eeg_pda: float, bold_pda: float) -> None:
        self.x.append(int(tr))
        self.eeg.append(float(eeg_pda))
        self.bold.append(float(bold_pda))
        self._refresh()

    def clear(self) -> None:
        self.x.clear(); self.eeg.clear(); self.bold.clear()
        self._refresh()

    def _series(self):
        def pts(ys):
            return [fch.LineChartDataPoint(x=float(self.x[i]), y=ys[i]) for i in range(len(self.x))]
        return [
            fch.LineChartData(points=pts(self.bold), color=BOLD_COLOR, stroke_width=2, curved=True,
                              rounded_stroke_cap=True),
            fch.LineChartData(points=pts(self.eeg), color=EEG_COLOR, stroke_width=2, curved=True,
                              rounded_stroke_cap=True),
        ]

    def _ranges(self):
        if not self.x:
            return (0, 100, -3, 3)
        vals = self.eeg + self.bold
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.15, 0.2)
        return (float(self.x[0]), float(max(self.x[-1], self.x[0] + 1)), lo - pad, hi + pad)

    def build(self) -> ft.Container:
        min_x, max_x, min_y, max_y = self._ranges()
        self.chart = fch.LineChart(
            data_series=self._series(), min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
            horizontal_grid_lines=fch.ChartGridLines(
                interval=1, color=ft.Colors.with_opacity(0.15, ft.Colors.GREY_500), width=1),
            left_axis=fch.ChartAxis(label_size=34, title=ft.Text("PDA (normalized)", size=11)),
            bottom_axis=fch.ChartAxis(label_size=22, title=ft.Text("TR", size=11)),
            bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.GREY_500), expand=True)
        legend = ft.Row([
            ft.Row([ft.Container(width=18, height=4, bgcolor=BOLD_COLOR, border_radius=2),
                    ft.Text("fMRI PDA (BOLD)", size=st.CAPTION)], spacing=6),
            ft.Row([ft.Container(width=18, height=4, bgcolor=EEG_COLOR, border_radius=2),
                    ft.Text("EPOC PDA (EEG decoder)", size=st.CAPTION)], spacing=6),
        ], alignment=ft.MainAxisAlignment.CENTER, spacing=st.GAP_LG)
        return ft.Container(content=ft.Column(
            [ft.Container(content=self.chart, height=self.height, expand=True), legend], spacing=st.GAP_SM),
            padding=st.GAP_SM)

    def _refresh(self):
        if self.chart is None:
            return
        self.chart.data_series = self._series()
        min_x, max_x, min_y, max_y = self._ranges()
        self.chart.min_x, self.chart.max_x, self.chart.min_y, self.chart.max_y = min_x, max_x, min_y, max_y
        with contextlib.suppress(Exception):
            self.chart.update()


class ComparisonRunner:
    def __init__(self, app, study: StudyConfig, participant: str, run: int):
        self.app = app
        self.page = app.page
        self.study = study
        self.participant = participant
        self.run = run
        self.engine: ComparisonEngine | None = None
        self._stim_stop = threading.Event()
        self.plot = _OverlayPlot()

    def build(self) -> ft.Control:
        self.status = ft.Text("Ready.", size=st.BODY, color=st.MUTED)
        self.r_text = ft.Text("r = —", size=st.METRIC, weight=ft.FontWeight.BOLD, color=st.ACCENT)
        self.btn_start = ft.FilledButton("Start comparison", icon=ft.Icons.COMPARE_ARROWS,
                                         on_click=lambda _: self._start())
        self.btn_stop = ft.OutlinedButton("Stop", icon=ft.Icons.STOP, disabled=True,
                                          on_click=lambda _: self._stop())

        header = ft.Container(content=ft.Row([
            ft.Row([ft.IconButton(ft.Icons.ARROW_BACK, on_click=lambda _: self._back()),
                    st.title("fMRI vs EPOC — ball comparison"),
                    st.caption(f"  {self.participant} · run {self.run}")],
                   spacing=st.GAP_SM, vertical_alignment=ft.CrossAxisAlignment.CENTER)],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN), padding=st.PAD, bgcolor=st.HEADER_BG)

        readout = ft.Container(content=ft.Column(
            [ft.Text("EEG ↔ BOLD  PDA correlation", size=st.CAPTION, color=st.MUTED), self.r_text],
            spacing=0, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=st.GAP, bgcolor=st.SUBTLE_BG, border_radius=8, border=ft.Border.all(1, st.PANEL_BORDER))

        controls = ft.Row([self.btn_start, self.btn_stop], spacing=st.GAP_SM)
        body = ft.Container(content=ft.Column([
            ft.Row([controls, readout], alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            st.caption("Two ball tasks, same run: left driven by the scanner BOLD, right by the "
                       "portable EEG decoder. Both tracks are normalized to comparable amplitude; "
                       "watch whether the EPOC ball tracks the fMRI ball."),
            ft.Container(content=self.plot.build(), bgcolor=st.SUBTLE_BG, border_radius=8,
                         border=ft.Border.all(1, st.PANEL_BORDER), padding=st.GAP_SM, expand=True),
            self.status,
        ], spacing=st.GAP, expand=True), padding=st.PAD, expand=True)

        # warn early if the BOLD file is missing for this subject
        if find_bold_npz(self.participant) is None:
            self.status.value = (f"No observed-BOLD file for '{self.participant}'. "
                                 "Comparison needs fsnr_eeg/results/cen_ceiling/cenmean_*_<subject>.npz")
            self.status.color = st.WARNING

        return ft.Column([header, body], spacing=0, expand=True)

    # ── lifecycle ────────────────────────────────────────────────────────
    def _start(self) -> None:
        if self.engine and self.engine.is_running():
            return
        replay = self.study.source.get("replay_path") or ""
        if not replay:
            self._set_status("This study has no replay_path (comparison needs the recorded EEG run).", st.ERROR)
            return
        self.plot.clear()
        self.engine = ComparisonEngine(self.participant, self.run, replay,
                                       model_path=self.study.decoder.get("model_path") or None,
                                       on_update=self._on_update, on_status=self._on_status,
                                       speed=float(self.study.source.get("replay_speed", 1.0)) or 1.0)
        self._ui(lambda: (self._set_disabled(self.btn_start, True), self._set_disabled(self.btn_stop, False)))
        self.engine.start()
        self._launch_dual_ball()

    def _stop(self) -> None:
        self._stim_stop.set()
        if self.engine:
            self.engine.stop()
        self._ui(lambda: (self._set_disabled(self.btn_start, False), self._set_disabled(self.btn_stop, True)))

    def _back(self) -> None:
        self._stop()
        self.app.show_study_manager()

    def _launch_dual_ball(self) -> None:
        from ..dispatch import get_dispatcher, has_dispatcher
        if not has_dispatcher():
            self.app.toast("No dispatcher — launch via mindwear-gui to show the ball windows. Plot only.")
            return
        try:
            import psychopy  # noqa: F401
        except Exception:
            self.app.toast("PsychoPy not available — showing the overlay plot only.")
            return
        self._stim_stop.clear()
        scale = float(self.study.feedback.get("scale_factor", 10.0))
        from ..stimulus import run_dual_ball
        get_dispatcher().submit(run_dual_ball, self.engine, self._stim_stop, scale)

    # ── engine callbacks (worker thread → Flet loop) ─────────────────────
    def _on_update(self, u) -> None:
        def apply():
            self.plot.add(u.tr, u.eeg_pda, u.bold_pda)
            if self.engine and self.engine.corr_pda == self.engine.corr_pda:   # not NaN
                self.r_text.value = f"r = {self.engine.corr_pda:+.2f}"
                self._safe(self.r_text.update)
        self._ui(apply)

    def _on_status(self, msg: str) -> None:
        color = st.ERROR if msg.startswith("ERROR") else st.MUTED
        self._ui(lambda: self._set_status_now(msg, color))
        if not self.engine or not self.engine.is_running():
            self._ui(lambda: (self._set_disabled(self.btn_start, False), self._set_disabled(self.btn_stop, True)))

    # ── ui helpers ───────────────────────────────────────────────────────
    def _ui(self, fn) -> None:
        try:
            self.page.run_thread(fn)
        except Exception:
            with contextlib.suppress(Exception):
                fn()

    def _set_disabled(self, c, v) -> None:
        c.disabled = v
        self._safe(c.update)

    def _set_status_now(self, msg, color=None) -> None:
        self.status.value = msg
        if color:
            self.status.color = color
        self._safe(self.status.update)

    def _set_status(self, msg, color=None) -> None:
        self._ui(lambda: self._set_status_now(msg, color))

    @staticmethod
    def _safe(fn) -> None:
        with contextlib.suppress(Exception):
            fn()
