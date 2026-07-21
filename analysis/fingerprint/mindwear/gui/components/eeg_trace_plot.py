"""Live multi-channel raw EEG scrolling trace — an EmotivPRO-style signal preview.

Stacked one line per channel (vertically offset so traces don't overlap), over a rolling
time window. Each channel is centered on a slow running mean before display only — the felt
sensors carry a large single-ended DC offset (thousands of µV) that would otherwise push the
whole trace off-screen; this centering is a display convenience and touches nothing the
decoder sees. Fed by :func:`session_engine.stream_contact` via :meth:`add_window`.
"""
from __future__ import annotations

import contextlib
import time

import flet as ft
import flet_charts as fch
import numpy as np

from .. import styles as st

OFFSET_UV = 120.0             # vertical spacing between channel traces
DC_ALPHA = 0.05                # running-mean smoothing for the display centering
MAX_POINTS_PER_CH = 60         # decimation cap — redraws must stay cheap for the chart to keep up
MIN_REFRESH_INTERVAL = 0.35    # seconds between actual chart redraws (throttled independent of
                               # how often add_window() is called, so data ingestion never stalls
                               # waiting on a slow render)


class EEGTracePlot:
    def __init__(self, window_sec: float = 10.0, height: float = 380):
        self.window_sec = float(window_sec)
        self.height = height
        self.channels: list[str] = []
        self._dc: dict[str, float] = {}
        self._buf: dict[str, list[tuple[float, float]]] = {}   # ch -> [(t_sec, centered_uv)]
        self._t_last: float | None = None
        self._last_refresh = 0.0
        self.chart: fch.LineChart | None = None

    # ── data ─────────────────────────────────────────────────────────────
    def set_channels(self, channels: list[str]) -> None:
        self.channels = list(channels)
        self._dc = {c: 0.0 for c in self.channels}
        self._buf = {c: [] for c in self.channels}
        self._t_last = None
        self._refresh(force=True)

    def clear(self) -> None:
        self.set_channels(self.channels)

    def add_window(self, channels: list[str], sfreq: float, X: np.ndarray) -> None:
        """X: [n_samp, n_ch] raw µV samples for one short window (newest data)."""
        if channels != self.channels:
            self.set_channels(channels)
        n = X.shape[0]
        t_end = self._t_last + n / sfreq if self._t_last is not None else 0.0
        self._t_last = t_end
        for ci, ch in enumerate(self.channels):
            pts = self._buf[ch]
            dc = self._dc[ch] if pts else float(X[0, ci])   # seed the running mean from sample 1
            for k in range(n):
                v = float(X[k, ci])
                dc = (1 - DC_ALPHA) * dc + DC_ALPHA * v
                t = t_end - (n - 1 - k) / sfreq
                pts.append((t, v - dc))
            self._dc[ch] = dc
        t_min = t_end - self.window_sec
        for ch in self.channels:
            pts = self._buf[ch]
            i = 0
            while i < len(pts) and pts[i][0] < t_min:
                i += 1
            self._buf[ch] = pts[i:]
        self._refresh()

    # ── build ────────────────────────────────────────────────────────────
    def _series(self) -> list[fch.LineChartData]:
        out = []
        for ci, ch in enumerate(self.channels):
            pts = self._buf[ch]
            step = max(1, len(pts) // MAX_POINTS_PER_CH)
            offset = -ci * OFFSET_UV
            data = [fch.LineChartDataPoint(x=t, y=v + offset) for t, v in pts[::step]]
            out.append(fch.LineChartData(points=data, color=st.ACCENT, stroke_width=1.3,
                                         curved=False, rounded_stroke_cap=True))
        return out

    def _y_range(self) -> tuple[float, float]:
        n = max(len(self.channels), 1)
        return (-(n - 0.3) * OFFSET_UV, OFFSET_UV * 0.8)

    def build(self) -> ft.Container:
        min_y, max_y = self._y_range()
        t_end = self._t_last or self.window_sec
        self.chart = fch.LineChart(
            data_series=self._series(),
            min_x=max(0.0, t_end - self.window_sec), max_x=max(t_end, self.window_sec),
            min_y=min_y, max_y=max_y,
            horizontal_grid_lines=fch.ChartGridLines(
                interval=OFFSET_UV, color=ft.Colors.with_opacity(0.1, ft.Colors.GREY_500), width=1),
            left_axis=fch.ChartAxis(label_size=0, title=ft.Text("", size=1)),
            bottom_axis=fch.ChartAxis(label_size=22, title=ft.Text("time (s)", size=11)),
            bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.GREY_500),
            expand=True,
        )
        # channel-name labels down the left edge — row height is derived from the same y-scale as
        # the chart (self.height px over the y_range span) so each label lines up with its band.
        px_per_uv = self.height / (max_y - min_y)
        row_h = OFFSET_UV * px_per_uv
        labels = ft.Column(
            [ft.Container(content=ft.Text(ch, size=st.CAPTION, color=st.MUTED, weight=ft.FontWeight.W_500),
                         height=row_h, alignment=ft.Alignment(-1, -1))
             for ch in self.channels] or [st.caption("waiting for channels…")],
            spacing=0, alignment=ft.MainAxisAlignment.START)
        return ft.Container(
            content=ft.Row([ft.Container(content=labels, width=48, height=self.height),
                            ft.Container(content=self.chart, height=self.height, expand=True)],
                           expand=True),
            padding=st.GAP_SM, expand=True,
        )

    def _refresh(self, force: bool = False) -> None:
        if self.chart is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_refresh) < MIN_REFRESH_INTERVAL:
            return
        self._last_refresh = now
        t_end = self._t_last or self.window_sec
        self.chart.data_series = self._series()
        self.chart.min_x, self.chart.max_x = max(0.0, t_end - self.window_sec), max(t_end, self.window_sec)
        self.chart.min_y, self.chart.max_y = self._y_range()
        with contextlib.suppress(Exception):
            self.chart.update()
