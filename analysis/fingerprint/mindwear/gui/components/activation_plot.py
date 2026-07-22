"""Live CEN / DMN / PDA activation plot (flet-charts LineChart).

Fixed-width timeseries: the x-axis is pinned to the whole run up front (``x_max``) and the line
fills **left → right** as TRs arrive — no rolling window, no x-rescaling. The session runner uses
**seconds** on x (real-time EEG); the comparison view keeps TR. Colors match the manuscript
(CEN red, DMN blue, PDA green). :meth:`add` is called once per decoded TR from the GUI thread.
"""
from __future__ import annotations

import contextlib

import flet as ft
import flet_charts as fch

from .. import styles as st

SERIES = ("cen", "dmn", "pda")
LABELS = {"cen": "CEN", "dmn": "DMN", "pda": "PDA = CEN − DMN"}


class ActivationPlot:
    def __init__(self, x_max: float, dx: float = 1.2, height: float = 320, x_label: str = "time (s)"):
        self.x_max = float(x_max)
        self.dx = float(dx)                     # x-step per TR (seconds)
        self.x_label = x_label
        self.height = height
        self._n = 0                             # emitted-TR counter → x = n * dx
        self._x: list[float] = []
        self._y: dict[str, list[float]] = {k: [] for k in SERIES}
        self.chart: fch.LineChart | None = None

    # ── data ─────────────────────────────────────────────────────────────
    def add(self, cen: float, dmn: float, pda: float) -> None:
        self._x.append(self._n * self.dx)
        self._n += 1
        self._y["cen"].append(float(cen))
        self._y["dmn"].append(float(dmn))
        self._y["pda"].append(float(pda))
        self._refresh()

    def clear(self) -> None:
        self._n = 0
        self._x.clear()
        for k in SERIES:
            self._y[k].clear()
        self._refresh()

    def reset(self, x_max: float | None = None) -> None:
        """Start a fresh run: drop all points (bounding render cost per run) and, optionally,
        re-pin the x-axis to the new run's length. Called at each block boundary."""
        if x_max is not None and x_max > 0:
            self.x_max = float(x_max)
        self.clear()

    # ── build ────────────────────────────────────────────────────────────
    def _series(self) -> list[fch.LineChartData]:
        out = []
        for k in SERIES:
            pts = [fch.LineChartDataPoint(x=self._x[i], y=self._y[k][i]) for i in range(len(self._x))]
            out.append(fch.LineChartData(points=pts, color=st.NETWORK_COLORS[k],
                                         stroke_width=2, curved=False, rounded_stroke_cap=True))
        return out

    def _y_range(self) -> tuple[float, float]:
        vals = [v for k in SERIES for v in self._y[k]]
        if not vals:
            return (-0.3, 0.3)
        lo, hi = min(vals), max(vals)
        rng = hi - lo
        pad = max(rng * 0.15, 0.05)
        if rng < 0.2:
            c = (hi + lo) / 2
            lo, hi = c - 0.1, c + 0.1
        return (lo - pad, hi + pad)

    def build(self) -> ft.Container:
        min_y, max_y = self._y_range()
        self.chart = fch.LineChart(
            data_series=self._series(),
            min_x=0, max_x=self.x_max, min_y=min_y, max_y=max_y,
            horizontal_grid_lines=fch.ChartGridLines(
                interval=0.1, color=ft.Colors.with_opacity(0.15, ft.Colors.GREY_500), width=1),
            left_axis=fch.ChartAxis(label_size=34, title=ft.Text("decoder output", size=11)),
            bottom_axis=fch.ChartAxis(label_size=22, title=ft.Text(self.x_label, size=11)),
            bgcolor=ft.Colors.with_opacity(0.04, ft.Colors.GREY_500),
            expand=True,
        )
        legend = ft.Row(
            [ft.Row([ft.Container(width=18, height=4, bgcolor=st.NETWORK_COLORS[k], border_radius=2),
                     ft.Text(LABELS[k], size=st.CAPTION)], spacing=6) for k in SERIES],
            alignment=ft.MainAxisAlignment.CENTER, spacing=st.GAP_LG,
        )
        return ft.Container(
            content=ft.Column([ft.Container(content=self.chart, height=self.height, expand=True), legend],
                              spacing=st.GAP_SM),
            padding=st.GAP_SM,
        )

    def _refresh(self) -> None:
        if self.chart is None:
            return
        self.chart.data_series = self._series()
        self.chart.min_x, self.chart.max_x = 0, self.x_max     # x stays fixed to the full run
        self.chart.min_y, self.chart.max_y = self._y_range()
        with contextlib.suppress(Exception):
            self.chart.update()
