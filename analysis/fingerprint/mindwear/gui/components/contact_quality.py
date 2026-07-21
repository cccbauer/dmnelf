"""Per-channel contact-quality panel (the connect_test readout, live).

During the connection check a short buffer of EEG is scored per channel: RMS amplitude and a
flat-line (poor-contact) flag. This renders one row per EPOC-12 channel with a colored bar —
green good / orange marginal / red flat — so the operator can re-wet felt sensors before a run.
"""
from __future__ import annotations

import contextlib

import flet as ft

from .. import styles as st

# uV RMS bands for scalp EEG on the EPOC (empirical; felt sensors run high)
FLAT_RMS = 0.5        # below this ≈ disconnected / flat line
GOOD_LO, GOOD_HI = 3.0, 60.0
BAR_FULL_UV = 80.0    # RMS mapped to full-width bar


def score_channel(rms: float) -> tuple[str, str]:
    """Return (verdict, color) for a channel RMS in µV."""
    if rms < FLAT_RMS:
        return "flat", st.ERROR
    if rms < GOOD_LO or rms > GOOD_HI:
        return "marginal", st.WARNING
    return "good", st.SUCCESS


class ContactQuality:
    def __init__(self, channels: list[str] | None = None):
        self.channels = channels or []
        self._rows: dict[str, tuple[ft.Container, ft.Text]] = {}
        self._col: ft.Column | None = None

    def build(self) -> ft.Container:
        self._col = ft.Column(spacing=st.GAP_XS, scroll=ft.ScrollMode.AUTO)
        self.set_channels(self.channels)
        return ft.Container(
            content=ft.Column([st.subsection("Contact quality"), self._col], spacing=st.GAP_SM),
            padding=st.GAP, bgcolor=st.SUBTLE_BG, border_radius=8,
            border=ft.Border.all(1, st.PANEL_BORDER),
        )

    def set_channels(self, channels: list[str]) -> None:
        self.channels = list(channels)
        self._rows.clear()
        if self._col is None:
            return
        self._col.controls.clear()
        for ch in self.channels:
            bar = ft.Container(width=0, height=10, bgcolor=st.NEUTRAL, border_radius=3)
            val = ft.Text("—", size=st.CAPTION, color=st.MUTED)
            track = ft.Container(content=bar, width=140, height=10, bgcolor=ft.Colors.with_opacity(0.12, st.NEUTRAL),
                                 border_radius=3)
            self._col.controls.append(ft.Row(
                [ft.Container(ft.Text(ch, size=st.CAPTION, weight=ft.FontWeight.W_500), width=44),
                 track, val],
                spacing=st.GAP_SM, vertical_alignment=ft.CrossAxisAlignment.CENTER))
            self._rows[ch] = (bar, val)
        with contextlib.suppress(Exception):
            self._col.update()

    def update(self, rms_by_channel: dict[str, float]) -> None:
        """Feed a dict of channel -> RMS µV; recolors and resizes each bar."""
        for ch, (bar, val) in self._rows.items():
            rms = rms_by_channel.get(ch)
            if rms is None:
                continue
            verdict, color = score_channel(rms)
            bar.width = max(4.0, min(rms / BAR_FULL_UV, 1.0) * 140.0)
            bar.bgcolor = color
            val.value = f"{rms:5.1f} µV"
            val.color = color
            with contextlib.suppress(Exception):
                bar.update()
                val.update()
