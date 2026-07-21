"""Head-map contact-quality view — electrode dots on a head silhouette (EmotivPRO-style).

Approximate top-down electrode positions for the EPOC-14 montage (front at top), colored by
the same RMS-based verdict as :mod:`contact_quality`'s bars, plus an overall good-contact
percentage. Positions are illustrative (a fixed schematic layout), not a real sensor projection.
"""
from __future__ import annotations

import contextlib

import flet as ft

from .. import styles as st
from .contact_quality import score_channel

# Normalized (x, y) in [0, 1]: x 0=left ear -> 1=right ear; y 0=forehead -> 1=occiput.
EPOC_POSITIONS: dict[str, tuple[float, float]] = {
    "AF3": (0.38, 0.10), "AF4": (0.62, 0.10),
    "F7": (0.14, 0.23), "F3": (0.37, 0.24), "F4": (0.63, 0.24), "F8": (0.86, 0.23),
    "FC5": (0.20, 0.39), "FC6": (0.80, 0.39),
    "T7": (0.06, 0.54), "T8": (0.94, 0.54),
    "P7": (0.16, 0.70), "P8": (0.84, 0.70),
    "O1": (0.40, 0.85), "O2": (0.60, 0.85),
}
DOT = 28.0


class HeadMap:
    def __init__(self, width: float = 260.0, height: float = 280.0):
        self.width = width
        self.height = height
        self.channels: list[str] = []
        self._dots: dict[str, ft.Container] = {}
        self.pct_text: ft.Text | None = None

    def set_channels(self, channels: list[str]) -> None:
        self.channels = list(channels)

    def _pos(self, ch: str) -> tuple[float, float]:
        return EPOC_POSITIONS.get(ch, (0.5, 0.5))

    def build(self) -> ft.Container:
        head = ft.Container(
            width=self.width * 0.8, height=self.height * 0.92,
            left=self.width * 0.1, top=self.height * 0.03,
            border_radius=999, bgcolor=ft.Colors.with_opacity(0.30, ft.Colors.BLUE_200),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.5, ft.Colors.BLUE_GREY_300)))
        self._dots = {}
        dots = []
        for ch in self.channels:
            x, y = self._pos(ch)
            dot = ft.Container(
                width=DOT, height=DOT, border_radius=DOT / 2, bgcolor=st.NEUTRAL,
                left=x * self.width - DOT / 2, top=y * self.height - DOT / 2,
                border=ft.Border.all(1.5, ft.Colors.with_opacity(0.6, ft.Colors.BLACK)),
                alignment=ft.Alignment(0, 0), tooltip=ch,
                content=ft.Text(ch, size=7, weight=ft.FontWeight.BOLD, color=ft.Colors.BLACK))
            self._dots[ch] = dot
            dots.append(dot)
        self.pct_text = ft.Text("—", size=26, weight=ft.FontWeight.BOLD, color=st.NEUTRAL)
        stack = ft.Stack([head, *dots, ft.Container(content=self.pct_text, right=2, bottom=0)],
                         width=self.width, height=self.height)
        return ft.Container(content=stack, alignment=ft.Alignment(0, 0), padding=st.GAP_SM)

    def update(self, rms_by_channel: dict) -> None:
        """Recolor dots + refresh the good-contact percentage. Cheap — no rebuild."""
        good = 0
        for ch, dot in self._dots.items():
            rms = rms_by_channel.get(ch)
            if rms is None:
                continue
            verdict, color = score_channel(rms)
            good += verdict == "good"
            dot.bgcolor = color
            with contextlib.suppress(Exception):
                dot.update()
        if self._dots and self.pct_text is not None:
            pct = round(100 * good / len(self._dots))
            self.pct_text.value = f"{pct}%"
            self.pct_text.color = st.SUCCESS if pct == 100 else (st.WARNING if pct >= 60 else st.ERROR)
            with contextlib.suppress(Exception):
                self.pct_text.update()
