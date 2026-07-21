"""Shared visual tokens for the MindWear GUI.

Mirrors pineuro's style module (type scale, spacing rhythm, semantic colors, text
factories) so the two operator consoles look like siblings, plus the network color
tokens this project standardizes on: **CEN red, DMN blue, PDA green** (matching the
manuscript figures).
"""
from __future__ import annotations

import flet as ft

APP_NAME = "MindWear"

# ── theme ────────────────────────────────────────────────────────────────────
SEED_COLOR = ft.Colors.INDIGO


def theme() -> ft.Theme:
    """One Theme, used for both ``page.theme`` and ``page.dark_theme`` — Flet derives the
    light/dark ColorScheme from ``color_scheme_seed`` per-mode, so role tokens below
    (HEADER_BG, SUBTLE_BG, ...) resolve correctly in either mode automatically."""
    return ft.Theme(color_scheme_seed=SEED_COLOR, visual_density=ft.VisualDensity.COMPACT)


# ── type scale ───────────────────────────────────────────────────────────────
TITLE = 22
SECTION = 16
SUBSECTION = 14
BODY = 14
CAPTION = 12
METRIC = 34  # big live-number readouts

ICON_SM, ICON_MD, ICON_LG, ICON_XL, ICON_HERO = 16, 20, 24, 48, 96

# ── spacing ──────────────────────────────────────────────────────────────────
GAP_XS, GAP_SM, GAP_MD, GAP, GAP_LG, PAD = 4, 8, 12, 16, 20, 20

# ── semantic colors ──────────────────────────────────────────────────────────
HEADER_BG = ft.Colors.PRIMARY_CONTAINER
ON_HEADER = ft.Colors.ON_PRIMARY_CONTAINER
ACCENT = ft.Colors.PRIMARY
MUTED = ft.Colors.ON_SURFACE_VARIANT
SIDEBAR_BG = ft.Colors.SURFACE_CONTAINER_HIGH
SUBTLE_BG = ft.Colors.SURFACE_CONTAINER_HIGHEST
PANEL_BORDER = ft.Colors.OUTLINE_VARIANT

SUCCESS = ft.Colors.GREEN_400
SUCCESS_BG = ft.Colors.with_opacity(0.15, ft.Colors.GREEN_400)
WARNING = ft.Colors.ORANGE_400
WARNING_BG = ft.Colors.with_opacity(0.15, ft.Colors.ORANGE_400)
ERROR = ft.Colors.ERROR
ERROR_BG = ft.Colors.ERROR_CONTAINER
INFO = ft.Colors.BLUE_400
NEUTRAL = ft.Colors.GREY_500

# ── network colors (CEN red / DMN blue / PDA green) — matches the paper ──────
CEN = ft.Colors.RED_600
DMN = ft.Colors.BLUE_600
PDA = ft.Colors.GREEN_600
NETWORK_COLORS = {"cen": CEN, "dmn": DMN, "pda": PDA}
NETWORK_HEX = {"cen": "#d7301f", "dmn": "#2c7fb8", "pda": "#31a354"}

# phase → chip color
PHASE_COLORS = {
    "connect": INFO,
    "calibrate": WARNING,
    "calib_review": ft.Colors.PURPLE_400,
    "ready": ft.Colors.PURPLE_400,
    "rest": ft.Colors.BLUE_GREY_400,
    "feedback": SUCCESS,
    "done": NEUTRAL,
}


# ── text factories ───────────────────────────────────────────────────────────
def title(text: str, **kw) -> ft.Text:
    return ft.Text(text, size=TITLE, weight=ft.FontWeight.BOLD, **kw)


def section(text: str, **kw) -> ft.Text:
    return ft.Text(text, size=SECTION, weight=ft.FontWeight.BOLD, **kw)


def subsection(text: str, **kw) -> ft.Text:
    return ft.Text(text, size=SUBSECTION, weight=ft.FontWeight.W_500, **kw)


def caption(text: str, **kw) -> ft.Text:
    kw.setdefault("color", MUTED)
    kw.setdefault("italic", True)
    return ft.Text(text, size=CAPTION, **kw)
