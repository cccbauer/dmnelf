#!/usr/bin/env python3
"""
flanker.py  —  Eriksen flanker trial generator for the "flanker" calibration block
------------------------------------------------------------------------------------
A classic executive-control/attention task: respond with the direction of the CENTER arrow in a
row of 5, ignoring congruent or incongruent flanking arrows. Used as the calibration block's
active "up" (task-positive/executive) pole, contrasting with the "self" (SRET) block's DMN pole.
"""
import random

# plain ASCII, not Unicode arrows (<-/->) — guaranteed to render distinctly in any PsychoPy font,
# unlike U+2190/U+2192 which can silently fall back to the same placeholder glyph in some fonts,
# making congruent and incongruent trials visually indistinguishable.
LEFT, RIGHT = "<", ">"
_DIRECTIONS = (LEFT, RIGHT)


def _arrows(direction: str, congruent: bool) -> str:
    flanker = direction if congruent else (LEFT if direction == RIGHT else RIGHT)
    return flanker * 2 + direction + flanker * 2


def trial_deck(rng=None, p_incongruent: float = 0.5):
    """Endless generator of shuffled (arrow_string, correct_direction) pairs.

    correct_direction in {"left", "right"} — the direction of the center arrow, which the
    participant reports (independent of the flankers). Reshuffles a fresh balanced deck (both
    directions x congruent/incongruent, weighted toward p_incongruent) each pass, so any block
    duration gets enough trials without an obvious repeating pattern."""
    rng = rng or random.Random()
    n_incongruent = max(1, round(4 * p_incongruent))
    n_congruent = max(1, 4 - n_incongruent)
    while True:
        deck = [(d, True) for d in _DIRECTIONS for _ in range(n_congruent)] + \
               [(d, False) for d in _DIRECTIONS for _ in range(n_incongruent)]
        rng.shuffle(deck)
        for d, congruent in deck:
            yield _arrows(d, congruent), ("left" if d == LEFT else "right")
