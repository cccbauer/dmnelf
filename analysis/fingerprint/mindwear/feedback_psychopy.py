#!/usr/bin/env python3
"""
feedback_psychopy.py  —  neurofeedback display (PsychoPy), matching the scanner paradigm
----------------------------------------------------------------------------------------
Two variable-height RED feedback bars driven by the PDA signal (z-scored to the rest baseline),
with a fixed BLUE target level — the same look as the MURFI rt-fMRI paradigm (Bauer 2020; the
red bars rise when CEN > DMN, i.e. when the participant up-regulates PDA toward the target).

PsychoPy is imported lazily so the decoder engine and tests run without it. Use NullFeedback for
headless runs.
"""


class NullFeedback:
    """Headless no-op display (for testing / log-only runs)."""
    def rest(self): pass
    def feedback(self, pda_z): pass
    def message(self, text): pass
    def close(self): pass


class PsychoPyFeedback:
    def __init__(self, target_z=1.0, full_scale_z=3.0, fullscreen=True, size=(1280, 720)):
        from psychopy import visual, core
        self._core = core
        self.target_z = target_z; self.full = full_scale_z
        self.win = visual.Window(size=size, color=(-0.4, -0.4, -0.4), units="norm",
                                 fullscr=fullscreen, allowGUI=False)
        self.bar_h = 1.6            # max bar half-height range (norm units, -0.8..0.8)
        self.bars = [visual.Rect(self.win, width=0.18, height=0.001, pos=(x, -0.8 + 0.0005),
                                 fillColor=(1, -0.5, -0.5), lineColor=None, anchor="bottom")
                     for x in (-0.28, 0.28)]
        # fixed blue target level line + label
        ty = -0.8 + self.bar_h * (target_z / self.full)
        self.target = visual.Line(self.win, start=(-0.5, ty), end=(0.5, ty),
                                  lineColor=(-0.5, -0.5, 1), lineWidth=4)
        self.msg = visual.TextStim(self.win, text="", pos=(0, 0.7), height=0.08, color=(1, 1, 1))
        self.win.flip()

    def _draw(self, pda_z=None, text=""):
        if pda_z is not None:
            frac = max(0.0, min(1.0, (pda_z + self.full) / (2 * self.full)))   # -full..+full -> 0..1
            h = max(0.001, self.bar_h * frac)
            for b in self.bars:
                b.height = h; b.draw()
            self.target.draw()
        if text:
            self.msg.text = text; self.msg.draw()
        self.win.flip()

    def rest(self):
        self._draw(text="Rest — relax, look at the cross")

    def feedback(self, pda_z):
        self._draw(pda_z=pda_z, text="Raise the bars (mental noting)")

    def message(self, text):
        self._draw(text=text)

    def close(self):
        try:
            self.win.close()
        except Exception:
            pass


def make_feedback(kind="psychopy", **kw):
    return PsychoPyFeedback(**kw) if kind == "psychopy" else NullFeedback()
