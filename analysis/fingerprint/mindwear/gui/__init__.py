"""Flet desktop GUI for the EPOC-X neurofeedback system.

Operator console that wraps the headless :class:`mindwear.session_engine.SessionEngine`:
study/protocol management, a source + contact-quality check, calibration, and a live
CEN / DMN / PDA activation plot during feedback runs. Mirrors the architecture of the
pineuro real-time fMRI GUI (Flet in a background thread; the PsychoPy stimulus on the
main thread via a dispatcher). Launch with ``mindwear-gui`` or ``python -m mindwear.gui.app``.
"""
