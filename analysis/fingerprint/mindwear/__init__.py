"""EPOC-X neurofeedback system.

Real-time pipeline (source → rt_features → frozen EFP decoder → PDA feedback) plus a Flet operator
console (:mod:`mindwear.gui`). The top-level acquisition/decoder modules (``session_engine``,
``sources``, ``decoder``, ``rt_features`` …) are also importable as flat modules when this
directory is on ``sys.path`` (how the scripts run themselves); the GUI is a proper subpackage.
"""
