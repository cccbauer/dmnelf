#!/usr/bin/env python3
"""Launch the MindWear operator console.

    python launch_gui.py            # from inside mindwear/
    # equivalently, from analysis/fingerprint/:
    python -m mindwear.gui.app

Both put the PsychoPy stimulus on the main thread (see mindwear/gui/app.py::main).
"""
import pathlib
import sys

# make `mindwear` importable as a package (parent = analysis/fingerprint)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mindwear.gui.app import main

if __name__ == "__main__":
    main()
