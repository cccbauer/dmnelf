"""Main-thread task dispatcher for PsychoPy compatibility.

Flet's ``ft.app()`` blocks whatever thread runs it, but PsychoPy (pyglet/Cocoa) must create
its window on the **main** thread on macOS. So we run ``ft.app()`` on a background thread and
keep the main thread pumping this dispatcher: background threads (the Flet callbacks) submit
the "open the stimulus window / run the task" callable via :meth:`submit` and get a ``Future``.

Adapted from pineuro's ``gui/dispatch.py`` — same single-code-path approach across platforms.
"""
from __future__ import annotations

import queue
import sys
from concurrent.futures import Future
from typing import Any

_dispatcher: "MainThreadDispatcher | None" = None


def get_dispatcher() -> "MainThreadDispatcher":
    if _dispatcher is None:
        raise RuntimeError("MainThreadDispatcher not initialised")
    return _dispatcher


def set_dispatcher(d: "MainThreadDispatcher") -> None:
    global _dispatcher
    _dispatcher = d


def has_dispatcher() -> bool:
    return _dispatcher is not None


class MainThreadDispatcher:
    """Execute callables on the main thread from any background thread."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[tuple[Any, tuple, dict, Future] | None]" = queue.Queue()

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Future:
        fut: Future = Future()
        self._queue.put((fn, args, kwargs, fut))
        return fut

    def shutdown(self) -> None:
        self._queue.put(None)

    def run_forever(self) -> None:
        """Block on the main thread, executing submitted callables until :meth:`shutdown`."""
        while True:
            item = self._queue.get()
            if item is None:
                break
            fn, args, kwargs, fut = item
            try:
                fut.set_result(fn(*args, **kwargs))
            except BaseException as exc:  # surface into the Future, keep the loop alive
                fut.set_exception(exc)
            _flush_cocoa_events()


def _flush_cocoa_events() -> None:
    """Pump the Core Foundation run loop so PsychoPy window open/close finalise (macOS only)."""
    if sys.platform != "darwin":
        return
    try:
        import ctypes
        import ctypes.util

        cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
        mode = ctypes.c_void_p.in_dll(cf, "kCFRunLoopDefaultMode")
        cf.CFRunLoopRunInMode.restype = ctypes.c_int32
        cf.CFRunLoopRunInMode.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_bool]
        cf.CFRunLoopRunInMode(mode, 0.2, False)
    except Exception:
        pass
