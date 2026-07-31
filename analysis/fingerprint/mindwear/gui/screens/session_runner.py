"""Session runner — the live operator console for one NF run.

Wraps a :class:`mindwear.session_engine.SessionEngine`: a connection + contact-quality check, then
Start/Stop of the calibrate → rest → feedback run, with a live CEN/DMN/PDA plot, big-number
readouts, a phase chip, a status log, and the per-run CSV path. If the study asks for a stimulus
and this process owns a main-thread dispatcher, a PsychoPy feedback window is driven off the engine.

Engine callbacks arrive on the engine's worker thread; Flet control ``.update()`` is safe to call
from there (it enqueues), so updates are applied directly and wrapped in ``suppress`` for shutdown.
"""
from __future__ import annotations

import contextlib
import sys
import threading
import time
from pathlib import Path

import flet as ft

from .. import styles as st
from ..components.activation_plot import ActivationPlot
from ..components.contact_quality import ContactQuality
from ..components.eeg_trace_plot import EEGTracePlot
from ..components.head_map import HeadMap
from ..models import StudyConfig

HERE = Path(__file__).resolve().parent.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

from session_engine import (  # noqa: E402
    DEFAULT_MODEL,
    EngineConfig,
    SessionEngine,
    adaptive_scale_factor,
    existing_calibration_path,
    score_contact,
    stream_contact,
    subject_artifacts,
    subject_dir,
)


class SessionRunner:
    def __init__(self, app, study: StudyConfig, participant: str, run: int, session: str = ""):
        self.app = app
        self.page = app.page
        self.study = study
        self.participant = participant
        self.run = run
        self.session = session         # BIDS ses- label (visit/day); "" = omitted from filenames
        self.engine: SessionEngine | None = None
        self._stim_stop = threading.Event()
        self._stim_active = False              # True while the PsychoPy stimulus is driving the gates
        self._preview_stop: threading.Event | None = None   # set while a live contact preview runs
        self.bad_channels: list[str] = []      # flagged at calibration, dropped during the run
        self.calibrated = False                # gate Start on a completed contact/QC check

        # plot spans ONE run/block at a time (rest+task seconds) and resets at each block boundary,
        # so its render cost stays flat across a 10-run protocol instead of growing to a stall.
        tr = self._model_tr()
        self._block_secs: dict[tuple[str, int], float] = {}
        for b in study.blocks():
            if b["kind"] == "calibration":
                continue
            key = (b.get("stage", ""), int(b.get("run", 0)) or 0)
            self._block_secs[key] = float(b.get("rest_sec", 0)) + float(b.get("task_sec", 0))
        first_block = max(self._block_secs.values(), default=330.0)
        self._cur_block_key: tuple[str, int] | None = None
        self.plot = ActivationPlot(x_max=first_block, dx=tr, height=320, x_label="time (s)")
        self.trace = EEGTracePlot()      # live raw-EEG preview shown during the contact check
        self.cq = ContactQuality()
        self.headmap = HeadMap()         # electrode-position contact-quality view (EmotivPRO-style)

    def _model_tr(self) -> float:
        try:
            mp = self.study.decoder.get("model_path") or str(DEFAULT_MODEL)
            return float(np.load(mp, allow_pickle=True)["tr"])
        except Exception:
            return 1.2

    # ── layout ───────────────────────────────────────────────────────────
    def _metric(self, key: str, label: str):
        val = ft.Text("—", size=st.METRIC, weight=ft.FontWeight.BOLD, color=st.NETWORK_COLORS.get(key, st.ACCENT))
        setattr(self, f"m_{key}", val)
        return ft.Container(
            content=ft.Column([ft.Text(label, size=st.CAPTION, color=st.MUTED), val], spacing=0,
                              horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=st.GAP, bgcolor=st.SUBTLE_BG, border_radius=8,
            border=ft.Border.all(1, st.PANEL_BORDER), width=130)

    def build(self) -> ft.Control:
        self.phase_chip = ft.Container(
            content=ft.Text("idle", size=st.CAPTION, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
            padding=ft.padding.symmetric(vertical=4, horizontal=12), border_radius=20, bgcolor=st.NEUTRAL)
        self.status = ft.Text("Ready.", size=st.BODY, color=st.MUTED)
        self.log = ft.ListView(spacing=2, auto_scroll=True, height=140)
        self.csv_text = st.caption("")

        self.btn_check = ft.OutlinedButton("Start preview (contact + QC)", icon=ft.Icons.SENSORS,
                                           on_click=lambda _: self._toggle_preview())
        self.bad_text = st.caption("")
        self.offset_text = st.caption("")   # raw/unreferenced-stream warning, see _on_contact_window
        start_label = "Start calibration" if self.study.session.get("calibrate", True) else "Start run"
        self.btn_start = ft.FilledButton(start_label, icon=ft.Icons.PLAY_ARROW, on_click=lambda _: self._start())
        self.btn_stop = ft.OutlinedButton("Stop", icon=ft.Icons.STOP, disabled=True, on_click=lambda _: self._stop())
        self.btn_next = ft.FilledButton("Next run", icon=ft.Icons.SKIP_NEXT, visible=False,
                                        on_click=lambda _: self._next_run())
        self.btn_exit = ft.OutlinedButton("Exit", icon=ft.Icons.LOGOUT, on_click=lambda _: self._back())
        self.chk_stim = ft.Checkbox(
            label="Show participant stimulus window",
            value=(self.study.feedback.get("mode", "ball") != "none"))

        metrics = ft.Row([self._metric("cen", "CEN"), self._metric("dmn", "DMN"),
                          self._metric("pda", "PDA"), self._metric("pdaz", "PDA z")],
                         spacing=st.GAP, alignment=ft.MainAxisAlignment.CENTER)
        # PDA-z uses a neutral accent (not a network color)
        self.m_pdaz.color = st.ACCENT

        self.run_caption = st.caption(self._participant_label())
        header = ft.Container(
            content=ft.Row([
                ft.Row([ft.IconButton(ft.Icons.ARROW_BACK, on_click=lambda _: self._back()),
                        st.title(f"{self.study.metadata.name}"), self.run_caption],
                       spacing=st.GAP_SM, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self.phase_chip,
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            padding=st.PAD, bgcolor=st.HEADER_BG)

        # swaps between the live raw-EEG trace (contact check) and the decoded CEN/DMN/PDA plot (run)
        self.plot_area = ft.Container(content=self.trace.build(), bgcolor=st.SUBTLE_BG, border_radius=8,
                                      border=ft.Border.all(1, st.PANEL_BORDER), padding=st.GAP_SM, expand=True)
        left = ft.Container(
            content=ft.Column([metrics, self.plot_area], spacing=st.GAP, expand=True),
            padding=st.PAD, expand=True)

        self.headmap_area = ft.Container(content=self.headmap.build(), alignment=ft.Alignment(0, 0))
        right = ft.Container(
            content=ft.Column([
                ft.Row([self.btn_check, self.btn_start, self.btn_stop, self.btn_next, self.btn_exit],
                       spacing=st.GAP_SM, wrap=True),
                self.bad_text,
                self.offset_text,
                self.chk_stim,
                ft.Divider(),
                self.headmap_area,
                self.cq.build(),
                ft.Divider(),
                st.subsection("Status"), self.status, self.csv_text,
                ft.Container(content=self.log, bgcolor=st.SUBTLE_BG, border_radius=8,
                             border=ft.Border.all(1, st.PANEL_BORDER), padding=st.GAP_SM),
            ], spacing=st.GAP_SM, scroll=ft.ScrollMode.AUTO),
            width=360, padding=st.PAD, bgcolor=st.SIDEBAR_BG)

        return ft.Column([header, ft.Row([left, right], spacing=0, expand=True,
                                         vertical_alignment=ft.CrossAxisAlignment.STRETCH)],
                         spacing=0, expand=True)

    def _participant_label(self, suffix: str | None = None) -> str:
        """Header caption: '  participant <id> [· ses-<session>] · <suffix or run <run>>'."""
        ses = f" · ses-{self.session}" if self.session else ""
        return f"  participant {self.participant}{ses} · {suffix if suffix is not None else f'run {self.run}'}"

    # ── engine config ────────────────────────────────────────────────────
    def _engine_config(self) -> EngineConfig:
        # a participant who already has a saved calibration for THIS session (from an earlier
        # manual run on the same visit/day) skips calibration/transfer entirely and reuses it —
        # see existing_calibration_path, StudyConfig.to_engine_config, build_blocks(first_run=...).
        # A different session label (a new visit/day) always calibrates fresh.
        calib_path = existing_calibration_path(self.participant, session=self.session)
        cfg = self.study.to_engine_config(self.participant, self.run,
                                          calib_path=str(calib_path) if calib_path else None,
                                          bids_session=self.session)
        cfg.bad_channels = list(self.bad_channels)
        return cfg

    # ── contact check (live preview: raw trace + continuously-updated RMS bars) ──────────
    def _toggle_preview(self) -> None:
        if self._preview_stop is not None:
            self._preview_stop.set()
            return
        self._show_trace()
        self.trace.clear()
        self._log("Starting live preview…")
        self._preview_stop = threading.Event()
        self._ui(lambda: (self._set_button(self.btn_check, "Stop preview", ft.Icons.STOP),
                          self._set_status_now("Live preview running.", st.MUTED)))

        def worker():
            stop_event = self._preview_stop
            res = stream_contact(self._engine_config(), on_window=self._on_contact_window,
                                 stop_event=stop_event, window_sec=0.25, on_status=self._log)
            self._preview_stop = None
            if res["error"]:
                self._set_status(f"Preview stopped: {res['error']}", st.ERROR)
                self._log(res["error"])
            else:
                self._log("Preview stopped.")
            self._ui(lambda: self._set_button(self.btn_check, "Start preview (contact + QC)", ft.Icons.SENSORS))

        threading.Thread(target=worker, daemon=True).start()

    def _on_contact_window(self, channels: list[str], sfreq: float, X: np.ndarray) -> None:
        """Runs on the preview worker thread — one short raw window (~0.25 s), continuously."""
        Xc = X - X.mean(axis=1, keepdims=True)                # common-average reference
        rms = np.sqrt(np.nanmean(Xc ** 2, axis=0))
        rms_by_ch = {ch: float(r) for ch, r in zip(channels, rms)}
        bad = score_contact(rms_by_ch)
        n_ch = len(channels); n_good = n_ch - len(bad)

        # raw (pre-CAR) per-channel mean vs. its own variation — calibrated µV EEG hovers near 0;
        # a mean many times larger than the channel's own SD usually means a raw/unreferenced LSL
        # stream (ADC counts) rather than the processed EEG output. CAR only removes the
        # cross-channel component at each instant, so this survives it and would otherwise go
        # unnoticed until it shows up as a null result in a full calibration/session.
        raw_mean = X.mean(axis=0); raw_sd = X.std(axis=0)
        offset = np.abs(raw_mean) > 20 * np.maximum(raw_sd, 1e-9)
        offset_channels = [ch for ch, o in zip(channels, offset) if o]

        def apply():
            if self.cq.channels != channels:
                self.cq.set_channels(channels)
            self.cq.update(rms_by_ch)
            is_new_trace_channels = self.trace.channels != channels
            self.trace.add_window(channels, sfreq, X)
            if is_new_trace_channels:
                self._show_trace_now()   # rebuild so the left-edge channel labels appear
            if self.headmap.channels != channels:
                self.headmap.set_channels(channels)
                self.headmap_area.content = self.headmap.build()
                self._safe(self.headmap_area.update)
            self.headmap.update(rms_by_ch)
            self.bad_channels = bad
            self.calibrated = n_good >= 6
            if bad:
                self.bad_text.value = (f"⚠ dropping {len(bad)} bad channel(s): "
                                       f"{', '.join(bad)}  ({n_good}/{n_ch} good)")
                self.bad_text.color = st.WARNING
            else:
                self.bad_text.value = f"✓ all {n_ch} channels good"
                self.bad_text.color = st.SUCCESS
            self._safe(self.bad_text.update)
            if offset_channels:
                self.offset_text.value = f"⚠ {len(offset_channels)} channel(s) look raw/unreferenced: {', '.join(offset_channels)}"
                self.offset_text.color = st.WARNING
            else:
                self.offset_text.value = ""
            self._safe(self.offset_text.update)
            if self.calibrated:
                self._set_status_now(f"Live preview — {n_good}/{n_ch} good @ {sfreq:g} Hz.", st.SUCCESS)
            else:
                self._set_status_now(f"Only {n_good} good channels — re-wet sensors.", st.ERROR)
        self._ui(apply)

    def _show_trace_now(self) -> None:
        """Assumes it's already running on the Flet loop (inside an _ui handler)."""
        self.plot_area.content = self.trace.build()
        self._safe(self.plot_area.update)

    def _show_trace(self) -> None:
        self._ui(self._show_trace_now)

    def _set_button(self, btn, text: str, icon) -> None:
        btn.content = text
        btn.icon = icon
        self._safe(btn.update)

    # ── run lifecycle ────────────────────────────────────────────────────
    def _start(self) -> None:
        if self.engine and self.engine.is_running():
            return
        calib_path = existing_calibration_path(self.participant)
        existing = subject_artifacts(self.participant)
        if calib_path is not None:
            # reusing a saved calibration: its own files (and the once-per-participant transfer
            # blocks, also skipped — see build_blocks(first_run=False)) are being kept, not
            # touched, so they don't count as a conflict. Only warn about files THIS run's own
            # feedback blocks would actually collide with (e.g. re-typing a run # already used).
            existing = [n for n in existing if "task-calibration" not in n
                       and "task-transferpre" not in n and "task-transferpost" not in n]
            if not existing:
                self._log(f"reusing saved calibration: {calib_path.name}")
                self._start_engine()
                return
        if existing:
            self._prompt_overwrite(existing)
            return
        self._start_engine()

    def _prompt_overwrite(self, existing: list[str]) -> None:
        """This subject already has saved recordings — one protocol session writes the same BIDS
        filenames (no session/run entity spans the protocol), so running again overwrites them."""
        def do_overwrite(_):
            self.page.pop_dialog()
            d = subject_dir(self.participant)
            for name in existing:
                with contextlib.suppress(Exception):
                    (d / name).unlink()
            self._start_engine()

        dlg = ft.AlertDialog(
            title=ft.Text(f"{self.participant} already has saved data"),
            content=ft.Container(content=ft.Column([
                ft.Text(f"{len(existing)} file(s) are already saved for this subject. Running "
                        "again overwrites them (use a new subject id to keep both):"),
                st.caption(", ".join(existing[:6]) + (" …" if len(existing) > 6 else "")),
            ], tight=True, spacing=st.GAP_SM), width=460),
            actions=[
                ft.TextButton("Cancel", on_click=lambda _: self.page.pop_dialog()),
                ft.OutlinedButton("Overwrite", icon=ft.Icons.DELETE_OUTLINE, on_click=do_overwrite),
            ])
        self.page.show_dialog(dlg)

    def _start_engine(self) -> None:
        if self._preview_stop is not None:
            self._preview_stop.set()
        self._cur_block_key = None      # first update re-pins the plot to the opening block
        self.plot.clear()
        cfg = self._engine_config()
        self.engine = SessionEngine(cfg, on_update=self._on_update, on_phase=self._on_phase,
                                    on_status=self._on_status)

        def apply():
            self._set_disabled(self.btn_start, True)
            self._set_disabled(self.btn_stop, False)
            self._set_disabled(self.btn_check, True)
            self.btn_next.visible = False
            self._safe(self.btn_next.update)
            self.plot_area.content = self.plot.build()
            self._safe(self.plot_area.update)
            self.plot._refresh()
        self._ui(apply)
        self.engine.start()
        if self.chk_stim.value:
            self._launch_stimulus()

    def _next_run(self) -> None:
        self.run += 1
        self.run_caption.value = self._participant_label()
        self._safe(self.run_caption.update)
        self._start()

    def _stop(self) -> None:
        self._stim_stop.set()
        if self._preview_stop is not None:
            self._preview_stop.set()
        if self.engine:
            self.engine.stop(join=False)
        self._ui(self._reset_buttons)

    def _back(self) -> None:
        self._stop()
        self.app.show_study_manager()

    def _reset_buttons(self) -> None:
        self._set_disabled(self.btn_start, False)
        self._set_disabled(self.btn_stop, True)
        self._set_disabled(self.btn_check, False)
        self.btn_next.visible = False   # _on_phase("done") re-shows it if a next run is available
        self._safe(self.btn_next.update)

    # ── engine callbacks (worker thread → marshalled onto Flet's loop) ───
    def _on_phase(self, phase: str) -> None:
        if phase == "done":
            self._stim_stop.set()

        def apply():
            self.phase_chip.content.value = phase
            self.phase_chip.bgcolor = st.PHASE_COLORS.get(phase, st.NEUTRAL)
            self._safe(self.phase_chip.update)
            if phase == "calib_review":
                self._show_calib_review()
            elif phase == "ready" and not self._stim_active:
                self._operator_ready()
            elif phase == "question" and not self._stim_active:
                self._operator_direction()
            elif phase == "ratings" and not self._stim_active:
                # ratings are participant-facing only (gui/stimulus.py); with no stimulus window
                # there's no one to ask, so don't deadlock the session waiting for them.
                if self.engine:
                    self.engine.ratings_submitted()
            elif phase == "run_choice":
                # mbNF, between feedback runs: always an operator decision (never shown to the
                # participant — the stimulus window just displays a "please wait" message during
                # this phase, see gui/stimulus.py), regardless of whether a stimulus is running.
                self._show_run_choice()
            if phase == "done":
                self._reset_buttons()
                if self.engine and self.engine.log_path:
                    self.csv_text.value = f"log: {self.engine.log_path.name}"
                    self._safe(self.csv_text.update)
                # A completed protocol session (calibration + all configured feedback runs, with
                # any mid-protocol recalibration already handled via the run_choice gate above) is
                # just finished. btn_next / _next_run() stay in place for the operator to
                # deliberately start another full session (new participant/day).
                self.btn_next.visible = False
                self._safe(self.btn_next.update)
                if not self.chk_stim.value:
                    # no stimulus window competing for focus — safe to refocus MindWear right away.
                    # When a stimulus IS shown, _stimulus_loop() does this once it actually finishes
                    # (ratings screen included), so it doesn't steal focus mid-questionnaire.
                    self._bring_app_to_front()
        self._ui(apply)

    def _bring_app_to_front(self) -> None:
        """Focus the MindWear window — called when a run finishes, since the PsychoPy stimulus
        window may currently have focus."""
        with contextlib.suppress(Exception):
            self.page.window.focused = True
            self.page.update()

    def _show_calib_review(self) -> None:
        """Calibration finished — pause and let the operator inspect it before proceeding.
        Assumes it's already running on the Flet loop."""
        s = (self.engine.calib_summary or {}) if self.engine else {}
        n_tr, n_feat = s.get("n_tr", 0), s.get("n_features", 0)
        seconds = n_tr * (self.engine.tr or 1.2) if self.engine else 0.0
        pct_flat = s.get("pct_flat", 0.0)
        flat_note = (f"{s.get('n_flat_features', 0)}/{n_feat} features ({pct_flat:.0f}%) barely "
                    "varied during calibration — fine if you held still, worth a re-check "
                    "otherwise." if pct_flat > 0 else "All features showed variation.")

        # rest/self/flanker (or rest/noting) QA (only present when the calibration ran cycles > 0
        # — see SessionEngine._run_cue_block / _score_calibration): did the calibration's task(s)
        # actually move the decoded PDA apart, and how does each compare to rest? Generalized over
        # whichever conditions/separations this calibration actually produced (calibration["type"]),
        # rather than hardcoding "self"/"flanker".
        means = s.get("pda_means") or {}
        sep_lines = []
        if len(means) > 1:
            parts = [f"{cond}={v:+.3f}" for cond, v in means.items()]
            sep_lines.append(f"PDA by condition: {'  '.join(parts)}")
            for key, val in s.items():
                if key.startswith("separation_") and val is not None:
                    a, b = key[len("separation_"):].split("_vs_")
                    # _score_calibration's d = mean(a) - mean(b), so it's negative whenever b
                    # happens to be the higher-PDA condition — always show it non-negative, with
                    # whichever condition is actually higher named first, so the sign never reads
                    # as "something's wrong" for what's really just a > b vs. b > a.
                    if val < 0:
                        a, b, val = b, a, -val
                    sep_lines.append(f"{a} vs. {b} separation: d={val:.2f}")

        def decide(retry: bool) -> None:
            self.page.pop_dialog()
            if self.engine:
                if retry:
                    self.engine.retry_calibration()
                else:
                    self.engine.confirm_calibration()

        dlg = ft.AlertDialog(
            title=ft.Text("Calibration complete — review"),
            content=ft.Container(content=ft.Column([
                ft.Text(f"{seconds:.0f}s of EEG collected, {n_feat} features fit."),
                st.caption(flat_note),
                *[st.caption(line) for line in sep_lines],
            ], tight=True, spacing=st.GAP_SM), width=420),
            actions=[
                ft.OutlinedButton("Repeat calibration", icon=ft.Icons.REPLAY,
                                  on_click=lambda _: decide(True)),
                ft.FilledButton("Continue to feedback", icon=ft.Icons.PLAY_ARROW,
                               on_click=lambda _: decide(False)),
            ])
        self.page.show_dialog(dlg)

    def _show_run_choice(self) -> None:
        """mbNF, between feedback runs: the participant just finished ratings for the run that
        ended (engine.block_run) — ask the operator whether to recalibrate before the next run
        or go straight to it, reusing the current calibration."""
        finished_run = self.engine.block_run if self.engine else 0

        def decide(recalibrate: bool) -> None:
            self.page.pop_dialog()
            if self.engine:
                if recalibrate:
                    self.engine.recalibrate_next_run()
                else:
                    self.engine.continue_next_run()

        dlg = ft.AlertDialog(
            title=ft.Text(f"Run {finished_run} complete"),
            content=ft.Container(
                content=ft.Text("Recalibrate before the next feedback run, or continue with the "
                                "current calibration?"),
                width=380),
            actions=[
                ft.OutlinedButton("Recalibrate", icon=ft.Icons.REPLAY,
                                  on_click=lambda _: decide(True)),
                ft.FilledButton("Continue to next run", icon=ft.Icons.PLAY_ARROW,
                               on_click=lambda _: decide(False)),
            ])
        self.page.show_dialog(dlg)

    def _on_status(self, msg: str) -> None:
        color = st.ERROR if msg.startswith("ERROR") else st.MUTED
        self._ui(lambda: (self._set_status_now(msg, color), self._log_now(msg)))

    def _on_update(self, u) -> None:
        def apply():
            key = (u.stage, u.run)
            if key != self._cur_block_key:      # new run/block → fresh plot (keeps render flat)
                self._cur_block_key = key
                self.plot.reset(self._block_secs.get(key))
                label = {"transferpre": "transfer (pre)", "transferpost": "transfer (post)"}.get(
                    u.stage, f"feedback run {u.run}")
                self.run_caption.value = self._participant_label(label)
                self._safe(self.run_caption.update)
            self.plot.add(u.cen, u.dmn, u.pda)
            self.m_cen.value = f"{u.cen:+.3f}"
            self.m_dmn.value = f"{u.dmn:+.3f}"
            self.m_pda.value = f"{u.pda:+.3f}"
            self.m_pdaz.value = "—" if u.pda_z != u.pda_z else f"{u.pda_z:+.2f}"   # NaN check
            for m in (self.m_cen, self.m_dmn, self.m_pda, self.m_pdaz):
                self._safe(m.update)
        self._ui(apply)

    # ── stimulus (PsychoPy on the main thread via dispatcher) ────────────
    def _launch_stimulus(self) -> None:
        from ..dispatch import get_dispatcher, has_dispatcher
        mode = self.study.feedback.get("mode", "ball")
        if mode == "none":
            return
        if not has_dispatcher():
            self.app.toast("No main-thread dispatcher — stimulus needs `mindwear-gui` launch. Running plot only.")
            return
        try:
            import psychopy  # noqa: F401
        except Exception:
            self.app.toast("PsychoPy not available — running operator plot only.")
            return
        self._stim_stop.clear()
        self._stim_active = True
        self._log(f"launching {mode} stimulus")
        get_dispatcher().submit(self._stimulus_loop, mode)   # runs on main thread until the run ends

    def _stimulus_loop(self, mode: str) -> None:
        from ..stimulus import run_stimulus
        fb_cfg = dict(self.study.feedback)
        fb_cfg["scale_factor"] = scale = adaptive_scale_factor(self.participant, self.run,
                                                                session=self.session)
        self._log(f"ball scale factor: {scale:.2f} (adaptive, from run {self.run - 1})"
                 if self.run > 1 else f"ball scale factor: {scale:.2f} (default, run 1)")
        try:
            run_stimulus(self.engine, mode, fb_cfg, self._stim_stop,
                         log_dir=Path(self._engine_config().log_dir),
                         participant=self.participant, run=self.run, session=self.session)
        except Exception as exc:
            self._log(f"stimulus error: {exc}")
        finally:
            self._stim_active = False
            # stimulus (incl. any ratings screen) has actually finished — safe to refocus now
            self._ui(self._bring_app_to_front)

    # ── operator-side gate fallbacks (only when no participant stimulus is driving) ──
    def _operator_ready(self) -> None:
        """No stimulus running: advance the participant-ready gate from the operator console."""
        if self.engine and self.engine.phase == "ready":
            self.engine.participant_ready()

    def _operator_direction(self) -> None:
        """No stimulus running: collect the R-mbNF up/down report via an operator dialog."""
        def answer(direction: str) -> None:
            self.page.pop_dialog()
            if self.engine:
                self.engine.answer_direction(direction)
        dlg = ft.AlertDialog(
            title=ft.Text("Participant report"),
            content=ft.Text("Did the participant's noting drive the ball up or down?"),
            actions=[ft.FilledButton("Up", icon=ft.Icons.ARROW_UPWARD, on_click=lambda _: answer("up")),
                     ft.OutlinedButton("Down", icon=ft.Icons.ARROW_DOWNWARD, on_click=lambda _: answer("down")),
                     ft.TextButton("Not sure", icon=ft.Icons.HELP_OUTLINE,
                                  on_click=lambda _: answer("not_sure"))])
        self.page.show_dialog(dlg)

    # ── ui helpers ───────────────────────────────────────────────────────
    def _ui(self, fn) -> None:
        """Run *fn* on Flet's event loop (from any thread) so control updates flush immediately."""
        try:
            self.page.run_thread(fn)
        except Exception:
            with contextlib.suppress(Exception):   # fallback: best-effort direct call
                fn()

    def _set_disabled(self, control, value: bool) -> None:
        control.disabled = value
        self._safe(control.update)

    # *_now variants assume they are already running on the Flet loop (inside an _ui handler)
    def _set_status_now(self, msg: str, color=None) -> None:
        self.status.value = msg
        if color:
            self.status.color = color
        self._safe(self.status.update)

    def _log_now(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log.controls.append(ft.Text(f"{ts}  {msg}", size=st.CAPTION, color=st.MUTED))
        if len(self.log.controls) > 200:
            del self.log.controls[0]
        self._safe(self.log.update)

    # thread-safe entry points (marshal onto the loop)
    def _set_status(self, msg: str, color=None) -> None:
        self._ui(lambda: self._set_status_now(msg, color))

    def _log(self, msg: str) -> None:
        self._ui(lambda: self._log_now(msg))

    @staticmethod
    def _safe(fn) -> None:
        with contextlib.suppress(Exception):
            fn()
