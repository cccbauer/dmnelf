"""MindWear operator console — Flet application shell + entry point.

Single-page app with three screens: **Study Manager → Study Editor → Session Runner**. Mirrors
pineuro's shell, including the macOS-safe launch (Flet on a background thread; the main thread runs
the :class:`~mindwear.gui.dispatch.MainThreadDispatcher` so the PsychoPy stimulus can open its window
there). Launch with ``mindwear-gui`` or ``python -m mindwear.gui.app``.
"""
from __future__ import annotations

import logging

import flet as ft

from . import styles as st
from .config_manager import ConfigManager
from .models import StudyConfig
from .screens.comparison_runner import ComparisonRunner
from .screens.data_browser import DataBrowser
from .screens.session_runner import SessionRunner
from .screens.study_editor import StudyEditor
from .screens.study_wizard import StudyWizard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("mindwear.gui")


class MindWearApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.config_manager = ConfigManager()
        self._setup_page()
        self.show_study_manager()

    def _setup_page(self) -> None:
        self.page.title = f"{st.APP_NAME} — Portable DMN/CEN Neurofeedback"
        self.page.window.width = 1400
        self.page.window.height = 900
        self.page.window.maximizable = True
        self.page.padding = 0
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.theme = st.theme()
        self.page.dark_theme = st.theme()

    # ── navigation ───────────────────────────────────────────────────────
    def _render(self, control: ft.Control) -> None:
        self.page.controls.clear()
        self.page.add(control)
        self.page.update()

    def show_study_manager(self) -> None:
        self._render(self._study_manager())

    def show_study_editor(self, study: StudyConfig | None) -> None:
        self._render(StudyEditor(self, study).build())

    def show_study_wizard(self) -> None:
        self._render(StudyWizard(self).build())

    def show_data_browser(self, study: StudyConfig | None = None) -> None:
        self._render(DataBrowser(self, study).build())

    def show_session_runner(self, study: StudyConfig, participant: str, run: int,
                            session: str = "") -> None:
        self._render(SessionRunner(self, study, participant, run, session).build())

    def show_comparison(self, study: StudyConfig, participant: str, run: int) -> None:
        self._render(ComparisonRunner(self, study, participant, run).build())

    # ── study manager screen ─────────────────────────────────────────────
    def _study_manager(self) -> ft.Control:
        header = ft.Container(
            content=ft.Row([
                ft.Row([ft.Icon(ft.Icons.PSYCHOLOGY, color=st.ACCENT, size=st.ICON_LG),
                        st.title(f"{st.APP_NAME} Study Manager")], spacing=st.GAP_SM),
                ft.Row([
                    ft.OutlinedButton("Data & Results", icon=ft.Icons.INSIGHTS,
                                      on_click=lambda _: self.show_data_browser()),
                    ft.FilledButton("New Study", icon=ft.Icons.ADD, on_click=lambda _: self.show_study_wizard()),
                ], spacing=st.GAP_SM),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            padding=st.PAD, bgcolor=st.HEADER_BG)
        body = ft.Container(content=self._study_list(), padding=st.PAD, expand=True)
        return ft.Column([header, body], spacing=0, expand=True)

    def _study_list(self) -> ft.Control:
        studies = self.config_manager.list_studies()
        if not studies:
            return ft.Container(
                content=ft.Column([
                    ft.Icon(ft.Icons.SCIENCE, size=st.ICON_HERO, color=ft.Colors.OUTLINE),
                    ft.Text("No studies yet", size=st.SECTION, color=st.MUTED),
                    st.caption("Create a study to configure the source, decoder, and feedback."),
                    ft.FilledButton("New Study", icon=ft.Icons.ADD, on_click=lambda _: self.show_study_wizard()),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=st.GAP_MD),
                alignment=ft.Alignment(0, 0), expand=True)
        return ft.GridView([self._study_card(m) for m in studies],
                           runs_count=3, max_extent=440, child_aspect_ratio=1.5,
                           spacing=st.GAP, run_spacing=st.GAP)

    def _study_card(self, meta) -> ft.Card:
        return ft.Card(content=ft.Container(
            content=ft.Column([
                st.section(meta.name, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                ft.Container(content=ft.Text(meta.description or "—", size=st.BODY, color=st.MUTED,
                                             max_lines=2, overflow=ft.TextOverflow.ELLIPSIS), height=44),
                ft.Divider(),
                ft.Row([
                    ft.FilledButton("Start", icon=ft.Icons.PLAY_ARROW,
                                    on_click=lambda _, n=meta.name: self._prompt_start(n)),
                    ft.OutlinedButton("Compare", icon=ft.Icons.COMPARE_ARROWS,
                                      tooltip="fMRI (BOLD) vs EPOC (EEG decoder), same run",
                                      on_click=lambda _, n=meta.name: self._prompt_start(n, mode="compare")),
                    ft.OutlinedButton("Edit", icon=ft.Icons.EDIT,
                                      on_click=lambda _, n=meta.name: self.show_study_editor(self.config_manager.load(n))),
                    ft.IconButton(ft.Icons.INSIGHTS, tooltip="Data & Results",
                                  on_click=lambda _, n=meta.name: self.show_data_browser(self.config_manager.load(n))),
                    ft.IconButton(ft.Icons.CONTENT_COPY, tooltip="Duplicate",
                                  on_click=lambda _, n=meta.name: self._duplicate(n)),
                    ft.IconButton(ft.Icons.DELETE_OUTLINE, tooltip="Delete", icon_color=st.ERROR,
                                  on_click=lambda _, n=meta.name: self._confirm_delete(n)),
                ], spacing=st.GAP_XS, wrap=True),
            ], spacing=st.GAP_SM),
            padding=st.GAP))

    # ── dialogs ──────────────────────────────────────────────────────────
    def _prompt_start(self, name: str, mode: str = "run") -> None:
        study = self.config_manager.load(name)
        is_cmp = mode == "compare"
        hint = "e.g. dmnelf005" if is_cmp else "e.g. rtbpd001"
        # prefill with the study's next subject id (basename + zero-padded index)
        prefill = "" if is_cmp else study.subject_id(1)
        pid = ft.TextField(label="Participant ID", hint_text=hint, value=prefill, autofocus=True)
        run = ft.TextField(label="Run #", value="1", width=100)
        # BIDS ses- label (a visit/day) — live sessions only; the offline fMRI-vs-EEG comparison
        # doesn't write any BIDS-named files, so it has no use for one.
        session = ft.TextField(label="Session (optional)", hint_text="e.g. 01, pre", width=160)

        def go(_):
            participant = (pid.value or "").strip()
            if not participant:
                pid.error_text = "required"; pid.update(); return
            try:
                r = int(run.value)
            except (TypeError, ValueError):
                r = 1
            self.page.pop_dialog()
            if is_cmp:
                self.show_comparison(study, participant, r)
            else:
                self.show_session_runner(study, participant, r, (session.value or "").strip())

        title = f"Compare fMRI vs EPOC — {name}" if is_cmp else f"Start session — {name}"
        note = (st.caption("Participant must have an observed-BOLD file "
                           "(fsnr_eeg/results/cen_ceiling/cenmean_*_<id>.npz), e.g. dmnelf005.")
                if is_cmp else st.caption("Runs the live NF session on this study's source."))
        fields = [pid, run, note] if is_cmp else [pid, run, session, note]
        btn = ft.FilledButton("Compare" if is_cmp else "Start",
                              icon=ft.Icons.COMPARE_ARROWS if is_cmp else ft.Icons.PLAY_ARROW, on_click=go)
        dlg = ft.AlertDialog(
            title=ft.Text(title),
            content=ft.Container(content=ft.Column(fields, tight=True, spacing=st.GAP_MD), width=420),
            actions=[ft.TextButton("Cancel", on_click=lambda _: self.page.pop_dialog()), btn])
        self.page.show_dialog(dlg)

    def _confirm_delete(self, name: str) -> None:
        def do(_):
            self.config_manager.delete(name)
            self.page.pop_dialog()
            self.toast(f"Deleted {name}")
            self.show_study_manager()
        dlg = ft.AlertDialog(
            title=ft.Text("Delete study?"),
            content=ft.Text(f"Permanently delete '{name}'?"),
            actions=[ft.TextButton("Cancel", on_click=lambda _: self.page.pop_dialog()),
                     ft.FilledButton("Delete", icon=ft.Icons.DELETE, on_click=do)])
        self.page.show_dialog(dlg)

    def _duplicate(self, name: str) -> None:
        self.config_manager.duplicate(name)
        self.toast(f"Duplicated {name}")
        self.show_study_manager()

    def toast(self, msg: str) -> None:
        sb = ft.SnackBar(ft.Text(msg))
        sb.open = True
        self.page.overlay.append(sb)
        self.page.update()


def main() -> None:
    """Entry point. Flet runs on a background thread; the main thread pumps the dispatcher so a
    PsychoPy stimulus window can be created there (required on macOS)."""
    import signal
    import threading

    from .dispatch import MainThreadDispatcher, set_dispatcher

    dispatcher = MainThreadDispatcher()
    set_dispatcher(dispatcher)

    # signal.signal() only works on the main thread; Flet registers handlers internally, which
    # fails from the background thread. Skip registration off-main-thread.
    _orig = signal.signal

    def _safe_signal(signum, handler):
        if threading.current_thread() is threading.main_thread():
            return _orig(signum, handler)
        return signal.getsignal(signum)

    signal.signal = _safe_signal

    flet_thread = threading.Thread(target=ft.app, args=(lambda page: MindWearApp(page),),
                                   daemon=True, name="FletApp")
    flet_thread.start()

    def _watch():
        flet_thread.join()
        dispatcher.shutdown()

    threading.Thread(target=_watch, daemon=True, name="FletExitWatcher").start()
    dispatcher.run_forever()

    # close the shared PsychoPy stimulus window on the main thread at shutdown
    try:
        from .stimulus import close_stimulus_window
        close_stimulus_window()
    except Exception:
        pass


if __name__ == "__main__":
    main()
