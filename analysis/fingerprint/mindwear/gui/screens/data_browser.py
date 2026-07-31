"""Data & Results screen — browse collected sessions per study and analyze R-mbNF outcomes.

Pick a study; the browser lists every subject with saved data under ``data/<subject>/``. Click a
subject to drill into its full per-run detail (direction-report **accuracy** and PDA **regulation** —
mean feedback-phase z), or rename/delete its saved data from either the list or the detail view.
Parsing and file management live in :mod:`mindwear.results` (flet-free).
"""
from __future__ import annotations

import sys
from pathlib import Path

import flet as ft

from .. import styles as st

_HERE = Path(__file__).resolve().parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from results import SubjectResult, delete_subject, rename_subject, study_results  # noqa: E402


def _fmt(x, spec="+.2f") -> str:
    return "—" if x is None or x != x else format(x, spec)


class DataBrowser:
    def __init__(self, app, study=None):
        self.app = app
        self.page = app.page
        self.cm = app.config_manager
        self.studies = self.cm.list_studies()
        names = [m.name for m in self.studies]
        self.study_name = study.metadata.name if study is not None else (names[0] if names else None)
        self.selected_subject: str | None = None

    # ── build ────────────────────────────────────────────────────────────
    def build(self) -> ft.Control:
        names = [m.name for m in self.studies]
        self.picker = ft.Dropdown(
            label="Study", value=self.study_name, width=320,
            options=[ft.dropdown.Option(n, n) for n in names],
            on_select=lambda _: self._on_pick())
        header = ft.Container(
            content=ft.Row([
                ft.Row([ft.IconButton(ft.Icons.ARROW_BACK, on_click=lambda _: self.app.show_study_manager()),
                        ft.Row([ft.Icon(ft.Icons.INSIGHTS, color=st.ACCENT, size=st.ICON_LG),
                                st.title("Data & Results")], spacing=st.GAP_SM)], spacing=st.GAP_SM),
                self.picker,
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            padding=st.PAD, bgcolor=st.HEADER_BG)

        self.body = ft.Container(content=self._body(), padding=st.PAD, expand=True)
        return ft.Column([header, self.body], spacing=0, expand=True)

    def _on_pick(self) -> None:
        self.study_name = self.picker.value
        self.selected_subject = None
        self._refresh()

    def _refresh(self) -> None:
        self.body.content = self._body()
        self.body.update()

    def _basename(self) -> str:
        if not self.study_name:
            return ""
        try:
            return (self.cm.load(self.study_name).subjects.get("basename") or "").strip()
        except Exception:
            return ""

    def _select(self, subject: str) -> None:
        self.selected_subject = subject
        self._refresh()

    def _back_to_list(self) -> None:
        self.selected_subject = None
        self._refresh()

    # ── body ─────────────────────────────────────────────────────────────
    def _body(self) -> ft.Control:
        if not self.study_name:
            return self._empty("No studies yet", "Create a study and run a session to collect data.")
        results = [r for r in study_results(self._basename()) if r.has_data]
        if not results:
            self.selected_subject = None
            return self._empty(
                f"No data for “{self.study_name}”",
                f"Nothing saved yet under a subject starting with “{self._basename() or '(any)'}”. "
                "Run a session, then come back.")

        if self.selected_subject:
            match = next((r for r in results if r.subject == self.selected_subject), None)
            if match is not None:
                return self._detail(match)
            self.selected_subject = None

        return self._list(results)

    def _list(self, results: list[SubjectResult]) -> ft.Control:
        n_reports = sum(r.n_reports for r in results)
        n_correct = sum(r.n_correct for r in results)
        pooled_acc = (n_correct / n_reports) if n_reports else None
        fb_z = [r.mean_feedback_pda_z for r in results if r.mean_feedback_pda_z == r.mean_feedback_pda_z]
        mean_z = sum(fb_z) / len(fb_z) if fb_z else None

        summary = ft.Row([
            self._tile("Subjects", str(len(results)), ft.Icons.PEOPLE_ALT),
            self._tile("Direction accuracy", _fmt(pooled_acc and pooled_acc * 100, ".0f") + "%"
                       if pooled_acc is not None else "—", ft.Icons.CHECK_CIRCLE,
                       sub=f"{n_correct}/{n_reports} reports · chance 50%",
                       color=self._acc_color(pooled_acc)),
            self._tile("Mean feedback z", _fmt(mean_z), ft.Icons.TRENDING_UP,
                       sub="PDA vs rest baseline"),
        ], spacing=st.GAP, wrap=True)

        rows = ft.Column([self._subject_row(r) for r in results],
                         spacing=st.GAP_SM, scroll=ft.ScrollMode.AUTO, expand=True)
        return ft.Column([summary, ft.Divider(), rows], spacing=st.GAP, expand=True)

    # ── pieces ─────────────────────────────────────────────────────────────
    def _empty(self, title: str, hint: str) -> ft.Control:
        return ft.Container(
            content=ft.Column([
                ft.Icon(ft.Icons.QUERY_STATS, size=st.ICON_HERO, color=ft.Colors.OUTLINE),
                ft.Text(title, size=st.SECTION, color=st.MUTED),
                st.caption(hint),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=st.GAP_MD),
            alignment=ft.Alignment(0, 0), expand=True)

    def _acc_color(self, acc):
        if acc is None:
            return None
        return st.SUCCESS if acc >= 0.6 else (st.WARNING if acc >= 0.4 else st.ERROR)

    def _tile(self, label, value, icon, sub="", color=None) -> ft.Control:
        return ft.Container(
            content=ft.Column([
                ft.Row([ft.Icon(icon, size=18, color=st.MUTED), st.caption(label)], spacing=st.GAP_XS),
                ft.Text(value, size=st.METRIC, weight=ft.FontWeight.BOLD,
                        color=color or ft.Colors.ON_SURFACE),
                st.caption(sub) if sub else ft.Container(height=0),
            ], spacing=st.GAP_XS),
            padding=st.PAD, bgcolor=st.SUBTLE_BG, border_radius=8, width=280,
            border=ft.Border.all(1, st.PANEL_BORDER))

    def _chip(self, text, icon, color=None) -> ft.Control:
        return ft.Container(
            content=ft.Row([ft.Icon(icon, size=14, color=color or st.MUTED),
                            ft.Text(text, size=st.CAPTION, color=color or st.MUTED)], spacing=4),
            padding=ft.Padding(10, 5, 10, 5), bgcolor=st.SUBTLE_BG, border_radius=20)

    def _summary_chips(self, r: SubjectResult) -> ft.Control:
        acc = r.accuracy
        return ft.Row([
            self._chip(f"{len(r.feedback_runs)} feedback runs", ft.Icons.REPLAY),
            self._chip(f"accuracy {(_fmt(acc * 100, '.0f') + '%') if acc is not None else '—'} "
                       f"({r.n_correct}/{r.n_reports})", ft.Icons.CHECK, self._acc_color(acc)),
            self._chip(f"mean z {_fmt(r.mean_feedback_pda_z)}", ft.Icons.TRENDING_UP),
        ], spacing=st.GAP_XS, wrap=True)

    def _subject_row(self, r: SubjectResult) -> ft.Control:
        row = ft.Row([
            ft.Icon(ft.Icons.PERSON, color=st.ACCENT),
            ft.Container(content=st.section(r.subject, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                        width=180),
            ft.Container(content=self._summary_chips(r), expand=True),
            ft.IconButton(ft.Icons.EDIT, tooltip="Rename",
                          on_click=lambda _, s=r.subject: self._prompt_rename(s)),
            ft.IconButton(ft.Icons.DELETE_OUTLINE, tooltip="Delete", icon_color=st.ERROR,
                          on_click=lambda _, s=r.subject: self._confirm_delete(s)),
            ft.Icon(ft.Icons.CHEVRON_RIGHT, color=st.MUTED),
        ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER,
           spacing=st.GAP_SM)

        return ft.Card(content=ft.Container(
            content=row, padding=st.PAD, ink=True,
            on_click=lambda _, s=r.subject: self._select(s)))

    def _detail(self, r: SubjectResult) -> ft.Control:
        acc = r.accuracy
        bar = ft.ProgressBar(value=acc if acc is not None else 0, bar_height=10,
                             color=self._acc_color(acc) or st.NEUTRAL,
                             bgcolor=ft.Colors.with_opacity(0.12, ft.Colors.GREY_500))

        top = ft.Row([
            ft.Row([ft.IconButton(ft.Icons.ARROW_BACK, tooltip="Back to subjects",
                                  on_click=lambda _: self._back_to_list()),
                    ft.Icon(ft.Icons.PERSON, color=st.ACCENT),
                    st.section(r.subject)], spacing=st.GAP_SM),
            ft.Row([
                ft.OutlinedButton("Rename", icon=ft.Icons.EDIT,
                                  on_click=lambda _, s=r.subject: self._prompt_rename(s)),
                ft.OutlinedButton("Delete", icon=ft.Icons.DELETE_OUTLINE,
                                  on_click=lambda _, s=r.subject: self._confirm_delete(s)),
            ], spacing=st.GAP_XS),
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

        return ft.Column([
            top,
            self._summary_chips(r),
            ft.Row([ft.Container(content=bar, expand=True),
                    st.caption("← 50% chance")], spacing=st.GAP_SM),
            ft.Divider(),
            ft.Container(content=self._run_table(r), expand=True),
        ], spacing=st.GAP_MD, expand=True)

    def _run_table(self, r: SubjectResult) -> ft.Control:
        cols = [ft.DataColumn(ft.Text(c, size=st.CAPTION)) for c in
                ("Block", "TRs", "mean PDA", "feedback z", "ball dir", "reported", "correct")]
        rows = []
        for run in r.runs:
            correct = ("—" if run.correct is None else ("✓" if run.correct else "✗"))
            ccolor = (st.SUCCESS if run.correct else st.ERROR) if run.correct is not None else st.MUTED
            cells = [
                ft.DataCell(ft.Text(run.label, size=st.CAPTION)),
                ft.DataCell(ft.Text(str(run.n_tr) if run.n_tr else "—", size=st.CAPTION)),
                ft.DataCell(ft.Text(_fmt(run.mean_pda), size=st.CAPTION)),
                ft.DataCell(ft.Text(_fmt(run.mean_pda_z), size=st.CAPTION)),
                ft.DataCell(ft.Text(run.true_direction or "—", size=st.CAPTION)),
                ft.DataCell(ft.Text(run.answer or "—", size=st.CAPTION)),
                ft.DataCell(ft.Text(correct, size=st.CAPTION, color=ccolor, weight=ft.FontWeight.BOLD)),
            ]
            rows.append(ft.DataRow(cells=cells))
        return ft.Container(content=ft.DataTable(columns=cols, rows=rows, column_spacing=24,
                                                 heading_row_height=32, data_row_max_height=34),
                            padding=ft.Padding(0, 0, 0, st.GAP))

    # ── management dialogs (destructive — confirm first) ──────────────────
    def _prompt_rename(self, subject: str) -> None:
        field = ft.TextField(label="New subject id", value=subject, autofocus=True)

        def go(_):
            new = (field.value or "").strip()
            if not new or new == subject:
                self.page.pop_dialog()
                return
            if not rename_subject(subject, new):
                field.error_text = "That id is blank or already taken"
                field.update()
                return
            self.page.pop_dialog()
            self.app.toast(f"Renamed {subject} → {new}")
            if self.selected_subject == subject:
                self.selected_subject = new
            self._refresh()

        dlg = ft.AlertDialog(
            title=ft.Text(f"Rename {subject}"),
            content=ft.Container(content=field, width=360),
            actions=[ft.TextButton("Cancel", on_click=lambda _: self.page.pop_dialog()),
                     ft.FilledButton("Rename", icon=ft.Icons.EDIT, on_click=go)])
        self.page.show_dialog(dlg)

    def _confirm_delete(self, subject: str) -> None:
        def do(_):
            delete_subject(subject)
            self.page.pop_dialog()
            self.app.toast(f"Deleted {subject}")
            if self.selected_subject == subject:
                self.selected_subject = None
            self._refresh()

        dlg = ft.AlertDialog(
            title=ft.Text("Delete subject?"),
            content=ft.Text(f"Permanently delete all saved data for '{subject}'? This cannot be undone."),
            actions=[ft.TextButton("Cancel", on_click=lambda _: self.page.pop_dialog()),
                     ft.FilledButton("Delete", icon=ft.Icons.DELETE, on_click=do)])
        self.page.show_dialog(dlg)
