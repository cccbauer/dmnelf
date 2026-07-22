"""Study editor screen — configure a protocol across tabs, then save.

Tabs: General · Source · Decoder · Session · Feedback. Fields are read back into the study's
``session_config`` on Save (config_manager writes the YAML). Kept deliberately flat — the whole
protocol is a handful of scalars.
"""
from __future__ import annotations

import flet as ft

from .. import styles as st
from ..models import MONTAGE_PRESETS, StudyConfig


class StudyEditor:
    def __init__(self, app, study: StudyConfig | None):
        self.app = app
        self.page = app.page
        self.is_new = study is None
        self.study = study or StudyConfig.new()

    # ── field helpers ────────────────────────────────────────────────────
    def _tf(self, label, value, hint="", width=None):
        return ft.TextField(label=label, value="" if value is None else str(value),
                            hint_text=hint, width=width, dense=True)

    def build(self) -> ft.Control:
        sc = self.study
        src, dec, sess, fb = sc.source, sc.decoder, sc.session, sc.feedback

        # General
        self.f_name = self._tf("Study name", sc.metadata.name)
        self.f_desc = ft.TextField(label="Description", value=sc.metadata.description,
                                   multiline=True, min_lines=2, max_lines=4)

        # Source
        self.f_source = ft.Dropdown(
            label="EEG source", value=src.get("type", "replay"),
            options=[ft.dropdown.Option("replay", "Replay (recorded .fif — no hardware)"),
                     ft.dropdown.Option("cortex", "Cortex API (live EPOC X, needs license)"),
                     ft.dropdown.Option("lsl", "LSL outlet (EmotivPRO)"),
                     ft.dropdown.Option("emokit", "Emokit dongle (EPOC/EPOC+ only)")])
        self.f_replay = self._tf("Replay .fif path", src.get("replay_path", ""),
                                 hint="testdata/dmnelf005_feedback_run-01_250Hz.fif")
        self.f_speed = self._tf("Replay speed (0 = as-fast)", src.get("replay_speed", 1.0), width=220)
        self.f_creds = self._tf("Credentials YAML (blank = mindwear/credentials.yaml)",
                                src.get("credentials_path", ""))

        # Decoder — montage preset picks the matching frozen model; "Custom" takes an explicit path.
        montage = dec.get("montage", "epoc12")
        self.f_montage = ft.Dropdown(
            label="Montage", value=montage,
            options=[ft.dropdown.Option(k, v["label"]) for k, v in MONTAGE_PRESETS.items()]
                    + [ft.dropdown.Option("custom", "Custom model path…")],
            on_select=lambda _: self._on_montage_change())
        shown_path = dec.get("model_path") or MONTAGE_PRESETS.get(montage, {}).get("model_path", "")
        self.f_model = self._tf("Model .npz", shown_path)
        self.f_model.disabled = montage != "custom"

        # Session
        self.f_calibrate = ft.Checkbox(label="Run a calibration phase first", value=bool(sess.get("calibrate", True)))
        self.f_calib = self._tf("Calibrate seconds", sess.get("calib_sec", 60.0), width=200)
        self.f_rest = self._tf("Rest baseline seconds", sess.get("rest_sec", 30.0), width=200)
        self.f_fb = self._tf("Feedback seconds", sess.get("feedback_sec", 300.0), width=200)
        self.f_runs = self._tf("Number of runs", sess.get("n_runs", 4), width=200)

        # Feedback
        self.f_mode = ft.Dropdown(
            label="Feedback display", value=fb.get("mode", "ball"),
            options=[ft.dropdown.Option("ball", "Ball task (scanner paradigm)"),
                     ft.dropdown.Option("bars", "Thermometer bars"),
                     ft.dropdown.Option("none", "None (operator plot only)")])
        self.f_scale = self._tf("Ball scale factor", fb.get("scale_factor", 10.0), width=200)
        self.f_targetz = self._tf("Target z (bars)", fb.get("target_z", 1.0), width=200)

        def _card(title, *children):
            return ft.Container(
                content=ft.Column([st.section(title), *children], spacing=st.GAP_SM),
                padding=st.PAD, bgcolor=st.SUBTLE_BG, border_radius=8,
                border=ft.Border.all(1, st.PANEL_BORDER))

        body = ft.Column([
            _card("General", self.f_name, self.f_desc),
            _card("EEG source", self.f_source,
                  st.caption("Replay streams a recorded run at real time — the whole pipeline runs with no "
                             "headset or license. Switch to LSL/Cortex for a live headset."),
                  self.f_replay, self.f_speed, self.f_creds),
            _card("Decoder", self.f_montage, self.f_model,
                  st.caption("The DMNELF-trained CEN/DMN ridge, frozen for the selected electrode montage. "
                             "Pick Custom to point at a different model .npz.")),
            _card("Session timing (legacy — protocol studies use the wizard's blocks)",
                  self.f_calibrate,
                  ft.Row([self.f_calib, self.f_rest], spacing=st.GAP, wrap=True),
                  ft.Row([self.f_fb, self.f_runs], spacing=st.GAP, wrap=True)),
            _card("Feedback display", self.f_mode,
                  ft.Row([self.f_scale, self.f_targetz], spacing=st.GAP, wrap=True),
                  st.caption("Ball task mirrors the MRI paradigm (CEN yellow top / DMN blue bottom). "
                             "PDA = CEN − DMN drives the ball.")),
        ], spacing=st.GAP, scroll=ft.ScrollMode.AUTO, expand=True)

        header = ft.Container(
            content=ft.Row([
                ft.Row([ft.IconButton(ft.Icons.ARROW_BACK, on_click=lambda _: self.app.show_study_manager()),
                        st.title("New Study" if self.is_new else f"Edit — {sc.metadata.name}")], spacing=st.GAP_SM),
                ft.FilledButton("Save", icon=ft.Icons.SAVE, on_click=lambda _: self._save()),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            padding=st.PAD, bgcolor=st.HEADER_BG)

        return ft.Column([header, ft.Container(content=body, padding=st.PAD, expand=True)],
                         spacing=0, expand=True)

    # ── decoder tab ──────────────────────────────────────────────────────
    def _on_montage_change(self) -> None:
        montage = self.f_montage.value
        preset = MONTAGE_PRESETS.get(montage)
        self.f_model.disabled = montage != "custom"
        if preset:
            self.f_model.value = preset["model_path"]
        self.f_model.update()

    # ── save ─────────────────────────────────────────────────────────────
    def _num(self, field, default):
        try:
            return float(field.value)
        except (TypeError, ValueError):
            return default

    def _save(self) -> None:
        name = (self.f_name.value or "").strip()
        if not name:
            self.f_name.error_text = "Name is required"
            self.f_name.update()
            return
        sc = self.study
        sc.metadata.name = name
        sc.metadata.description = self.f_desc.value or ""
        sc.source.update({
            "type": self.f_source.value,
            "replay_path": (self.f_replay.value or "").strip(),
            "replay_speed": self._num(self.f_speed, 1.0),
            "credentials_path": (self.f_creds.value or "").strip(),
        })
        montage = self.f_montage.value
        model_path = (self.f_model.value or "").strip() if montage == "custom" else ""
        sc.decoder.update({"montage": montage, "model_path": model_path})
        sc.session.update({
            "calibrate": bool(self.f_calibrate.value),
            "calib_sec": self._num(self.f_calib, 60.0),
            "rest_sec": self._num(self.f_rest, 30.0),
            "feedback_sec": self._num(self.f_fb, 300.0),
            "n_runs": int(self._num(self.f_runs, 4)),
        })
        sc.feedback.update({
            "mode": self.f_mode.value,
            "scale_factor": self._num(self.f_scale, 10.0),
            "target_z": self._num(self.f_targetz, 1.0),
        })
        self.app.config_manager.save(sc)
        self.app.toast(f"Saved study: {name}")
        self.app.show_study_manager()
