"""New-study wizard — pick a headset, then assemble the protocol.

A guided alternative to the raw tabbed StudyEditor: choose the EEG headset (which sets the
acquisition source + decoder montage) and the neurofeedback protocol (mbNF or R-mbNF, with the
transfer blocks / #feedback-runs / timings). Contact-quality + calibration are always included.
Produces a StudyConfig, saves it, and hands off to the Study Manager (edit fine details there).
"""
from __future__ import annotations

import copy

import flet as ft

from .. import styles as st
from ..models import (
    DEFAULT_PROTOCOL,
    DEFAULT_SESSION_CONFIG,
    HEADSET_PRESETS,
    MONTAGE_PRESETS,
    PROTOCOL_TYPES,
    StudyConfig,
)


class StudyWizard:
    def __init__(self, app):
        self.app = app
        self.page = app.page

    def _num(self, field, default):
        try:
            return float(field.value)
        except (TypeError, ValueError):
            return default

    def build(self) -> ft.Control:
        self.f_name = ft.TextField(label="Study name", value="New Study", autofocus=True)
        self.f_desc = ft.TextField(label="Brief description", multiline=True, min_lines=2, max_lines=4,
                                   hint_text="One or two lines describing the study.")
        self.f_basename = ft.TextField(label="Subject basename", value="sub", width=200,
                                       hint_text="e.g. sub -> sub01, sub02")
        self.f_nsubj = ft.TextField(label="Number of subjects", value="1", width=200)

        # 1 — headset (sets source + default decoder montage)
        self.f_headset = ft.RadioGroup(
            value="epocx",
            content=ft.Column([
                ft.Radio(value=k, label=f"{v['label']}   →   {MONTAGE_PRESETS[v['montage']]['label']}")
                for k, v in HEADSET_PRESETS.items()
            ], spacing=st.GAP_XS))

        # 1b — EPOC-X decoder variant: overrides the headset's default montage, only for EPOC-X
        # (the only headset with more than one trained model to choose between).
        self.f_decoder_variant = ft.RadioGroup(
            value="epoc12",
            content=ft.Column([
                ft.Radio(value="epoc12", label=f"{MONTAGE_PRESETS['epoc12']['label']} (default)"),
                ft.Radio(value="epoc_dual", label=MONTAGE_PRESETS["epoc_dual"]["label"]),
            ], spacing=st.GAP_XS))

        # 2 — protocol
        self.f_protocol = ft.RadioGroup(
            value="mbNF",
            content=ft.Column([
                ft.Radio(value="mbNF", label="mbNF — mindfulness NF (veridical feedback)"),
                ft.Radio(value="R-mbNF", label="R-mbNF — randomized direction + up/down report (accuracy test)"),
            ], spacing=st.GAP_XS))

        fb = DEFAULT_PROTOCOL["feedback"]
        tp = DEFAULT_PROTOCOL["transfer_pre"]
        cal = DEFAULT_PROTOCOL["calibration"]
        self.f_transfer_pre = ft.Checkbox(label="Transfer (pre) — static targets, no feedback", value=True)
        self.f_transfer_post = ft.Checkbox(label="Transfer (post) — static targets, no feedback", value=True)
        self.f_nruns = ft.TextField(label="Feedback runs", value=str(fb["n_runs"]), width=140)
        self.f_rest = ft.TextField(label="Rest per block (s)", value=str(int(tp["rest_sec"])), width=170)
        self.f_task = ft.TextField(label="Noting task per block (s)", value=str(int(tp["task_sec"])), width=200)
        # calibration cycles rest -> flanker -> rest -> self ("induction") or rest -> noting
        # ("noting" — the design the frozen ridge was actually trained on), `cycles` times — drives
        # the decoded PDA to both poles rather than just collecting quiet rest.
        self.f_calib_type = ft.RadioGroup(
            value=cal.get("type", "induction"),
            content=ft.Row([
                ft.Radio(value="induction", label="Induction (self-reference + flanker)"),
                ft.Radio(value="noting", label="Rest vs. mental noting"),
            ], spacing=st.GAP, wrap=True))
        self.f_calib_rest = ft.TextField(label="Rest per cue (s)", value=str(int(cal["rest_sec"])), width=140)
        self.f_calib_cycles = ft.TextField(label="Cycles", value=str(int(cal["cycles"])), width=100)
        self.f_calib_self = ft.TextField(label="Self-reference block (s)", value=str(int(cal["self_sec"])), width=200)
        self.f_calib_flanker = ft.TextField(label="Flanker block (s)", value=str(int(cal["flanker_sec"])), width=170)
        self.f_calib_noting = ft.TextField(label="Noting block (s)", value=str(int(cal["noting_sec"])), width=170)

        def card(title, *children):
            return ft.Container(
                content=ft.Column([st.section(title), *children], spacing=st.GAP_SM),
                padding=st.PAD, bgcolor=st.SUBTLE_BG, border_radius=8,
                border=ft.Border.all(1, st.PANEL_BORDER))

        always = ft.Container(
            content=ft.Row([ft.Icon(ft.Icons.CHECK_CIRCLE, color=st.SUCCESS, size=st.ICON_SM),
                            st.caption("Always included: contact-quality check + calibration "
                                       "(rest / self-reference / flanker cycles).", italic=False)],
                           spacing=st.GAP_SM),
            padding=ft.padding.symmetric(vertical=st.GAP_XS))

        body = ft.Column([
            card("Study", self.f_name, self.f_desc,
                 ft.Row([self.f_basename, self.f_nsubj], spacing=st.GAP, wrap=True)),
            card("1 · Headset", st.caption("Sets the acquisition source and decoder montage."), self.f_headset),
            card("1b · EPOC-X decoder variant",
                 st.caption("Only applies when the headset above is Emotiv EPOC X; ignored for "
                            "EPOC Flex / Brain Products, which use the research-cap montage. "
                            "Dual electrode: single-electrode CEN(P8)/DMN(O1) ridges — zero-shot "
                            "LOSO r=0.118/0.082, both better than the default 12-channel montage's "
                            "own zero-shot LOSO (r=0.080/0.058) in the same validated procedure; "
                            "PDA is derived as CEN−DMN rather than independently decoded."),
                 self.f_decoder_variant),
            card("2 · Protocol",
                 always,
                 self.f_protocol,
                 ft.Divider(),
                 st.subsection("Calibration"),
                 self.f_calib_type,
                 ft.Row([self.f_calib_rest, self.f_calib_cycles, self.f_calib_self,
                        self.f_calib_flanker, self.f_calib_noting], spacing=st.GAP, wrap=True),
                 st.caption("Induction: self-reference ('does this word describe you?') drives "
                            "DMN/PDA down, flanker (attention/executive-control) drives CEN/PDA up. "
                            "Noting: rest vs. mental noting — the same rest-then-task design the "
                            "frozen decoder was actually trained on in the scanner. Either way, "
                            "cycles=0 falls back to a single flat rest window."),
                 ft.Divider(),
                 st.subsection("Blocks (each: rest → noting task)"),
                 ft.Row([self.f_transfer_pre]), ft.Row([self.f_transfer_post]),
                 ft.Row([self.f_nruns, self.f_rest, self.f_task], spacing=st.GAP, wrap=True),
                 st.caption("R-mbNF flips the PDA→ball direction at random each feedback run and asks "
                            "'up or down?' after each — agreement with the true direction scores accuracy.")),
        ], spacing=st.GAP, scroll=ft.ScrollMode.AUTO, expand=True)

        header = ft.Container(
            content=ft.Row([
                ft.Row([ft.IconButton(ft.Icons.ARROW_BACK, on_click=lambda _: self.app.show_study_manager()),
                        st.title("New Study")], spacing=st.GAP_SM),
                ft.FilledButton("Create", icon=ft.Icons.CHECK, on_click=lambda _: self._create()),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            padding=st.PAD, bgcolor=st.HEADER_BG)

        return ft.Column([header, ft.Container(content=body, padding=st.PAD, expand=True)],
                         spacing=0, expand=True)

    def _create(self) -> None:
        name = (self.f_name.value or "").strip()
        if not name:
            self.f_name.error_text = "Name is required"; self.f_name.update(); return

        headset = self.f_headset.value
        preset = HEADSET_PRESETS[headset]
        rest = self._num(self.f_rest, 30.0)
        task = self._num(self.f_task, 150.0)
        calib_type = self.f_calib_type.value
        calib_rest = self._num(self.f_calib_rest, 15.0)
        calib_cycles = int(self._num(self.f_calib_cycles, 3))
        calib_self = self._num(self.f_calib_self, 30.0)
        calib_flanker = self._num(self.f_calib_flanker, 45.0)
        calib_noting = self._num(self.f_calib_noting, 60.0)
        n_runs = int(self._num(self.f_nruns, 1))

        sc = StudyConfig.new(name)
        sc.metadata.description = (self.f_desc.value or "").strip()
        cfg = copy.deepcopy(DEFAULT_SESSION_CONFIG)
        cfg["headset"] = headset
        cfg["subjects"] = {"basename": (self.f_basename.value or "sub").strip(),
                           "n_subjects": int(self._num(self.f_nsubj, 1))}
        cfg["source"]["type"] = preset["source"]
        montage = self.f_decoder_variant.value if headset == "epocx" else preset["montage"]
        cfg["decoder"]["montage"] = montage
        cfg["decoder"]["model_path"] = ""
        cfg["protocol"] = {
            "type": self.f_protocol.value,
            "calibration": {"type": calib_type, "rest_sec": calib_rest, "cycles": calib_cycles,
                            "self_sec": calib_self, "flanker_sec": calib_flanker,
                            "noting_sec": calib_noting},
            "transfer_pre": {"enabled": bool(self.f_transfer_pre.value), "rest_sec": rest, "task_sec": task},
            "feedback": {"rest_sec": rest, "task_sec": task, "n_runs": max(1, n_runs)},
            "transfer_post": {"enabled": bool(self.f_transfer_post.value), "rest_sec": rest, "task_sec": task},
        }
        # legacy session mirror — only a defensive fallback now (session_engine.EngineConfig.calib_sec
        # is used only if a calibration block dict were missing rest_sec, which build_blocks() never
        # does); the stimulus's calibration countdown reads the engine's live calib_cue_sec instead.
        cfg["session"].update({"calibrate": True, "calib_sec": calib_rest, "rest_sec": rest,
                               "feedback_sec": task, "n_runs": max(1, n_runs)})
        cfg["feedback"]["mode"] = "ball"
        sc.session_config = cfg

        self.app.config_manager.save(sc)
        self.app.toast(f"Created study: {name}")
        self.app.show_study_manager()
