#!/usr/bin/env python3
"""
eeg_preproc.py
Automated EEG preprocessing pipeline for DMNELF simultaneous EEG-fMRI data.

Usage:
    python eeg_preproc.py --subject sub-dmnelf009 --task rest --run 01
    python eeg_preproc.py --all
    python eeg_preproc.py --subject sub-dmnelf009
    python eeg_preproc.py --all --overwrite

Steps:
    1. Load minimally preprocessed EDF (BVA gradient correction, 1kHz)
    2. Detect ECG channel, compute R-peaks (NeuroKit2)
    3. Auto-detect bad channels (variance + HF noise z-score)
    4. Annotate noisy edges (scanner ramp artifact)
    5. Bandpass filter (1-40 Hz)
    6. BCG correction using MNE create_ecg_epochs
    7. Downsample to 250 Hz
    8. ICA (29 components, ICLabel + cardiac + EOG correlation)
    9. Interpolate bad channels
    10. Average reference
    11. Save FIF + QC images (raw + preproc)
"""

import argparse
import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
EEG_RAW_ROOT = Path("/projects/swglab/data/DMNELF/rawdata_eeg")
EEG_OUT_ROOT = Path("/projects/swglab/data/DMNELF/derivatives/eeg_preprocessed")

SUBJECTS_EEG = [
    "sub-dmnelf001", "sub-dmnelf004", "sub-dmnelf005", "sub-dmnelf006",
    "sub-dmnelf007", "sub-dmnelf008", "sub-dmnelf009", "sub-dmnelf010",
    "sub-dmnelf011", "sub-dmnelf012", "sub-dmnelf013",
    "sub-dmnelf1001", "sub-dmnelf1002", "sub-dmnelf1003",
]

TASKS = {
    "rest":      ["01", "02"],
    "shortrest": ["01"],
    "feedback":  ["01", "02", "03", "04"],
}

MISSING_RUNS = {
    "sub-dmnelf1002": [("rest", "02")],
    "sub-dmnelf1003": [("feedback", "04")],
}

# ── Preprocessing parameters ───────────────────────────────────────────────
SFREQ_TARGET    = 250.0  # default, overridden by --sfreq argument
HIGHPASS        = 1.0
LOWPASS         = 40.0
N_ICA           = 29
BCG_TMIN        = -0.2
BCG_TMAX        = 0.6
BAD_CH_Z        = 3.0
BAD_HF_Z        = 2.5
EDGE_SEARCH_SEC = 5.0
EDGE_THRESH     = 3.0


# ═══════════════════════════════════════════════════════════════════════════
# PREPROCESSING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_run(subject, task, run, overwrite=False, sfreq_target=250.0):
    import mne
    mne.set_log_level('WARNING')

    session  = "ses-dmnelf"
    eeg_dir  = EEG_RAW_ROOT / subject / session / "eeg"
    out_dir  = EEG_OUT_ROOT / subject / session / "eeg"
    qc_dir   = out_dir / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    edf_file   = eeg_dir / (subject + "_" + session
                            + "_task-" + task + "_run-" + run
                            + "_desc-bvaAC1kHz_eeg.edf")
    sfreq_tag  = str(int(sfreq_target)) + "Hz"
    
    output_fif = out_dir / (subject + "_" + session
                            + "_task-" + task + "_run-" + run
                            + "_desc-preproc" + sfreq_tag + "_eeg.fif")

    tag = subject + " task-" + task + " run-" + run
    print("\n" + "=" * 60)
    print(tag)
    print("=" * 60)

    if output_fif.exists() and not overwrite:
        print("  EXISTS (skip): " + output_fif.name)
        return True

    if not edf_file.exists():
        print("  MISSING EDF: " + edf_file.name)
        return False

    # ── 1. Load ────────────────────────────────────────────
    print("  Loading EDF...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)

    ecg_ch_name = None
    for ch in raw.ch_names:
        if any(x in ch.upper() for x in ['ECG', 'EKG', 'HEART', 'CARDIO']):
            raw.set_channel_types({ch: 'ecg'})
            ecg_ch_name = ch
            break

    print("  sfreq=" + str(raw.info['sfreq']) + "Hz"
          + "  ch=" + str(len(raw.ch_names))
          + "  dur=" + str(round(raw.times[-1], 1)) + "s"
          + ("  ECG=" + ecg_ch_name if ecg_ch_name else "  no ECG"))

    # ── 2. R-peak detection ────────────────────────────────
    cardiac_events = None
    cardiac_freq   = 1.2
    avg_hr         = 72.0

    if ecg_ch_name is not None:
        try:
            import neurokit2 as nk
            ecg_data    = raw.get_data(picks=ecg_ch_name)[0]
            ecg_cleaned = nk.ecg_clean(ecg_data, sampling_rate=raw.info['sfreq'])
            _, rpeaks   = nk.ecg_peaks(ecg_cleaned, sampling_rate=raw.info['sfreq'])
            r_idx       = rpeaks['ECG_R_Peaks']
            if len(r_idx) > 1:
                avg_hr         = len(r_idx) / (raw.times[-1] / 60)
                cardiac_freq   = avg_hr / 60
                cardiac_events = r_idx
                print("  R-peaks=" + str(len(r_idx))
                      + "  HR=" + str(round(avg_hr, 1)) + "BPM")
        except Exception as e:
            print("  R-peak detection failed: " + str(e))

    # ── 3. Bad channel detection ───────────────────────────
    eeg_data     = raw.get_data(picks='eeg')
    eeg_ch_names = [ch for ch in raw.ch_names if ch != ecg_ch_name]

    ch_std   = eeg_data.std(axis=1)
    z_scores = (ch_std - ch_std.mean()) / (ch_std.std() + 1e-10)

    raw_hf   = raw.copy().filter(40, 50, picks='eeg', verbose=False)
    hf_data  = raw_hf.get_data(picks='eeg')
    hf_power = hf_data.var(axis=1)
    hf_z     = (hf_power - hf_power.mean()) / (hf_power.std() + 1e-10)
    del raw_hf

    auto_bads = list(set(
        [eeg_ch_names[i] for i, z in enumerate(z_scores) if abs(z) > BAD_CH_Z] +
        [eeg_ch_names[i] for i, z in enumerate(hf_z)     if z > BAD_HF_Z]
    ))
    if auto_bads:
        raw.info['bads'] = auto_bads
        print("  Bad channels: " + str(auto_bads))
    else:
        print("  Bad channels: none")

    # ── 4. Annotate noisy edges ────────────────────────────
    rms        = np.sqrt((eeg_data**2).mean(axis=0))
    sfreq      = raw.info['sfreq']
    n_search   = int(EDGE_SEARCH_SEC * sfreq)
    mid_start  = int(len(rms) * 0.1)
    mid_end    = int(len(rms) * 0.9)
    stable_rms = np.median(rms[mid_start:mid_end])
    threshold  = stable_rms * EDGE_THRESH

    onset_end = 0.0
    for i in range(n_search):
        if rms[i] > threshold:
            onset_end = (i + 1) / sfreq
        else:
            break

    offset_start = raw.times[-1]
    for i in range(len(rms)-1, len(rms)-n_search-1, -1):
        if rms[i] > threshold:
            offset_start = i / sfreq
        else:
            break

    new_annots = []
    if onset_end > 0.1:
        new_annots.append((0.0, onset_end, 'BAD_edge_start'))
    if offset_start < raw.times[-1] - 0.1:
        new_annots.append((offset_start, raw.times[-1] - offset_start, 'BAD_edge_end'))
    if new_annots:
        from mne import Annotations
        annots = Annotations(
            onset=[a[0] for a in new_annots],
            duration=[a[1] for a in new_annots],
            description=[a[2] for a in new_annots],
            orig_time=raw.info.get('meas_date')
        )
        raw.set_annotations(raw.annotations + annots)
        print("  Edge annotations: " + str(len(new_annots)))

    # ── 5. QC image — raw ─────────────────────────────────
    print("  Saving raw QC image...")
    _save_qc_raw(raw, eeg_ch_names, auto_bads,
                 qc_dir, subject, session, task, run)

    # ── 6. Filter ─────────────────────────────────────────
    print("  Filtering " + str(HIGHPASS) + "-" + str(LOWPASS) + "Hz...")
    raw_filtered = raw.copy().filter(
        HIGHPASS, LOWPASS, picks='eeg',
        method='fir', verbose=False
    )

    # ── 7. BCG correction via ECG epochs ──────────────────
    if ecg_ch_name is not None and cardiac_events is not None:
        print("  BCG correction using NeuroKit2 R-peaks...")
        try:
            # Build events array from our NeuroKit2 R-peaks
            # Adjust R-peak indices to filtered data (same sfreq at this point)
            events_bcg = np.column_stack([
                cardiac_events,
                np.zeros(len(cardiac_events), dtype=int),
                np.ones(len(cardiac_events),  dtype=int)
            ])
            epochs_bcg = mne.Epochs(
                raw_filtered, events_bcg, event_id=1,
                tmin=BCG_TMIN, tmax=BCG_TMAX,
                baseline=None, preload=True,
                picks='eeg',
                reject_by_annotation=True,
                verbose=False
            )
            n_epochs = len(epochs_bcg)
            print("  BCG epochs: " + str(n_epochs))
            if n_epochs > 10:
                bcg_template  = epochs_bcg.average()
                template_data = bcg_template.get_data()  # (n_eeg, n_times)
                n_tmpl        = template_data.shape[1]
                eeg_picks     = mne.pick_types(raw_filtered.info, eeg=True)
                raw_data      = raw_filtered.get_data(picks='eeg').copy()
                sfreq_f       = raw_filtered.info['sfreq']
                n_applied     = 0
                for r_samp in cardiac_events:
                    onset = r_samp + int(BCG_TMIN * sfreq_f)
                    end   = onset + n_tmpl
                    if onset >= 0 and end <= raw_data.shape[1]:
                        raw_data[:, onset:end] -= template_data
                        n_applied += 1
                raw_filtered._data[eeg_picks, :] = raw_data
                print("  BCG applied to " + str(n_applied) + " heartbeats")
            else:
                print("  BCG skipped (too few clean epochs: " + str(n_epochs) + ")")
        except Exception as e:
            print("  BCG failed: " + str(e))
    elif ecg_ch_name is None:
        print("  BCG skipped — no ECG channel found (flag for QC)")
    else:
        print("  BCG skipped — no R-peaks detected (flag for QC)")

    # ── 8. Downsample ─────────────────────────────────────
    if raw_filtered.info['sfreq'] > sfreq_target:
        print("  Downsampling to " + str(sfreq_target) + "Hz...")
        raw_filtered.resample(sfreq_target, verbose=False)

    # ── 9. Set montage ────────────────────────────────────
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw_filtered.set_montage(montage, on_missing='ignore', verbose=False)
    except Exception as e:
        print("  Montage failed: " + str(e))

    # ── 10. ICA ───────────────────────────────────────────
    from mne.preprocessing import ICA
    n_eeg        = len(mne.pick_types(raw_filtered.info, eeg=True))
    n_components = min(N_ICA, n_eeg - 1)
    print("  ICA (" + str(n_components) + " components)...")

    ica = ICA(
        n_components=n_components,
        method='picard',
        random_state=42,
        max_iter=500,
        fit_params=dict(ortho=False, extended=True),
        verbose=False
    )

    ica.fit(raw_filtered, picks='eeg',
            reject_by_annotation=True, verbose=False)

    artifact_components = []

    # ICLabel
    try:
        from mne_icalabel import label_components
        ic_labels = label_components(raw_filtered, ica, method='iclabel')
        labels    = ic_labels['labels']
        artifact_components = [i for i, l in enumerate(labels) if l != 'brain']
        print("  ICLabel artifacts: " + str(artifact_components))
    except Exception as e:
        print("  ICLabel failed: " + str(e))
        labels = None

    # Cardiac detection via CTPS (same as Mexico pipeline)
    cardiac_components = []
    if ecg_ch_name is not None:
        try:
            ecg_epochs_ica = mne.preprocessing.create_ecg_epochs(
                raw_filtered,
                ch_name=ecg_ch_name,
                verbose=False
            )
            cardiac_components, ecg_scores = ica.find_bads_ecg(
                ecg_epochs_ica,
                ch_name=ecg_ch_name,
                method='ctps',
                threshold=0.9,
                verbose=False
            )
            cardiac_components = [int(i) for i in cardiac_components]
            if cardiac_components:
                print("  Cardiac CTPS: " + str(cardiac_components))
            else:
                # Fall back to correlation if CTPS finds nothing
                best = int(np.argmax(np.abs(ecg_scores)))
                if abs(ecg_scores[best]) > 0.2:
                    cardiac_components = [best]
                    print("  Cardiac fallback (best corr): IC" + str(best)
                          + "  score=" + str(round(float(ecg_scores[best]), 3)))
        except Exception as e:
            print("  Cardiac CTPS failed: " + str(e))

    # EOG detection using Fp1/Fp2 as proxy
    eog_components = []
    try:
        eog_chs = [ch for ch in ['Fp1', 'Fp2'] if ch in raw_filtered.ch_names]
        if eog_chs:
            eog_proxy = raw_filtered.get_data(picks=eog_chs).mean(axis=0)
            sources   = ica.get_sources(raw_filtered).get_data()
            for idx in range(n_components):
                min_len = min(len(sources[idx]), len(eog_proxy))
                corr = abs(np.corrcoef(
                    sources[idx][:min_len], eog_proxy[:min_len]
                )[0, 1])
                if corr > 0.5:
                    eog_components.append(idx)
            if eog_components:
                print("  EOG ICA: " + str(eog_components))
    except Exception as e:
        print("  EOG ICA detection failed: " + str(e))

    artifact_components = sorted(set(
        artifact_components + cardiac_components + eog_components
    ))

# Combine: ICLabel + cardiac + EOG
    # Priority: cardiac and EOG always included
    # ICLabel capped at 20% on top
    priority   = sorted(set(cardiac_components + eog_components))
    iclabel_extra = [c for c in artifact_components
                     if c not in priority]
    max_iclabel = int(n_components * 0.20)
    if len(iclabel_extra) > max_iclabel:
        print("  WARNING: ICLabel flagged " + str(len(iclabel_extra))
              + " extra components — capping at " + str(max_iclabel))
        iclabel_extra = iclabel_extra[:max_iclabel]

    artifact_components = sorted(set(priority + iclabel_extra))
    print("  Final exclusions: " + str(artifact_components)
          + "  (cardiac=" + str(cardiac_components)
          + "  eog=" + str(eog_components)
          + "  iclabel=" + str(iclabel_extra) + ")")
    
    if artifact_components:
        ica.exclude = artifact_components
        raw_clean   = raw_filtered.copy()
        ica.apply(raw_clean, verbose=False)
        print("  ICA excluded: " + str(artifact_components))
    else:
        raw_clean = raw_filtered
        print("  ICA: no components excluded")

    # ── 11. Interpolate + reference ───────────────────────
    if raw_clean.info['bads']:
        raw_clean.interpolate_bads(reset_bads=True, verbose=False)
        print("  Bad channels interpolated")
    raw_clean.set_eeg_reference('average', projection=False, verbose=False)
    print("  Average reference applied")

    # ── 12. QC image — preproc ────────────────────────────
    print("  Saving preproc QC image...")
    _save_qc_preproc(raw_filtered, raw_clean,
                     qc_dir, subject, session, task, run, cardiac_freq)

    # ── 13. Save FIF ──────────────────────────────────────
    raw_clean.save(str(output_fif), overwrite=True, verbose=False)
    fsize = os.path.getsize(str(output_fif)) / 1e6
    print("  Saved: " + output_fif.name + "  (" + str(round(fsize, 1)) + "MB)")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# QC HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _save_qc_raw(raw, eeg_ch_names, auto_bads,
                 qc_dir, subject, session, task, run):
    import mne
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(20, 14))

    # Panel 1: channel std
    eeg_data = raw.get_data(picks='eeg')
    ch_std   = eeg_data.std(axis=1)
    colors   = ['red' if c in auto_bads else 'steelblue' for c in eeg_ch_names]
    axes[0].bar(range(len(ch_std)), ch_std * 1e6, color=colors)
    axes[0].set_title(subject + " task-" + task + " run-" + run
                      + " — Channel std (red=bad)")
    axes[0].set_xticks(range(len(eeg_ch_names)))
    axes[0].set_xticklabels(eeg_ch_names, rotation=90, fontsize=7)
    axes[0].set_ylabel("Std (µV)")

    # Panel 2: whole run traces
    sfreq  = raw.info['sfreq']
    decim  = max(1, int(sfreq / 50))
    d_dec  = eeg_data[:, ::decim]
    t_dec  = raw.times[::decim]
    offset = 0
    yticks = []
    for i, ch in enumerate(eeg_ch_names):
        color = 'red' if ch in auto_bads else 'black'
        axes[1].plot(t_dec, d_dec[i] * 1e6 + offset,
                     color=color, linewidth=0.3, alpha=0.7)
        yticks.append((offset, ch))
        offset += 150
    for ann in raw.annotations:
        if 'BAD' in ann['description']:
            axes[1].axvspan(ann['onset'],
                            ann['onset'] + ann['duration'],
                            alpha=0.25, color='red')
    axes[1].set_yticks([y[0] for y in yticks])
    axes[1].set_yticklabels([y[1] for y in yticks], fontsize=7)
    axes[1].set_title("Whole run — raw EEG (red=bad, shading=BAD annotation)")
    axes[1].set_xlabel("Time (s)")

    # Panel 3: PSD
    psd   = raw.compute_psd(fmax=50, picks='eeg', verbose=False)
    psds  = psd.get_data()
    freqs = psd.freqs
    axes[2].semilogy(freqs, psds.mean(axis=0), linewidth=1.5)
    axes[2].fill_between(freqs, psds.min(axis=0), psds.max(axis=0), alpha=0.2)
    axes[2].set_title("PSD — before preprocessing")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Power (V²/Hz)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = qc_dir / (subject + "_ses-dmnelf_task-" + task
                      + "_run-" + run + "_qc_raw.png")
    fig.savefig(str(fname), dpi=100)
    plt.close()
    print("    qc_raw saved")


def _save_qc_preproc(raw_before, raw_after,
                     qc_dir, subject, session, task, run, cardiac_freq):
    import mne
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(20, 22))

    # Panel 1: PSD comparison
    psd_b = raw_before.compute_psd(fmax=50, picks='eeg', verbose=False)
    psd_a = raw_after.compute_psd(fmax=50, picks='eeg', verbose=False)
    axes[0].semilogy(psd_b.freqs, psd_b.get_data().mean(axis=0),
                     'b--', linewidth=1.5, alpha=0.7, label='Before')
    axes[0].semilogy(psd_a.freqs, psd_a.get_data().mean(axis=0),
                     'g-',  linewidth=1.5, label='After')
    axes[0].axvspan(cardiac_freq - 0.2, cardiac_freq + 0.2,
                    alpha=0.15, color='red', label='cardiac')
    axes[0].set_title(subject + " task-" + task + " run-" + run
                      + " — PSD comparison")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power (V²/Hz)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: whole run RMS before
    d_b   = raw_before.get_data(picks='eeg')
    rms_b = np.sqrt((d_b**2).mean(axis=0))
    axes[1].plot(raw_before.times, rms_b * 1e6, 'b-', linewidth=0.5)
    for ann in raw_before.annotations:
        if 'BAD' in ann['description']:
            axes[1].axvspan(ann['onset'],
                            ann['onset'] + ann['duration'],
                            alpha=0.3, color='red')
    axes[1].set_title("Whole run RMS — before")
    axes[1].set_ylabel("RMS (µV)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: whole run RMS after
    d_a   = raw_after.get_data(picks='eeg')
    rms_a = np.sqrt((d_a**2).mean(axis=0))
    axes[2].plot(raw_after.times, rms_a * 1e6, 'g-', linewidth=0.5)
    axes[2].set_title("Whole run RMS — after")
    axes[2].set_ylabel("RMS (µV)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylim(axes[1].get_ylim())
    axes[2].grid(True, alpha=0.3)

    # Panel 4: whole run traces before
    eeg_ch_b = [ch for ch in raw_before.ch_names
                if mne.channel_type(raw_before.info,
                   raw_before.ch_names.index(ch)) == 'eeg']
    sfreq_b  = raw_before.info['sfreq']
    decim_b  = max(1, int(sfreq_b / 50))
    d_b_dec  = d_b[:8, ::decim_b]
    t_b_dec  = raw_before.times[::decim_b]
    offset   = 0
    yticks_b = []
    for i in range(min(8, d_b_dec.shape[0])):
        axes[3].plot(t_b_dec, d_b_dec[i] * 1e6 + offset,
                     linewidth=0.3, color='steelblue', alpha=0.8)
        yticks_b.append((offset, eeg_ch_b[i] if i < len(eeg_ch_b) else str(i)))
        offset += 200
    axes[3].set_yticks([y[0] for y in yticks_b])
    axes[3].set_yticklabels([y[1] for y in yticks_b], fontsize=8)
    axes[3].set_title("Whole run — before (first 8 channels)")
    axes[3].set_xlabel("Time (s)")

    # Panel 5: whole run traces after
    eeg_ch_a = [ch for ch in raw_after.ch_names
                if mne.channel_type(raw_after.info,
                   raw_after.ch_names.index(ch)) == 'eeg']
    sfreq_a  = raw_after.info['sfreq']
    decim_a  = max(1, int(sfreq_a / 50))
    d_a_dec  = d_a[:8, ::decim_a]
    t_a_dec  = raw_after.times[::decim_a]
    offset   = 0
    yticks_a = []
    for i in range(min(8, d_a_dec.shape[0])):
        axes[4].plot(t_a_dec, d_a_dec[i] * 1e6 + offset,
                     linewidth=0.3, color='green', alpha=0.8)
        yticks_a.append((offset, eeg_ch_a[i] if i < len(eeg_ch_a) else str(i)))
        offset += 200
    axes[4].set_yticks([y[0] for y in yticks_a])
    axes[4].set_yticklabels([y[1] for y in yticks_a], fontsize=8)
    axes[4].set_title("Whole run — after (first 8 channels)")
    axes[4].set_xlabel("Time (s)")

    plt.tight_layout()
    fname = qc_dir / (subject + "_ses-dmnelf_task-" + task
                      + "_run-" + run + "_qc_preproc.png")
    fig.savefig(str(fname), dpi=100)
    plt.close()
    print("    qc_preproc saved")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import mne
    parser = argparse.ArgumentParser(description="DMNELF EEG preprocessing")
    parser.add_argument("--subject",   type=str, default=None)
    parser.add_argument("--task",      type=str, default=None)
    parser.add_argument("--run",       type=str, default=None)
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sfreq", type=float, default=250.0,
                        help="Target sampling frequency (default: 250)")
    args = parser.parse_args()

    if args.all or (args.subject and not args.task) or \
       (not args.subject and not args.task and not args.run):
        subjects = [args.subject] if args.subject else SUBJECTS_EEG
        n_ok = 0
        n_fail = 0
        for subject in subjects:
            for task, runs in TASKS.items():
                for run in runs:
                    if subject in MISSING_RUNS:
                        if (task, run) in MISSING_RUNS[subject]:
                            continue
                    ok = preprocess_run(subject, task, run,
                                        args.overwrite, args.sfreq)
                    if ok:
                        n_ok += 1
                    else:
                        n_fail += 1
        print("\n" + "=" * 60)
        print("DONE  OK=" + str(n_ok) + "  FAILED=" + str(n_fail))
        print("=" * 60)

    elif args.subject and args.task and args.run:
        preprocess_run(args.subject, args.task, args.run, args.overwrite, args.sfreq)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()