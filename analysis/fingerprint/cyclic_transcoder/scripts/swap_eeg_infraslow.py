"""
swap_eeg_infraslow.py
---------------------
Build a cyclic_features_infraslow tree from the existing cyclic_features by
REPLACING only the `eeg_block` (1-40Hz baseline) with the per-TR block-mean of
the INFRASLOW EEG (desc-preproc500HzISp01). fmri_features, pda, and all metadata
are copied unchanged, so the transcoder sees identical fMRI/PDA but infraslow EEG.

Avoids re-running the heavy DiFuMo/mask fMRI extraction. The IS .fif has the same
sample count as the baseline .fif, so block-averaging over n_volumes aligns 1:1.

Usage:
  python swap_eeg_infraslow.py --src .../cyclic_features --dst .../cyclic_features_infraslow \
      --eeg-root .../eeg_preprocessed --desc preproc500HzISp01 [--samples-per-tr 600]
"""
import argparse, warnings
from pathlib import Path
import numpy as np, mne
warnings.simplefilter("ignore"); mne.set_log_level("ERROR")


def block_mean(raw, spt, n_vol):
    x = raw.get_data(picks="eeg")
    need = n_vol * spt
    if x.shape[1] < need:
        raise ValueError(f"EEG {x.shape[1]} < need {need}")
    x = x[:, :need].reshape(x.shape[0], n_vol, spt).mean(2).T   # (n_vol, n_ch)
    mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-8
    return ((x - mu) / sd).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--eeg-root", required=True)
    ap.add_argument("--session", default="ses-dmnelf")
    ap.add_argument("--desc", default="preproc500HzISp01")
    ap.add_argument("--samples-per-tr", type=int, default=600)
    a = ap.parse_args()
    src = Path(a.src); dst = Path(a.dst); eroot = Path(a.eeg_root)

    n_ok = n_skip = n_miss = 0
    for npz in sorted(src.glob("sub-*/*_features.npz")):
        d = dict(np.load(npz, allow_pickle=True))
        subj = str(d["subject"]); task = str(d["task"]); run = str(d["run"])
        n_vol = d["eeg_block"].shape[0]
        fif = (eroot / f"sub-{subj}" / a.session / "eeg" /
               f"sub-{subj}_{a.session}_task-{task}_run-{int(run):02d}_desc-{a.desc}_eeg.fif")
        if not fif.exists():
            print(f"  [miss fif] {npz.name} -> {fif.name}"); n_miss += 1; continue
        try:
            raw = mne.io.read_raw_fif(str(fif), preload=True, verbose=False)
            new_block = block_mean(raw, a.samples_per_tr, n_vol)
            if new_block.shape != d["eeg_block"].shape:
                print(f"  [shape!] {npz.name} {new_block.shape} vs {d['eeg_block'].shape}"); n_skip += 1; continue
            d["eeg_block"] = new_block
            out = dst / npz.parent.name / npz.name
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, **d)
            n_ok += 1
        except Exception as e:
            print(f"  [err] {npz.name}: {e}"); n_skip += 1
    print(f"\nDONE swap: ok={n_ok} skip={n_skip} miss_fif={n_miss}")


if __name__ == "__main__":
    main()
