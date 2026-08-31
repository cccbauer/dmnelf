"""
predict_pda.py
--------------
Apply the trained cyclic transcoder to feedback runs:
    EEG (feedback) → transcoded fMRI → predicted PDA (CEN - DMN)

Also saves the full transcoded fMRI parcel timeseries so you can
compare against MURFI real-time output.

Usage:
    python predict_pda.py --subject dmnelf001 --config config.yaml
    python predict_pda.py --all --config config.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import make_predict_loader
from models import build_model, CyclicTranscoder


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Machine-aware features_dir: cluster + compute nodes have /projects/swglab,
    # the Mac does not. Lets one committed config work on both machines.
    d = cfg.get("data", {})
    if "features_dir_cluster" in d and "features_dir_local" in d:
        d["features_dir"] = (d["features_dir_cluster"]
                             if Path("/projects/swglab").exists()
                             else d["features_dir_local"])
    return cfg


def load_best_model(cfg, left_out_subject, device):
    """Load best checkpoint for the given left-out subject."""
    ckpt_root = Path(cfg["project"]["base_dir"]) / cfg["training"]["checkpoint_dir"]
    ckpt_path = ckpt_root / ("loocv_" + left_out_subject + "_best.pt")

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}\n"
            "Run train.py first."
        )

    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded checkpoint (val_loss={ckpt['best_val_loss']:.5f}): {ckpt_path.name}")
    return model


def predict_subject(subject, cfg, task="feedback"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # In LOOCV, the model trained WITH all other subjects is used to predict
    # for the left-out subject on unseen feedback data
    model = load_best_model(cfg, left_out_subject=subject, device=device)

    out_dir = (
        Path(cfg["data"]["features_dir"])
        / f"sub-{subject}"
        / "predictions"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = make_predict_loader(cfg, subject=subject, task=task)
    window_trs = cfg["data"]["windowing"]["window_trs"]

    all_pda = []
    all_pda_true = []
    all_fmri_hat = []
    all_fmri_true = []

    with torch.no_grad():
        for batch in loader:
            eeg  = batch["eeg"].to(device)   # (B, 31, T)
            fmri = batch["fmri"].to(device)  # (B, 66, T)
            pda  = batch["pda"]              # (B, T)  ground-truth CEN-DMN

            fmri_hat = model.eeg_to_fmri(eeg)  # (B, 66, T)
            pda_pred = (
                fmri_hat[:, CyclicTranscoder.CEN_IDX, :]
                - fmri_hat[:, CyclicTranscoder.DMN_IDX, :]
            )  # (B, T)

            all_pda.append(pda_pred.cpu().numpy())
            all_pda_true.append(pda.numpy())
            all_fmri_hat.append(fmri_hat.cpu().numpy())
            all_fmri_true.append(fmri.cpu().numpy())

    # Flatten windows to a single timeseries in temporal order. Windows are
    # non-overlapping (stride=window_trs) and the loader is shuffle=False, so
    # sample-major row-major flatten == temporal order:
    #   (B, T) -> [s0t0..s0tN, s1t0..s1tN, ...]
    def flatten_time(chunks):                       # list of (B, T) -> (total_T,)
        return np.concatenate([w.reshape(-1) for w in chunks])

    def flatten_parcels(chunks):                    # list of (B, P, T) -> (P, total_T)
        # transpose to (P, B, T) then reshape (P, B*T): keeps sample-major time,
        # so column ordering matches flatten_time above.
        return np.concatenate(
            [w.transpose(1, 0, 2).reshape(w.shape[1], -1) for w in chunks], axis=1
        )

    pda_arr       = flatten_time(all_pda)          # (total_T,)  predicted CEN-DMN
    pda_true_arr  = flatten_time(all_pda_true)     # (total_T,)  true CEN-DMN
    fmri_hat_arr  = flatten_parcels(all_fmri_hat)  # (66, total_T)
    fmri_true_arr = flatten_parcels(all_fmri_true) # (66, total_T)

    # Pearson correlation between predicted and true PDA (honest, full length)
    if len(pda_arr) > 1 and len(pda_true_arr) == len(pda_arr):
        r = np.corrcoef(pda_arr, pda_true_arr)[0, 1]
        print(f"  PDA correlation (pred vs true): r = {r:.4f}")

    out_path = out_dir / f"sub-{subject}_task-{task}_pda_prediction.npz"
    np.savez(
        out_path,
        pda_predicted=pda_arr,        # (T,)
        pda_true=pda_true_arr,        # (T,)  ground-truth CEN-DMN, aligned with pred
        fmri_predicted=fmri_hat_arr,  # (66, T)
        fmri_true=fmri_true_arr,      # (66, T)
        subject=subject,
        task=task,
        difumo_dim=64,
        dmn_idx=CyclicTranscoder.DMN_IDX,
        cen_idx=CyclicTranscoder.CEN_IDX,
    )
    print(f"  Saved: {out_path.name}  (T={pda_arr.shape[0]})")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--task", type=str, default="feedback")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    excluded = set(cfg["data"]["subjects"].get("exclude", []))
    all_subjects = [s for s in cfg["data"]["subjects"]["all"] if s not in excluded]

    if args.all:
        for subj in all_subjects:
            print(f"\n=== {subj} ===")
            try:
                predict_subject(subj, cfg, task=args.task)
            except FileNotFoundError as e:
                print(f"  [skip] {e}")
    elif args.subject:
        predict_subject(args.subject, cfg, task=args.task)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
