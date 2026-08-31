"""
train.py
--------
Leave-one-subject-out training loop for the cyclic transcoder.

Usage:
    python train.py --left-out dmnelf001 --config config.yaml
    python train.py --left-out dmnelf001 --config config.yaml --resume
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml

# Make sure local modules are on path when called from SLURM
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import make_loocv_loaders
from models import build_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[CPU] No GPU found — training will be slow")
    return dev


def save_checkpoint(state, path):
    torch.save(state, path)


def load_checkpoint(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["best_val_loss"]


def set_seed(seed):
    """Seed every RNG the training loop actually draws from: Python's random
    (unused directly here, but dataset.py's worker_init_fn needs it seeded
    consistently), NumPy (dataset.py's __getitem__ time-shift augmentation),
    and PyTorch (model weight init at build_model() time, plus the DataLoader
    shuffle sampler and Dropout layers, all of which draw from torch's global
    RNG). CPU-only today (see get_device()) so no CUDA/cudnn seeding needed,
    but harmless to set for whenever this moves to GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, weights, device, train=True):
    model.train(train)
    totals = {k: 0.0 for k in ["eeg_cycle", "fmri_cycle", "eeg_transcoder", "fmri_transcoder", "pda", "total"]}
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            eeg  = batch["eeg"].to(device)    # (B, 31, T)
            fmri = batch["fmri"].to(device)   # (B, 66, T)
            pda  = batch["pda"].to(device)    # (B, T)

            losses = model.compute_losses(eeg, fmri, pda, weights)

            if train:
                optimizer.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            for k in totals:
                totals[k] += losses[k].item()
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train(left_out_subject, cfg, resume=False, seed=None):
    if seed is not None:
        set_seed(seed)
        print(f"  seed = {seed}")
    device = get_device()
    t_cfg = cfg["training"]
    ckpt_root = Path(cfg["project"]["base_dir"]) / t_cfg["checkpoint_dir"]
    ckpt_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_root / ("loocv_" + left_out_subject + "_best.pt")
    last_path  = ckpt_root / ("loocv_" + left_out_subject + "_last.pt")

    # --- Data ---
    print(f"\n[LOOCV] left-out = {left_out_subject}")
    train_loader, val_loader = make_loocv_loaders(
        cfg,
        left_out_subject=left_out_subject,
        batch_size=t_cfg["batch_size"],
        num_workers=cfg["slurm"]["cpus_per_task"] // 2,
        seed=seed,
    )
    print(f"  train windows: {len(train_loader.dataset)}")
    print(f"  val   windows: {len(val_loader.dataset)}")

    # --- Model ---
    model = build_model(cfg).to(device)
    params = model.n_params()
    print(f"  model params: {params['total']:,}  "
          f"(eeg_dec={params['eeg_decoder']:,}  fmri_dec={params['fmri_decoder']:,})")

    optimizer = optim.Adam(
        model.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )

    weights = t_cfg["loss_weights"]
    n_epochs = t_cfg["n_epochs"]
    val_interval = t_cfg.get("val_interval", 5)
    patience = t_cfg["early_stopping_patience"]
    # Which val loss drives checkpointing / early stop / LR plateau.
    # "pda" = the prediction target; "total" = legacy reconstruction objective.
    monitor = t_cfg.get("monitor_metric", "total")
    print(f"  monitoring val '{monitor}' for checkpoint selection")

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if resume and last_path.exists():
        start_epoch, best_val_loss = load_checkpoint(last_path, model, optimizer)
        print(f"  resumed from epoch {start_epoch}, best_val={best_val_loss:.5f}")

    # --- Loop ---
    log_path = ckpt_root / ("loocv_" + left_out_subject + "_log.tsv")
    with open(log_path, "a") as log_f:
        if start_epoch == 0:
            log_f.write("epoch\ttrain_total\tval_total\tval_pda\tlr\n")

        for epoch in range(start_epoch, n_epochs):
            t0 = time.time()
            train_losses = run_epoch(model, train_loader, optimizer, weights, device, train=True)

            # Validate every val_interval epochs
            if (epoch + 1) % val_interval == 0 or epoch == n_epochs - 1:
                val_losses = run_epoch(model, val_loader, optimizer, weights, device, train=False)
                scheduler.step(val_losses[monitor])

                improved = val_losses[monitor] < best_val_loss
                if improved:
                    best_val_loss = val_losses[monitor]
                    patience_counter = 0
                    save_checkpoint(
                        {"epoch": epoch + 1, "model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "best_val_loss": best_val_loss},
                        ckpt_path,
                    )
                    marker = " ✓"
                else:
                    patience_counter += val_interval
                    marker = ""

                lr_now = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                print(
                    f"  ep {epoch+1:4d}/{n_epochs}"
                    f"  train={train_losses['total']:.4f}"
                    f"  val={val_losses['total']:.4f}"
                    f"  pda={val_losses['pda']:.4f}"
                    f"  lr={lr_now:.2e}"
                    f"  {elapsed:.1f}s"
                    f"{marker}"
                )
                log_f.write(
                    f"{epoch+1}\t{train_losses['total']:.6f}\t"
                    f"{val_losses['total']:.6f}\t{val_losses['pda']:.6f}\t"
                    f"{lr_now:.2e}\n"
                )
                log_f.flush()

                # Save last checkpoint for resuming
                save_checkpoint(
                    {"epoch": epoch + 1, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "best_val_loss": best_val_loss},
                    last_path,
                )

                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            else:
                # Print training loss only
                if (epoch + 1) % 10 == 0:
                    print(
                        f"  ep {epoch+1:4d}/{n_epochs}"
                        f"  train={train_losses['total']:.4f}"
                    )

    print(f"\n  Best val loss: {best_val_loss:.5f}")
    print(f"  Checkpoint: {ckpt_path}")
    return ckpt_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-out", required=True,
                        help="Subject ID to hold out, e.g. dmnelf001")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (weight init, DataLoader "
                             "shuffling, augmentation). Omit for the old unseeded behavior.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(args.left_out, cfg, resume=args.resume, seed=args.seed)


if __name__ == "__main__":
    main()