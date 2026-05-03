"""
dataset.py
----------
PyTorch Dataset that loads pre-extracted .npz feature files and returns
sliding windows for training.

Each sample:
    eeg   : (31, window_trs)   float32
    fmri  : (66, window_trs)   float32  — DiFuMo64 + [DMN_mean, CEN_mean]
    pda   : (window_trs,)      float32  — CEN - DMN ground truth
    meta  : dict with subject, task, run, start_tr
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CyclicTranscoderDataset(Dataset):
    """
    Loads windowed EEG + fMRI feature pairs for one set of subjects/tasks.

    Args
    ----
    features_dir : str | Path
        Root of pre-extracted features (contains sub-*/…_features.npz)
    subjects : list[str]
        Subject IDs to include (e.g. ['dmnelf001', 'dmnelf002'])
    tasks : list[str]
        Task names to include (e.g. ['rest'])
    window_trs : int
        Window length in TRs
    stride_trs : int
        Sliding stride in TRs
    augment : bool
        If True, apply light time-shift augmentation during training
    """

    def __init__(
        self,
        features_dir,
        subjects,
        tasks,
        window_trs=120,
        stride_trs=10,
        augment=False,
    ):
        self.features_dir = Path(features_dir)
        self.window_trs = window_trs
        self.stride_trs = stride_trs
        self.augment = augment

        # Build index: list of (npz_path, start_tr)
        self.index = []
        self._load_npz_cache = {}

        for subj in subjects:
            subj_dir = self.features_dir / f"sub-{subj}"
            if not subj_dir.exists():
                continue
            for npz_path in sorted(subj_dir.glob("*.npz")):
                # Check task match
                task_match = any(f"_task-{t}_" in npz_path.name for t in tasks)
                if not task_match:
                    continue

                # Load to get length
                data = np.load(npz_path, allow_pickle=True)
                n_trs = data["eeg_block"].shape[0]

                # Build window start positions
                starts = list(range(0, n_trs - window_trs + 1, stride_trs))
                if not starts:
                    continue  # run too short

                for start in starts:
                    self.index.append((npz_path, start, str(data["subject"]), str(data["task"]), str(data["run"])))

                # Cache the loaded arrays
                self._load_npz_cache[str(npz_path)] = {
                    "eeg": data["eeg_block"].astype(np.float32),   # (T, 31)
                    "fmri": data["fmri_features"].astype(np.float32),  # (T, 66)
                    "pda": data["pda"].astype(np.float32),          # (T,)
                }

        if len(self.index) == 0:
            raise ValueError(
                f"No windows found in {features_dir} for subjects={subjects}, tasks={tasks}"
            )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        npz_path, start, subject, task, run = self.index[idx]
        cache = self._load_npz_cache[str(npz_path)]

        end = start + self.window_trs

        eeg = cache["eeg"][start:end]   # (window_trs, 31)
        fmri = cache["fmri"][start:end] # (window_trs, 66)
        pda = cache["pda"][start:end]   # (window_trs,)

        # Light augmentation: random time-shift up to ±5 TRs
        if self.augment:
            shift = np.random.randint(-5, 6)
            if shift != 0:
                eeg = np.roll(eeg, shift, axis=0)
                fmri = np.roll(fmri, shift, axis=0)
                pda = np.roll(pda, shift, axis=0)

        # Transpose to (features, time) for Conv1d
        eeg_t = torch.from_numpy(eeg.T.copy())   # (31, window_trs)
        fmri_t = torch.from_numpy(fmri.T.copy()) # (66, window_trs)
        pda_t = torch.from_numpy(pda.copy())      # (window_trs,)

        return {
            "eeg": eeg_t,
            "fmri": fmri_t,
            "pda": pda_t,
            "subject": subject,
            "task": task,
        }


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def make_loocv_loaders(cfg, left_out_subject, batch_size=8, num_workers=4):
    """
    Return (train_loader, val_loader) for leave-one-subject-out split.

    train : all subjects except left_out_subject, using train tasks
    val   : left_out_subject only, using validate tasks
    """
    excluded = set(cfg["data"]["subjects"].get("exclude", []))
    all_subjects = [s for s in cfg["data"]["subjects"]["all"] if s not in excluded]

    train_subjects = [s for s in all_subjects if s != left_out_subject]
    val_subjects = [left_out_subject]

    features_dir = cfg["data"]["features_dir"]
    window_trs = cfg["data"]["windowing"]["window_trs"]
    stride_trs = cfg["data"]["windowing"]["stride_trs"]
    train_tasks = cfg["data"]["tasks"]["train"]
    val_tasks = cfg["data"]["tasks"]["validate"] or cfg["data"]["tasks"]["train"]

    train_ds = CyclicTranscoderDataset(
        features_dir=features_dir,
        subjects=train_subjects,
        tasks=train_tasks,
        window_trs=window_trs,
        stride_trs=stride_trs,
        augment=True,
    )
    val_ds = CyclicTranscoderDataset(
        features_dir=features_dir,
        subjects=val_subjects,
        tasks=val_tasks,
        window_trs=window_trs,
        stride_trs=stride_trs,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def make_predict_loader(cfg, subject, task="feedback", batch_size=8, num_workers=2):
    """Return a loader for inference on a single subject/task (no windowing stride needed)."""
    features_dir = cfg["data"]["features_dir"]
    window_trs = cfg["data"]["windowing"]["window_trs"]

    ds = CyclicTranscoderDataset(
        features_dir=features_dir,
        subjects=[subject],
        tasks=[task],
        window_trs=window_trs,
        stride_trs=window_trs,   # non-overlapping for clean reconstruction
        augment=False,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
