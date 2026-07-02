#!/usr/bin/env python3
"""Diagnose HMM state collapse: soft alpha, hard occupancy, covariance spread."""
import sys
from pathlib import Path
import numpy as np

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/projects/swglab/data/DMNELF/analysis/fingerprint/dmn_hmm_detection/results/group_k12")

d = np.load(RES / "state_probabilities.npz", allow_pickle=True)
alpha = d["alpha"]
K = alpha[0].shape[1]

soft = np.zeros(K); occ = np.zeros(K); mx = np.zeros(K); n = 0
# also count how many timepoints have max-alpha > 0.5 (confident assignment)
conf = 0
for a in alpha:
    a = np.asarray(a, dtype=np.float64)
    soft += a.sum(0); n += a.shape[0]
    am = a.argmax(1); occ += np.bincount(am, minlength=K)
    mx = np.maximum(mx, a.max(0))
    conf += int((a.max(1) > 0.5).sum())

print("n_timepoints:", n)
print("mean soft alpha per state:", np.round(soft / n, 4))
print("max  soft alpha per state:", np.round(mx, 4))
print("hard occupancy per state :", np.round(occ / n, 4))
print(f"fraction of tp with max-alpha>0.5: {conf/n:.3f}")

# Covariance spread: are state covariances distinct or collapsed?
try:
    from osl_dynamics.models import load
    model = load(str(RES / "trained_model"))
    covs = model.get_covariances()  # (K, C, C)
    print("\ncovariance shape:", covs.shape)
    # pairwise Frobenius distance between state covariances
    K2 = covs.shape[0]
    diffs = []
    for i in range(K2):
        for j in range(i + 1, K2):
            diffs.append(np.linalg.norm(covs[i] - covs[j]))
    diffs = np.array(diffs)
    print("pairwise cov Frobenius dist: min=%.3f mean=%.3f max=%.3f" %
          (diffs.min(), diffs.mean(), diffs.max()))
    traces = np.array([np.trace(covs[k]) for k in range(K2)])
    print("cov traces per state:", np.round(traces, 2))
    # transition probability matrix stickiness
    trans = model.get_trans_prob()
    print("trans-prob diagonal (stickiness):", np.round(np.diag(trans), 3))
except Exception as e:
    print("\n[cov/trans inspection skipped]:", repr(e))
