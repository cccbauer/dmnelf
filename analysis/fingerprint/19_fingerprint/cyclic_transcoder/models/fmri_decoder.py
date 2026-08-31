"""
fmri_decoder.py  (G2)
---------------------
Maps fMRI parcel space → latent source space.
Learns to DECONVOLVE the hemodynamic response — converting slow BOLD
signal back into a faster neural-timescale latent representation.

Input:  (batch, n_fmri_features, T)   — 66 parcels at TR resolution
Output: (batch, latent_dim, T)

Architecture: stacked temporal Conv1d with large kernel (27 TRs ≈ 32 s)
to capture the full HRF shape. Non-causal (symmetric padding) because
we have the full run available during training.

Physical interpretation: fMRI decoder ≈ inverse of HRF convolution.
The fMRI encoder (R2) learns the forward HRF; this module undoes it.
"""

import torch
import torch.nn as nn


class FMRIDecoder(nn.Module):
    """
    G2 : fMRI parcel time series → latent source space.

    Temporal deconvolution. Each Conv1d operates along the time axis,
    mixing information across a ~32 s window to invert the HRF.
    """

    def __init__(
        self,
        n_fmri_features: int = 66,
        latent_dim: int = 64,
        n_layers: int = 6,
        n_features: int = 32,
        kernel_size: int = 27,   # ~32 s at TR=1.2 s
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = kernel_size // 2  # same-length output (non-causal)

        layers = []
        in_ch = n_fmri_features

        for i in range(n_layers):
            out_ch = latent_dim if i == n_layers - 1 else n_features
            layers.append(
                nn.Conv1d(
                    in_ch, out_ch,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm1d(out_ch))
            if i < n_layers - 1:
                layers.append(nn.ELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.Tanh())
            in_ch = out_ch

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x       : (batch, n_fmri_features, T)
        returns : (batch, latent_dim, T)
        """
        return self.net(x)
