"""
fmri_encoder.py  (R2)
---------------------
Maps latent source space → fMRI parcel space.
Learns the HEMODYNAMIC RESPONSE FUNCTION (HRF) as a data-driven
temporal convolution — no canonical HRF assumed.

Input:  (batch, latent_dim, T)
Output: (batch, n_fmri_features, T)  — 66 parcels at TR resolution

Architecture: stacked CAUSAL temporal Conv1d (past → future only),
because the HRF is a strictly causal process: neural activity at time t
causes BOLD response at t + lag, never before.

To visualise the learned HRF: feed a unit impulse to this module
and read off the output (analogous to Appendix D in Liu et al. 2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution: output at t depends only on input at t' ≤ t.
    Achieved by left-padding with (kernel_size - 1) zeros, then trimming.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        # Left-pad only so future timesteps cannot see future inputs
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class FMRIEncoder(nn.Module):
    """
    R2 : latent source space → fMRI parcel time series.

    Temporal forward model (HRF convolution).
    All convolutions are causal — past drives future, never the reverse.

    The learned impulse response of this module is the data-driven HRF.
    Extract it with FMRIEncoder.get_hrf(device).
    """

    def __init__(
        self,
        latent_dim: int = 64,
        n_fmri_features: int = 66,
        n_layers: int = 6,
        n_features: int = 32,
        kernel_size: int = 27,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim

        layers = []
        in_ch = latent_dim

        for i in range(n_layers):
            out_ch = n_fmri_features if i == n_layers - 1 else n_features
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size=kernel_size))
            if i < n_layers - 1:
                layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.ELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            # No activation on final layer — BOLD is z-scored, unbounded
            in_ch = out_ch

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x       : (batch, latent_dim, T)
        returns : (batch, n_fmri_features, T)
        """
        return self.net(x)

    @torch.no_grad()
    def get_hrf(self, n_timepoints: int = 50, device: str = "cpu") -> torch.Tensor:
        """
        Visualise the learned HRF by feeding a unit impulse.
        Returns (n_fmri_features, n_timepoints) tensor.

        Usage:
            hrf = model.fmri_encoder.get_hrf(n_timepoints=50)
            import matplotlib.pyplot as plt
            plt.plot(hrf.mean(0).numpy())   # mean across parcels
        """
        self.eval()
        impulse = torch.zeros(1, self.latent_dim, n_timepoints, device=device)
        impulse[:, :, 0] = 1.0  # unit impulse at t=0
        out = self.forward(impulse)  # (1, n_fmri_features, n_timepoints)
        return out.squeeze(0).cpu()
