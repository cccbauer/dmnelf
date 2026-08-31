"""
eeg_encoder.py  (R1)
--------------------
Maps latent source space → EEG channel space.
Implements the EEG forward model (source → electrode).

Input:  (batch, latent_dim, T)
Output: (batch, n_channels, T)

Mirror of EEGDecoder (G1). Pure spatial / channel mixing, no temporal mixing.
"""

import torch
import torch.nn as nn


class EEGEncoder(nn.Module):
    """
    R1 : latent source space → EEG electrode space.

    Spatial forward model. Learned channel mixing from source dims to
    electrode dims — analogous to the EEG lead-field matrix.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        n_channels: int = 31,
        n_layers: int = 4,
        n_features: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_ch = latent_dim

        for i in range(n_layers):
            out_ch = n_channels if i == n_layers - 1 else n_features
            layers.append(
                nn.Conv1d(
                    in_ch, out_ch,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                )
            )
            if i < n_layers - 1:
                layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.ELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            # No activation on final layer — EEG is z-scored, unbounded
            in_ch = out_ch

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x       : (batch, latent_dim, T)
        returns : (batch, n_channels, T)
        """
        return self.net(x)
