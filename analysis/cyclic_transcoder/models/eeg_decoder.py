"""
eeg_decoder.py  (G1)
--------------------
Maps EEG channel space → latent source space.
Operates spatially (across channels) at each timepoint — pure channel mixing.

Input:  (batch, 31, T)
Output: (batch, latent_dim, T)

Architecture: stacked Conv1d with kernel_size=3, acting on the channel dim.
No temporal information crosses between timepoints — the temporal
structure in the source space comes entirely from the data, not the model.
"""

import torch
import torch.nn as nn


class EEGDecoder(nn.Module):
    """
    G1 : EEG → latent source space.

    Spatial inverse model (electrode space → source space).
    Analogous to EEG source localisation forward/inverse models,
    but learned end-to-end.
    """

    def __init__(
        self,
        n_channels: int = 31,
        latent_dim: int = 64,
        n_layers: int = 4,
        n_features: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_ch = n_channels

        for i in range(n_layers):
            out_ch = latent_dim if i == n_layers - 1 else n_features
            layers.append(
                nn.Conv1d(
                    in_ch, out_ch,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # same padding
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm1d(out_ch))
            if i < n_layers - 1:
                layers.append(nn.ELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.Tanh())  # bounded output in final layer
            in_ch = out_ch

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, n_channels, T)
        returns : (batch, latent_dim, T)
        """
        return self.net(x)
