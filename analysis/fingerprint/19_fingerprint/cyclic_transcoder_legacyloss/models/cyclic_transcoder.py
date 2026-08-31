"""
cyclic_transcoder.py
--------------------
Assembles the four modules into the full cyclic transcoder,
computes all four consistency losses, and exposes the two
transcoding directions (EEG→fMRI, fMRI→EEG).

Loss structure (Liu et al. 2020, Section 2.1):
    loss1  EEG cycle:         R1(G1(E)) ≈ E        — EEG  reconstruction
    loss2  fMRI cycle:        R2(G2(F)) ≈ F        — fMRI reconstruction
    loss3  fMRI→EEG xcode:   R1(G2(F)) ≈ E        — fMRI decodes to EEG
    loss4  EEG→fMRI xcode:   R2(G1(E)) ≈ F        — EEG  decodes to fMRI
    loss5  PDA supervision:   PDA_pred  ≈ PDA_true  — direct CEN-DMN signal

The latent source spaces:
    S_eeg  = G1(E)   — source estimated from EEG
    S_fmri = G2(F)   — source estimated from fMRI
    (both should converge toward the same underlying neural state)
"""

import torch
import torch.nn as nn

from .eeg_decoder import EEGDecoder
from .eeg_encoder import EEGEncoder
from .fmri_decoder import FMRIDecoder
from .fmri_encoder import FMRIEncoder


class CyclicTranscoder(nn.Module):
    """
    Full cyclic convolutional transcoder.

    Attributes
    ----------
    eeg_decoder  : G1  — EEG  → latent source
    eeg_encoder  : R1  — latent source → EEG
    fmri_decoder : G2  — fMRI → latent source
    fmri_encoder : R2  — latent source → fMRI

    PDA indices into the fMRI feature vector:
        idx 64 : DMN personal mask mean
        idx 65 : CEN personal mask mean
        PDA = fmri[:, 65, :] - fmri[:, 64, :]
    """

    DMN_IDX = 64
    CEN_IDX = 65

    def __init__(
        self,
        n_eeg_channels: int = 31,
        n_fmri_features: int = 66,
        latent_dim: int = 64,
        eeg_n_layers: int = 4,
        eeg_n_features: int = 32,
        eeg_kernel_size: int = 3,
        fmri_n_layers: int = 6,
        fmri_n_features: int = 32,
        fmri_kernel_size: int = 27,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.eeg_decoder = EEGDecoder(
            n_channels=n_eeg_channels,
            latent_dim=latent_dim,
            n_layers=eeg_n_layers,
            n_features=eeg_n_features,
            kernel_size=eeg_kernel_size,
            dropout=dropout,
        )
        self.eeg_encoder = EEGEncoder(
            latent_dim=latent_dim,
            n_channels=n_eeg_channels,
            n_layers=eeg_n_layers,
            n_features=eeg_n_features,
            kernel_size=eeg_kernel_size,
            dropout=dropout,
        )
        self.fmri_decoder = FMRIDecoder(
            n_fmri_features=n_fmri_features,
            latent_dim=latent_dim,
            n_layers=fmri_n_layers,
            n_features=fmri_n_features,
            kernel_size=fmri_kernel_size,
            dropout=dropout,
        )
        self.fmri_encoder = FMRIEncoder(
            latent_dim=latent_dim,
            n_fmri_features=n_fmri_features,
            n_layers=fmri_n_layers,
            n_features=fmri_n_features,
            kernel_size=fmri_kernel_size,
            dropout=dropout,
        )

        self._mse = nn.MSELoss()

    # ------------------------------------------------------------------
    # Core transcoding directions
    # ------------------------------------------------------------------

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        """EEG → latent source.  eeg: (B, 31, T) → (B, latent_dim, T)"""
        return self.eeg_decoder(eeg)

    def encode_fmri(self, fmri: torch.Tensor) -> torch.Tensor:
        """fMRI → latent source.  fmri: (B, 66, T) → (B, latent_dim, T)"""
        return self.fmri_decoder(fmri)

    def eeg_to_fmri(self, eeg: torch.Tensor) -> torch.Tensor:
        """EEG → (latent) → fMRI.  Returns transcoded fMRI (B, 66, T)."""
        return self.fmri_encoder(self.eeg_decoder(eeg))

    def fmri_to_eeg(self, fmri: torch.Tensor) -> torch.Tensor:
        """fMRI → (latent) → EEG.  Returns transcoded EEG (B, 31, T)."""
        return self.eeg_encoder(self.fmri_decoder(fmri))

    def predict_pda(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        From raw EEG, predict PDA = CEN - DMN.
        eeg     : (B, 31, T)
        returns : (B, T)
        """
        fmri_hat = self.eeg_to_fmri(eeg)
        return fmri_hat[:, self.CEN_IDX, :] - fmri_hat[:, self.DMN_IDX, :]

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_losses(
        self,
        eeg: torch.Tensor,
        fmri: torch.Tensor,
        pda_true: torch.Tensor,
        weights: dict,
    ) -> dict:
        """
        Compute all five losses.

        Parameters
        ----------
        eeg      : (B, 31, T)
        fmri     : (B, 66, T)
        pda_true : (B, T)
        weights  : dict with keys eeg_cycle, fmri_cycle,
                   eeg_transcoder, fmri_transcoder

        Returns
        -------
        dict with individual losses and 'total'
        """
        # --- Latent source spaces ---
        s_eeg = self.eeg_decoder(eeg)    # G1(E)
        s_fmri = self.fmri_decoder(fmri) # G2(F)

        # --- Four cycle paths ---
        eeg_hat_cycle   = self.eeg_encoder(s_eeg)    # R1(G1(E))  — EEG cycle
        fmri_hat_cycle  = self.fmri_encoder(s_fmri)  # R2(G2(F))  — fMRI cycle
        eeg_hat_xcode   = self.eeg_encoder(s_fmri)   # R1(G2(F))  — fMRI→EEG transcode
        fmri_hat_xcode  = self.fmri_encoder(s_eeg)   # R2(G1(E))  — EEG→fMRI transcode

        # --- Individual losses ---
        loss_eeg_cycle  = self._mse(eeg_hat_cycle,  eeg)
        loss_fmri_cycle = self._mse(fmri_hat_cycle, fmri)
        loss_eeg_xcode  = self._mse(eeg_hat_xcode,  eeg)
        loss_fmri_xcode = self._mse(fmri_hat_xcode, fmri)

        # --- PDA supervision on transcoded fMRI (EEG→fMRI path) ---
        pda_pred = (
            fmri_hat_xcode[:, self.CEN_IDX, :]
            - fmri_hat_xcode[:, self.DMN_IDX, :]
        )
        loss_pda = self._mse(pda_pred, pda_true)

        # --- Weighted total ---
        # NOTE: PDA (CEN-DMN on the EEG->fMRI transcode) is the actual prediction
        # target. Earlier configs omitted it, so the model was never supervised on
        # it; weights.get keeps backward compat (missing key => 0 contribution).
        total = (
            weights["eeg_cycle"]       * loss_eeg_cycle
            + weights["fmri_cycle"]    * loss_fmri_cycle
            + weights["eeg_transcoder"]  * loss_eeg_xcode
            + weights["fmri_transcoder"] * loss_fmri_xcode
            + weights.get("pda", 0.0)    * loss_pda
        )

        return {
            "eeg_cycle":        loss_eeg_cycle,
            "fmri_cycle":       loss_fmri_cycle,
            "eeg_transcoder":   loss_eeg_xcode,
            "fmri_transcoder":  loss_fmri_xcode,
            "pda":              loss_pda,
            "total":            total,
        }

    # ------------------------------------------------------------------
    # Parameter count helper
    # ------------------------------------------------------------------

    def n_params(self) -> dict:
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "eeg_decoder":  count(self.eeg_decoder),
            "eeg_encoder":  count(self.eeg_encoder),
            "fmri_decoder": count(self.fmri_decoder),
            "fmri_encoder": count(self.fmri_encoder),
            "total":        count(self),
        }


# ------------------------------------------------------------------
# Factory from config dict
# ------------------------------------------------------------------

def build_model(cfg: dict) -> CyclicTranscoder:
    m = cfg["model"]
    return CyclicTranscoder(
        n_eeg_channels=m["n_eeg_channels"],
        n_fmri_features=m["n_fmri_features"],
        latent_dim=m["latent_dim"],
        eeg_n_layers=m["eeg_n_layers"],
        eeg_n_features=m["eeg_n_features"],
        eeg_kernel_size=m["eeg_kernel_size"],
        fmri_n_layers=m["fmri_n_layers"],
        fmri_n_features=m["fmri_n_features"],
        fmri_kernel_size=m["fmri_kernel_size"],
    )
