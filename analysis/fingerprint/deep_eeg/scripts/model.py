#!/usr/bin/env python3
"""R-EEGNet (Stabile 2025): EEGNet Blocks 1-2 + linear regression head. Torch."""
import torch, torch.nn as nn


class REEGNet(nn.Module):
    def __init__(self, n_ch=31, n_samp=1200, n_out=2, F1=8, D=2, F2=16,
                 kern=40, p=0.5):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),   # temporal (freq filters)
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),           # depthwise spatial
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(p))
        self.b2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),  # separable: depthwise
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),                          # pointwise
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(p))
        with torch.no_grad():
            flat = self.b2(self.b1(torch.zeros(1, 1, n_ch, n_samp))).flatten(1).shape[1]
        self.head = nn.Linear(flat, n_out, bias=False)                          # linear regression head

    def forward(self, x):           # x: [B, n_ch, n_samp]
        return self.head(self.b2(self.b1(x.unsqueeze(1))).flatten(1))


if __name__ == "__main__":
    m = REEGNet()
    print("params:", sum(p.numel() for p in m.parameters()))
    print("out:", m(torch.zeros(4, 31, 1200)).shape)
