"""PhysNet-style rPPG model + losses + HR-from-waveform.

Input  : video clip (B, 3, T, H, W) in [0,1]
Output : rPPG waveform (B, T)   -- spatial dims pooled away, temporal length kept

Trained with negative-Pearson loss against the ground-truth PPG waveform
(SCAMPS d_ppg, resampled+z-normalized to T). HR = FFT peak of the predicted
waveform. Replaces the failed HR-bin classifier (see notes/POSTMORTEM_rppg_3dconv.md).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class ConvBlock3d(nn.Module):
    def __init__(self, cin, cout, k=(3, 3, 3), p=(1, 1, 1)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, k, stride=1, padding=p),
            nn.BatchNorm3d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PhysNet(nn.Module):
    """3D-CNN encoder that keeps the temporal axis and pools space away.

    All temporal kernels are stride 1 (no temporal downsampling) so the output
    waveform has the same length T as the input clip."""

    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.stem = ConvBlock3d(3, w, k=(1, 5, 5), p=(0, 2, 2))
        self.pool_s1 = nn.MaxPool3d((1, 2, 2))
        self.b1 = ConvBlock3d(w, 2 * w)
        self.b2 = ConvBlock3d(2 * w, 2 * w)
        self.pool_s2 = nn.MaxPool3d((1, 2, 2))
        self.b3 = ConvBlock3d(2 * w, 4 * w)
        self.b4 = ConvBlock3d(4 * w, 4 * w)
        self.pool_s3 = nn.MaxPool3d((1, 2, 2))
        self.b5 = ConvBlock3d(4 * w, 4 * w)
        self.b6 = ConvBlock3d(4 * w, 4 * w)
        self.spatial = nn.AdaptiveAvgPool3d((None, 1, 1))   # (B,C,T,1,1)
        self.head = nn.Conv3d(4 * w, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # (B,3,T,H,W)
        x = self.pool_s1(self.stem(x))
        x = self.pool_s2(self.b2(self.b1(x)))
        x = self.pool_s3(self.b4(self.b3(x)))
        x = self.b6(self.b5(x))
        x = self.spatial(x)                                 # (B,C,T,1,1)
        x = self.head(x)                                    # (B,1,T,1,1)
        return x[:, 0, :, 0, 0]                             # (B,T)


def neg_pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - Pearson r between predicted and GT waveform, averaged over the batch."""
    p = pred - pred.mean(dim=1, keepdim=True)
    t = target - target.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = torch.sqrt((p * p).sum(dim=1) * (t * t).sum(dim=1) + 1e-8)
    return (1.0 - num / den).mean()


def hr_from_wave(wave: np.ndarray, fs: float,
                 lo_hz: float = 0.7, hi_hz: float = 3.0) -> float:
    """HR (bpm) = peak of the power spectrum in the physiological band.
    fs = effective sampling rate of the waveform (clip_frames / clip_seconds)."""
    w = np.asarray(wave, dtype=np.float64)
    w = w - w.mean()
    n = len(w)
    if n < 4:
        return float("nan")
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    ps = np.abs(np.fft.rfft(w * np.hanning(n))) ** 2
    band = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not band.any():
        return float("nan")
    peak = freqs[band][int(np.argmax(ps[band]))]
    return float(peak * 60.0)
