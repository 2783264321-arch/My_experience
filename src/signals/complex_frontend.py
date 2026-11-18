"""Complex STFT front-end helpers for the complex coarse/fine pilot.

The operator is deliberately small and explicit:
- native 0.7 ms complex STFT at the existing 512 frame centers;
- frame-center phase correction (no window-start phase ramp);
- 50%-overlap linear-Hann complex/power bands for the full-band coarse view.

Power is always derived from the same complex coefficients. Coarse power is the
weighted average PSD over a band; coarse Re/Im is the weighted average complex
coefficient. No log/complex averaging is mixed.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .data import N_TIME, WINDOW_MS


def hann_window(n: int) -> torch.Tensor:
    return torch.hann_window(n, periodic=True, dtype=torch.float32)


def complex_stft(x: torch.Tensor, fs: float, win_len: int, hop: int,
                 n_ext: int) -> torch.Tensor:
    """Right-pad, center, bilateral complex STFT, fftshift, PSD scale.

    Returns complex64 [win_len, 512]. The PSD scale is
    Z / sqrt(Fs * sum(w^2)), so |Z|^2 matches the existing power path.
    The frame-center phase term is removed after fftshift.
    """
    if x.numel() < n_ext:
        x = F.pad(x, (0, n_ext - x.numel()))
    elif x.numel() > n_ext:
        x = x[:n_ext]
    win = hann_window(win_len)
    z = torch.stft(
        x, n_fft=win_len, hop_length=hop, win_length=win_len, window=win,
        center=True, pad_mode="constant", normalized=False,
        onesided=False, return_complex=True,
    )
    z = torch.fft.fftshift(z, dim=0)
    f_hz = (torch.arange(win_len, dtype=torch.float64) - win_len / 2.0) * fs / win_len
    z = z * torch.exp(1j * 2.0 * math.pi * f_hz[:, None] * (win_len / 2.0) / fs)
    scale = math.sqrt(fs * float(win.square().sum()))
    return (z / scale).to(torch.complex64)


def overlap_bands(z_fine: torch.Tensor, n_out: int, fs: float,
                  nfft: int) -> tuple[torch.Tensor, torch.Tensor]:
    """50%-overlap linear-Hann complex/power bands, weighted average.

    Band center (b+0.5)*Nband; support width 2*Nband. We divide by the weight
    sum so the coarse power is an average PSD and the complex value is an
    average complex coefficient. This matches the existing coarse power's
    frequency-average convention and makes the complex and power channels share
    the same weights.
    """
    nfreq, ncols = z_fine.shape
    nband = nfreq / float(n_out)
    z_out = torch.zeros((n_out, ncols), dtype=torch.complex64)
    p_out = torch.zeros((n_out, ncols), dtype=torch.float32)
    for b in range(n_out):
        center = (b + 0.5) * nband
        r0 = max(0, int(math.floor(center - nband)))
        r1 = min(nfreq, int(math.ceil(center + nband)))
        if r1 <= r0:
            continue
        u = (torch.arange(r0, r1, dtype=torch.float64) + 0.5 -
             (center - nband)) / (2.0 * nband)
        w = 0.5 - 0.5 * torch.cos(2.0 * math.pi * u)
        w_sum = float(w.sum().clamp_min(1e-12))
        z_out[b] = (z_fine[r0:r1] * w[:, None]).sum(dim=0).to(torch.complex64) / w_sum
        p_out[b] = (z_fine[r0:r1].abs().square() * w[:, None]).sum(dim=0) / w_sum
    return z_out, p_out


def normalize_log_power(power: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    d = 10.0 * torch.log10(torch.clamp(power, min=torch.finfo(torch.float32).tiny))
    return ((d - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0)


def normalize_complex(z: torch.Tensor, scale: float) -> torch.Tensor:
    """Return [2,H,W] float32, Re/Im jointly scaled and clipped."""
    x = torch.view_as_real(z).permute(2, 0, 1) / max(float(scale), 1e-12)
    return x.clamp(-1.0, 1.0)


def fine_window_len(fs: float) -> int:
    L = int(round(fs * WINDOW_MS / 1000.0))
    return L + (L % 2)
