"""Fast on-the-fly STFT power using scipy.fft on strided frames.

Semantics match src/signals/data.py::stft_power (torch.stft, center=True,
pad_mode='constant', periodic Hann, unnormalized FFT, fftshift, |X|^2):

    Fs = B MHz, L = round(Fs * 0.7 ms), h = ceil(N / 511),
    N_ext = 511 h, input right-padded with zeros, centered with L//2 zeros
    on both sides, exactly 512 columns.

The result is kept time-major [512, F] so fine frequency slices and coarse
aggregation can be produced without transposing the full array.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.fft as sfft
import torch

from .data import N_TIME, WINDOW_MS, frequency_starts


def _hann_periodic(L: int) -> np.ndarray:
    # 0.5 - 0.5 cos(2 pi n / L), matching torch.hann_window(..., periodic=True)
    n = np.arange(L, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / L)).astype(np.float32)


def stft_power_fast(
    iq: torch.Tensor,
    B_mhz: float,
    fft_threads: int = 1,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Return (power [512,F] float32 time-major, window, h, L, F).

    power is already fftshifted along the frequency (last) axis.
    """
    n = int(iq.numel())
    fs = B_mhz * 1e6
    L = int(round(fs * WINDOW_MS / 1000.0))
    if L % 2:
        L += 1
    h = math.ceil(n / (N_TIME - 1))
    n_ext = (N_TIME - 1) * h
    if n_ext < n:
        raise ValueError(f"n_ext={n_ext} < n={n}")
    pad = n_ext - n
    half = L // 2
    total = n + pad + 2 * half
    x = np.zeros(total, dtype=np.complex64)
    x[half:half + n] = iq.numpy()
    win = _hann_periodic(L)
    frames = np.lib.stride_tricks.sliding_window_view(x, L)[::h]
    if frames.shape[0] != N_TIME:
        raise RuntimeError(f"expected {N_TIME} frames, got {frames.shape[0]}")
    fr = frames * win  # contiguous complex64 [512,L]
    z = sfft.fft(fr, axis=-1, overwrite_x=True, workers=fft_threads)
    F = z.shape[-1]
    p = np.empty(z.shape, dtype=np.float32)
    np.multiply(z.real, z.real, out=p)
    # z.imag * z.imag allocates a temporary; acceptable and still faster than
    # torch's complex abs path here.
    p += z.imag * z.imag
    p *= np.float32(1.0 / (fs * float(np.sum(win.astype(np.float64) ** 2))))
    # fftshift along frequency
    half_f = F // 2
    p = np.concatenate([p[:, half_f:], p[:, :half_f]], axis=1)
    return p, win, h, L, F


def normalize_slice(
    power: np.ndarray,
    lo: float,
    hi: float,
) -> torch.Tensor:
    """log10-power normalization for a [512,H] float32 slice -> [H,512] torch."""
    denom = max(hi - lo, 1e-6)
    out = np.empty_like(power, dtype=np.float32)
    np.log10(np.maximum(power, np.float32(np.finfo(np.float32).tiny)), out=out)
    out *= np.float32(10.0)
    out -= np.float32(lo)
    out /= np.float32(denom)
    np.clip(out, 0.0, 1.0, out=out)
    return torch.from_numpy(np.ascontiguousarray(out.T))


def aggregate_frequency_fast(
    power: np.ndarray,
    out_h: int = 2048,
) -> np.ndarray:
    """Area-weighted frequency mean for [512,F] -> [512,out_h]."""
    f = power.shape[1]
    if f == out_h:
        return power.copy()
    if f < out_h:
        # not used by the current corpus (smallest F is 3500) but keep a guard
        idx = np.linspace(0, f - 1, out_h)
        lo = np.floor(idx).astype(np.int64)
        hi = np.minimum(lo + 1, f - 1)
        w = (idx - lo).astype(np.float32)
        return power[:, lo] * (1 - w) + power[:, hi] * w
    bounds = np.linspace(0, f, out_h + 1).round().astype(np.int64)
    sums = np.add.reduceat(power, bounds[:-1], axis=1)
    counts = (bounds[1:] - bounds[:-1]).astype(np.float32)
    sums /= counts
    return sums
