"""Three C3 coarse front-ends for the Baseline-0 complex-input comparison.

All three keep the original C1 power channel (the 0.7 ms area-weighted
full-band power image). Only the two extra channels differ:

  overlap : 0.7 ms complex STFT -> 2048 50%-overlap linear-Hann complex bands
  sparse  : 2048-point complex STFT at the original 512 frame centres,
            half-bin shifted so row k is centred on (k+0.5-1024)*Fs/2048
  dense   : 2048-point STFT at ~10 us hop, adjacent-frame normalized complex
            correlation, expected row-centre phase removed, averaged into the
            original 512 columns by frame-pair midpoint

The half-bin shift is essential: the original coarse power rows are centred at
(k+0.5)*Fs/2048 after frequency aggregation, while an unshifted 2048-point FFT
has bins at integer multiples of Fs/2048. A -0.5-bin time modulation moves the
short-window bins onto the coarse row centres.

Frame-centre convention follows src/signals/complex_frontend.py:
  Z_ref[k,j] = Z[k,j] * exp(+j 2 pi f_k (L/2) / Fs)
so the phase reference is the frame centre, not the window start. For adjacent
frames the expected residual rotation is removed with exp(-j 2 pi f_k hop / Fs)
(the sign is verified on synthetic tones in scripts/validate_c3_frontends.py).
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .complex_frontend import complex_stft, fine_window_len, overlap_bands
from .data import N_TIME, WINDOW_MS

COARSE_NFFT = 2048
DENSE_HOP_US = 10.0


def _hann(n: int) -> torch.Tensor:
    return torch.hann_window(n, periodic=True, dtype=torch.float32)


def halfbin_short_stft(
    x: torch.Tensor,
    fs: float,
    hop: int,
    n_ext: int,
    nfft: int = COARSE_NFFT,
    half_bin: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilateral, fftshifted 2048-point STFT on the original 512 grid.

    Returns (z [nfft, frames] complex64 PSD-scaled, f_hz [nfft]).
    The frames are centred at j*hop; the input is right-padded to n_ext and
    centre-padded by nfft//2, matching torch.stft(center=True).
    """
    if x.numel() < n_ext:
        x = F.pad(x, (0, n_ext - x.numel()))
    elif x.numel() > n_ext:
        x = x[:n_ext]
    x = F.pad(x, (nfft // 2, nfft // 2))
    frames = x.unfold(0, nfft, hop)  # [nframes, nfft]
    win = _hann(nfft)
    n = torch.arange(nfft, dtype=torch.float64)
    mod = torch.exp(-1j * math.pi * n / nfft) if half_bin else torch.ones(nfft, dtype=torch.complex128)
    seg = frames.to(torch.complex64) * win.to(torch.complex64)[None, :] * mod.to(torch.complex64)[None, :]
    z = torch.fft.fftshift(torch.fft.fft(seg, dim=1), dim=1).T  # [nfft, nframes]
    if half_bin:
        f_hz = (torch.arange(nfft, dtype=torch.float64) - nfft / 2.0 + 0.5) * fs / nfft
    else:
        f_hz = (torch.arange(nfft, dtype=torch.float64) - nfft / 2.0) * fs / nfft
    # Remove the window-start phase, leaving the frame-centre phase.
    z = z * torch.exp(1j * 2.0 * math.pi * f_hz[:, None] * (nfft / 2.0) / fs)
    scale = math.sqrt(fs * float(win.square().sum()))
    return (z / scale).to(torch.complex64), f_hz


def overlap_complex(record, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Complex 50%-overlap bands from the 0.7 ms STFT; no power channel here."""
    fs = record.B_mhz * 1e6
    h = math.ceil(record.N / (N_TIME - 1))
    n_ext = (N_TIME - 1) * h
    z = complex_stft(iq, fs, fine_window_len(fs), h, n_ext)
    fbins = z.shape[0]
    zc, _ = overlap_bands(z, COARSE_NFFT, fs, fbins)
    return zc, z


def sparse_complex(record, iq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """2048-point complex STFT at the original 512 frame centres."""
    fs = record.B_mhz * 1e6
    h = math.ceil(record.N / (N_TIME - 1))
    n_ext = (N_TIME - 1) * h
    z, f_hz = halfbin_short_stft(iq, fs, h, n_ext, COARSE_NFFT, half_bin=True)
    return z, f_hz


def dense_hop_samples(fs: float, hop_us: float = DENSE_HOP_US) -> int:
    return max(1, int(round(fs * hop_us / 1e6)))


def dense_correlation(
    record,
    iq: torch.Tensor,
    hop_us: float = DENSE_HOP_US,
) -> tuple[torch.Tensor, dict]:
    """Adjacent-frame normalized complex correlation on the 512-column grid.

    Returns corr [2,2048,512] float32 (Re, Im) and diagnostics.  Each dense
    frame pair is assigned to the nearest original column by its midpoint
    time.  The last column must have at least one valid pair; the builder
    asserts this for every sample.
    """
    fs = record.B_mhz * 1e6
    h = math.ceil(record.N / (N_TIME - 1))
    n_ext = (N_TIME - 1) * h
    hop = dense_hop_samples(fs, hop_us)
    z, f_hz = halfbin_short_stft(iq, fs, hop, n_ext, COARSE_NFFT, half_bin=True)
    nfreq, nframes = z.shape
    if nframes < 2:
        raise RuntimeError(f"dense STFT has {nframes} frames; need at least 2")
    num = z[:, 1:] * torch.conj(z[:, :-1])
    # Empirical sign from the frame-centre convention: for a stationary tone
    # the raw pair phase is +2 pi f hop / Fs, so remove it with the minus sign.
    num = num * torch.exp(-1j * 2.0 * math.pi * f_hz[:, None] * hop / fs)
    den = torch.sqrt(
        (z[:, 1:].abs().square() * z[:, :-1].abs().square()).clamp_min(1e-30)
    )
    corr = (num / den).to(torch.complex64)
    # Pair midpoint times; nearest original column.
    j = torch.arange(nframes - 1, dtype=torch.float64)
    t_mid = (j + 0.5) * hop
    cols = torch.round(t_mid / h).to(torch.long).clamp(0, N_TIME - 1)
    sums = torch.zeros((nfreq, N_TIME), dtype=torch.complex64)
    counts = torch.zeros((N_TIME,), dtype=torch.float32)
    sums.index_add_(1, cols, corr)
    counts.index_add_(0, cols, torch.ones(len(cols), dtype=torch.float32))
    safe = counts.clamp_min(1.0)
    avg = sums / safe[None, :]
    # Empty columns are not expected; keep a zero and record them.
    empty = (counts == 0).nonzero(as_tuple=False).flatten().tolist()
    out = torch.view_as_real(avg).permute(2, 0, 1).to(torch.float32)  # [2,H,W]
    out = out.clamp(-1.0, 1.0)
    last_js = torch.nonzero(cols == N_TIME - 1, as_tuple=False).flatten()
    last_centers = (last_js.to(torch.float64)[:, None] * hop +
                    torch.tensor([0.0, 1.0], dtype=torch.float64)[None, :])
    diag = {
        "dense_hop_samples": int(hop),
        "dense_hop_us": float(hop / fs * 1e6),
        "dense_frames": int(nframes),
        "pairs": int(nframes - 1),
        "columns_with_pairs": int((counts > 0).sum()),
        "empty_columns": empty,
        "last_column_pairs": int(counts[-1]),
        "first_column_pairs": int(counts[0]),
        "min_column_pairs": int(counts.min()),
        "max_column_pairs": int(counts.max()),
        "last_column_pair_indices": [int(v) for v in last_js[-4:].tolist()],
        "last_column_pair_midpoints_us": [
            float(((int(v) + 0.5) * hop) / fs * 1e6) for v in last_js[-4:].tolist()
        ],
        "last_column_frames_within_n_ext": bool(
            len(last_centers) == 0 or float(last_centers.max()) <= float(n_ext)
        ),
        "pair_phase_sign": "-",
    }
    if counts[-1] <= 0:
        raise RuntimeError("dense correlation has no valid pair for the last column")
    if empty:
        raise RuntimeError(f"dense correlation has empty columns: {empty[:8]}")
    return out, diag


def normalize_re_im(z: torch.Tensor, scale: float) -> torch.Tensor:
    """[2,H,W] float32 Re/Im, jointly scaled and clipped."""
    x = torch.view_as_real(z).permute(2, 0, 1).to(torch.float32) / max(float(scale), 1e-12)
    return x.clamp(-1.0, 1.0)


def complex_scale(zs: Iterable[torch.Tensor], max_values: int = 20000) -> float:
    vals = [z.abs().flatten()[::max(1, z.numel() // max_values)].numpy() for z in zs]
    return float(np.percentile(np.concatenate(vals), 99.5)) if vals else 1.0


def dense_correlation_aggregate(
    record,
    iq: torch.Tensor,
    hop_us: float = DENSE_HOP_US,
) -> tuple[torch.Tensor, dict]:
    """Dense correlation with numerator/energy aggregated before normalization.

    This is the alternative discussed for weak signals: sum the phase-corrected
    conjugate products and the pair energies per original column, then divide
    once. It is intentionally separate from dense_correlation(); the production
    cache keeps the per-pair-normalized-then-average operator.
    """
    fs = record.B_mhz * 1e6
    h = math.ceil(record.N / (N_TIME - 1))
    n_ext = (N_TIME - 1) * h
    hop = dense_hop_samples(fs, hop_us)
    z, f_hz = halfbin_short_stft(iq, fs, hop, n_ext, COARSE_NFFT, half_bin=True)
    nfreq, nframes = z.shape
    if nframes < 2:
        raise RuntimeError(f"dense STFT has {nframes} frames; need at least 2")
    num = z[:, 1:] * torch.conj(z[:, :-1])
    num = (num * torch.exp(-1j * 2.0 * math.pi * f_hz[:, None] * hop / fs)).to(torch.complex64)
    p0 = z[:, :-1].abs().square()
    p1 = z[:, 1:].abs().square()
    energy_pair = torch.sqrt((p0 * p1).clamp_min(1e-30))
    j = torch.arange(nframes - 1, dtype=torch.float64)
    t_mid = (j + 0.5) * hop
    cols = torch.round(t_mid / h).to(torch.long).clamp(0, N_TIME - 1)
    sums = torch.zeros((nfreq, N_TIME), dtype=torch.complex64)
    energy = torch.zeros((nfreq, N_TIME), dtype=torch.float32)
    counts = torch.zeros((N_TIME,), dtype=torch.float32)
    sums.index_add_(1, cols, num)
    energy.index_add_(1, cols, energy_pair)
    counts.index_add_(0, cols, torch.ones(len(cols), dtype=torch.float32))
    avg = sums / energy.clamp_min(1e-12)
    out = torch.view_as_real(avg).permute(2, 0, 1).to(torch.float32).clamp(-1.0, 1.0)
    diag = {
        "dense_hop_samples": int(hop),
        "dense_frames": int(nframes),
        "pairs": int(nframes - 1),
        "last_column_pairs": int(counts[-1]),
        "min_column_pairs": int(counts.min()),
        "empty_columns": (counts == 0).nonzero(as_tuple=False).flatten().tolist(),
        "normalization": "aggregate numerator / sum pair sqrt(P_j P_j+1)",
    }
    if counts[-1] <= 0 or diag["empty_columns"]:
        raise RuntimeError("dense aggregate correlation has empty columns")
    return out, diag
