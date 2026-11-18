#!/usr/bin/env python3
"""Small controlled comparison of complex coarse-front-end candidates.

Candidates:
  sparse   : 2048-point complex STFT at 512 fine frame centers
  overlap  : 0.7 ms complex STFT -> 2048 complex bands with 50%-overlap Hann
  rect     : 0.7 ms complex STFT -> 2048 rectangular bands
  single   : 0.7 ms complex STFT -> 2048 non-overlapping Hann bands

Controlled tests:
  - band-center / band-crossover / observation-edge single tones;
  - same-band dual tones with relative phase 0, pi/2, pi;
  - short bursts at a frame center and half-way between centers;
  - real IQ on the same 14 samples as the previous round (overlap coherence).

Front-end only. No detector training, no original data modification.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.signals.data import N_TIME, SPLIT_MANIFEST, load_iq, load_manifest
from scripts.validate_complex_coarse_frontend import (
    COARSE_NFFT, FINE_MS, complex_stft, dense_pool,
    fine_frequency_pool, fine_frequency_pool_hann, gt_geometry,
)

ROOT = Path(__file__).resolve().parents[1]


def fine_frequency_pool_overlap(z_fine: torch.Tensor, n_out: int, fs: float,
                                nfft: int) -> tuple[torch.Tensor, torch.Tensor]:
    """50%-overlap linear-Hann complex bands and weighted power bands.

    Each band has center (b+0.5)*Nband and support of width 2*Nband.
    Adjacent linear-Hann windows sum to one, so a tone at a crossover is
    split 0.5/0.5 between two adjacent bands, not nulled. Power uses the same
    linear weights (a weighted band average).
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
        # Unit gain at the band center, zero at the two band edges. Adjacent
        # bands overlap 50%; linear Hann coefficients sum to one, while the
        # same weights on power are power-complementary (sum w = 1).
        w = 0.5 - 0.5 * torch.cos(2.0 * math.pi * u)
        z_out[b] = (z_fine[r0:r1] * w[:, None]).sum(dim=0).to(torch.complex64)
        p_out[b] = (z_fine[r0:r1].abs().square() * w[:, None]).sum(dim=0)
    return z_out, p_out



def fine_frequency_pool_overlap_norm(z_fine: torch.Tensor, n_out: int,
                                     nfft: int):
    """Sum-normalized 50%-overlap bands for a 0..1 coherence diagnostic."""
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
        w = w / w.sum().clamp_min(1e-12)
        z_out[b] = (z_fine[r0:r1] * w[:, None]).sum(dim=0).to(torch.complex64)
        p_out[b] = (z_fine[r0:r1].abs().square() * w[:, None]).sum(dim=0)
    return z_out, p_out

def coherent_scale(fs: float, win: torch.Tensor) -> float:
    return math.sqrt(fs * float(win.square().sum())) / float(win.sum())


def single_hann_unit(z_fine: torch.Tensor, n_out: int, nfft: int):
    """Non-overlapping unit-center-gain Hann bands (baseline)."""
    nfreq, ncols = z_fine.shape
    nband = nfreq / float(n_out)
    z_out = torch.zeros((n_out, ncols), dtype=torch.complex64)
    p_out = torch.zeros((n_out, ncols), dtype=torch.float32)
    for b in range(n_out):
        r0 = int(round(b * nband)); r1 = int(round((b + 1) * nband))
        r1 = min(nfreq, max(r1, r0 + 1))
        n = r1 - r0
        if n <= 0:
            continue
        w = torch.hann_window(n, periodic=False, dtype=torch.float32)
        z_out[b] = (z_fine[r0:r1] * w[:, None]).sum(dim=0).to(torch.complex64)
        p_out[b] = (z_fine[r0:r1].abs().square() * w[:, None].square()).sum(dim=0)
    return z_out, p_out


def rect_unit(z_fine: torch.Tensor, n_out: int, nfft: int):
    """Non-overlapping unit-gain rectangular bands (baseline)."""
    nfreq, ncols = z_fine.shape
    nband = nfreq / float(n_out)
    z_out = torch.zeros((n_out, ncols), dtype=torch.complex64)
    p_out = torch.zeros((n_out, ncols), dtype=torch.float32)
    for b in range(n_out):
        r0 = int(round(b * nband)); r1 = int(round((b + 1) * nband))
        r1 = min(nfreq, max(r1, r0 + 1))
        z_out[b] = z_fine[r0:r1].sum(dim=0).to(torch.complex64)
        p_out[b] = z_fine[r0:r1].abs().square().sum(dim=0)
    return z_out, p_out


def band_rows(fs: float, nband: int, f0: float) -> int:
    centers = (np.arange(nband) + 0.5) * fs / nband - fs / 2.0
    return int(np.argmin(np.abs(centers - f0)))


def tone_case(fs: float, N: int, f0: float):
    h = math.ceil(N / (N_TIME - 1))
    n_ext = (N_TIME - 1) * h
    fine_L = int(round(fs * FINE_MS / 1000.0))
    if fine_L % 2:
        fine_L += 1
    n = np.arange(N, dtype=np.float64)
    x = torch.from_numpy(np.exp(1j * 2 * np.pi * f0 * n / fs + 0.3).astype(np.complex64))
    zf = complex_stft(x, fs, fine_L, h, n_ext)
    zs = complex_stft(x, fs, COARSE_NFFT, h, n_ext)
    zo, po = fine_frequency_pool_overlap(zf, COARSE_NFFT, fs, fine_L)
    zh, _ = single_hann_unit(zf, COARSE_NFFT, fine_L)
    zr, _ = rect_unit(zf, COARSE_NFFT, fine_L)
    b = band_rows(fs, COARSE_NFFT, f0)
    j = 256
    coh_f = coherent_scale(fs, torch.hann_window(fine_L, periodic=True))
    coh_c = coherent_scale(fs, torch.hann_window(COARSE_NFFT, periodic=True))

    def amps(z, coh):
        out = {}
        for off in (-1, 0, 1):
            k = min(COARSE_NFFT - 1, max(0, b + off))
            out[str(off)] = float(abs(z[k, j]) * coh)
        out["sum_adjacent"] = out["-1"] + out["0"] + out["1"]
        return out

    return {
        "f0_hz": float(f0), "band": int(b), "column": j,
        "sparse": amps(zs, coh_c),
        "overlap": amps(zo, coh_f),
        "single_hann": amps(zh, coh_f),
        "rect": amps(zr, coh_f),
    }


def dual_tone_case(fs: float, N: int, f_center: float, delta: float, phase: float):
    h = math.ceil(N / (N_TIME - 1))
    n_ext = (N_TIME - 1) * h
    fine_L = int(round(fs * FINE_MS / 1000.0))
    if fine_L % 2:
        fine_L += 1
    n = np.arange(N, dtype=np.float64)
    f1, f2 = f_center - delta / 2.0, f_center + delta / 2.0
    x = torch.from_numpy(
        (np.exp(1j * 2 * np.pi * f1 * n / fs) +
         np.exp(1j * 2 * np.pi * f2 * n / fs + 1j * phase)).astype(np.complex64))
    zf = complex_stft(x, fs, fine_L, h, n_ext)
    zs = complex_stft(x, fs, COARSE_NFFT, h, n_ext)
    zo, po = fine_frequency_pool_overlap(zf, COARSE_NFFT, fs, fine_L)
    zh, _ = single_hann_unit(zf, COARSE_NFFT, fine_L)
    zr, pr = rect_unit(zf, COARSE_NFFT, fine_L)
    b = band_rows(fs, COARSE_NFFT, f_center)
    j = 256
    coh_f = coherent_scale(fs, torch.hann_window(fine_L, periodic=True))
    coh_c = coherent_scale(fs, torch.hann_window(COARSE_NFFT, periodic=True))
    return {
        "phase_rad": float(phase), "band": int(b),
        "sparse_amp": float(abs(zs[b, j]) * coh_c),
        "overlap_amp": float(abs(zo[b, j]) * coh_f),
        "single_hann_amp": float(abs(zh[b, j]) * coh_f),
        "rect_amp": float(abs(zr[b, j]) * coh_f),
        "overlap_power": float(po[b, j]),
        "rect_power": float(pr[b, j]),
    }


def burst_case(fs: float, N: int, f0: float, dur_ms: float, offset_samples: float):
    h = math.ceil(N / (N_TIME - 1))
    n_ext = (N_TIME - 1) * h
    fine_L = int(round(fs * FINE_MS / 1000.0))
    if fine_L % 2:
        fine_L += 1
    n = np.arange(N, dtype=np.float64)
    start = int(round(256 * h + offset_samples))
    length = int(round(fs * dur_ms / 1000.0))
    end = min(N, start + length)
    x = (1e-3 * (np.random.default_rng(1).standard_normal(N) +
                 1j * np.random.default_rng(2).standard_normal(N))).astype(np.complex64)
    x[start:end] += np.exp(1j * 2 * np.pi * f0 * np.arange(start, end) / fs + 0.3)
    xt = torch.from_numpy(x)
    zf = complex_stft(xt, fs, fine_L, h, n_ext)
    zs = complex_stft(xt, fs, COARSE_NFFT, h, n_ext)
    dense_hop = min(1024, max(256, h // 2))
    zd = complex_stft(xt, fs, COARSE_NFFT, dense_hop, n_ext)
    _, p_dp = dense_pool(zd, h, dense_hop, N_TIME, fs, COARSE_NFFT)
    zo, po = fine_frequency_pool_overlap(zf, COARSE_NFFT, fs, fine_L)
    p_fp = rect_unit(zf, COARSE_NFFT, fine_L)[1]
    b = band_rows(fs, COARSE_NFFT, f0)
    # coarse row band around f0
    r0, r1 = max(0, b - 2), min(COARSE_NFFT, b + 3)
    profiles = {
        "sparse": zs[r0:r1].abs().square().mean(0).numpy(),
        "dense_power": p_dp[r0:r1].mean(0).numpy(),
        "overlap_power": po[r0:r1].mean(0).numpy(),
        "fine_power": p_fp[r0:r1].mean(0).numpy(),
    }
    true_c1 = start / h
    true_c2 = (end - 1) / h
    # noise reference away from the burst
    mask = np.ones(N_TIME, dtype=bool)
    lo = max(0, int(math.floor(true_c1)) - 30)
    hi = min(N_TIME, int(math.ceil(true_c2)) + 30)
    mask[lo:hi] = False
    out = {"dur_ms": dur_ms, "offset_samples": offset_samples,
           "true_cols": [true_c1, true_c2], "profiles": {}}
    for name, p in profiles.items():
        noise = float(np.median(p[mask])) if mask.any() else 1e-12
        peak = float(np.max(p))
        thr = max(0.1 * peak, 10.0 * noise, 1e-12)
        idx = np.where(p >= thr)[0]
        a, bb = (int(idx[0]), int(idx[-1])) if len(idx) else (-1, -1)
        out["profiles"][name] = {
            "peak_snr": float(peak / max(noise, 1e-12)),
            "support": [a, bb],
            "start_err_ms": ((a - true_c1) * h / fs * 1000.0) if a >= 0 else None,
            "end_err_ms": ((bb - true_c2) * h / fs * 1000.0) if bb >= 0 else None,
        }
    return out


def real_iq_overlap(ids):
    fit = load_manifest(SPLIT_MANIFEST, "fit")
    by_id = {r.sample_id: r for r in fit}
    out = []
    for sid in ids:
        rec = by_id.get(sid)
        if rec is None:
            continue
        fs = rec.B_mhz * 1e6
        h = math.ceil(rec.N / (N_TIME - 1))
        n_ext = (N_TIME - 1) * h
        fine_L = int(round(fs * FINE_MS / 1000.0))
        if fine_L % 2:
            fine_L += 1
        iq = load_iq(rec.bin_path)
        zf = complex_stft(iq, fs, fine_L, h, n_ext)
        zo, po = fine_frequency_pool_overlap_norm(zf, COARSE_NFFT, fine_L)
        for sig in rec.signals:
            hh, t1, t2, y1, y2 = gt_geometry(rec, sig)
            c1 = int(np.clip(round(y1 * COARSE_NFFT), 0, COARSE_NFFT - 1))
            c2 = int(np.clip(round(y2 * COARSE_NFFT), c1 + 1, COARSE_NFFT))
            lo = max(0, int(round(t1 / h)))
            hi = min(N_TIME, int(round(t2 / h)) + 1)
            if hi <= lo:
                continue
            coh = (zo[c1:c2, lo:hi].abs().square().mean().item() /
                   max(po[c1:c2, lo:hi].mean().item(), 1e-30))
            out.append({
                "sample_id": sid, "signal_id": int(sig["signal_id"]),
                "class": int(sig["class"]), "B_mhz": rec.B_mhz,
                "T_ms": rec.T_ms, "overlap_coherence": float(coh),
            })
    vals = [x["overlap_coherence"] for x in out]
    return {
        "n": len(out),
        "coherence_kind": "sum-normalized 50%-overlap weights; 0..1",
        "median": float(np.median(vals)) if vals else None,
        "p10": float(np.percentile(vals, 10)) if vals else None,
        "p90": float(np.percentile(vals, 90)) if vals else None,
        "signals": out,
    }


def main():
    fs = 80e6
    N_short = 1_600_000
    N_long = 12_000_000
    f_boundary = 5_000_000.0
    f_center = f_boundary + fs / COARSE_NFFT / 2.0
    out = {
        "config": {"fs_hz": fs, "coarse_nfft": COARSE_NFFT,
                   "fine_ms": FINE_MS, "overlap": "50% linear Hann"},
        "band_tones": [], "dual_tones": [], "bursts": [],
    }
    t0 = time.perf_counter()
    for f0 in [-39.9e6, -39.5e6, f_boundary, f_center,
               f_boundary + fs / COARSE_NFFT, 39.5e6, 39.9e6]:
        out["band_tones"].append(tone_case(fs, N_short, f0))
        print("tone", f0, flush=True)
    for phase in [0.0, math.pi / 2, math.pi]:
        out["dual_tones"].append(dual_tone_case(fs, N_short, f_center, 5e3, phase))
        print("dual", phase, flush=True)
    for dur in [0.1, 0.3, 1.0]:
        for off in [0.0, math.ceil(N_long / (N_TIME - 1)) / 2.0]:
            out["bursts"].append(burst_case(fs, N_long, f_center, dur, off))
            print("burst", dur, off, flush=True)
    ids = ["train/29", "train/68", "train/103", "train/14", "train/15",
           "train/108", "train/270", "train/130", "train/214",
           "train/33", "train/188", "train/151", "train/357", "train/342"]
    out["real_iq_overlap"] = real_iq_overlap(ids)
    out["elapsed_s"] = time.perf_counter() - t0
    path = ROOT / "reports/complex_coarse_candidate_comparison.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:12000])
    print("wrote", path)


if __name__ == "__main__":
    main()
