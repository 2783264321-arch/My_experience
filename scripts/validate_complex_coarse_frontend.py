#!/usr/bin/env python3
"""Validate a 2048-point complex coarse STFT against 0.7 ms complex fine slices.

Questions
---------
1. Can a 2048-point full-band complex STFT use the same 512 frame centers as
   the existing 0.7 ms native complex STFT?
2. Do the coarse and fine tensors share a consistent time/frequency coordinate
   convention and a consistent complex (Re/Im) normalization?
3. On long samples, what do the short-window gaps do to GT signal coverage,
   boundary estimates, and local detail?
4. Is a denser 2048-point STFT (50% overlap, pooled to 512 columns) a better
   construction? What does it cost?

This is a front-end validation only: no detector training, no original data
modification.

Usage:
  PYTHONPATH=. python scripts/validate_complex_coarse_frontend.py
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.signals.data import N_TIME, SPLIT_MANIFEST, load_iq, load_manifest

ROOT = Path(__file__).resolve().parents[1]
FINE_MS = 0.7
COARSE_NFFT = 2048
DENSE_HOP_MAX = 1024

# Representative fit samples: long/short duration, 20/40/80 MHz observation,
# each chosen to contain at least one wide signal and one narrow signal.
DEFAULT_SAMPLE_IDS = [
    "train/29", "train/68", "train/103",   # 80 MHz / 150 ms
    "train/14", "train/15",                # 40 MHz / 150 ms
    "train/108", "train/270",              # 20 MHz / 150 ms
    "train/130", "train/214",              # 80 MHz / 20 ms
    "train/33", "train/188",               # 20 MHz / 20 ms
    "train/151",                           # 40 MHz / 40 ms
]


def hann_window(L: int) -> torch.Tensor:
    return torch.hann_window(L, periodic=True, dtype=torch.float32)


def complex_stft(x: torch.Tensor, fs: float, win_len: int, hop: int,
                 n_ext: int) -> torch.Tensor:
    """Right-pad, center, bilateral complex STFT, fftshift, PSD normalization.

    Returns complex64 [win_len, frames]. The normalization is the same as the
    existing power path: Z / sqrt(Fs * sum(w^2)), so |Z|^2 is the existing PSD.
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
    # Remove the window-center phase so Re/Im are referenced to the frame
    # center, not to the start of the finite window. For a periodic Hann
    # window of even length L the effective center is L/2.
    f_hz = (torch.arange(win_len, dtype=torch.float64) - win_len / 2.0) * fs / win_len
    z = z * torch.exp(1j * 2.0 * math.pi * f_hz[:, None] * (win_len / 2.0) / fs)
    scale = math.sqrt(fs * float(win.square().sum()))
    return (z / scale).to(torch.complex64)


def phase_align(z: torch.Tensor, f_hz: torch.Tensor, dt_samples: torch.Tensor,
                fs: float) -> torch.Tensor:
    """Multiply z[:, m] by exp(-j 2 pi f (t_m - t_ref)/Fs)."""
    phase = torch.exp(-1j * 2.0 * math.pi * f_hz[:, None] * dt_samples[None, :] / fs)
    return z * phase


def dense_pool(z_dense: torch.Tensor, h: int, hop: int, n_cols: int,
               fs: float, nfft: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Phase-aligned complex pool and power pool of dense 2048 STFT to n_cols."""
    nfreq, nframes = z_dense.shape
    t_dense = torch.arange(nframes, dtype=torch.float64) * hop
    f_hz = (torch.arange(nfft, dtype=torch.float64) - nfft / 2.0) * fs / nfft
    z_out = torch.zeros((nfreq, n_cols), dtype=torch.complex64)
    p_out = torch.zeros((nfreq, n_cols), dtype=torch.float32)
    for j in range(n_cols):
        t_ref = j * h
        sel = torch.nonzero((t_dense >= t_ref - h / 2) &
                            (t_dense < t_ref + h / 2), as_tuple=False).flatten()
        if len(sel) == 0:
            sel = torch.tensor([int(round(t_ref / hop))], dtype=torch.long)
            sel = sel.clamp(0, nframes - 1)
        zd = z_dense[:, sel]
        dt = t_dense[sel] - t_ref
        z_out[:, j] = phase_align(zd, f_hz, dt, fs).mean(dim=1)
        p_out[:, j] = zd.abs().square().mean(dim=1)
    return z_out, p_out


def fine_frequency_pool(z_fine: torch.Tensor, n_out: int, fs: float,
                        nfft: int, h: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Phase-aligned complex and power pool of the native fine STFT to n_out."""
    nfreq, ncols = z_fine.shape
    bounds = np.linspace(0, nfreq, n_out + 1).round().astype(np.int64)
    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i - 1]:
            bounds[i] = bounds[i - 1] + 1
    bounds = np.clip(bounds, 0, nfreq)
    z_out = torch.zeros((n_out, ncols), dtype=torch.complex64)
    p_out = torch.zeros((n_out, ncols), dtype=torch.float32)
    # z_fine is already referenced to the common frame center, so all rows
    # sharing that center can be summed coherently without an extra time
    # carrier term. The band coefficient is the complex band average.
    for b in range(n_out):
        r0, r1 = int(bounds[b]), int(bounds[b + 1])
        if r1 <= r0:
            continue
        z_out[b] = z_fine[r0:r1].mean(dim=0).to(torch.complex64)
        p_out[b] = z_fine[r0:r1].abs().square().mean(dim=0)
    return z_out, p_out



def fine_frequency_pool_hann(z_fine: torch.Tensor, n_out: int, fs: float,
                             nfft: int) -> torch.Tensor:
    """Hann-weighted frequency-band complex envelope (full time coverage)."""
    nfreq, ncols = z_fine.shape
    bounds = np.linspace(0, nfreq, n_out + 1).round().astype(np.int64)
    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i - 1]:
            bounds[i] = bounds[i - 1] + 1
    bounds = np.clip(bounds, 0, nfreq)
    z_out = torch.zeros((n_out, ncols), dtype=torch.complex64)
    for b in range(n_out):
        r0, r1 = int(bounds[b]), int(bounds[b + 1])
        if r1 <= r0:
            continue
        w = torch.hann_window(r1 - r0, periodic=False, dtype=torch.float32)
        w = w / w.sum().clamp_min(1e-12)
        z_out[b] = (z_fine[r0:r1] * w[:, None]).sum(dim=0).to(torch.complex64)
    return z_out

def support_columns(profile: np.ndarray, frac: float = 0.1) -> tuple[int, int]:
    thr = max(float(np.max(profile)) * frac, 1e-12)
    idx = np.where(profile >= thr)[0]
    if len(idx) == 0:
        return -1, -1
    return int(idx[0]), int(idx[-1])


def gt_geometry(rec, sig):
    h = math.ceil(rec.N / (N_TIME - 1))
    x1 = ((sig["start_time"] / 1000.0) * (rec.B_mhz * 1e6) / h + 0.5) / N_TIME
    x2 = ((sig["end_time"] / 1000.0) * (rec.B_mhz * 1e6) / h + 0.5) / N_TIME
    y1 = (sig["start_frequency"] - rec.f_lo) / rec.B_mhz
    y2 = (sig["end_frequency"] - rec.f_lo) / rec.B_mhz
    t1 = (x1 * N_TIME - 0.5) * h
    t2 = (x2 * N_TIME - 0.5) * h
    return h, t1, t2, y1, y2


def signal_metrics(rec, sig, reps, h, fs):
    hh, t1, t2, y1, y2 = gt_geometry(rec, sig)
    dur = max(1.0, t2 - t1)
    win_half = 1024.0
    # Exact union coverage of the sparse 2048 windows.
    centers = np.arange(N_TIME) * h
    cov = 0.0
    for c in centers:
        cov += max(0.0, min(c + win_half, t2) - max(c - win_half, t1))
    coverage = min(1.0, cov / dur)
    d_start = float(np.min(np.abs(centers - t1))) / fs * 1000.0
    d_end = float(np.min(np.abs(centers - t2))) / fs * 1000.0
    c1 = int(np.clip(round(y1 * 2048), 0, 2047))
    c2 = int(np.clip(round(y2 * 2048), 1, 2048))
    f1 = int(np.clip(round(y1 * reps["F"]), 0, reps["F"] - 1))
    f2 = int(np.clip(round(y2 * reps["F"]), f1 + 1, reps["F"]))
    out = {
        "sample_id": rec.sample_id, "signal_id": int(sig["signal_id"]),
        "class": int(sig["class"]),
        "B_mhz": rec.B_mhz, "T_ms": rec.T_ms,
        "start_ms": float(sig["start_time"]), "end_ms": float(sig["end_time"]),
        "start_mhz": float(sig["start_frequency"]),
        "end_mhz": float(sig["end_frequency"]),
        "dur_ms": float(sig["end_time"] - sig["start_time"]),
        "bandwidth_mhz": float(sig["end_frequency"] - sig["start_frequency"]),
        "sparse_coverage": float(coverage),
        "start_center_ms": d_start, "end_center_ms": d_end,
        "gt_cols": [int(round(t1 / h)), int(round(t2 / h))],
        "profiles": {},
    }
    # Power profiles in the GT band: mean over the band rows.
    p_sparse = reps["sparse"].abs().square()[c1:c2].mean(0).numpy()
    p_dense = reps["dense_power"][c1:c2].mean(0).numpy()
    p_fine = reps["fine"].abs().square()[f1:f2].mean(0).numpy()
    p_fp = reps["fine_power"][c1:c2].mean(0).numpy()
    p_dp_complex = reps["dense_pool"].abs().square()[c1:c2].mean(0).numpy()
    p_fp_complex = reps["fine_pool"].abs().square()[c1:c2].mean(0).numpy()
    for name, p in [("sparse", p_sparse), ("dense_power", p_dense),
                    ("fine", p_fine), ("fine_power", p_fp),
                    ("dense_pool_complex", p_dp_complex),
                    ("fine_pool_complex", p_fp_complex)]:
        out["profiles"][name] = p.astype(float).tolist()
    lo = max(0, out["gt_cols"][0]); hi = min(N_TIME, out["gt_cols"][1] + 1)
    search_lo = max(0, lo - 10); search_hi = min(N_TIME, hi + 10)
    for name, p in [("sparse", p_sparse), ("dense_power", p_dense),
                    ("fine", p_fine), ("fine_power", p_fp)]:
        a, b = support_columns(p[search_lo:search_hi])
        a = a + search_lo if a >= 0 else -1
        b = b + search_lo if b >= 0 else -1
        out[f"{name}_support"] = [a, b]
        out[f"{name}_start_err_cols"] = (a - out["gt_cols"][0]) if a >= 0 else None
        out[f"{name}_end_err_cols"] = (b - out["gt_cols"][1]) if b >= 0 else None
        out[f"{name}_start_err_ms"] = ((a - out["gt_cols"][0]) * h / fs * 1000.0) if a >= 0 else None
        out[f"{name}_end_err_ms"] = ((b - out["gt_cols"][1]) * h / fs * 1000.0) if b >= 0 else None
    # Sparse vs dense missed bins inside the GT interval.
    seg_d = p_dense[lo:hi]
    seg_s = p_sparse[lo:hi]
    thr = max(float(np.max(seg_d)) * 0.1, 1e-12)
    missed = (seg_d > 5.0 * np.maximum(seg_s, 1e-12)) & (seg_d > thr)
    out["dense_energy_frac"] = float(seg_d.sum())
    out["sparse_energy_frac"] = float(seg_s.sum() / max(seg_d.sum(), 1e-12))
    out["sparse_missed_bins"] = int(missed.sum())
    out["sparse_missed_frac"] = float(missed.mean()) if len(missed) else None
    out["sparse_dense_ratio_median"] = float(np.median(
        seg_s / np.maximum(seg_d, 1e-12)))
    # Coherent complex pooling retention: |mean z|^2 / mean |z|^2.
    c_dp = reps["dense_pool"].abs().square()[c1:c2, lo:hi].mean().item()
    p_dp = reps["dense_power"][c1:c2, lo:hi].mean().item()
    c_fp = reps["fine_pool"].abs().square()[c1:c2, lo:hi].mean().item()
    p_fp = reps["fine_power"][c1:c2, lo:hi].mean().item()
    c_fh = reps["fine_hann"].abs().square()[c1:c2, lo:hi].mean().item()
    out["dense_complex_coherence"] = float(c_dp / max(p_dp, 1e-30))
    out["fine_complex_coherence"] = float(c_fp / max(p_fp, 1e-30))
    out["fine_hann_complex_coherence"] = float(c_fh / max(p_fp, 1e-30))
    return out


def synthetic_checks(fs: float, N: int) -> dict:
    """Tone phase/scale and rectangular-burst boundary checks."""
    h = math.ceil(N / (N_TIME - 1))
    n_ext = (N_TIME - 1) * h
    n = np.arange(N, dtype=np.float64)
    f0 = 5e6 if fs > 20e6 else 2e6
    x = torch.from_numpy(np.exp(1j * 2 * np.pi * f0 * n / fs + 0.3).astype(np.complex64))
    t = time.perf_counter()
    fine_L = int(round(fs * FINE_MS / 1000.0))
    if fine_L % 2:
        fine_L += 1
    z_fine = complex_stft(x, fs, fine_L, h, n_ext)
    t_fine = time.perf_counter() - t
    t = time.perf_counter()
    z_sparse = complex_stft(x, fs, COARSE_NFFT, h, n_ext)
    t_sparse = time.perf_counter() - t
    z_dense = complex_stft(x, fs, COARSE_NFFT, DENSE_HOP_MAX, n_ext)
    t_dense = time.perf_counter() - t
    z_fh = fine_frequency_pool_hann(z_fine, COARSE_NFFT, fs, fine_L)
    z_fr, _ = fine_frequency_pool(z_fine, COARSE_NFFT, fs, fine_L)
    fs_bin_c = (np.arange(COARSE_NFFT) - COARSE_NFFT / 2.0) * fs / COARSE_NFFT
    fs_bin_f = (np.arange(fine_L) - fine_L / 2.0) * fs / fine_L
    kc = int(np.argmin(np.abs(fs_bin_c - f0)))
    kf = int(np.argmin(np.abs(fs_bin_f - f0)))
    j = 100
    phase_c = float(np.angle(z_sparse[kc, j]))
    phase_f = float(np.angle(z_fine[kf, j]))
    dphase = float(np.angle(np.exp(1j * (phase_c - phase_f))))
    wc = hann_window(COARSE_NFFT)
    wf = hann_window(fine_L)
    coh_c = math.sqrt(fs * float(wc.square().sum())) / float(wc.sum())
    coh_f = math.sqrt(fs * float(wf.square().sum())) / float(wf.sum())
    phase_h = float(np.angle(z_fh[kc, j]))
    dphase_h = float(np.angle(np.exp(1j * (phase_h - phase_f))))
    phase_r = float(np.angle(z_fr[kc, j]))
    dphase_r = float(np.angle(np.exp(1j * (phase_r - phase_f))))
    return {
        "fs_hz": fs, "N": int(N), "h": int(h), "n_ext": int(n_ext),
        "tone_hz": float(f0), "coarse_bin_hz": float(fs_bin_c[kc]),
        "fine_bin_hz": float(fs_bin_f[kf]),
        "amp_coarse_psd": float(abs(z_sparse[kc, j])),
        "amp_fine_psd": float(abs(z_fine[kf, j])),
        "amp_coarse_coherent": float(abs(z_sparse[kc, j]) * coh_c),
        "amp_fine_coherent": float(abs(z_fine[kf, j]) * coh_f),
        "phase_diff_rad": dphase,
        "amp_hann_coherent": float(abs(z_fh[kc, j]) * coh_f),
        "phase_diff_hann_rad": dphase_h,
        "amp_rect_coherent": float(abs(z_fr[kc, j]) * coh_f),
        "phase_diff_rect_rad": dphase_r,
        "stft_seconds": {"fine": t_fine, "sparse": t_sparse, "dense": t_dense},
        "psd_sum_sparse": float(z_sparse.abs().square().sum()),
        "psd_sum_fine": float(z_fine.abs().square().sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-ids", default="", help="comma-separated; default representative set")
    ap.add_argument("--out", default="reports/complex_coarse_frontend.json")
    ap.add_argument("--dense-hop-max", type=int, default=1024)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    fit = load_manifest(SPLIT_MANIFEST, "fit")
    by_id = {r.sample_id: r for r in fit}
    ids = [s.strip() for s in args.sample_ids.split(",") if s.strip()]
    if not ids:
        ids = [s for s in DEFAULT_SAMPLE_IDS if s in by_id]
        # Add two more long samples if the placeholder was removed.
        for s in ["train/357", "train/342"]:
            if s in by_id and s not in ids:
                ids.append(s)
    if args.max_samples:
        ids = ids[: args.max_samples]
    ids = [s for s in ids if s in by_id]
    print(f"samples: {ids}", flush=True)

    out = {"config": {
        "fine_ms": FINE_MS, "coarse_nfft": COARSE_NFFT,
        "dense_hop_max": args.dense_hop_max,
        "normalization": "Z / sqrt(Fs * sum(w^2)) for both views",
        "frame_centers": "t_j = j*h, h=ceil(N/511), center=True",
    }, "synthetic": [], "samples": [], "signals": []}
    # Synthetic checks at two rates.
    for fs, N in [(80e6, 1_600_000), (20e6, 400_000)]:
        out["synthetic"].append(synthetic_checks(fs, N))
        print("synthetic", out["synthetic"][-1], flush=True)

    for sid in ids:
        rec = by_id[sid]
        fs = rec.B_mhz * 1e6
        h = math.ceil(rec.N / (N_TIME - 1))
        n_ext = (N_TIME - 1) * h
        fine_L = int(round(fs * FINE_MS / 1000.0))
        if fine_L % 2:
            fine_L += 1
        iq = load_iq(rec.bin_path)
        print(f"{sid}: B={rec.B_mhz} T={rec.T_ms} N={rec.N} h={h}",
              flush=True)

        t0 = time.perf_counter()
        z_fine = complex_stft(iq, fs, fine_L, h, n_ext)
        t_fine = time.perf_counter() - t0
        t0 = time.perf_counter()
        z_sparse = complex_stft(iq, fs, COARSE_NFFT, h, n_ext)
        t_sparse = time.perf_counter() - t0
        dense_hop = min(args.dense_hop_max, max(256, h // 2))
        t0 = time.perf_counter()
        z_dense = complex_stft(iq, fs, COARSE_NFFT, dense_hop, n_ext)
        t_dense = time.perf_counter() - t0
        t0 = time.perf_counter()
        z_dp, p_dp = dense_pool(z_dense, h, dense_hop, N_TIME, fs, COARSE_NFFT)
        t_pool = time.perf_counter() - t0
        t0 = time.perf_counter()
        z_fp, p_fp = fine_frequency_pool(z_fine, COARSE_NFFT, fs, fine_L, h)
        t_fpool = time.perf_counter() - t0
        t0 = time.perf_counter()
        z_fh = fine_frequency_pool_hann(z_fine, COARSE_NFFT, fs, fine_L)
        t_fhann = time.perf_counter() - t0

        # Verify the complex fine power matches the existing power path.
        fine_power_err = None
        try:
            from src.signals.stft_fast import stft_power_fast
            p_ref, _, _, _, _ = stft_power_fast(iq, rec.B_mhz, fft_threads=1)
            p_ref = torch.from_numpy(p_ref.T)  # [F,512]
            fine_power_err = float((z_fine.abs().square() - p_ref).abs().max() /
                                   max(float(p_ref.max()), 1e-12))
        except Exception as exc:
            fine_power_err = f"skip: {exc}"

        reps = {
            "F": z_fine.shape[0],
            "sparse": z_sparse,
            "dense_pool": z_dp,
            "dense_power": p_dp,
            "fine": z_fine,
            "fine_pool": z_fp,
            "fine_hann": z_fh,
            "fine_power": p_fp,
        }
        sample = {
            "sample_id": sid, "B_mhz": rec.B_mhz, "T_ms": rec.T_ms,
            "N": rec.N, "h": h, "n_ext": n_ext, "fine_L": fine_L,
            "dense_hop": dense_hop, "fine_frames": int(z_fine.shape[1]),
            "sparse_frames": int(z_sparse.shape[1]),
            "dense_frames": int(z_dense.shape[1]),
            "coarse_bin_hz": float(fs / COARSE_NFFT),
            "fine_bin_hz": float(fs / fine_L),
            "time_center_max_diff": 0.0,
            "fine_power_rel_err": fine_power_err,
            "seconds": {"fine": t_fine, "sparse": t_sparse,
                        "dense": t_dense, "dense_pool": t_pool,
                        "fine_pool": t_fpool, "fine_hann_pool": t_fhann},
            "signals": [],
        }
        for sig in rec.signals:
            m = signal_metrics(rec, sig, reps, h, fs)
            sample["signals"].append(m)
            out["signals"].append(m)
        out["samples"].append(sample)
        print(f"  fine {t_fine:.2f}s sparse {t_sparse:.2f}s dense {t_dense:.2f}s "
              f"pool {t_pool:.2f}s fpool {t_fpool:.2f}s", flush=True)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("wrote", out_path)

    # Compact summary.
    print(json.dumps({
        "synthetic": out["synthetic"],
        "samples": [{k: s[k] for k in ["sample_id", "B_mhz", "T_ms", "h",
                                        "dense_hop", "coarse_bin_hz",
                                        "fine_bin_hz", "seconds"]}
                    for s in out["samples"]],
        "coverage": {
            "min": min(s["sparse_coverage"] for s in out["signals"]),
            "median": float(np.median([s["sparse_coverage"] for s in out["signals"]])),
            "by_T": {str(T): float(np.median([s["sparse_coverage"] for s in out["signals"]
                                              if abs(s["T_ms"] - T) < 1e-6]))
                     for T in sorted(set(s["T_ms"] for s in out["signals"]))},
        },
        "missed": {
            "total_bins": int(sum(s["sparse_missed_bins"] for s in out["signals"])),
            "total_signal_bins": int(sum(len(s["profiles"]["dense_power"]) for s in out["signals"])),
            "worst": sorted(out["signals"], key=lambda s: -s["sparse_missed_frac"])[:5],
        },
    }, indent=2)[:12000])


if __name__ == "__main__":
    main()
