#!/usr/bin/env python3
"""Validate the three C3 front-ends before building the training cache.

Checks requested for this round:
  - half-bin frequency alignment of the two short-window routes;
  - last-column valid frame pairs for the dense correlation route;
  - power-channel equality with the original C1 coarse path;
  - front-end timing for cache-size planning.

Synthetic tones are placed exactly on a coarse row centre. The unshifted
2048-point FFT is included only to show the half-bin split that the shift
removes.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.signals.c3_frontend import (
    COARSE_NFFT, DENSE_HOP_US, dense_correlation, halfbin_short_stft,
    overlap_complex, sparse_complex,
)
from src.signals.data import SPLIT_MANIFEST, build_sample_views, load_iq, load_manifest

ROOT = Path(__file__).resolve().parents[1]


def synthetic_case(fs_mhz: float, T_ms: float, b: int) -> dict:
    fs = fs_mhz * 1e6
    N = int(round(fs * T_ms / 1000.0))
    h = math.ceil(N / 511)
    n = torch.arange(N, dtype=torch.float64)
    f0 = (b - COARSE_NFFT / 2.0 + 0.5) * fs / COARSE_NFFT
    x = torch.from_numpy(
        np.exp(1j * 2 * np.pi * f0 * n.numpy() / fs + 0.3).astype(np.complex64)
    )
    rec = SimpleNamespace(B_mhz=float(fs_mhz), T_ms=float(T_ms), N=int(N))
    z_sparse, _ = sparse_complex(rec, x)
    j = min(10, z_sparse.shape[1] - 1)
    top = torch.topk(z_sparse[:, j].abs(), 3)
    # Unshifted control: a tone at a half-bin centre splits between two bins.
    xp = torch.nn.functional.pad(x, (0, (511 * h) - N))
    xp = torch.nn.functional.pad(xp, (COARSE_NFFT // 2, COARSE_NFFT // 2))
    frames = xp.unfold(0, COARSE_NFFT, h)
    win = torch.hann_window(COARSE_NFFT, periodic=True)
    z_int = torch.fft.fftshift(
        torch.fft.fft(frames * win[None, :], dim=1), dim=1
    ).T
    f_int = (torch.arange(COARSE_NFFT, dtype=torch.float64) - COARSE_NFFT / 2.0) * fs / COARSE_NFFT
    z_int = z_int * torch.exp(1j * 2 * math.pi * f_int[:, None] * (COARSE_NFFT / 2.0) / fs)
    top_int = torch.topk(z_int[:, j].abs(), 3)
    dense, diag = dense_correlation(rec, x, hop_us=DENSE_HOP_US)
    return {
        "fs_mhz": fs_mhz, "T_ms": T_ms, "N": N, "h": h,
        "target_row": b, "tone_hz": float(f0),
        "sparse_peak_rows": top.indices.tolist(),
        "sparse_peak_ratio": float(top.values[0] / max(float(top.values[1]), 1e-30)),
        "unshifted_peak_rows": top_int.indices.tolist(),
        "unshifted_peak_ratio": float(top_int.values[0] / max(float(top_int.values[1]), 1e-30)),
        "dense_row_corr_re": float(dense[0, b].mean()),
        "dense_row_corr_im": float(dense[1, b].mean()),
        "dense_row_corr_mag": float(dense[:, b].norm(dim=0).mean()),
        "dense": diag,
    }


def real_case(rec, norm, fit_index, pmm) -> dict:
    iq = load_iq(rec.bin_path)
    t = time.perf_counter(); zo, _ = overlap_complex(rec, iq); to = time.perf_counter() - t
    t = time.perf_counter(); zs, _ = sparse_complex(rec, iq); ts = time.perf_counter() - t
    t = time.perf_counter(); zd, diag = dense_correlation(rec, iq); td = time.perf_counter() - t
    bits = np.asarray(pmm[fit_index]).copy()
    p = torch.from_numpy(bits).view(torch.bfloat16).float()[0]
    ref, _, _ = build_sample_views(rec, norm, [])
    power_exact = bool(torch.equal(p, ref.tensor[0].bfloat16().float()))
    return {
        "sample_id": rec.sample_id, "B_mhz": rec.B_mhz, "T_ms": rec.T_ms,
        "N": rec.N, "h": math.ceil(rec.N / 511),
        "overlap_seconds": to, "sparse_seconds": ts, "dense_seconds": td,
        "shapes": {"overlap": list(zo.shape), "sparse": list(zs.shape),
                   "dense": list(zd.shape)},
        "overlap_abs_p99_5": float(torch.quantile(zo.abs().flatten(), 0.995)),
        "sparse_abs_p99_5": float(torch.quantile(zs.abs().flatten(), 0.995)),
        "dense_mean_mag": float(zd.norm(dim=0).mean()),
        "dense_frac_mag_gt_0_99": float((zd.norm(dim=0) > 0.99).float().mean()),
        "dense": diag,
        "power_exact": power_exact,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/c3_frontend_validation.json")
    ap.add_argument("--md", default="reports/c3_frontend_validation.md")
    args = ap.parse_args()

    norm_cfg = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm_cfg["lo"]), float(norm_cfg["hi"]))
    fit = load_manifest(SPLIT_MANIFEST, "fit")
    by_id = {r.sample_id: r for r in fit}
    idx = {r.sample_id: i for i, r in enumerate(fit)}
    pmm = np.load(ROOT / "cache/coarse_fit_bf16.npy", mmap_mode="r")
    ids = ["train/29", "train/14", "train/108", "train/130", "train/33"]
    for r in fit:
        if r.B_mhz == 5.0:
            ids.append(r.sample_id)
            break

    synt = [synthetic_case(fs, 20.0, b)
            for fs in (20.0, 40.0, 80.0)
            for b in (500, 1000, 1500, 1900)]
    print(f"synthetic {len(synt)} cases", flush=True)
    real = [real_case(by_id[s], norm, idx[s], pmm) for s in ids]
    print(f"real {len(real)} samples", flush=True)

    total_s = sum(r["overlap_seconds"] + r["sparse_seconds"] + r["dense_seconds"] for r in real)
    per_sample = total_s / max(len(real), 1)
    estimated_cache_s = per_sample * (1536 + 384)
    result = {
        "synthetic": synt, "real": real,
        "timing": {
            "real_samples": len(real),
            "sum_seconds": total_s,
            "per_sample_seconds": per_sample,
            "estimated_1536_fit_plus_384_val_seconds": estimated_cache_s,
            "note": "includes overlap + sparse + dense, one after another; not a pipeline total",
        },
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# C3 front-end validation", "",
        "## Synthetic tone at a coarse row centre", "",
        "| fs | T | target row | sparse peak | ratio | unshifted peak | ratio | dense corr (Re,Im) | dense last pairs |",
        "| ---: | ---: | ---: | --- | ---: | --- | ---: | --- | ---: |",
    ]
    for s in synt:
        lines.append(
            f"| {s['fs_mhz']:g} MHz | {s['T_ms']:g} ms | {s['target_row']} | "
            f"{s['sparse_peak_rows']} | {s['sparse_peak_ratio']:.2f} | "
            f"{s['unshifted_peak_rows']} | {s['unshifted_peak_ratio']:.2f} | "
            f"({s['dense_row_corr_re']:.4f}, {s['dense_row_corr_im']:.2e}) | "
            f"{s['dense']['last_column_pairs']} |"
        )
    lines += [
        "", "## Real samples", "",
        "| sample | B/T | h | overlap s | sparse s | dense s | power exact | dense last pairs | min pairs |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for r in real:
        lines.append(
            f"| {r['sample_id']} | {r['B_mhz']:g}/{r['T_ms']:g} | {r['h']} | "
            f"{r['overlap_seconds']:.2f} | {r['sparse_seconds']:.2f} | "
            f"{r['dense_seconds']:.2f} | {r['power_exact']} | "
            f"{r['dense']['last_column_pairs']} | {r['dense']['min_column_pairs']} |"
        )
    lines += [
        "", "## Timing estimate", "",
        f"- {len(real)} samples: overlap+sparse+dense = {total_s:.1f} s, "
        f"{per_sample:.2f} s/sample.",
        f"- 1536 fit + 384 val estimate: **{estimated_cache_s/60:.1f} min** "
        f"for all three front-ends if built sequentially.",
    ]
    md = ROOT / args.md
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"wrote {out} and {md}", flush=True)


if __name__ == "__main__":
    main()
