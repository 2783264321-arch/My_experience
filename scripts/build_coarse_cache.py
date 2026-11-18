#!/usr/bin/env python3
"""Build a BF16-bit-pattern cache of full-band coarse views.

The cache stores exactly the values the BF16 autocast training path sees:
normalized coarse tensor -> bfloat16 -> int16 bit patterns. The dataset expands
the bits back to float32 without changing the represented values, so a cached
training step is numerically identical to the on-the-fly f32 -> autocast(bf16)
path.

Usage:
  SPACENET_FFT_THREADS=1 PYTHONPATH=. \
    .venv/bin/python scripts/build_coarse_cache.py \
      --split fit --output cache/coarse_fit_bf16.npy --check-samples 32
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from src.signals.data import (
    COARSE_H, N_TIME, SPLIT_MANIFEST, WINDOW_MS, build_sample_views,
    coarse_targets, file_sha256, load_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def verify_samples(mm, records, norm, check_samples: int, seed: int) -> dict:
    rng = random.Random(seed)
    n = len(records)
    picks = sorted({0, n - 1, *[rng.randrange(n) for _ in range(max(0, check_samples - 2))]})
    max_abs = 0.0
    mismatched = 0
    label_mismatch = 0
    for idx in picks:
        bits = np.asarray(mm[idx]).copy()
        cached = torch.from_numpy(bits).view(torch.bfloat16).float()
        coarse, _, _ = build_sample_views(records[idx], norm, [])
        onfly = coarse.tensor
        onfly_bf16 = onfly.to(torch.bfloat16).float()
        if cached.shape != onfly_bf16.shape:
            raise SystemExit(f"shape mismatch at {idx}: {cached.shape} vs {onfly_bf16.shape}")
        diff = (cached - onfly_bf16).abs()
        max_abs = max(max_abs, float(diff.max()))
        if not torch.equal(cached, onfly_bf16):
            mismatched += 1
        c_boxes, c_labels, _, _ = coarse_targets(records[idx])
        if not (torch.equal(c_boxes, coarse.boxes) and torch.equal(c_labels, coarse.labels)):
            label_mismatch += 1
    return {
        "n_checked": len(picks),
        "indices": picks,
        "bf16_bit_exact": mismatched == 0,
        "mismatched_tensors": mismatched,
        "max_abs_diff": max_abs,
        "targets_exact": label_mismatch == 0,
        "target_mismatches": label_mismatch,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["fit", "val"], default="fit")
    ap.add_argument("--output", default="cache/coarse_fit_bf16.npy")
    ap.add_argument("--limit", type=int, default=0,
                    help="only use the first N manifest rows (0 = all)")
    ap.add_argument("--check-samples", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to rebuild")
    out.parent.mkdir(parents=True, exist_ok=True)

    normalizer = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(normalizer["lo"]), float(normalizer["hi"]))
    records = load_manifest(SPLIT_MANIFEST, split=args.split)
    if args.limit:
        records = records[: args.limit]
    n = len(records)
    shape = (n, 1, COARSE_H, N_TIME)
    print(f"building {args.split} coarse BF16 cache: {n} samples, shape={shape}, "
          f"output={out}", flush=True)

    mm = np.lib.format.open_memmap(out, mode="w+", dtype=np.int16, shape=shape)
    t0 = time.perf_counter()
    for i, record in enumerate(records):
        coarse, _, _ = build_sample_views(record, norm, [])
        bits = coarse.tensor[0].to(torch.bfloat16).view(torch.int16).numpy()
        mm[i, 0] = bits
        if (i + 1) % 50 == 0:
            mm.flush()
        if (i + 1) % args.log_every == 0 or i + 1 == n:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            eta = (n - i - 1) / max(rate, 1e-9)
            print(f"  {i + 1}/{n}  {rate:.1f} samples/s  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)
    mm.flush()
    del mm
    build_s = time.perf_counter() - t0

    mm = np.load(out, mmap_mode="r")
    verification = verify_samples(mm, records, norm, args.check_samples, args.seed)
    del mm
    meta = {
        "format": "coarse-bf16-bits-v1",
        "dtype": "bfloat16_bits",
        "n_samples": n,
        "shape": list(shape),
        "split": args.split,
        "sample_ids": [r.sample_id for r in records],
        "window_ms": WINDOW_MS,
        "n_time": N_TIME,
        "coarse_h": COARSE_H,
        "normalizer_lo": norm[0],
        "normalizer_hi": norm[1],
        "normalizer_sha256": file_sha256(ROOT / "configs/normalizer.json"),
        "split_manifest": str(SPLIT_MANIFEST),
        "split_manifest_sha256": file_sha256(SPLIT_MANIFEST),
        "git_commit": git_commit(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "build_sample_views(record, normalizer, fine_indices=[])",
        "build_seconds": build_s,
        "verification": verification,
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024 ** 3:.2f} GiB) and {meta_path}", flush=True)
    print(json.dumps(verification, indent=2), flush=True)
    if not verification["bf16_bit_exact"] or not verification["targets_exact"]:
        raise SystemExit("cache verification failed; do not use this cache")


if __name__ == "__main__":
    main()
