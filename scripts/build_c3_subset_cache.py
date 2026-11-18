#!/usr/bin/env python3
"""Build a complex-only C3 cache for the matched front-end pilot.

The cache stores only the two extra channels [N,2,2048,512] float16. The
power channel is always read from the original Baseline-0 BF16 cache, so all
C1/C3 arms share bitwise-identical power inputs.

Frontends:
  overlap : 0.7 ms complex STFT -> 50%-overlap linear-Hann complex bands
  sparse  : half-bin-aligned 2048-point complex STFT at the 512 frame centres
  dense   : adjacent-frame normalized complex correlation at ~10 us hop,
            average into the 512 original columns by pair midpoint

A fixed stratified subset (default 1536 fit / 384 val) keeps the 8-hour pilot
within disk and compute limits. The subset IDs and power row indices are saved
once in cache/c3_subset.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.signals.c3_frontend import (
    COARSE_NFFT, dense_correlation, normalize_re_im, overlap_complex,
    sparse_complex, complex_scale, DENSE_HOP_US,
)
from src.signals.data import (
    COARSE_H, N_TIME, SPLIT_MANIFEST, build_sample_views, load_iq,
    load_manifest,
)
from scripts.build_complex_pilot_cache import select_subset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POWER = "cache/coarse_fit_bf16.npy"


def build_subset(fit, val, fit_n, val_n, seed):
    fit_sel = select_subset(fit, fit_n, seed)
    val_sel = select_subset(val, val_n, seed + 1)
    fit_index = {r.sample_id: i for i, r in enumerate(fit)}
    return {
        "fit_ids": [r.sample_id for r in fit_sel],
        "val_ids": [r.sample_id for r in val_sel],
        "fit_power_index": [fit_index[r.sample_id] for r in fit_sel],
        "seed": seed,
        "fit_n": len(fit_sel),
        "val_n": len(val_sel),
    }


def build_val_power(records, norm, out_path: Path):
    if out_path.exists() and out_path.with_suffix(".json").exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.int16,
        shape=(len(records), 1, COARSE_H, N_TIME),
    )
    t0 = time.perf_counter()
    for i, rec in enumerate(records):
        coarse, _, _ = build_sample_views(rec, norm, [])
        mm[i, 0] = coarse.tensor[0].to(torch.bfloat16).view(torch.int16).numpy()
        if (i + 1) % 50 == 0:
            mm.flush()
    mm.flush(); del mm
    meta = {
        "format": "coarse-bf16-bits-v1",
        "dtype": "bfloat16_bits",
        "shape": [len(records), 1, COARSE_H, N_TIME],
        "sample_ids": [r.sample_id for r in records],
        "normalizer_lo": norm[0], "normalizer_hi": norm[1],
        "source": "build_sample_views(record, normalizer, fine_indices=[])",
        "build_seconds": time.perf_counter() - t0,
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote val power cache {out_path}", flush=True)


def calibrate(frontend, records, norm, n):
    if frontend == "dense":
        return 1.0
    vals = []
    for rec in records[:n]:
        iq = load_iq(rec.bin_path)
        if frontend == "overlap":
            z, _ = overlap_complex(rec, iq)
        elif frontend == "sparse":
            z, _ = sparse_complex(rec, iq)
        else:
            raise ValueError(frontend)
        vals.append(z)
        print(f"cal {frontend} {rec.sample_id}", flush=True)
    return complex_scale(vals)


def build_frontend(frontend, rec, scale, dense_hop_us):
    iq = load_iq(rec.bin_path)
    if frontend == "overlap":
        z, _ = overlap_complex(rec, iq)
        x = normalize_re_im(z, scale)
        diag = {"clip_frac": float((x.abs() >= 1).float().mean())}
    elif frontend == "sparse":
        z, _ = sparse_complex(rec, iq)
        x = normalize_re_im(z, scale)
        diag = {"clip_frac": float((x.abs() >= 1).float().mean())}
    elif frontend == "dense":
        x, diag = dense_correlation(rec, iq, hop_us=dense_hop_us)
    else:
        raise ValueError(frontend)
    return x, diag


def verify_cache(mm, records, norm, frontend, scale, dense_hop_us, check=4):
    rng = random.Random(1729)
    picks = sorted({0, len(records) - 1, *[rng.randrange(len(records)) for _ in range(check)]})
    out = []
    for i in picks:
        x, diag = build_frontend(frontend, records[i], scale, dense_hop_us)
        cached = torch.from_numpy(np.asarray(mm[i]).copy()).float()
        diff = (cached - x.to(torch.float16).float()).abs().max().item()
        out.append({
            "index": i, "sample_id": records[i].sample_id,
            "max_abs_diff_f16": diff, "diag": diag,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", choices=["overlap", "sparse", "dense"], required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--fit-n", type=int, default=1536)
    ap.add_argument("--val-n", type=int, default=384)
    ap.add_argument("--cal-n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--dense-hop-us", type=float, default=DENSE_HOP_US)
    ap.add_argument("--power-cache", default=DEFAULT_POWER)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / f"cache/c3_{args.frontend}"
    if not out.is_absolute():
        out = ROOT / out
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)

    norm_cfg = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm_cfg["lo"]), float(norm_cfg["hi"]))
    fit = load_manifest(SPLIT_MANIFEST, "fit")
    val = load_manifest(SPLIT_MANIFEST, "val")
    subset_path = ROOT / "cache/c3_subset.json"
    if subset_path.exists():
        subset = json.loads(subset_path.read_text())
    else:
        subset = build_subset(fit, val, args.fit_n, args.val_n, args.seed)
        subset_path.write_text(json.dumps(subset, indent=2))
    by_fit = {r.sample_id: r for r in fit}
    by_val = {r.sample_id: r for r in val}
    fit_recs = [by_fit[s] for s in subset["fit_ids"]]
    val_recs = [by_val[s] for s in subset["val_ids"]]

    val_power = ROOT / "cache/c3_subset_val_power_bf16.npy"
    build_val_power(val_recs, norm, val_power)

    # Power equality check against the original path (first and last).
    pmm = np.load(ROOT / args.power_cache, mmap_mode="r")
    for k in (0, len(fit_recs) - 1):
        i = subset["fit_power_index"][k]
        bits = np.asarray(pmm[i]).copy()
        p = torch.from_numpy(bits).view(torch.bfloat16).float()[0]
        ref, _, _ = build_sample_views(fit_recs[k], norm, [])
        if not torch.equal(p, ref.tensor[0].bfloat16().float()):
            raise SystemExit(f"power cache mismatch at fit index {i}")
    print("power channel exact against build_sample_views", flush=True)

    scale = calibrate(args.frontend, fit_recs, norm, min(args.cal_n, len(fit_recs)))
    print(f"{args.frontend} scale={scale}", flush=True)

    meta = {
        "format": "c3-complex-f16-v1",
        "frontend": args.frontend,
        "channels": 2,
        "dtype": "float16",
        "shape": [len(fit_recs), 2, COARSE_H, N_TIME],
        "normalizer_lo": norm[0], "normalizer_hi": norm[1],
        "scale": scale,
        "dense_hop_us": args.dense_hop_us,
        "subset_path": str(subset_path),
        "fit_ids": subset["fit_ids"],
        "val_ids": subset["val_ids"],
        "power_cache": str(ROOT / args.power_cache),
        "val_power_cache": str(val_power),
        "power_is_original_c1": True,
        "per_sample": {},
        "build_seconds": {},
    }
    for split, recs in [("fit", fit_recs), ("val", val_recs)]:
        n = len(recs)
        path = out / f"{split}_complex.npy"
        mm = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float16,
            shape=(n, 2, COARSE_H, N_TIME),
        )
        t0 = time.perf_counter()
        diags = {}
        for i, rec in enumerate(recs):
            x, diag = build_frontend(args.frontend, rec, scale, args.dense_hop_us)
            mm[i] = x.numpy().astype(np.float16)
            diags[rec.sample_id] = diag
            if (i + 1) % 20 == 0:
                mm.flush()
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / max(elapsed, 1e-9)
                print(f"{split} {i+1}/{n} {rate:.1f}/s eta={(n-i-1)/max(rate,1e-9):.0f}s",
                      flush=True)
        mm.flush(); del mm
        meta["per_sample"][split] = diags
        meta["build_seconds"][split] = time.perf_counter() - t0
        if args.frontend == "dense":
            bad = [sid for sid, d in diags.items()
                   if d.get("last_column_pairs", 0) <= 0
                   or d.get("empty_columns")
                   or not d.get("last_column_frames_within_n_ext", False)]
            if bad:
                raise SystemExit(f"dense last/empty column failures: {bad[:5]}")
        check = verify_cache(np.load(path, mmap_mode="r"),
                             recs, norm, args.frontend, scale,
                             args.dense_hop_us, check=4)
        meta.setdefault("verification", {})[split] = check
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out} cache and meta", flush=True)


if __name__ == "__main__":
    main()
