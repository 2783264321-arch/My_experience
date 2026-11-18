#!/usr/bin/env python3
"""Build a small paired C1/C3 cache from a fixed stratified fit/val subset.

Cache layout per split:
  coarse.npy  [N,3,2048,512] float16: channel 0 power, 1 Re, 2 Im
  fine.npy    [N,3,2048,512] float16: same for one fixed fine slice
  meta.json   sample IDs, boxes/labels, fine starts, normalization scales

Channel 0 is bitwise shared by C1 and C3. Coarse power and complex come from
the same 0.7 ms STFT and the same 50%-overlap linear-Hann weights.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.signals.complex_frontend import (
    complex_stft, fine_window_len, normalize_complex, normalize_log_power,
    overlap_bands,
)
from src.signals.data import (
    COARSE_H, FINE_H, N_TIME, SPLIT_MANIFEST, _label_boxes,
    frequency_starts, h_for_n, load_iq, load_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def select_subset(records, n, seed):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for r in records:
        buckets[(round(r.B_mhz, 3), round(r.T_ms, 3))].append(r)
    keys = sorted(buckets)
    for k in keys:
        rng.shuffle(buckets[k])
    out = []
    i = 0
    while len(out) < n and any(buckets[k] for k in keys):
        k = keys[i % len(keys)]
        i += 1
        if buckets[k]:
            out.append(buckets[k].pop())
    return out


def pick_fine_start(rec, F, h):
    starts = frequency_starts(F)
    best = (0, -1, -1.0)
    for j, start in enumerate(starts):
        boxes, labels, _, _ = _label_boxes(rec.signals, rec, F, h, start)
        if len(labels) == 0:
            score = (-1, 0.0)
        else:
            area = float((boxes[:, 2] - boxes[:, 0]).clamp_min(0).sum() *
                         (boxes[:, 3] - boxes[:, 1]).clamp_min(0).sum())
            score = (len(labels), area)
        if score > (best[1], best[2]):
            best = (j, score[0], score[1])
    return best[0], starts[best[0]]


def build_one(rec, lo, hi, scale_c, scale_f):
    fs = rec.B_mhz * 1e6
    h = h_for_n(rec.N)
    n_ext = (N_TIME - 1) * h
    F = int(round(rec.B_mhz * 700))
    L = fine_window_len(fs)
    iq = load_iq(rec.bin_path)
    z = complex_stft(iq, fs, L, h, n_ext)
    zc, pc = overlap_bands(z, COARSE_H, fs, F)
    # coarse labels
    cb, cl, cids, ctr = _label_boxes(rec.signals, rec, F, h, None)
    # fixed fine slice
    fj, start = pick_fine_start(rec, F, h)
    zf = z[start:start + FINE_H]
    pb = _label_boxes(rec.signals, rec, F, h, start)
    fb, fl, fids, ftr = pb
    coarse = np.empty((3, COARSE_H, N_TIME), dtype=np.float16)
    coarse[0] = normalize_log_power(pc, lo, hi).numpy().astype(np.float16)
    coarse[1:3] = normalize_complex(zc, scale_c).numpy().astype(np.float16)
    fine = np.empty((3, FINE_H, N_TIME), dtype=np.float16)
    fine[0] = normalize_log_power(zf.abs().square(), lo, hi).numpy().astype(np.float16)
    fine[1:3] = normalize_complex(zf, scale_f).numpy().astype(np.float16)
    return {
        "sample_id": rec.sample_id, "B_mhz": rec.B_mhz, "T_ms": rec.T_ms,
        "N": rec.N, "h": h, "F": F, "fine_index": int(fj),
        "fine_start": int(start),
        "coarse_power_clip": float(((normalize_log_power(pc, lo, hi) <= 0) |
                                    (normalize_log_power(pc, lo, hi) >= 1)).float().mean()),
        "fine_power_clip": float(((normalize_log_power(zf.abs().square(), lo, hi) <= 0) |
                                  (normalize_log_power(zf.abs().square(), lo, hi) >= 1)).float().mean()),
        "coarse_complex_clip": float((normalize_complex(zc, scale_c).abs() >= 1).float().mean()),
        "fine_complex_clip": float((normalize_complex(zf, scale_f).abs() >= 1).float().mean()),
        "coarse_boxes": cb.tolist(), "coarse_labels": cl.tolist(),
        "fine_boxes": fb.tolist(), "fine_labels": fl.tolist(),
        "coarse_ids": [int(x) for x in cids], "fine_ids": [int(x) for x in fids],
        "coarse_truncated": [bool(x) for x in ctr],
        "fine_truncated": [bool(x) for x in ftr],
    }, coarse, fine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/complex_pilot_cache")
    ap.add_argument("--fit-n", type=int, default=256)
    ap.add_argument("--val-n", type=int, default=128)
    ap.add_argument("--cal-n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1729)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    norm = json.loads((ROOT / "configs/normalizer.json").read_text())
    lo, hi = float(norm["lo"]), float(norm["hi"])

    fit = load_manifest(SPLIT_MANIFEST, "fit")
    val = load_manifest(SPLIT_MANIFEST, "val")
    fit_sel = select_subset(fit, args.fit_n, args.seed)
    val_sel = select_subset(val, args.val_n, args.seed + 1)
    print(f"fit {len(fit_sel)} val {len(val_sel)}", flush=True)

    # Calibration scales from fit only; no val leakage.
    cal = fit_sel[: min(args.cal_n, len(fit_sel))]
    cvals, fvals = [], []
    for rec in cal:
        fs = rec.B_mhz * 1e6
        h = h_for_n(rec.N); n_ext = (N_TIME - 1) * h
        F = int(round(rec.B_mhz * 700)); L = fine_window_len(fs)
        z = complex_stft(load_iq(rec.bin_path), fs, L, h, n_ext)
        zc, _ = overlap_bands(z, COARSE_H, fs, F)
        fj, start = pick_fine_start(rec, F, h)
        zf = z[start:start + FINE_H]
        cvals.append(zc.abs().flatten()[::max(1, zc.numel() // 20000)].numpy())
        fvals.append(zf.abs().flatten()[::max(1, zf.numel() // 20000)].numpy())
        print("cal", rec.sample_id, flush=True)
    scale_c = float(np.percentile(np.concatenate(cvals), 99.5))
    scale_f = float(np.percentile(np.concatenate(fvals), 99.5))
    print(f"scales coarse={scale_c:.6g} fine={scale_f:.6g}", flush=True)

    for split, recs in [("fit", fit_sel), ("val", val_sel)]:
        n = len(recs)
        coarse = np.lib.format.open_memmap(out / f"{split}_coarse.npy", mode="w+",
                                           dtype=np.float16,
                                           shape=(n, 3, COARSE_H, N_TIME))
        fine = np.lib.format.open_memmap(out / f"{split}_fine.npy", mode="w+",
                                         dtype=np.float16,
                                         shape=(n, 3, FINE_H, N_TIME))
        metas = []
        t0 = time.perf_counter()
        for i, rec in enumerate(recs):
            meta, c, f = build_one(rec, lo, hi, scale_c, scale_f)
            coarse[i] = c
            fine[i] = f
            metas.append(meta)
            if (i + 1) % 20 == 0 or i + 1 == n:
                print(f"{split} {i+1}/{n} {time.perf_counter()-t0:.0f}s", flush=True)
        coarse.flush(); fine.flush()
        (out / f"{split}_meta.json").write_text(json.dumps({
            "normalizer": norm, "scale_coarse": scale_c,
            "scale_fine": scale_f, "samples": metas,
        }, indent=2))
    (out / "subset.json").write_text(json.dumps({
        "fit": [r.sample_id for r in fit_sel],
        "val": [r.sample_id for r in val_sel],
        "seed": args.seed,
    }, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
