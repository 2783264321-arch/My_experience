#!/usr/bin/env python3
"""Collect same-frame power baselines for Baseline-3 round 1.

For every frame in reports/local_iq_round1/frames.npz (GT box, both families)
this rebuilds the two power representations at exactly the frame's GT-box
geometry + 10% physical margin:

  coarse      : the existing 0.7 ms full-band coarse view (normalized lo/hi),
                cropped at the GT box + 10% margin, resized to 128x256 --
                the same recipe as collect_proposal_crops.crop_coarse.
  local_power : log power of the local 0.1 ms complex STFT over the same
                GT-box band/time + 10% margin -- the same recipe as
                collect_local_proposal_crops.local_stft_crop.

The published separability-study crops used detector proposal boxes; those
numbers stay as literature reference. The paired comparison in round 1 uses
these GT-geometry crops so that IQ features and power baselines describe
exactly the same region.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.signals.data import SPLIT_MANIFEST, build_sample_views, load_manifest
from scripts.collect_local_proposal_crops import local_stft_crop, load_iq_segment
from scripts.collect_proposal_crops import crop_coarse

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "reports/local_iq_round1/frames.npz"


def gt_box_norm(rec, f1, f2, t1, t2):
    """GT box as normalized global xyxy, same convention as the pipeline."""
    fs = rec.B_mhz * 1e6
    h = math.ceil(rec.N / 511)
    x1 = ((t1 / 1000.0) * fs / h + 0.5) / 512
    x2 = ((t2 / 1000.0) * fs / h + 0.5) / 512
    y1 = (f1 - rec.f_lo) / rec.B_mhz
    y2 = (f2 - rec.f_lo) / rec.B_mhz
    return [min(1.0, max(0.0, x1)), min(1.0, max(0.0, y1)),
            min(1.0, max(0.0, x2)), min(1.0, max(0.0, y2))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/local_iq_round1/power_reps.npz")
    ap.add_argument("--win-ms", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args()

    z = np.load(FRAMES)
    sid = z["sample_id"].astype(str)
    labels = z["label"].astype(np.int64)
    n_frames = len(sid)
    order = np.argsort(sid, kind="stable")

    fit = load_manifest(SPLIT_MANIFEST, split="fit")
    by_id = {r.sample_id: r for r in fit}
    norm_cfg = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm_cfg["lo"]), float(norm_cfg["hi"]))

    coarse_reps, power_reps = [], []
    keep = []
    started = time.perf_counter()
    done = 0
    cur = None
    tensor = None
    for rank, fi in enumerate(order, 1):
        s = str(sid[fi])
        rec = by_id.get(s)
        if rec is None:
            continue
        if s != cur:
            cur = s
            coarse_view, _, _ = build_sample_views(rec, norm, [])
            tensor = coarse_view.tensor
        f1 = float(z["f1_mhz"][fi]); f2 = float(z["f2_mhz"][fi])
        t1 = float(z["t1_ms"][fi]); t2 = float(z["t2_ms"][fi])
        box = gt_box_norm(rec, f1, f2, t1, t2)
        c = crop_coarse(tensor, box, margin=args.margin)
        if c is None:
            continue
        # local geometry: GT box + 10% margin (mirrors crop_coarse semantics)
        dur = max(t2 - t1, 1e-6)
        bw = max(f2 - f1, 1e-6)
        ta = max(0.0, t1 - args.margin * dur)
        tb = min(float(rec.T_ms), t2 + args.margin * dur)
        fa = max(rec.f_lo, f1 - args.margin * bw)
        fb = min(rec.f_hi, f2 + args.margin * bw)
        fs = rec.B_mhz * 1e6
        i0 = max(0, int(round(ta / 1000.0 * fs)))
        i1 = min(int(rec.N), int(round(tb / 1000.0 * fs)))
        if i1 - i0 < 16 or fb <= fa:
            continue
        seg = load_iq_segment(rec, i0, i1)
        _, p = local_stft_crop(seg, rec, fa, fb, args.win_ms, 128, 256)
        if p is None:
            continue
        coarse_reps.append(c.numpy().astype(np.float16))
        power_reps.append(p.astype(np.float16))
        keep.append(fi)
        done += 1
        if rank % args.progress_every == 0:
            print(f"{rank}/{n_frames} kept={done} "
                  f"{time.perf_counter()-started:.0f}s", flush=True)

    keep = np.array(keep, dtype=np.int64)
    out = {
        "coarse": np.stack(coarse_reps).astype(np.float16),
        "local_power": np.stack(power_reps).astype(np.float16),
    }
    for k in z.files:
        out[k] = z[k][keep]
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    counts = np.bincount(labels[keep], minlength=14)
    print(json.dumps({
        "out": str(out_path), "frames": int(len(keep)),
        "class_counts": {str(i): int(c) for i, c in enumerate(counts) if c},
        "seconds": round(time.perf_counter() - started, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
