#!/usr/bin/env python3
"""Collect aligned local time-frequency crops for 0-5 modulation separability.

The frozen detector's fit proposal boxes are reused from
`reports/proposal_crops_fit.npz`. For each unique, accurately localized
proposal (one best crop per GT box, IoU >= --iou-min) this script builds:

  coarse       : the existing 0.7 ms coarse power crop (128x256)
  local_power  : log power of a local complex STFT crop
  local_complex: real/imag of the *same* local complex STFT crop

The local power and complex arrays come from exactly the same selected
complex coefficients and the same [128, 256] grid. The only difference is
the representation: |z|^2 in log scale vs. Re(z)/Im(z).

Local geometry:
  - time window: proposal time box + 10% of its duration
  - frequency band: proposal frequency box + 10% of its bandwidth
  - STFT window: --win-ms (default 0.1 ms), periodic Hann
  - local grid: 128 frequency rows x 256 time columns

Normalization:
  - local_complex: global phase removed, then one RMS scale per crop
  - local_power: log10 power; the trainer standardizes each crop
  - coarse: stored as in proposal_crops_fit.npz; trainer standardizes

The shared sample-level split is intentionally *not* applied here. All
representations are collected for the same proposal indices so the trainer
can use one split for every representation and every bandwidth group.
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

from src.signals.data import SPLIT_MANIFEST, load_iq, load_manifest

ROOT = Path(__file__).resolve().parents[1]


def dedup_best(z, iou_min: float):
    """One highest-score crop per (sample_id, GT box), then IoU filter."""
    sid = z["sample_ids"].astype(str)
    gt = z["gt_boxes"]
    iou = z["ious"]
    score = z["scores"]
    best = {}
    for i, (s, b) in enumerate(zip(sid, gt)):
        key = (s, tuple(np.round(b, 6)))
        if key not in best or (score[i], iou[i]) > (score[best[key]], iou[best[key]]):
            best[key] = i
    idx = np.array(sorted(best.values()), dtype=np.int64)
    idx = idx[iou[idx] >= iou_min]
    return idx


def signal_boxes(rec):
    """Exact normalized GT boxes and physical boxes, same convention as gt_global."""
    h = math.ceil(rec.N / 511)
    boxes = []
    for sig in rec.signals:
        x1 = ((sig["start_time"] / 1000.0) * (rec.B_mhz * 1e6) / h + 0.5) / 512
        x2 = ((sig["end_time"] / 1000.0) * (rec.B_mhz * 1e6) / h + 0.5) / 512
        y1 = (sig["start_frequency"] - rec.f_lo) / rec.B_mhz
        y2 = (sig["end_frequency"] - rec.f_lo) / rec.B_mhz
        boxes.append([min(1.0, max(0.0, x1)), min(1.0, max(0.0, y1)),
                      min(1.0, max(0.0, x2)), min(1.0, max(0.0, y2))])
    return boxes


def overlap_flags(rec, gt_box, gt_label):
    """Return the GT instance id and time-frequency overlap flags for one box."""
    boxes = signal_boxes(rec)
    if not boxes:
        return -1, 0, 0, 0, 0, 0.0, 0.0
    dists = [float(np.abs(np.asarray(b) - gt_box).sum()) for b in boxes]
    gi = int(np.argmin(dists))
    if dists[gi] > 1e-3:
        # Should not happen for proposal_crops_fit.npz, but keep a safe fallback.
        cand = [i for i, s in enumerate(rec.signals) if int(s["class"]) == int(gt_label)]
        if not cand:
            return -1, 0, 0, 0, 0, 0.0, 0.0
        gi = int(min(cand, key=lambda i: dists[i]))
    s = rec.signals[gi]
    area = max(0.0, float(s["end_time"]) - float(s["start_time"])) * \
           max(0.0, float(s["end_frequency"]) - float(s["start_frequency"]))
    any_ov = wifi_ov = wifi_cross = wifi_same = 0
    any_frac = 0.0
    wifi_frac = 0.0
    for j, o in enumerate(rec.signals):
        if j == gi:
            continue
        inter_t = max(0.0, min(float(s["end_time"]), float(o["end_time"])) -
                      max(float(s["start_time"]), float(o["start_time"])))
        inter_f = max(0.0, min(float(s["end_frequency"]), float(o["end_frequency"])) -
                      max(float(s["start_frequency"]), float(o["start_frequency"])))
        inter = inter_t * inter_f
        if inter <= 0.0:
            continue
        frac = inter / area if area > 0 else 0.0
        any_ov = 1
        any_frac = max(any_frac, frac)
        if int(o["class"]) in range(6):
            wifi_ov = 1
            wifi_frac = max(wifi_frac, frac)
            if int(o["class"]) == int(s["class"]):
                wifi_same = 1
            else:
                wifi_cross = 1
    return (int(s["signal_id"]), any_ov, wifi_ov, wifi_cross, wifi_same,
            float(wifi_frac), float(any_frac))


def local_geometry(rec, box, margin):
    """Return (i0, i1, f1_mhz, f2_mhz) for a proposal box."""
    h = math.ceil(rec.N / 511)
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(1.0, max(0.0, x1)); x2 = min(1.0, max(0.0, x2))
    y1 = min(1.0, max(0.0, y1)); y2 = min(1.0, max(0.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    t1 = (x1 * 512.0 - 0.5) * h
    t2 = (x2 * 512.0 - 0.5) * h
    if t2 <= t1:
        return None
    mt = (t2 - t1) * margin
    t1 = max(0.0, t1 - mt)
    t2 = min(float(rec.N), t2 + mt)
    i0 = max(0, int(round(t1)))
    i1 = min(int(rec.N), int(round(t2)))
    if i1 - i0 < 2:
        return None
    f1 = rec.f_lo + y1 * rec.B_mhz
    f2 = rec.f_lo + y2 * rec.B_mhz
    mf = (f2 - f1) * margin
    f1 = max(rec.f_lo, f1 - mf)
    f2 = min(rec.f_hi, f2 + mf)
    if f2 <= f1:
        return None
    return i0, i1, f1, f2


def load_iq_segment(rec, i0, i1):
    """Read only the needed interleaved float16 I/Q segment from the .bin file."""
    n = max(0, i1 - i0)
    raw = np.fromfile(rec.bin_path, dtype="<f2", count=2 * n, offset=4 * i0)
    m = len(raw) // 2
    iq = np.empty(m, dtype=np.complex64)
    iq.real = raw[0::2][:m]
    iq.imag = raw[1::2][:m]
    return torch.from_numpy(iq)


def local_stft_crop(seg, rec, f1, f2, win_ms, height, width):
    """Return (complex [2,H,W] float32, log power [H,W] float32)."""
    fs = rec.B_mhz * 1e6
    L = int(round(fs * win_ms / 1000.0))
    if L < 8:
        L = 8
    if L % 2:
        L += 1
    if seg.numel() < L:
        seg = F.pad(seg, (0, L - seg.numel()))
    seg_len = int(seg.numel())
    # Uniformly sample about `width` windows across the local segment. This
    # keeps the STFT small for long bursts while preserving time coverage.
    hop = max(1, (seg_len - L) // max(1, width - 1))
    win = torch.hann_window(L, periodic=True, dtype=torch.float32)
    z = torch.stft(
        seg, n_fft=L, hop_length=hop, win_length=L, window=win,
        center=False, pad_mode="constant", normalized=False,
        onesided=False, return_complex=True,
    )
    z = torch.fft.fftshift(z, dim=0)  # [L, T], -fs/2 .. +fs/2

    fc = (rec.f_lo + rec.f_hi) / 2.0
    r1 = int(round((f1 - fc) / rec.B_mhz * L + L / 2.0))
    r2 = int(round((f2 - fc) / rec.B_mhz * L + L / 2.0))
    r1 = max(0, min(L - 2, r1))
    r2 = max(r1 + 2, min(L, r2))
    z = z[r1:r2]
    if z.numel() == 0:
        return None, None

    # Remove the arbitrary global phase (the magnitude is unchanged).
    flat = z.reshape(-1)
    ref = flat[int(torch.argmax(flat.abs()))]
    if float(ref.abs()) > 0:
        z = z * torch.exp(-1j * torch.angle(ref))

    # Real/imag on the shared local grid.
    zz = torch.view_as_real(z).permute(2, 0, 1).unsqueeze(0)  # [1,2,F,T]
    zz = F.interpolate(zz.float(), size=(height, width), mode="bilinear",
                       align_corners=False)[0]
    scale = zz.square().mean().sqrt().clamp_min(1e-8)
    zz = zz / scale

    # Power from the same complex coefficients, same output grid.
    p = z.abs().square()[None, None]
    p = F.interpolate(p.float(), size=(height, width), mode="bilinear",
                      align_corners=False)[0, 0]
    p = torch.log10(p + 1e-12)
    return zz.numpy().astype(np.float32), p.numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", default="reports/proposal_crops_fit.npz")
    ap.add_argument("--out", default="reports/proposal_local_win{win}.npz")
    ap.add_argument("--win-ms", default="0.1",
                    help="comma-separated local STFT windows in ms, e.g. 0.05,0.1,0.2")
    ap.add_argument("--iou-min", type=float, default=0.75)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--max-crops", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()

    prop_path = ROOT / args.proposal if not Path(args.proposal).is_absolute() else Path(args.proposal)
    # Load all arrays once; NpzFile is lazy and re-reading the 500 MB
    # coarse crop array for every proposal would dominate the run.
    with np.load(prop_path) as zf:
        z = {k: zf[k] for k in zf.files}
    idx = dedup_best(z, args.iou_min)
    if args.max_crops:
        idx = idx[: args.max_crops]
    print(f"proposals: {len(z['labels'])} total, {len(idx)} unique IoU>={args.iou_min}",
          flush=True)

    fit = load_manifest(SPLIT_MANIFEST, split="fit")
    by_id = {r.sample_id: r for r in fit}

    wins = [float(v) for v in str(args.win_ms).split(",") if v.strip()]
    if not wins:
        raise SystemExit("no windows requested")
    win_tags = [f"{int(round(w * 1000)):04d}" for w in wins]
    power = {t: [] for t in win_tags}
    comp = {t: [] for t in win_tags}
    coarse, labels, sample_ids, pred_classes, scores, ious = [], [], [], [], [], []
    boxes, gt_boxes, gt_ids = [], [], []
    ov_any, ov_wifi, ov_cross, ov_same = [], [], [], []
    ov_wifi_frac, ov_any_frac = [], []
    started = time.perf_counter()
    done = 0
    skipped = 0
    for ci, i in enumerate(idx, 1):
        sid = str(z["sample_ids"][i])
        rec = by_id.get(sid)
        if rec is None:
            skipped += 1
            continue
        geom = local_geometry(rec, z["boxes"][i], args.margin)
        if geom is None:
            skipped += 1
            continue
        i0, i1, f1, f2 = geom
        seg = load_iq_segment(rec, i0, i1)
        crops = {}
        ok = True
        for w, tag in zip(wins, win_tags):
            c, p = local_stft_crop(
                seg, rec, f1, f2, w, args.height, args.width,
            )
            if c is None:
                ok = False
                break
            crops[tag] = (c, p)
        if not ok:
            skipped += 1
            continue
        gid, oa, ow, oc, os_, owf, oaf = overlap_flags(
            rec, z["gt_boxes"][i], int(z["labels"][i]),
        )
        coarse.append(z["crops"][i].astype(np.float16))
        for tag in win_tags:
            c, p = crops[tag]
            comp[tag].append(c.astype(np.float16))
            power[tag].append(p.astype(np.float16))
        labels.append(int(z["labels"][i]))
        sample_ids.append(sid)
        pred_classes.append(int(z["pred_classes"][i]))
        scores.append(float(z["scores"][i]))
        ious.append(float(z["ious"][i]))
        boxes.append([float(v) for v in z["boxes"][i]])
        gt_boxes.append([float(v) for v in z["gt_boxes"][i]])
        gt_ids.append(gid)
        ov_any.append(oa); ov_wifi.append(ow); ov_cross.append(oc); ov_same.append(os_)
        ov_wifi_frac.append(owf); ov_any_frac.append(oaf)
        done += 1
        if ci % args.progress_every == 0 or ci == len(idx):
            print(f"{ci}/{len(idx)} crops, kept={done}, skipped={skipped}, "
                  f"{time.perf_counter()-started:.0f}s", flush=True)

    if not done:
        raise SystemExit("no crops collected")

    base_out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    base_out.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for w, tag in zip(wins, win_tags):
        if "{win}" in str(base_out):
            out = Path(str(base_out).replace("{win}", tag))
        elif len(wins) == 1:
            out = base_out
        else:
            out = base_out.with_name(base_out.stem + f"_win{tag}" + base_out.suffix)
        np.savez(
            out,
            coarse=np.stack(coarse).astype(np.float16),
            local_power=np.stack(power[tag]).astype(np.float16),
            local_complex=np.stack(comp[tag]).astype(np.float16),
            labels=np.array(labels, dtype=np.int64),
            sample_ids=np.array(sample_ids),
            pred_classes=np.array(pred_classes, dtype=np.int64),
            scores=np.array(scores, dtype=np.float32),
            ious=np.array(ious, dtype=np.float32),
            boxes=np.array(boxes, dtype=np.float32),
            gt_boxes=np.array(gt_boxes, dtype=np.float32),
            gt_signal_ids=np.array(gt_ids, dtype=np.int64),
            overlap_any=np.array(ov_any, dtype=np.int8),
            overlap_wifi=np.array(ov_wifi, dtype=np.int8),
            overlap_wifi_cross=np.array(ov_cross, dtype=np.int8),
            overlap_wifi_same=np.array(ov_same, dtype=np.int8),
            overlap_wifi_frac_max=np.array(ov_wifi_frac, dtype=np.float32),
            overlap_any_frac_max=np.array(ov_any_frac, dtype=np.float32),
            meta=np.array([json.dumps({
                "source": str(prop_path), "win_ms": w,
                "iou_min": args.iou_min, "margin": args.margin,
                "height": args.height, "width": args.width,
                "complex_norm": "global phase removed; per-crop joint RMS",
                "power_norm": "log10(|z|^2); trainer standardizes per crop",
                "coarse_norm": "as in proposal_crops_fit.npz; trainer standardizes",
            })]),
        )
        written.append(str(out))
    counts = np.bincount(np.array(labels), minlength=6)
    print(json.dumps({
        "out": written, "crops": done, "skipped": skipped,
        "samples": len(set(sample_ids)),
        "class_counts": counts.tolist(),
        "overlap_wifi": int(np.sum(ov_wifi)),
        "overlap_any": int(np.sum(ov_any)),
        "win_ms": wins,
        "seconds": time.perf_counter() - started,
    }, indent=2))
    print("wrote", written)


if __name__ == "__main__":
    main()
