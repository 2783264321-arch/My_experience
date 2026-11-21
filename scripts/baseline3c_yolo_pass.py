#!/usr/bin/env python3
"""Phase C' stage 1: run frozen YOLO on full val, cache per-view predictions.

Same inference protocol as fine_0_5_filter_comparison.py (frozen
best_corrected.pt, conf 0.001, per-view NMS IoU 0.5 max 300, boundary mask,
global class-wise NMS IoU 0.5 max 100), but saves per-sample predictions
with view-source labels so the cascade stage can operate without re-running
the detector.

Output: runs_local/baseline3c/yolo_val_cache.npz with per-sample arrays.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.signals.data import SPLIT_MANIFEST, load_manifest
from src.signals.infer import (
    Prediction, _boundary_mask, _map_view_dets, _nms_view_batch,
    global_classwise_nms, gt_global,
)
from src.signals.model import build_model
from src.signals.validate import FullViewDataset, _collate_one

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "runs_local/baseline3c"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="best_corrected.pt")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch-views", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--max-det-view", type=int, default=300)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    run = ROOT / "runs/yolo11s_c1_directreg_2048x512"
    ckpt = run / args.checkpoint
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    norm_cfg = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm_cfg["lo"]), float(norm_cfg["hi"]))
    records = load_manifest(SPLIT_MANIFEST, split="val")
    if args.max_samples:
        records = records[: args.max_samples]
    model = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["ema"] if ck.get("ema") is not None else ck["model"])
    model.eval().cuda()

    ds = FullViewDataset(records, norm)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=args.workers,
        collate_fn=_collate_one, pin_memory=False,
        prefetch_factor=4 if args.workers else None,
        persistent_workers=args.workers > 0,
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    all_c0 = []          # filtered: coarse + fine(6-13 only), global NMS
    all_both = []         # unfiltered: coarse + all fine, global NMS
    all_gt = []
    meta = []             # per-sample metadata for IQ extraction
    sample_ids = []
    started = time.perf_counter()

    for ni, item in enumerate(loader):
        rec = records[item["index"]]
        F = int(round(rec.B_mhz * 700))
        views = list(zip(item["tensors"], item["view_types"], item["starts"]))
        coarse_parts, fine_parts = [], []

        for begin in range(0, len(views), args.batch_views):
            chunk = views[begin:begin + args.batch_views]
            images = torch.stack([v[0] for v in chunk], 0).cuda()
            dets = _nms_view_batch(
                model, images, conf=args.conf, iou=args.iou,
                max_det=args.max_det_view, multi_label=False,
            )
            for (tensor, vt, st), det in zip(chunk, dets):
                pred = _map_view_dets(det, vt, st, F)
                bm = _boundary_mask(pred, vt, st, F)
                if bm.any():
                    post = Prediction(pred.boxes[~bm], pred.scores[~bm],
                                      pred.labels[~bm])
                else:
                    post = pred
                (coarse_parts if vt == "coarse" else fine_parts).append(post)

        # both: all coarse + all fine
        from scripts.view_source_attribution import cat_parts
        both = global_classwise_nms(
            cat_parts(coarse_parts + fine_parts),
            iou=args.iou, max_det=args.max_det)
        all_both.append((both.boxes.numpy(), both.scores.numpy(),
                         both.labels.numpy()))

        # filtered (C0): remove fine predictions with class 0-5
        fine_kept = []
        for p in fine_parts:
            if len(p.labels) == 0:
                continue
            mask = p.labels > 5
            if mask.any():
                fine_kept.append(Prediction(
                    p.boxes[mask], p.scores[mask], p.labels[mask]))
        c0 = global_classwise_nms(
            cat_parts(coarse_parts + fine_kept),
            iou=args.iou, max_det=args.max_det)
        all_c0.append((c0.boxes.numpy(), c0.scores.numpy(), c0.labels.numpy()))

        # GT
        h = (rec.N + 510) // 511
        gt = gt_global(rec, h)
        all_gt.append((gt.boxes.numpy(), gt.labels.numpy()))

        # metadata for IQ extraction
        sample_ids.append(rec.sample_id)
        meta.append({
            "sample_id": rec.sample_id, "B_mhz": rec.B_mhz,
            "T_ms": rec.T_ms, "N": rec.N, "f_lo": rec.f_lo,
            "f_hi": rec.f_hi,
        })

        if (ni + 1) % 50 == 0 or ni + 1 == len(records):
            n_c0 = sum(len(p[0]) for p in all_c0)
            print(f"{ni+1}/{len(records)} c0_preds={n_c0} "
                  f"{time.perf_counter()-started:.0f}s", flush=True)

    # save
    def pack(arrays, prefix):
        boxes = np.concatenate([a[0] for a in arrays]) if arrays else np.zeros((0, 4))
        scores = np.concatenate([a[1] for a in arrays]) if arrays else np.zeros(0)
        labels = np.concatenate([a[2] for a in arrays]) if arrays else np.zeros(0, dtype=np.int64)
        counts = np.array([len(a[0]) for a in arrays], dtype=np.int64)
        return {f"{prefix}_boxes": boxes, f"{prefix}_scores": scores,
                f"{prefix}_labels": labels, f"{prefix}_counts": counts}

    out = {}
    out.update(pack(all_c0, "c0"))
    out.update(pack(all_both, "both"))
    gt_boxes = np.concatenate([a[0] for a in all_gt]) if all_gt else np.zeros((0, 4))
    gt_labels = np.concatenate([a[1] for a in all_gt]) if all_gt else np.zeros(0, dtype=np.int64)
    gt_counts = np.array([len(a[0]) for a in all_gt], dtype=np.int64)
    out.update({"gt_boxes": gt_boxes, "gt_labels": gt_labels,
                "gt_counts": gt_counts})
    out["sample_ids"] = np.array(sample_ids)
    out["meta"] = np.array([json.dumps(m) for m in meta])
    out["n_samples"] = np.array(len(records))

    path = OUTDIR / "yolo_val_cache.npz"
    np.savez_compressed(path, **out)
    n_c0 = sum(len(a[0]) for a in all_c0)
    n_both = sum(len(a[0]) for a in all_both)
    print(f"\nwrote {path}: {len(records)} samples, "
          f"c0={n_c0}, both={n_both}, "
          f"gt={sum(len(a[0]) for a in all_gt)}")
    print(f"total time: {time.perf_counter()-started:.0f}s")


if __name__ == "__main__":
    main()
