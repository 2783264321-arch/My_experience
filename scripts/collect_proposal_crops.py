#!/usr/bin/env python3
"""Collect coarse-view proposal crops from fit for a 0-5 proposal classifier.

Freezes best_corrected.pt. For each selected fit sample:
  - build the coarse view (STFT + full-band aggregation);
  - run the detector on the coarse view (per-view NMS 300);
  - match each predicted box to its best GT at IoU>=0.5;
  - keep up to N boxes per GT whose GT class is 0-5;
  - crop the coarse view around the predicted box + 10% context, resize to 128x256.

No fine views, no GT boxes for proposals; training labels are GT classes.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.signals.data import SPLIT_MANIFEST, build_sample_views, load_manifest
from src.signals.infer import _map_view_dets, _nms_view_batch, gt_global
from src.signals.metrics import iou_matrix
from src.signals.model import build_model

ROOT = Path(__file__).resolve().parents[1]


class CoarseOnly(Dataset):
    def __init__(self, records, normalizer):
        self.records = list(records)
        self.normalizer = normalizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        coarse, _, meta = build_sample_views(self.records[idx], self.normalizer, [])
        return {"tensor": coarse.tensor, "index": idx, "meta": meta}


def crop_coarse(tensor, box, margin=0.1, size=(128, 256)):
    """tensor [1,2048,512]; box normalized global xyxy."""
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(1e-4, x2 - x1)
    h = max(1e-4, y2 - y1)
    x1 = max(0.0, x1 - margin * w)
    x2 = min(1.0, x2 + margin * w)
    y1 = max(0.0, y1 - margin * h)
    y2 = min(1.0, y2 + margin * h)
    px1 = max(0, min(511, int(x1 * 512)))
    px2 = max(px1 + 1, min(512, int(np.ceil(x2 * 512))))
    py1 = max(0, min(2047, int(y1 * 2048)))
    py2 = max(py1 + 1, min(2048, int(np.ceil(y2 * 2048))))
    crop = tensor[0, py1:py2, px1:px2]
    if crop.numel() == 0:
        return None
    out = F.interpolate(
        crop[None, None].float(), size=size, mode="bilinear", align_corners=False,
    )[0, 0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="yolo11s_c1_directreg_2048x512")
    ap.add_argument("--checkpoint", default="best_corrected.pt")
    ap.add_argument("--out", default="reports/proposal_crops_fit.npz")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--extra-b20-30", type=int, default=800)
    ap.add_argument("--cap-per-gt", type=int, default=3)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    norm_cfg = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm_cfg["lo"]), float(norm_cfg["hi"]))
    fit = load_manifest(SPLIT_MANIFEST, split="fit")
    base = [r for r in fit if r.B_mhz >= 40]
    pool = [r for r in fit if r.B_mhz in (20.0, 30.0)]
    rng = random.Random(1729)
    extra = rng.sample(pool, min(args.extra_b20_30, len(pool)))
    records = sorted(base + extra, key=lambda r: r.sample_id)
    if args.max_samples:
        records = records[: args.max_samples]
    print(f"fit collection: {len(records)} samples (B>=40 {len(base)}, B20/30 {len(extra)})", flush=True)

    run = ROOT / "runs" / args.run
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = run / ckpt
    model = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["ema"] if ck.get("ema") is not None else ck["model"])
    model.eval()

    ds = CoarseOnly(records, norm)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=args.workers,
        collate_fn=lambda b: b[0], pin_memory=False,
        prefetch_factor=4 if args.workers else None,
        persistent_workers=args.workers > 0,
    )
    crops, labels, sample_ids, pred_classes, scores, ious, boxes, gt_boxes = [], [], [], [], [], [], [], []
    started = time.perf_counter()
    for ni, item in enumerate(loader):
        rec = records[item["index"]]
        F = int(round(rec.B_mhz * 700))
        tensor = item["tensor"]
        images = tensor.unsqueeze(0).cuda()
        dets = _nms_view_batch(model, images, conf=0.001, iou=0.5, max_det=300, multi_label=False)
        pred = _map_view_dets(dets[0], "coarse", 0, F)
        if len(pred.boxes) == 0:
            continue
        h = (rec.N + 510) // 511
        gt = gt_global(rec, h)
        if len(gt.boxes) == 0:
            continue
        iou = iou_matrix(gt.boxes, pred.boxes.numpy())
        by_gt = {}
        for pi in range(len(pred.boxes)):
            gi = int(np.argmax(iou[:, pi]))
            best = float(iou[gi, pi])
            if best < 0.5 or int(gt.labels[gi]) > 5:
                continue
            by_gt.setdefault(gi, []).append((float(pred.scores[pi]), pi, best))
        for gi, cand in by_gt.items():
            cand.sort(key=lambda x: (-x[0], -x[2]))
            for score, pi, best in cand[: args.cap_per_gt]:
                crop = crop_coarse(
                    tensor, pred.boxes[pi].numpy(), margin=args.margin,
                    size=(args.height, args.width),
                )
                if crop is None:
                    continue
                crops.append(crop.numpy().astype(np.float16))
                labels.append(int(gt.labels[gi]))
                sample_ids.append(rec.sample_id)
                pred_classes.append(int(pred.labels[pi]))
                scores.append(float(pred.scores[pi]))
                ious.append(best)
                boxes.append([float(x) for x in pred.boxes[pi].numpy()])
                gt_boxes.append([float(x) for x in gt.boxes[gi].numpy()])
        if (ni + 1) % 100 == 0 or ni + 1 == len(records):
            print(f"{ni+1}/{len(records)} samples, crops={len(crops)}, "
                  f"{time.perf_counter()-started:.0f}s", flush=True)

    if not crops:
        raise SystemExit("no crops collected")
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        crops=np.stack(crops).astype(np.float16),
        labels=np.array(labels, dtype=np.int64),
        sample_ids=np.array(sample_ids),
        pred_classes=np.array(pred_classes, dtype=np.int64),
        scores=np.array(scores, dtype=np.float32),
        ious=np.array(ious, dtype=np.float32),
        boxes=np.array(boxes, dtype=np.float32),
        gt_boxes=np.array(gt_boxes, dtype=np.float32),
        meta=np.array([json.dumps({
            "margin": args.margin, "height": args.height, "width": args.width,
            "checkpoint": str(ckpt), "cap_per_gt": args.cap_per_gt,
        })]),
    )
    counts = {str(c): int((np.array(labels) == c).sum()) for c in range(6)}
    wrong = int((np.array(labels) != np.array(pred_classes)).sum())
    print(json.dumps({
        "samples": len(records), "crops": len(crops),
        "class_counts": counts,
        "wrong_original_class": wrong,
        "crop_shape": list(np.stack(crops).shape[1:]),
        "seconds": time.perf_counter() - started,
    }, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
