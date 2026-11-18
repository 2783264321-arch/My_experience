#!/usr/bin/env python3
"""Stage-wise class-agnostic recall: pre-boundary, post-boundary, post-NMS, caps.

Answers where the first-ceiling loss occurs:
  R_pre        per-view NMS, before internal-boundary filtering
  R_post       after boundary filtering
  R_nms        after global class-wise NMS (no max-det cap)
  R_final100   after the per-sample 100-box cap

Also reports R at caps 10/50/100/300/1000, and per signal-bandwidth/duration
group. Uses the same model/NMS/coordinate functions as production inference.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.signals.data import SPLIT_MANIFEST, load_manifest
from src.signals.infer import (
    Prediction, _boundary_mask, _map_view_dets, _nms_view_batch,
    global_classwise_nms, gt_global,
)
from src.signals.metrics import ImagePredictions, ImageTargets, _match_single
from src.signals.model import build_model
from src.signals.validate import FullViewDataset, _collate_one
from scripts.analyze_signal_characteristics import (
    DIMS, time_ratio_group, bw_abs_group, bw_rel_group, dur_group,
)

ROOT = Path(__file__).resolve().parents[1]


def cat_parts(parts):
    if not parts:
        return Prediction(torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long))
    return Prediction(
        torch.cat([p.boxes for p in parts]),
        torch.cat([p.scores for p in parts]),
        torch.cat([p.labels for p in parts]),
    )


def match_once(gt, pred, cap):
    gt_obj = ImageTargets(gt.boxes.numpy(), gt.labels.numpy())
    pred_obj = ImagePredictions(pred.boxes.numpy(), pred.scores.numpy(), pred.labels.numpy())
    matches, _ = _match_single(
        gt_obj, pred_obj, 0.5, class_aware=False, max_dets=cap,
    )
    return matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="yolo11s_c1_directreg_2048x512")
    ap.add_argument("--checkpoint", default="best_corrected.pt")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-views", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-det-view", type=int, default=300)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()
    run = ROOT / "runs" / args.run
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = run / ckpt_path
    norm_cfg = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm_cfg["lo"]), float(norm_cfg["hi"]))
    records = load_manifest(SPLIT_MANIFEST, split="val")
    if args.max_samples:
        records = records[: args.max_samples]
    model = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"))
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["ema"] if ck.get("ema") is not None else ck["model"]
    model.load_state_dict(sd)
    model.eval()

    ds = FullViewDataset(records, norm)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=args.workers,
        collate_fn=_collate_one, pin_memory=False,
        prefetch_factor=4 if args.workers else None,
        persistent_workers=args.workers > 0,
    )
    caps = [10, 50, 100, 300, 1000]
    stages = ["pre_boundary", "post_boundary", "post_nms"] + [f"final@{c}" for c in caps]
    overall_gt = 0
    overall_matched = {s: 0 for s in stages}
    n_gt = {d: defaultdict(int) for d in DIMS}
    matched = {s: {d: defaultdict(int) for d in DIMS} for s in stages}
    total_seconds = 0.0
    import time
    for ni, item in enumerate(loader):
        t0 = time.perf_counter()
        rec = records[item["index"]]
        F = int(round(rec.B_mhz * 700))
        views = list(zip(item["tensors"], item["view_types"], item["starts"]))
        pre_parts, post_parts = [], []
        for begin in range(0, len(views), args.batch_views):
            chunk = views[begin:begin + args.batch_views]
            images = torch.stack([v[0] for v in chunk], 0).cuda()
            dets = _nms_view_batch(
                model, images, conf=args.conf, iou=0.5,
                max_det=args.max_det_view, multi_label=False,
            )
            for (tensor, vt, st), det in zip(chunk, dets):
                pred = _map_view_dets(det, vt, st, F)
                bm = _boundary_mask(pred, vt, st, F)
                pre_parts.append(pred)
                if bm.any():
                    post_parts.append(Prediction(pred.boxes[~bm], pred.scores[~bm], pred.labels[~bm]))
                else:
                    post_parts.append(pred)
        pre = cat_parts(pre_parts)
        post = cat_parts(post_parts)
        nms_all = global_classwise_nms(post, iou=0.5, max_det=100000)
        h = (rec.N + 510) // 511
        gt = gt_global(rec, h)
        # GT group labels in signal order
        group_labels = {}
        for d, fn in DIMS.items():
            labels = []
            for s in rec.signals:
                bw = float(s["end_frequency"]) - float(s["start_frequency"])
                dur = float(s["end_time"]) - float(s["start_time"])
                labels.append(fn(bw if d == "abs_bw" else
                                 bw / rec.B_mhz if d == "rel_bw" else
                                 dur if d == "duration" else dur / rec.T_ms))
                n_gt[d][labels[-1]] += 1
            group_labels[d] = labels
        overall_gt += len(gt.boxes)
        stage_preds = {
            "pre_boundary": (pre, 100000),
            "post_boundary": (post, 100000),
            "post_nms": (nms_all, 100000),
        }
        for cap in caps:
            stage_preds[f"final@{cap}"] = (Prediction(
                nms_all.boxes[:cap], nms_all.scores[:cap], nms_all.labels[:cap]
            ), cap)
        for s, (pred, cap) in stage_preds.items():
            matches = match_once(gt, pred, cap)
            overall_matched[s] += len(matches)
            for pi, gi in matches:
                for d in DIMS:
                    matched[s][d][group_labels[d][gi]] += 1
        total_seconds += time.perf_counter() - t0
        if (ni + 1) % 100 == 0 or ni + 1 == len(records):
            print(f"{ni+1}/{len(records)} samples, {total_seconds:.0f}s", flush=True)
        # early progress for a small test
        if args.max_samples and ni + 1 == len(records):
            break

    out = {"n_samples": len(records), "n_gt": overall_gt, "overall": {}, "dims": {}}
    for s in stages:
        out["overall"][s] = {
            "matched": int(overall_matched[s]),
            "recall": float(overall_matched[s] / overall_gt) if overall_gt else None,
        }
    for d in DIMS:
        out["dims"][d] = {}
        for g in sorted(n_gt[d]):
            out["dims"][d][g] = {
                "n_gt": int(n_gt[d][g]),
                "recall": {
                    s: (float(matched[s][d][g] / n_gt[d][g]) if n_gt[d][g] else None)
                    for s in stages
                },
            }
    path = run / "stage5_val" / "stage_recall.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "n_gt": overall_gt,
        "overall": out["overall"],
        "abs_bw": {g: v["recall"] for g, v in out["dims"]["abs_bw"].items()},
        "duration": {g: v["recall"] for g, v in out["dims"]["duration"].items()},
    }, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
