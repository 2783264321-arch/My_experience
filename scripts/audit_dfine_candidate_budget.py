"""Audit D-FINE candidate coverage and budgets with one frozen val pass."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from torchvision.ops import box_convert

from dfine_query_ablation import CKPTS, ROOT, load_model
from src.signals.data import N_TIME, SPLIT_MANIFEST, load_manifest
from src.signals.infer import (
    Prediction,
    _boundary_mask,
    _map_view_dets,
    filter_fine_classes,
    global_classwise_nms,
    gt_global,
)
from src.signals.metrics import ImagePredictions, _match_single, iou_matrix
from src.signals.validate import FullViewDataset, _collate_one
from validate_dual import FINE_0_5, _eval_set


BUDGETS = (100, 300, 600)
THRESHOLDS = (0.5, 0.75, 0.9, 0.95)


def views(loader, records):
    for item in loader:
        index = item["index"]
        rows = round(records[index].B_mhz * 700)
        for image, kind, start in zip(item["tensors"], item["view_types"], item["starts"]):
            yield index, image, kind, start, rows


def decode_variants(model, imgs):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(imgs)
        sizes = torch.tensor([[512, 2048]] * len(imgs), device=imgs.device)
        raw = box_convert(outputs["pred_boxes"], "cxcywh", "xyxy")
        raw = raw * sizes.repeat(1, 2).unsqueeze(1)
        post = model.postprocessor
        previous_budget = post.num_top_queries
        try:
            candidates = {}
            for budget in BUDGETS:
                post.num_top_queries = budget
                results = post(outputs, sizes)
                candidates[budget] = [
                    torch.cat((r["boxes"], r["scores"][:, None], r["labels"][:, None]), dim=1)
                    [r["scores"] >= 0.001].float()
                    for r in results
                ]
        finally:
            post.num_top_queries = previous_budget
    return raw.float().cpu(), candidates


def concat(parts):
    return Prediction(
        torch.cat([p.boxes for p in parts]),
        torch.cat([p.scores for p in parts]),
        torch.cat([p.labels for p in parts]),
    )


def coverage_summary(gt, boxes, n_gt, observed, stage, labels=None):
    ious = iou_matrix(gt.boxes.numpy(), boxes)
    gt_labels = gt.labels.numpy()
    if labels is not None:
        ious = np.where(gt_labels[:, None] == labels[None, :], ious, 0)
    for threshold in THRESHOLDS:
        feasible = ious >= threshold
        hit = feasible.any(axis=1)
        # The one-to-one count is class-agnostic when labels is None.
        unique = 0
        if labels is None:
            gi, pi = linear_sum_assignment(feasible, maximize=True)
            unique = int(feasible[gi, pi].sum())
        slot = observed[stage][str(threshold)]
        slot["gt"] += len(gt_labels)
        slot["one_to_one_matches"] += unique if labels is None else 0
        for cls in range(14):
            slot["each_gt_covered_by_class"][cls] += int((hit & (gt_labels == cls)).sum())
    if stage == "all_300_query_boxes":
        for cls in gt_labels:
            n_gt[int(cls)] += 1


def oracle_labels(gt, pred):
    """Diagnostic only. Change matched labels, keeping boxes and scores fixed."""
    labels = pred.labels.copy()
    matches, _ = _match_single(gt, pred, 0.5, class_aware=False, max_dets=100)
    for pi, gi in matches:
        labels[pi] = gt.labels[gi]
    return ImagePredictions(pred.boxes, pred.scores, labels)


def predict(args, records, gts):
    model, _, state = load_model(args.checkpoint, "cuda")
    cfg = json.loads((ROOT / "configs/normalizer.json").read_text())
    dataset = FullViewDataset(records, (cfg["lo"], cfg["hi"]), view_mode="all")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                        collate_fn=_collate_one, prefetch_factor=4 if args.workers else None,
                        persistent_workers=args.workers > 0)
    raw_boxes = defaultdict(list)
    decoded300 = defaultdict(list)
    parts = {k: defaultdict(list) for k in BUDGETS}
    n_views = 0
    stream = views(loader, records)
    while chunk := list(islice(stream, 8)):
        imgs = torch.stack([item[1] for item in chunk]).cuda()
        raw, decoded = decode_variants(model, imgs)
        for j, (index, _, kind, start, rows) in enumerate(chunk):
            placeholder = torch.zeros((len(raw[j]), 2))
            mapped = _map_view_dets(torch.cat((raw[j], placeholder), dim=-1), kind, start, rows)
            raw_boxes[index].append(mapped.boxes)
            for budget in BUDGETS:
                pred = _map_view_dets(decoded[budget][j], kind, start, rows)
                if budget == 300:
                    decoded300[index].append(pred)
                keep = ~_boundary_mask(pred, kind, start, rows)
                pred = Prediction(pred.boxes[keep], pred.scores[keep], pred.labels[keep])
                parts[budget][index].append(filter_fine_classes(pred, kind, FINE_0_5))
        n_views += len(chunk)

    coverage = defaultdict(lambda: defaultdict(lambda: {
        "gt": 0, "one_to_one_matches": 0, "each_gt_covered_by_class": [0] * 14,
    }))
    by_class = [0] * 14
    predictions = {k: [] for k in BUDGETS}
    for i, gt in enumerate(gts):
        raw = torch.cat(raw_boxes.pop(i)).numpy()
        coverage_summary(gt, raw, by_class, coverage, "all_300_query_boxes")
        dec = concat(decoded300.pop(i))
        coverage_summary(gt, dec.boxes.numpy(), by_class, coverage, "decoded_k300_conf001")
        coverage_summary(gt, dec.boxes.numpy(), by_class, coverage,
                         "decoded_k300_conf001_matching_class", dec.labels.numpy())
        for budget in BUDGETS:
            premerge = concat(parts[budget].pop(i))
            final = global_classwise_nms(premerge, iou=0.5, max_det=100)
            if budget == 300:
                coverage_summary(gt, premerge.boxes.numpy(), by_class, coverage, "k300_premerge")
                coverage_summary(gt, final.boxes.numpy(), by_class, coverage, "k300_merged")
            predictions[budget].append(ImagePredictions(final.boxes.numpy(), final.scores.numpy(),
                                                       final.labels.numpy()))

    serializable = {stage: dict(rows) for stage, rows in coverage.items()}
    for stage in serializable:
        if stage.endswith("matching_class"):
            for row in serializable[stage].values():
                del row["one_to_one_matches"]
    return predictions, serializable, by_class, n_views, state.get("epoch")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=CKPTS["round25"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=0, help="optional local smoke limit")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    records = load_manifest(SPLIT_MANIFEST, split="val")
    if args.max_samples:
        records = records[:args.max_samples]
    gts = [gt_global(r, math.ceil(r.N / (N_TIME - 1))) for r in records]
    started = time.perf_counter()
    preds, coverage, n_gt, n_views, epoch = predict(args, records, gts)
    predict_s = time.perf_counter() - started
    oracle = [oracle_labels(gt, p) for gt, p in zip(gts, preds[300])]
    metrics = {budget: _eval_set(gts, pred) for budget, pred in preds.items()}
    results = {
        "split": "val", "samples": len(records), "views": n_views,
        "checkpoint": str(args.checkpoint), "checkpoint_epoch_zero_based": epoch,
        "batch_views": 8, "conf": 0.001, "nms_iou": 0.5, "max_detections": 100,
        "predict_and_merge_s": predict_s, "n_gt_by_class": n_gt,
        "candidate_coverage": coverage, "budget_metrics": metrics,
        "relabeled_k300_metrics": _eval_set(gts, oracle),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: [v["map50"], v["map50_95"]] for k, v in metrics.items()}))


if __name__ == "__main__":
    main()
