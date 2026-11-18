"""Full-sample validation: all fine slices + coarse view per source, strict mAP@0.5.

The validator is deliberately independent of Ultralytics' COCO validator: the
fitness returned here is the class-aware mAP@0.5 over complete source samples,
which is what stage 4 requires for best-checkpoint selection.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import N_TIME, SampleRecord, build_sample_views, frequency_starts
from .infer import (
    Prediction, _boundary_mask, _map_view_dets, _nms_view_batch,
    filter_fine_classes, global_classwise_nms, gt_global,
)
from .metrics import ImagePredictions, ImageTargets, evaluate


class FullViewDataset(Dataset):
    """One item = every requested view of one source sample, computed on the fly.

    view_mode="all" keeps the delivered coarse + all fine slices. view_mode=
    "coarse" builds only the full-band view, which is the deployment input of
    the Group-A single-image baseline.
    """

    def __init__(self, records: Sequence[SampleRecord], normalizer: tuple[float, float],
                 view_mode: str = "all"):
        if view_mode not in ("all", "coarse"):
            raise ValueError(f"unsupported view_mode: {view_mode!r}")
        self.records = list(records)
        self.normalizer = normalizer
        self.view_mode = view_mode

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        if self.view_mode == "coarse":
            coarse, _, meta = build_sample_views(record, self.normalizer, [])
            views = [coarse]
        else:
            n_fine = len(frequency_starts(int(round(record.B_mhz * 700))))
            coarse, fines, meta = build_sample_views(
                record, self.normalizer, list(range(n_fine))
            )
            views = [coarse] + fines
        return {
            "index": idx,
            "tensors": [v.tensor for v in views],
            "view_types": [v.view_type for v in views],
            "starts": [v.start for v in views],
            "meta": meta,
        }


def _collate_one(batch):
    return batch[0]


@dataclass
class ValResult:
    map50: float
    map50_agnostic: float
    localization_recall: Optional[float]
    classification_accuracy: Optional[float]
    n_gt: int
    n_pred: int
    per_class: Dict[str, dict]
    confusion: List[List[int]]
    stats: Dict[str, float]
    elapsed_s: float


def _run_view_chunk(
    model,
    chunk: List[dict],
    batch_views: int,
    conf: float,
    iou: float,
    max_det_view: int,
    multi_label: bool,
    drop_fine_classes: list[int] | None = None,
) -> Dict[int, Prediction]:
    """NMS a chunk of views, remap coordinates and drop internal-boundary boxes."""
    images = torch.stack([b["tensor"] for b in chunk], 0).cuda(non_blocking=True)
    dets = _nms_view_batch(
        model, images, conf=conf, iou=iou, max_det=max_det_view,
        multi_label=multi_label,
    )
    grouped: Dict[int, List[Prediction]] = {}
    n_boundary = 0
    for b, det in zip(chunk, dets):
        F = b["F"]
        pred = _map_view_dets(det, b["view_type"], b["start"], F)
        bm = _boundary_mask(pred, b["view_type"], b["start"], F)
        if bm.any():
            n_boundary += int(bm.sum())
            pred = Prediction(pred.boxes[~bm], pred.scores[~bm], pred.labels[~bm])
        pred = filter_fine_classes(pred, b["view_type"], drop_fine_classes)
        grouped.setdefault(b["index"], []).append(pred)
    for idx in grouped:
        parts = grouped[idx]
        grouped[idx] = Prediction(
            torch.cat([p.boxes for p in parts]),
            torch.cat([p.scores for p in parts]),
            torch.cat([p.labels for p in parts]),
        )
    return grouped, n_boundary


def run_validation(
    model,
    records: Sequence[SampleRecord],
    normalizer: tuple[float, float],
    batch_views: int = 8,
    workers: int = 4,
    conf: float = 0.001,
    iou: float = 0.5,
    max_det_view: int = 300,
    max_det: int = 100,
    multi_label: bool = False,
    drop_fine_classes: Optional[Sequence[int]] = None,
    log_interval_s: float = 60.0,
    max_samples: Optional[int] = None,
    view_mode: str = "all",
) -> tuple[ValResult, List[ImageTargets], List[ImagePredictions]]:
    t_start = time.perf_counter()
    was_training = model.training
    model.eval()
    recs = list(records[:max_samples] if max_samples else records)
    ds = FullViewDataset(recs, normalizer, view_mode=view_mode)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=workers,
        collate_fn=_collate_one, pin_memory=False,
        prefetch_factor=4 if workers else None,
        persistent_workers=workers > 0,
    )
    gts = [gt_global(r, math.ceil(r.N / (N_TIME - 1))) for r in recs]
    parts: List[List[Prediction]] = [[] for _ in recs]
    buffer: List[dict] = []
    n_views = 0
    n_boundary = 0
    last_log = time.perf_counter()

    def flush(chunk):
        nonlocal n_boundary
        merged, nb = _run_view_chunk(
            model, chunk, batch_views, conf, iou, max_det_view, multi_label,
            list(drop_fine_classes) if drop_fine_classes else None,
        )
        n_boundary += nb
        for idx, pred in merged.items():
            parts[idx].append(pred)

    for item in loader:
        F = int(round(recs[item["index"]].B_mhz * 700))
        for tensor, vt, st in zip(item["tensors"], item["view_types"], item["starts"]):
            buffer.append({
                "index": item["index"], "tensor": tensor,
                "view_type": vt, "start": st, "F": F,
            })
        while len(buffer) >= batch_views:
            chunk, buffer = buffer[:batch_views], buffer[batch_views:]
            n_views += len(chunk)
            flush(chunk)
        if log_interval_s and time.perf_counter() - last_log > log_interval_s:
            print(f"  val {item['index'] + 1}/{len(recs)} samples, "
                  f"{n_views} views, {time.perf_counter() - t_start:.0f}s", flush=True)
            last_log = time.perf_counter()
    if buffer:
        n_views += len(buffer)
        flush(buffer)

    preds: List[ImagePredictions] = []
    for i, plist in enumerate(parts):
        if plist:
            cat = Prediction(
                torch.cat([p.boxes for p in plist]),
                torch.cat([p.scores for p in plist]),
                torch.cat([p.labels for p in plist]),
            )
            final = global_classwise_nms(cat, iou=iou, max_det=max_det)
        else:
            final = Prediction(torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long))
        preds.append(ImagePredictions(
            final.boxes.numpy(), final.scores.numpy(), final.labels.numpy(),
        ))
    n_pred = int(sum(len(p.boxes) for p in preds))
    result = evaluate(gts, preds, iou_thresholds=(0.5,), max_dets=max_det)
    per_iou = result["per_iou"]["0.5"]
    elapsed = time.perf_counter() - t_start
    vr = ValResult(
        map50=float(per_iou["map_class_aware"] or 0.0),
        map50_agnostic=float(per_iou["ap_class_agnostic"] or 0.0),
        localization_recall=result["classification"]["localization_recall"],
        classification_accuracy=result["classification"]["classification_accuracy"],
        n_gt=int(per_iou["n_gt"]),
        n_pred=n_pred,
        per_class=per_iou["per_class"],
        confusion=result["classification"]["confusion_gt_x_pred"],
        stats={"n_views": n_views, "n_samples": len(recs),
               "n_boundary_dropped": n_boundary, "view_mode": view_mode},
        elapsed_s=elapsed,
    )
    if was_training:
        model.train()
    return vr, gts, preds
