"""Formal dual-protocol validation for YOLO26l: unfiltered + fine-0-5 filtered.

One inference pass; merge/evaluate twice. Reports mAP@0.5, mAP@0.5:0.95,
class-agnostic AP, recall, classification accuracy, and WiFi/SRRC groups.
"""
from __future__ import annotations

import json, math, time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.signals.data import N_TIME, build_sample_views, frequency_starts
from src.signals.infer import (Prediction, _boundary_mask, _map_view_dets,
                               _nms_view_batch, filter_fine_classes,
                               global_classwise_nms, gt_global)
from src.signals.metrics import ImagePredictions, ImageTargets, evaluate
from src.signals.validate import FullViewDataset, _collate_one

FINE_0_5 = [0, 1, 2, 3, 4, 5]
ALL_IOU = tuple(np.round(np.arange(0.5, 1.0, 0.05), 2))
GROUPS = {
    "WiFi_0-5": list(range(6)),
    "WiFi_20MHz_0-2": [0, 1, 2],
    "WiFi_40MHz_3-5": [3, 4, 5],
    "SRRC_10-11": [10, 11],
    "Narrow_9_12_13": [9, 12, 13],
}


def _ap(all_result, thr):
    return float(all_result["per_iou"][str(thr)]["map_class_aware"] or 0.0)


def _eval_set(gts, preds):
    r05 = evaluate(gts, preds, iou_thresholds=(0.5,), max_dets=100)
    r75 = evaluate(gts, preds, iou_thresholds=(0.75,), max_dets=100)
    rfull = evaluate(gts, preds, iou_thresholds=ALL_IOU, max_dets=100)
    map50 = float(r05["per_iou"]["0.5"]["map_class_aware"] or 0.0)
    map50_95 = float(np.mean([_ap(rfull, t) for t in ALL_IOU]))
    out = {
        "map50": map50,
        "map50_95": map50_95,
        "ap50_agnostic": float(r05["per_iou"]["0.5"]["ap_class_agnostic"] or 0.0),
        "ap75_agnostic": float(r75["per_iou"]["0.75"]["ap_class_agnostic"] or 0.0),
        "localization_recall": r05["classification"]["localization_recall"],
        "classification_accuracy": r05["classification"]["classification_accuracy"],
        "n_gt": int(r05["per_iou"]["0.5"]["n_gt"]),
        "n_pred": int(sum(len(p.boxes) for p in preds)),
        "per_class": r05["per_iou"]["0.5"]["per_class"],
        "confusion": r05["classification"]["confusion_gt_x_pred"],
        "groups": {},
    }
    for name, classes in GROUPS.items():
        keep = set(classes)
        g = [ImageTargets(gt.boxes[np.isin(gt.labels, list(keep))],
                          gt.labels[np.isin(gt.labels, list(keep))]) for gt in gts]
        # class-agnostic localization: keep ALL predictions, restrict only GT.
        # Filtering predictions by group class before matching would count a
        # correct box with a wrong (out-of-group) class as a localization miss.
        rg_all = evaluate(g, preds, iou_thresholds=(0.5, 0.75), max_dets=100)
        # strict group AP still uses only the group's predicted classes
        p_grp = [ImagePredictions(pr.boxes[np.isin(pr.labels, list(keep))],
                                  pr.scores[np.isin(pr.labels, list(keep))],
                                  pr.labels[np.isin(pr.labels, list(keep))]) for pr in preds]
        rg = evaluate(g, p_grp, iou_thresholds=(0.5,), max_dets=100)
        rf = evaluate(g, p_grp, iou_thresholds=ALL_IOU, max_dets=100)
        out["groups"][name] = {
            "n_gt": int(rg["per_iou"]["0.5"]["n_gt"]),
            "map50": float(rg["per_iou"]["0.5"]["map_class_aware"] or 0.0),
            "map50_95": float(np.mean([_ap(rf, t) for t in ALL_IOU])),
            "ap50_agnostic": float(rg_all["per_iou"]["0.5"]["ap_class_agnostic"] or 0.0),
            "ap75_agnostic": float(rg_all["per_iou"]["0.75"]["ap_class_agnostic"] or 0.0),
            "recall": rg_all["classification"]["localization_recall"],
            "recall_strict_predclass": rg["classification"]["localization_recall"],
        }
    return out


def run_dual(model, records: Sequence, normalizer, batch_views=8, workers=8,
             conf=0.001, iou=0.5, max_det_view=300, max_det=100,
             max_samples: Optional[int] = None, log_interval_s=120.0,
             infer_fn=None, nms_method="nms", soft_sigma=0.5):
    t0 = time.perf_counter()
    recs = list(records[:max_samples] if max_samples else records)
    ds = FullViewDataset(recs, normalizer, view_mode="all")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=workers,
                        collate_fn=_collate_one, pin_memory=False,
                        prefetch_factor=4 if workers else None,
                        persistent_workers=workers > 0)
    gts = [gt_global(r, math.ceil(r.N / (N_TIME - 1))) for r in recs]
    # per-view predictions (mapped, boundary filtered), grouped by sample
    parts: List[List[tuple[Prediction, str]]] = [[] for _ in recs]
    buffer: List[dict] = []
    n_views = n_boundary = 0
    was_training = model.training
    model.eval()
    last = time.perf_counter()

    def flush(chunk):
        nonlocal n_boundary
        images = torch.stack([b["tensor"] for b in chunk], 0).cuda(non_blocking=True)
        if infer_fn is None:
            dets = _nms_view_batch(model, images, conf=conf, iou=iou,
                                   max_det=max_det_view, multi_label=False)
        else:
            dets = infer_fn(images)
        for b, det in zip(chunk, dets):
            p = _map_view_dets(det, b["view_type"], b["start"], b["F"])
            bm = _boundary_mask(p, b["view_type"], b["start"], b["F"])
            if bm.any():
                n_boundary += int(bm.sum())
                p = Prediction(p.boxes[~bm], p.scores[~bm], p.labels[~bm])
            parts[b["index"]].append((p, b["view_type"]))

    for item in loader:
        F = int(round(recs[item["index"]].B_mhz * 700))
        for tensor, vt, st in zip(item["tensors"], item["view_types"], item["starts"]):
            buffer.append({"index": item["index"], "tensor": tensor,
                           "view_type": vt, "start": st, "F": F})
        while len(buffer) >= batch_views:
            chunk, buffer = buffer[:batch_views], buffer[batch_views:]
            n_views += len(chunk); flush(chunk)
        if log_interval_s and time.perf_counter() - last > log_interval_s:
            print(f"  val {item['index']+1}/{len(recs)} samples, {n_views} views, "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)
            last = time.perf_counter()
    if buffer:
        n_views += len(buffer); flush(buffer)

    def merge(drop):
        preds = []
        for plist in parts:
            m = []
            for p, vt in plist:
                q = filter_fine_classes(p, vt, drop)
                m.append(q)
            cat = Prediction(torch.cat([x.boxes for x in m]),
                             torch.cat([x.scores for x in m]),
                             torch.cat([x.labels for x in m]))
            if nms_method == "soft":
                from src.signals.infer import soft_classwise_nms
                final = soft_classwise_nms(cat, iou=iou, sigma=soft_sigma, max_det=max_det)
            else:
                final = global_classwise_nms(cat, iou=iou, max_det=max_det)
            preds.append(ImagePredictions(final.boxes.numpy(), final.scores.numpy(), final.labels.numpy()))
        return preds

    out = {"n_samples": len(recs), "n_views": n_views,
           "n_boundary_dropped": n_boundary,
           "settings": {}}
    for name, drop in [("unfiltered", None), ("filtered_0_5", FINE_0_5)]:
        preds = merge(drop)
        res = _eval_set(gts, preds)
        res["n_pred"] = int(sum(len(p.boxes) for p in preds))
        out["settings"][name] = res
        out[f"preds_{name}"] = preds
    out["elapsed_s"] = time.perf_counter() - t0
    if was_training:
        model.train()
    return out
