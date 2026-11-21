#!/usr/bin/env python3
"""Ignore-aware AP50 for a bandwidth subset (does not modify production eval).

Used by the SRRC Stage 0 audit. For a class c:
  * valid GT   = class c with annotation bandwidth in [9.5, 10.5] MHz
  * ignored GT = class c outside that bandwidth
  * predictions = unchanged YOLO class-c predictions

Matching rule (explicit):
  1. valid GT are matched first, one-to-one, score-sorted, max_dets=100;
  2. a prediction that matched a valid GT is TP;
  3. an unmatched prediction with IoU>=0.5 to ANY ignored GT is excluded
     from both TP and FP (not one-to-one; all such duplicates are ignored);
  4. all other unmatched predictions are FP;
  5. n_gt = number of valid GT; 101-point AP over the kept TP/FP list.

The production `src/signals/metrics.py` is not modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.signals.metrics import (
    ImagePredictions, ImageTargets, _match_single, iou_matrix,
)

IOU_THR = 0.5
N_AP = 101


@dataclass
class IgnoreAPResult:
    ap50: Optional[float]
    n_valid_gt: int
    n_ignored_gt: int
    n_pred: int
    n_tp: int
    n_fp: int
    n_ignored_pred: int
    n_matched_valid: int


def _ap101(tp: np.ndarray, n_gt: int) -> float:
    if n_gt == 0 or len(tp) == 0:
        return 0.0
    tp_cum = np.cumsum(tp.astype(np.float64))
    fp_cum = np.cumsum((~tp).astype(np.float64))
    rec = tp_cum / n_gt
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    ap = 0.0
    for t in np.linspace(0, 1, N_AP):
        mask = rec >= t
        ap += float(prec[mask].max()) if mask.any() else 0.0
    return ap / N_AP


def ignore_aware_ap50(
    images: Sequence[ImagePredictions],
    targets: Sequence[ImageTargets],
    ignored_boxes: Sequence[np.ndarray],
    cls: int,
    iou_thr: float = IOU_THR,
    max_dets: int = 100,
) -> IgnoreAPResult:
    """Compute ignore-aware class-aware AP50.

    ignored_boxes[i] is an (M,4) array of ignored GT boxes for class cls in
    image i. ``targets`` must contain only VALID GT (the caller removes
    ignored GT from targets and passes them separately). Boxes must be in
    the same normalized xyxy convention. The function does not mutate any
    input.
    """
    assert len(images) == len(targets) == len(ignored_boxes)
    n_valid_gt = 0
    n_ignored_gt = 0
    entries = []  # (-score, img, pred_idx, tp)
    n_tp = n_fp = n_ignored_pred = 0
    n_matched_valid = 0

    for img, (pred, gt, ign) in enumerate(zip(images, targets, ignored_boxes)):
        valid = gt.labels == cls
        valid_gt = ImageTargets(gt.boxes[valid], gt.labels[valid])
        n_valid_gt += len(valid_gt.boxes)
        n_ignored_gt += len(ign)
        pidx = np.where(pred.labels == cls)[0]
        if len(pidx):
            pidx = pidx[np.argsort(-pred.scores[pidx], kind="stable")][:max_dets]
        if len(pidx) == 0:
            continue
        # valid GT matching first (greedy one-to-one, score order)
        matches, unmatched = _match_single(
            valid_gt, pred, iou_thr, class_aware=True, class_id=cls,
            max_dets=max_dets)
        matched = {m[0] for m in matches}
        n_matched_valid += len(matched)
        ign = np.asarray(ign, float).reshape(-1, 4)
        for pi in pidx:
            if pi in matched:
                entries.append((-float(pred.scores[pi]), img, int(pi), True))
                n_tp += 1
                continue
            # unmatched with valid GT: ignore if it overlaps an ignored GT
            if len(ign):
                ious = iou_matrix(ign, pred.boxes[pi][None, :])[:, 0]
                if float(ious.max()) >= iou_thr:
                    n_ignored_pred += 1
                    continue
            entries.append((-float(pred.scores[pi]), img, int(pi), False))
            n_fp += 1

    entries.sort()
    tp = np.asarray([e[3] for e in entries], bool)
    return IgnoreAPResult(
        ap50=_ap101(tp, n_valid_gt) if n_valid_gt else None,
        n_valid_gt=n_valid_gt,
        n_ignored_gt=n_ignored_gt,
        n_pred=len(entries) + n_ignored_pred,
        n_tp=n_tp,
        n_fp=n_fp,
        n_ignored_pred=n_ignored_pred,
        n_matched_valid=n_matched_valid,
    )


def drop_gt_ap50(
    images: Sequence[ImagePredictions],
    targets: Sequence[ImageTargets],
    cls: int,
    keep_valid_mask: Sequence[np.ndarray],
    iou_thr: float = IOU_THR,
    max_dets: int = 100,
) -> Dict[str, object]:
    """Old, known-biased protocol: physically drop ignored GT, keep all preds."""
    kept_targets = []
    for t, keep in zip(targets, keep_valid_mask):
        keep = np.asarray(keep, bool)
        kept_targets.append(ImageTargets(t.boxes[keep], t.labels[keep]))
    # use the production reference AP for this class
    from src.signals.metrics import evaluate
    ev = evaluate(kept_targets, images, iou_thresholds=(iou_thr,),
                  max_dets=max_dets)
    pc = ev["per_iou"][str(iou_thr)]["per_class"].get(str(cls))
    return {"ap50": None if pc is None else pc["ap"],
            "n_gt": 0 if pc is None else pc["n_gt"],
            "n_pred": 0 if pc is None else pc["n_pred"]}
