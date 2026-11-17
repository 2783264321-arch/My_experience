#!/usr/bin/env python3
"""Reference evaluator for SpaceNet-style 2D time-frequency boxes.

Protocol (v0.1):
- boxes are continuous [x1, y1, x2, y2] in normalized canvas coordinates.
- no pixel rounding; no class-based pre-filtering before matching.
- greedy one-to-one matching, predictions sorted by score descending.
- class-aware AP: per-class predictions matched to same-class GT.
- class-agnostic AP: all predictions matched to all GT.
- classification: class-agnostic location matching at IoU=0.5, then class comparison.
- AP is 101-point interpolated AP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class ImageTargets:
    boxes: np.ndarray  # (N,4) x1,y1,x2,y2
    labels: np.ndarray  # (N,)


@dataclass
class ImagePredictions:
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU for (N,4) and (M,4) continuous boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.maximum(0.0, np.minimum(ax2[:, None], bx2[None, :]) - np.maximum(ax1[:, None], bx1[None, :]))
    ih = np.maximum(0.0, np.minimum(ay2[:, None], by2[None, :]) - np.maximum(ay1[:, None], by1[None, :]))
    inter = iw * ih
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _match_single(
    gt: ImageTargets,
    pred: ImagePredictions,
    iou_thr: float,
    class_aware: bool,
    class_id: int | None = None,
    max_dets: int = 100,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Greedy one-to-one matching within one image.

    Returns (matched (pred_idx, gt_idx), unmatched pred indices).
    """
    if class_aware and class_id is not None:
        pidx = np.where(pred.labels == class_id)[0]
        gidx = np.where(gt.labels == class_id)[0]
    else:
        pidx = np.arange(len(pred.boxes))
        gidx = np.arange(len(gt.boxes))
    if len(pidx):
        order = np.argsort(-pred.scores[pidx], kind="stable")
        pidx = pidx[order][:max_dets]
    if len(pidx) == 0:
        return [], []
    if len(gidx) == 0:
        return [], list(pidx.tolist())
    ious = iou_matrix(gt.boxes[gidx], pred.boxes[pidx])
    used_gt = set()
    matches = []
    unmatched = []
    for k, pi in enumerate(pidx):
        best = -1.0
        best_j = -1
        for j, gi in enumerate(gidx):
            if j in used_gt:
                continue
            if ious[j, k] > best:
                best = float(ious[j, k])
                best_j = j
        if best >= iou_thr and best_j >= 0:
            used_gt.add(best_j)
            matches.append((int(pi), int(gidx[best_j])))
        else:
            unmatched.append(int(pi))
    return matches, unmatched


def _precision_recall(tp: np.ndarray, fp: np.ndarray, n_gt: int):
    if n_gt == 0:
        return None, None
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / n_gt
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return prec, rec


def _ap_101(prec: np.ndarray | None, rec: np.ndarray | None) -> float | None:
    if prec is None or rec is None:
        return None
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = rec >= t
        ap += float(np.max(prec[mask])) if mask.any() else 0.0
    return ap / 101.0


def evaluate(
    gts: Sequence[ImageTargets],
    preds: Sequence[ImagePredictions],
    iou_thresholds: Sequence[float] = (0.3, 0.5, 0.7),
    max_dets: int = 100,
) -> Dict[str, object]:
    """Evaluate a dataset of images. Returns AP/classification diagnostics."""
    assert len(gts) == len(preds)
    out: Dict[str, object] = {"iou_thresholds": list(iou_thresholds), "per_iou": {}}

    for thr in iou_thresholds:
        # class-agnostic AP
        all_tp, all_fp = [], []
        n_gt_all = 0
        for gt, pred in zip(gts, preds):
            n_gt_all += len(gt.boxes)
            matches, unmatched = _match_single(gt, pred, thr, class_aware=False, max_dets=max_dets)
            matched_pred = {m[0] for m in matches}
            for pi in range(len(pred.boxes)):
                all_tp.append(pi in matched_pred)
                all_fp.append(pi not in matched_pred)
        # sort globally by score for AP
        score_order = []
        for img, pred in enumerate(preds):
            for pi, sc in enumerate(pred.scores):
                score_order.append((-float(sc), img, pi))
        score_order.sort()
        # Recompute TP/FP in global score order with image-local greedy matching
        matched_by_img = {}
        for gt, pred in zip(gts, preds):
            matches, _ = _match_single(gt, pred, thr, class_aware=False, max_dets=max_dets)
            matched_by_img[id(pred)] = {m[0] for m in matches}
        tp, fp = [], []
        for _, img, pi in score_order:
            is_tp = pi in matched_by_img.get(id(preds[img]), set())
            tp.append(is_tp)
            fp.append(not is_tp)
        prec, rec = _precision_recall(np.asarray(tp), np.asarray(fp), n_gt_all)
        ap_agnostic = _ap_101(prec, rec)

        per_class = {}
        for cls in range(14):
            n_gt_cls = sum(int((gt.labels == cls).sum()) for gt in gts)
            if n_gt_cls == 0:
                continue
            entries = []
            for img, pred in enumerate(preds):
                matches, _ = _match_single(gts[img], pred, thr, class_aware=True, class_id=cls, max_dets=max_dets)
                matched = {m[0] for m in matches}
                for pi in np.where(pred.labels == cls)[0]:
                    entries.append((-float(pred.scores[pi]), img, int(pi), int(pi) in matched))
            entries.sort()
            tp = [e[3] for e in entries]
            fp = [not e[3] for e in entries]
            prec, rec = _precision_recall(np.asarray(tp), np.asarray(fp), n_gt_cls)
            per_class[str(cls)] = {
                "ap": _ap_101(prec, rec),
                "n_gt": n_gt_cls,
                "n_pred": len(entries),
            }
        vals = [v["ap"] for v in per_class.values() if v["ap"] is not None]
        out["per_iou"][str(thr)] = {
            "ap_class_agnostic": ap_agnostic,
            "map_class_aware": float(np.mean(vals)) if vals else None,
            "per_class": per_class,
            "n_gt": n_gt_all,
        }

    # classification and localization using class-agnostic matching at 0.5
    thr = 0.5
    correct = 0
    matched_total = 0
    gt_total = 0
    confusion = np.zeros((14, 14), dtype=np.int64)
    for gt, pred in zip(gts, preds):
        gt_total += len(gt.boxes)
        matches, _ = _match_single(gt, pred, thr, class_aware=False, max_dets=max_dets)
        matched_total += len(matches)
        for pi, gi in matches:
            pcls = int(pred.labels[pi])
            gcls = int(gt.labels[gi])
            if 0 <= pcls < 14 and 0 <= gcls < 14:
                confusion[gcls, pcls] += 1
            correct += int(pcls == gcls)
    out["classification"] = {
        "iou_threshold": thr,
        "matched": matched_total,
        "gt": gt_total,
        "localization_recall": (matched_total / gt_total) if gt_total else None,
        "classification_accuracy": (correct / matched_total) if matched_total else None,
        "confusion_gt_x_pred": confusion.tolist(),
    }
    return out


def _gt(boxes, labels):
    return ImageTargets(np.asarray(boxes, dtype=float), np.asarray(labels, dtype=int))


def _pred(boxes, scores, labels):
    return ImagePredictions(
        np.asarray(boxes, dtype=float), np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    )


def _self_test():
    import json

    b = [0.1, 0.1, 0.4, 0.4]
    gt = [_gt([b], [1])]
    pred = [_pred([b], [0.9], [1])]
    perfect = evaluate(gt, pred)
    assert perfect["per_iou"]["0.5"]["ap_class_agnostic"] == 1.0
    assert perfect["per_iou"]["0.5"]["map_class_aware"] == 1.0
    assert perfect["classification"]["classification_accuracy"] == 1.0

    missing = evaluate(gt, [_pred([], [], [])])
    assert missing["per_iou"]["0.5"]["ap_class_agnostic"] == 0.0

    false_pos = evaluate(
        [_gt([b], [1]), _gt([], [])],
        [_pred([b], [0.9], [1]), _pred([b], [0.95], [1])],
    )
    assert 0.0 < false_pos["per_iou"]["0.5"]["ap_class_agnostic"] < 1.0

    wrong_cls = evaluate(gt, [_pred([b], [0.9], [2])])
    assert wrong_cls["per_iou"]["0.5"]["ap_class_agnostic"] == 1.0
    assert wrong_cls["per_iou"]["0.5"]["map_class_aware"] == 0.0
    assert wrong_cls["classification"]["classification_accuracy"] == 0.0
    assert wrong_cls["classification"]["localization_recall"] == 1.0

    # two same-location GT, one prediction: one match only
    gt2 = [_gt([b, b], [1, 1])]
    p2 = [_pred([b], [0.9], [1])]
    m2 = evaluate(gt2, p2)
    assert m2["classification"]["matched"] == 1
    assert abs(m2["classification"]["localization_recall"] - 0.5) < 1e-9

    # narrow and boundary boxes preserve continuous coordinates
    narrow = [0.5000, 0.0, 0.5001, 1.0]
    gt3 = [_gt([narrow], [12])]
    p3 = [_pred([narrow], [0.8], [12])]
    m3 = evaluate(gt3, p3)
    assert m3["per_iou"]["0.5"]["map_class_aware"] == 1.0

    print(json.dumps({
        "perfect": perfect["per_iou"]["0.5"],
        "wrong_class": wrong_cls["per_iou"]["0.5"],
        "same_location": m2["classification"],
        "narrow": m3["per_iou"]["0.5"],
    }, ensure_ascii=False, indent=2))
    print("eval self-test OK")


if __name__ == "__main__":
    _self_test()
