"""SAHI-style tensor inference, coordinate remap and class-wise global NMS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torchvision.ops import box_iou, nms as tv_nms
from ultralytics.utils.nms import non_max_suppression

from .data import (
    COARSE_H, FINE_H, N_TIME, SampleRecord, SpaceNetViews, View,
    build_sample_views, frequency_starts,
)


def parse_class_set(spec: str | None) -> list[int] | None:
    """Parse '0-5', '0,1,2', or '0-5,10' into a sorted class list."""
    if not spec:
        return None
    out: set[int] = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def filter_fine_classes(pred: "Prediction", view_type: str,
                        drop_classes: list[int] | None) -> "Prediction":
    """Drop predictions from fine views whose predicted class is in drop_classes."""
    if view_type != "fine" or not drop_classes or len(pred.boxes) == 0:
        return pred
    drop = torch.tensor(drop_classes, dtype=pred.labels.dtype, device=pred.labels.device)
    keep = ~torch.isin(pred.labels, drop)
    if keep.all():
        return pred
    return Prediction(pred.boxes[keep], pred.scores[keep], pred.labels[keep])


@dataclass
class Prediction:
    boxes: torch.Tensor   # normalized xyxy global
    scores: torch.Tensor
    labels: torch.Tensor


def _nms_view_batch(model, images: torch.Tensor, conf=0.001, iou=0.5, max_det=300,
                    multi_label: bool = False):
    model.eval()
    with torch.no_grad():
        preds, _ = model(images)
        # 2048x512 has 21504 raw anchors per view; give NMS an explicit generous budget.
        dets = non_max_suppression(
            preds, conf_thres=conf, iou_thres=iou, nc=14,
            max_det=max_det, multi_label=multi_label, max_time_img=20.0,
            max_nms=max(30000, max_det),
        )
    return dets


def _map_view_dets(det: torch.Tensor, view_type: str, start: int, F: int) -> Prediction:
    """Clip a view's xyxy pixel boxes to the view, then map to global coordinates.

    The plan requires fine predictions to be trimmed to the view boundary
    before remapping. Without this clamp, DirectReg boxes that extend past the
    view map to frequencies outside the original band (observed in stage 5).
    """
    if len(det) == 0:
        return Prediction(torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long))
    boxes = det[:, :4].clone()
    view_h = FINE_H if view_type == "fine" else COARSE_H
    # basic indexing gives views; clamp each coordinate in place
    boxes[:, 0].clamp_(0.0, float(N_TIME))
    boxes[:, 2].clamp_(0.0, float(N_TIME))
    boxes[:, 1].clamp_(0.0, float(view_h))
    boxes[:, 3].clamp_(0.0, float(view_h))
    boxes[:, [0, 2]] /= N_TIME
    if view_type == "fine":
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] + start) / F
    else:
        boxes[:, [1, 3]] /= COARSE_H
    return Prediction(boxes.cpu(), det[:, 4].cpu(), det[:, 5].long().cpu())


def _boundary_mask(pred: Prediction, view_type: str, start: int, F: int, margin: int = 16) -> torch.Tensor:
    """True for a fine box close to an internal frequency slice boundary."""
    if view_type != "fine" or len(pred.boxes) == 0:
        return torch.zeros((len(pred.boxes),), dtype=torch.bool)
    y1 = pred.boxes[:, 1] * F
    y2 = pred.boxes[:, 3] * F
    left = start
    right = start + FINE_H
    near_left = (left > 0) & (y1 <= left + margin)
    near_right = (right < F) & (y2 >= right - margin)
    return near_left | near_right


def soft_classwise_nms(pred: "Prediction", iou: float = 0.5, sigma: float = 0.5,
                       score_thr: float = 1e-3, max_det: int = 100) -> "Prediction":
    """Gaussian Soft-NMS per class, then global top-k.

    Mask-based implementation: select the current max, decay all still-alive
    boxes, and continue. This preserves the exact greedy order of the previous
    torch.cat implementation without O(n^2) tensor copying.
    """
    if len(pred.boxes) == 0:
        return pred
    kb, ks, kl = [], [], []
    for cls in sorted(set(pred.labels.tolist())):
        idx = (pred.labels == cls).nonzero(as_tuple=False).view(-1)
        b = pred.boxes[idx].clone()
        s = pred.scores[idx].clone()
        alive = torch.ones(len(b), dtype=torch.bool, device=b.device)
        while bool(alive.any()):
            masked = s.masked_fill(~alive, float("-inf"))
            j = int(torch.argmax(masked))
            kb.append(b[j]); ks.append(s[j]); kl.append(cls)
            alive[j] = False
            if not bool(alive.any()):
                break
            ov = box_iou(b[j].unsqueeze(0), b[alive])[0]
            s[alive] = s[alive] * torch.exp(-(ov ** 2) / max(sigma, 1e-6))
            alive &= s >= score_thr
    boxes = torch.stack(kb); scores = torch.stack(ks)
    labels = torch.tensor(kl, dtype=pred.labels.dtype, device=pred.labels.device)
    order = torch.argsort(scores, descending=True)[:max_det]
    return Prediction(boxes[order], scores[order], labels[order])

def global_classwise_nms(pred: Prediction, iou: float = 0.5, max_det: int = 100) -> Prediction:
    """One-to-one class-wise NMS; boxes are global normalized xyxy."""
    if len(pred.boxes) == 0:
        return pred
    keep = []
    for cls in sorted(set(pred.labels.tolist())):
        idx = (pred.labels == cls).nonzero(as_tuple=False).view(-1)
        keep_cls = tv_nms(pred.boxes[idx], pred.scores[idx], iou)
        keep.extend(idx[keep_cls].tolist())
    keep = torch.tensor(keep, dtype=torch.long)
    if len(keep) == 0:
        return Prediction(torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long))
    order = torch.argsort(pred.scores[keep], descending=True)[:max_det]
    keep = keep[order]
    return Prediction(pred.boxes[keep], pred.scores[keep], pred.labels[keep])


def predict_views(
    model,
    views: Sequence[View],
    Fbins: int,
    batch_views: int = 8,
    conf: float = 0.001,
    iou: float = 0.5,
    max_det: int = 100,
    max_det_view: int = 300,
    drop_fine_classes: list[int] | None = None,
) -> tuple[Prediction, dict]:
    """Run NMS/mapping/merge on an already-built list of views."""
    all_boxes, all_scores, all_labels = [], [], []
    n_boundary = 0
    n_raw = 0
    for begin in range(0, len(views), batch_views):
        chunk = views[begin:begin + batch_views]
        images = torch.stack([v.tensor for v in chunk], 0).cuda()
        dets = _nms_view_batch(model, images, conf=conf, iou=iou, max_det=max_det_view)
        for det, view in zip(dets, chunk):
            n_raw += len(det)
            pred = _map_view_dets(det, view.view_type, view.start, Fbins)
            bm = _boundary_mask(pred, view.view_type, view.start, Fbins)
            n_boundary += int(bm.sum())
            if bm.any():
                pred = Prediction(pred.boxes[~bm], pred.scores[~bm], pred.labels[~bm])
            pred = filter_fine_classes(pred, view.view_type, drop_fine_classes)
            all_boxes.append(pred.boxes)
            all_scores.append(pred.scores)
            all_labels.append(pred.labels)
    if all_boxes:
        pred = Prediction(torch.cat(all_boxes), torch.cat(all_scores), torch.cat(all_labels))
    else:
        pred = Prediction(torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long))
    merged = global_classwise_nms(pred, iou=iou, max_det=max_det)
    stats = {
        "n_views": len(views), "n_raw_dets": n_raw,
        "n_after_boundary": len(pred.boxes), "n_boundary_dropped": n_boundary,
        "n_final": len(merged.boxes),
    }
    return merged, stats


def predict_sample(
    model,
    record: SampleRecord,
    normalizer: tuple[float, float],
    batch_views: int = 8,
    conf: float = 0.001,
    iou: float = 0.5,
    max_det: int = 100,
    max_det_view: int = 300,
    drop_fine_classes: list[int] | None = None,
    view_mode: str = "all",
) -> tuple[Prediction, dict]:
    """Full-sample inference.

    view_mode="all" uses one coarse + all fine slices (C). view_mode="coarse"
    uses only the full-band view (B coarse-only / A deployment).
    """
    Fbins = int(round(record.B_mhz * 700))
    if view_mode == "coarse":
        coarse, _, _ = build_sample_views(record, normalizer, [])
        views = [coarse]
    elif view_mode == "all":
        starts = frequency_starts(Fbins)
        coarse, fines, _ = build_sample_views(record, normalizer, list(range(len(starts))))
        views = [coarse] + fines
    else:
        raise ValueError(f"unsupported view_mode: {view_mode!r}")
    return predict_views(
        model, views, Fbins, batch_views=batch_views, conf=conf, iou=iou,
        max_det=max_det, max_det_view=max_det_view,
        drop_fine_classes=drop_fine_classes,
    )


def gt_global(record: SampleRecord, h: int) -> Prediction:
    """GT normalized global xyxy using the same pixel-center time convention."""
    boxes, labels, ids = [], [], []
    Fbins = int(round(record.B_mhz * 700))
    for sig in record.signals:
        x1 = ((sig["start_time"] / 1000.0) * (record.B_mhz * 1e6) / h + 0.5) / N_TIME
        x2 = ((sig["end_time"] / 1000.0) * (record.B_mhz * 1e6) / h + 0.5) / N_TIME
        y1 = (sig["start_frequency"] - record.f_lo) / record.B_mhz
        y2 = (sig["end_frequency"] - record.f_lo) / record.B_mhz
        boxes.append([min(1.0, max(0.0, x1)), min(1.0, max(0.0, y1)),
                      min(1.0, max(0.0, x2)), min(1.0, max(0.0, y2))])
        labels.append(int(sig["class"]))
        ids.append(int(sig["signal_id"]))
    return Prediction(torch.tensor(boxes, dtype=torch.float32),
                      torch.ones(len(boxes), dtype=torch.float32),
                      torch.tensor(labels, dtype=torch.long))
