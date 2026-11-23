#!/usr/bin/env python3
"""D-FINE-L round25/50 query-level inference and merge ablation.

For each checkpoint, one full val inference pass preserves the per-view
query id and all 14 class scores. It then compares:
  - original merge (cross-class candidates kept);
  - same-view same-query only the highest-scoring class before NMS.

Also reports Wi-Fi 0-2 true-class rank distributions and FM missed-stage
counts. No training, no test.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_convert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/detector_upgrade"))
from adapters import build_dfine
from src.signals.data import N_TIME, SPLIT_MANIFEST, load_manifest
from src.signals.infer import (
    FINE_H, Prediction, _boundary_mask, _map_view_dets, global_classwise_nms,
    gt_global,
)
from src.signals.metrics import ImagePredictions, ImageTargets, iou_matrix
from src.signals.validate import FullViewDataset, _collate_one
from validate_dual import FINE_0_5, _eval_set

OUT = ROOT / "reports/detector_upgrade"
CKPT_DIR = Path("/root/autodl-tmp/worktrees/formal-dfine/runs/dfine_l_c1_2048x512")
CKPTS = {"round25": CKPT_DIR / "best.pt", "round50": CKPT_DIR / "last.pt"}


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = ck["ema"] if ck.get("ema") is not None else ck["model"]
    model, info = build_dfine()
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model, info, ck


def decode_batch(model, images):
    """Return per-image det [n,6] (pixel xyxy, score, label), qid, class_scores."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(images)
    logits = outputs["pred_logits"]
    boxes = box_convert(outputs["pred_boxes"], "cxcywh", "xyxy")
    scale = torch.tensor([[512.0, 2048.0, 512.0, 2048.0]],
                         device=boxes.device).repeat(boxes.shape[0], 1)
    boxes = boxes * scale.unsqueeze(1)
    scores = torch.sigmoid(logits)
    B, Q, C = scores.shape
    top_scores, index = torch.topk(scores.flatten(1), model.postprocessor.num_top_queries,
                                   dim=-1)
    labels = torch.remainder(index, C)
    qid = torch.div(index, C, rounding_mode="floor")
    boxes = boxes.gather(1, qid.unsqueeze(-1).repeat(1, 1, 4))
    out = []
    for b in range(B):
        keep = top_scores[b] >= 0.001
        cls_all = scores[b].gather(0, qid[b].unsqueeze(-1).repeat(1, C))
        det = torch.cat([boxes[b], top_scores[b][:, None],
                         labels[b].float()[:, None], qid[b].float()[:, None]], 1)[keep]
        out.append({"det": det[:, :6], "qid": qid[b][keep],
                    "class_scores": cls_all[keep]})
    return out


def to_prediction(h):
    return Prediction(h["boxes"], h["scores"], h["labels"])


def concat_hyp(hs):
    if not hs:
        return (torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long),
                torch.zeros((0,), dtype=torch.long), torch.zeros((0, 14)))
    return (torch.cat([h["boxes"] for h in hs]).float().cpu(),
            torch.cat([h["scores"] for h in hs]).float().cpu(),
            torch.cat([h["labels"] for h in hs]).long().cpu(),
            torch.cat([h["qid"] for h in hs]).long().cpu(),
            torch.cat([h["class_scores"] for h in hs]).float().cpu())


def dedup_query(hs):
    """Keep the highest-score class per (view_id, qid)."""
    best = {}
    for h in hs:
        v = h["view_id"]
        for j, q in enumerate(h["qid"].tolist()):
            key = (v, q)
            s = float(h["scores"][j])
            if key not in best or s > best[key][0]:
                best[key] = (s, h, j)
    out = []
    for s, h, j in best.values():
        out.append({"view_id": h["view_id"],
                    "boxes": h["boxes"][j:j + 1],
                    "scores": h["scores"][j:j + 1],
                    "labels": h["labels"][j:j + 1],
                    "qid": h["qid"][j:j + 1],
                    "class_scores": h["class_scores"][j:j + 1]})
    return out


def max_iou(boxes, target):
    if len(boxes) == 0:
        return 0.0
    return float(iou_matrix(np.asarray(boxes), target[None]).max())


def evaluate_predictions(gts, preds):
    return _eval_set(gts, preds)


def run_checkpoint(ckpt, name, device, batch_views=8, workers=8, max_samples=0):
    records = load_manifest(SPLIT_MANIFEST, split="val")
    if max_samples:
        records = records[:max_samples]
    model, info, ck = load_model(ckpt, device)
    ds = FullViewDataset(records, (json.loads((ROOT / "configs/normalizer.json").read_text())["lo"],
                                   json.loads((ROOT / "configs/normalizer.json").read_text())["hi"]),
                         view_mode="all")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=workers,
                        collate_fn=_collate_one, pin_memory=False,
                        prefetch_factor=4 if workers else None,
                        persistent_workers=workers > 0)
    gts = [gt_global(r, math.ceil(r.N / (N_TIME - 1))) for r in records]
    samples = [[] for _ in records]
    raw_boxes = [[] for _ in records]
    buffer = []
    view_counter = 0
    t0 = time.perf_counter()
    last = time.perf_counter()
    for item in loader:
        rec = records[item["index"]]
        F = int(round(rec.B_mhz * 700))
        for tensor, vt, st in zip(item["tensors"], item["view_types"], item["starts"]):
            buffer.append({"index": item["index"], "tensor": tensor, "view_type": vt,
                           "start": st, "F": F})
        while len(buffer) >= batch_views:
            chunk, buffer = buffer[:batch_views], buffer[batch_views:]
            images = torch.stack([b["tensor"] for b in chunk], 0).to(device, non_blocking=True)
            dec = decode_batch(model, images)
            for b, d in zip(chunk, dec):
                p = _map_view_dets(d["det"], b["view_type"], b["start"], b["F"])
                raw_boxes[b["index"]].extend(p.boxes.numpy().tolist())
                bm = _boundary_mask(p, b["view_type"], b["start"], b["F"]).numpy()
                keep = ~bm
                if b["view_type"] == "fine":
                    keep &= ~np.isin(p.labels.numpy(), FINE_0_5)
                if keep.any():
                    idx = np.flatnonzero(keep)
                    samples[b["index"]].append({
                        "view_id": view_counter,
                        "boxes": p.boxes[idx], "scores": p.scores[idx],
                        "labels": p.labels[idx], "qid": d["qid"][idx],
                        "class_scores": d["class_scores"][idx],
                    })
                view_counter += 1
        if time.perf_counter() - last > 120:
            print(f"  {name} {item['index']+1}/{len(records)} samples, "
                  f"{view_counter} views, {time.perf_counter()-t0:.0f}s", flush=True)
            last = time.perf_counter()
    if buffer:
        chunk = buffer
        images = torch.stack([b["tensor"] for b in chunk], 0).to(device, non_blocking=True)
        dec = decode_batch(model, images)
        for b, d in zip(chunk, dec):
            p = _map_view_dets(d["det"], b["view_type"], b["start"], b["F"])
            raw_boxes[b["index"]].extend(p.boxes.numpy().tolist())
            bm = _boundary_mask(p, b["view_type"], b["start"], b["F"]).numpy()
            keep = ~bm
            if b["view_type"] == "fine":
                keep &= ~np.isin(p.labels.numpy(), FINE_0_5)
            if keep.any():
                idx = np.flatnonzero(keep)
                samples[b["index"]].append({
                    "view_id": view_counter,
                    "boxes": p.boxes[idx], "scores": p.scores[idx],
                    "labels": p.labels[idx], "qid": d["qid"][idx],
                    "class_scores": d["class_scores"][idx],
                })
            view_counter += 1
    elapsed = time.perf_counter() - t0

    original, query = [], []
    n_orig_cand = n_query_cand = 0
    for i in range(len(records)):
        hs = samples[i]
        boxes, scores, labels, qid, cls = concat_hyp(hs)
        cat = Prediction(boxes, scores, labels)
        final = global_classwise_nms(cat, iou=0.5, max_det=100)
        original.append(ImagePredictions(final.boxes.numpy(), final.scores.numpy(),
                                         final.labels.numpy()))
        n_orig_cand += len(final.boxes)
        hs2 = dedup_query(hs)
        boxes2, scores2, labels2, qid2, cls2 = concat_hyp(hs2)
        final2 = global_classwise_nms(Prediction(boxes2, scores2, labels2),
                                      iou=0.5, max_det=100)
        query.append(ImagePredictions(final2.boxes.numpy(), final2.scores.numpy(),
                                      final2.labels.numpy()))
        n_query_cand += len(final2.boxes)

    # Wi-Fi 0-2 true class rank and FM stage, using post-filter hypotheses.
    rank = Counter()
    fm_stage = Counter()
    fm_correct = 0
    fm_total = 0
    for i, gt in enumerate(gts):
        gy = gt.labels.numpy(); gb = gt.boxes.numpy()
        hs = samples[i]
        boxes, scores, labels, qid, cls = concat_hyp(hs)
        boxes_np = boxes.numpy(); labels_np = labels.numpy(); scores_np = scores.numpy()
        cls_np = cls.numpy()
        for j, g in enumerate(gy):
            if g in (0, 1, 2):
                if len(boxes_np) == 0:
                    rank["missing"] += 1
                    continue
                ious = iou_matrix(boxes_np, gb[j:j + 1]).reshape(-1)
                idx = np.flatnonzero(ious >= 0.5)
                if len(idx) == 0:
                    rank["missing"] += 1
                    continue
                best = idx[np.argmax(scores_np[idx])]
                s = cls_np[best]
                order = np.argsort(-s)
                true_rank = int(np.where(order == int(g))[0][0]) + 1
                rank[f"rank{true_rank}"] += 1
            elif g == 13:
                fm_total += 1
                final = original[i]
                io_final = iou_matrix(final.boxes, gb[j:j + 1]).reshape(-1) if len(final.boxes) else np.zeros(0)
                io_post = iou_matrix(boxes_np, gb[j:j + 1]).reshape(-1) if len(boxes_np) else np.zeros(0)
                raw = np.asarray(raw_boxes[i], float).reshape(-1, 4)
                io_raw = iou_matrix(raw, gb[j:j + 1]).reshape(-1) if len(raw) else np.zeros(0)
                hit_final = (io_final >= 0.5).any()
                hit_correct = hit_final and (final.labels[io_final >= 0.5] == 13).any()
                if hit_correct:
                    fm_correct += 1
                    fm_stage["detected_correct"] += 1
                elif hit_final:
                    fm_stage["box_wrong_class"] += 1
                elif (io_post >= 0.5).any():
                    fm_stage["nms_or_top100"] += 1
                elif (io_raw >= 0.5).any():
                    fm_stage["boundary_or_filter"] += 1
                else:
                    fm_stage["no_raw_candidate"] += 1

    result = {
        "checkpoint": str(ckpt), "name": name, "epoch": ck.get("epoch"),
        "elapsed_s": elapsed, "n_views": view_counter,
        "original": evaluate_predictions(gts, original),
        "query_dedup": evaluate_predictions(gts, query),
        "n_final_orig": n_orig_cand, "n_final_query": n_query_cand,
        "wifi_0_2_rank": dict(rank),
        "fm": {"total": fm_total, "correct": fm_correct, "stage": dict(fm_stage)},
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT / "dfine_query_ablation.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-views", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()
    results = {}
    for name, ckpt in CKPTS.items():
        print(f"=== {name} {ckpt}", flush=True)
        results[name] = run_checkpoint(ckpt, name, args.device,
                                       args.batch_views, args.workers,
                                       args.max_samples)
        print(json.dumps({name: {
            "elapsed_s": results[name]["elapsed_s"],
            "orig_map50": results[name]["original"]["map50"],
            "query_map50": results[name]["query_dedup"]["map50"],
            "orig_cand": results[name]["n_final_orig"],
            "query_cand": results[name]["n_final_query"],
            "wifi_rank": results[name]["wifi_0_2_rank"],
            "fm": results[name]["fm"],
        }}, indent=2), flush=True)
        (args.out).parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
