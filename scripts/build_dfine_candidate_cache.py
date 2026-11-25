#!/usr/bin/env python3
"""Cache fixed D-FINE-L round25 candidates for the local-IQ A/B/C ablation.

One forward pass per split. Candidates are the per-view K=300 query/class
pairs after the existing boundary and fine-class filters, i.e. exactly the
pool that enters global class-wise NMS in the current pipeline.

Each candidate stores its view, query id, global box, original score, all 14
class scores and its best GT match (diagnostic / fit labels only). The cache
is the fixed candidate identity used by A/B/C; B/C may change class scores but
not boxes or candidate membership.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/detector_upgrade"))
sys.path.insert(0, str(ROOT / "scripts"))

from dfine_query_ablation import CKPTS, decode_batch, load_model
from src.signals.data import N_TIME, SPLIT_MANIFEST, load_manifest
from src.signals.infer import _boundary_mask, _map_view_dets, gt_global
from src.signals.metrics import iou_matrix
from src.signals.validate import FullViewDataset, _collate_one
from validate_dual import FINE_0_5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("fit", "val"), default="val")
    ap.add_argument("--checkpoint", type=Path, default=CKPTS["round25"])
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--batch-views", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    records = load_manifest(SPLIT_MANIFEST, split=args.split)
    if args.max_samples:
        records = records[:args.max_samples]
    gts = [gt_global(r, math.ceil(r.N / (N_TIME - 1))) for r in records]
    norm = json.loads((ROOT / "configs/normalizer.json").read_text())
    ds = FullViewDataset(records, (norm["lo"], norm["hi"]), view_mode="all")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers,
                        collate_fn=_collate_one, pin_memory=False,
                        prefetch_factor=4 if args.workers else None,
                        persistent_workers=args.workers > 0)
    model, _, ck = load_model(args.checkpoint, args.device)

    # Per sample: list of candidate chunks in view order.
    chunks = [[] for _ in records]
    buf = []
    n_views = 0
    t0 = time.perf_counter()
    last = time.perf_counter()

    def flush(batch):
        nonlocal n_views
        imgs = torch.stack([b["tensor"] for b in batch], 0).to(args.device)
        dec = decode_batch(model, imgs)
        for b, d in zip(batch, dec):
            p = _map_view_dets(d["det"], b["view_type"], b["start"], b["F"])
            bm = _boundary_mask(p, b["view_type"], b["start"], b["F"])
            keep = ~bm.numpy()
            if b["view_type"] == "fine":
                keep &= ~np.isin(p.labels.numpy(), FINE_0_5)
            if not keep.any():
                n_views += 1
                continue
            idx = np.flatnonzero(keep)
            idx_t = torch.from_numpy(idx).long()
            nn = len(idx)
            chunks[b["index"]].append({
                "view_type": np.full(nn, 1 if b["view_type"] == "fine" else 0, dtype=np.int8),
                "view_start": np.full(nn, b["start"], dtype=np.int32),
                "qid": d["qid"][idx_t].long().cpu().numpy().astype(np.int16),
                "boxes": p.boxes[idx].float().cpu().numpy().astype(np.float32),
                "scores": p.scores[idx].float().cpu().numpy().astype(np.float32),
                "labels": p.labels[idx].long().cpu().numpy().astype(np.int8),
                "class_scores": d["class_scores"][idx_t].float().cpu().numpy().astype(np.float32),
            })
            n_views += 1

    for item in loader:
        rec = records[item["index"]]
        F = int(round(rec.B_mhz * 700))
        for tensor, vt, st in zip(item["tensors"], item["view_types"], item["starts"]):
            buf.append({"index": item["index"], "tensor": tensor,
                        "view_type": vt, "start": st, "F": F})
        while len(buf) >= args.batch_views:
            batch, buf = buf[:args.batch_views], buf[args.batch_views:]
            flush(batch)
        if time.perf_counter() - last > 120:
            print(f"  {args.split} {item['index']+1}/{len(records)} samples, "
                  f"{n_views} views, {time.perf_counter()-t0:.0f}s", flush=True)
            last = time.perf_counter()
    if buf:
        flush(buf)

    # Flatten and match each candidate to its best GT (class-agnostic IoU).
    keys = ("view_type", "view_start", "qid", "boxes", "scores", "labels",
            "class_scores", "gt_label", "gt_iou", "sample_index")
    arrays = {k: [] for k in keys}
    offsets = [0]
    for i, parts in enumerate(chunks):
        if not parts:
            offsets.append(offsets[-1])
            continue
        cat = {k: np.concatenate([p[k] for p in parts]) for k in
               ("view_type", "view_start", "qid", "boxes", "scores", "labels",
                "class_scores")}
        gt = gts[i]
        if len(gt.boxes):
            ious = iou_matrix(cat["boxes"], gt.boxes.numpy())
            best = ious.argmax(axis=1)
            best_iou = ious[np.arange(len(best)), best].astype(np.float32)
            gt_label = gt.labels.numpy()[best].astype(np.int8)
            gt_label[best_iou < 0.5] = -1
        else:
            best_iou = np.zeros((len(cat["scores"]),), dtype=np.float32)
            gt_label = np.full((len(cat["scores"]),), -1, dtype=np.int8)
        cat["gt_iou"] = best_iou
        cat["gt_label"] = gt_label
        cat["sample_index"] = np.full((len(cat["scores"]),), i, dtype=np.int32)
        for k in keys:
            arrays[k].append(cat[k])
        offsets.append(offsets[-1] + len(cat["scores"]))

    out = {k: (np.concatenate(v) if v else np.zeros((0,), dtype=np.float32))
           for k, v in arrays.items()}
    out["sample_offsets"] = np.asarray(offsets, dtype=np.int64)
    out["sample_ids"] = np.asarray([r.sample_id for r in records])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    meta = {
        "format": "dfine-fixed-candidates-v1",
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch_zero_based": ck.get("epoch"),
        "samples": len(records),
        "views": n_views,
        "candidates": int(len(out["scores"])),
        "candidate_filter": "K300, conf>=0.001, boundary drop, fine 0-5 drop",
        "fine_0_5": FINE_0_5,
        "normalizer": norm,
        "batch_views": args.batch_views,
        "workers": args.workers,
        "elapsed_s": time.perf_counter() - t0,
        "columns": list(out),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n",
                                             encoding="utf-8")
    print(json.dumps({k: meta[k] for k in
                      ("split", "samples", "views", "candidates", "elapsed_s")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
