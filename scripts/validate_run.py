#!/usr/bin/env python3
"""Stage-5 full validation of a trained run on the frozen val split.

Runs every fine slice + the coarse view for each val source, then reports:
- strict class-aware mAP@0.5 (the checkpoint fitness)
- class-agnostic AP@0.5, localization recall, classification accuracy
- per-class AP, per-B/T AP, confusion matrix
- per-sample predictions and GT (npz) for error analysis

test is not accepted by this script.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.signals.data import (
    SPLIT_MANIFEST, load_manifest, frequency_starts, N_TIME,
)
from src.signals.metrics import ImagePredictions, ImageTargets, evaluate
from src.signals.infer import parse_class_set
from src.signals.model import build_model
from src.signals.validate import run_validation

ROOT = Path(__file__).resolve().parents[1]


def load_weights(model, ckpt_path: Path, which: str = "ema") -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    key = "ema" if which == "ema" and ckpt.get("ema") is not None else "model"
    sd = ckpt[key]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint key mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return {
        "checkpoint": str(ckpt_path),
        "weights": key,
        "epoch": ckpt.get("epoch"),
        "best_fitness": ckpt.get("best_fitness"),
        "metrics": ckpt.get("metrics", {}),
        "nc": int(model.model[-1].nc),
        "reg_max": int(model.model[-1].reg_max),
        "end2end": bool(model.model[-1].end2end),
        "first_conv": list(model.model[0].conv.weight.shape),
    }


def group_metrics(gts, preds, indices):
    if not indices:
        return {"n_samples": 0}
    g = [gts[i] for i in indices]
    p = [preds[i] for i in indices]
    r = evaluate(g, p, iou_thresholds=(0.5,), max_dets=100)
    pi = r["per_iou"]["0.5"]
    return {
        "n_samples": len(indices),
        "n_gt": int(pi["n_gt"]),
        "n_pred": int(sum(len(x.boxes) for x in p)),
        "map50": pi["map_class_aware"],
        "ap50_agnostic": pi["ap_class_agnostic"],
        "per_class": pi["per_class"],
        "localization_recall": r["classification"]["localization_recall"],
        "classification_accuracy": r["classification"]["classification_accuracy"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="yolo11s_c1_directreg_2048x512")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--weights", choices=["ema", "model"], default="ema")
    ap.add_argument("--split", choices=["val"], default="val")
    ap.add_argument("--output", default="")
    ap.add_argument("--batch-views", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--max-det-view", type=int, default=300)
    ap.add_argument("--multi-label", action="store_true")
    ap.add_argument("--drop-fine-classes", default="")
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    run_dir = ROOT / "runs" / args.run
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = run_dir / ckpt_path
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    out_dir = Path(args.output) if args.output else run_dir / "stage5_val"
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_cfg = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm_cfg["lo"]), float(norm_cfg["hi"]))
    records = load_manifest(SPLIT_MANIFEST, split="val")
    if args.max_samples:
        records = records[: args.max_samples]
    expected_views = sum(
        len(frequency_starts(int(round(r.B_mhz * 700)))) + 1 for r in records
    )

    model = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"))
    info = load_weights(model, ckpt_path, which=args.weights)
    if info["nc"] != 14 or info["reg_max"] != 1 or info["end2end"]:
        raise SystemExit(f"wrong head: {info}")
    if info["first_conv"] != [32, 1, 3, 3]:
        raise SystemExit(f"wrong input stem: {info['first_conv']}")

    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    vr, gts, preds = run_validation(
        model, records, norm,
        batch_views=args.batch_views, workers=args.workers,
        conf=args.conf, iou=args.iou,
        max_det_view=args.max_det_view, max_det=args.max_det,
        multi_label=args.multi_label, log_interval_s=60,
        drop_fine_classes=parse_class_set(args.drop_fine_classes),
    )
    if vr.stats["n_views"] != expected_views:
        raise SystemExit(
            f"view count mismatch: {vr.stats['n_views']} != {expected_views}"
        )

    # per-B/T groups
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[(round(r.B_mhz, 4), round(r.T_ms, 4))].append(i)
    bt = {}
    for (B, T), idx in sorted(groups.items()):
        bt[f"{B:g}MHz_{T:g}ms"] = group_metrics(gts, preds, idx)

    # also group by bandwidth only
    by_b = defaultdict(list)
    for i, r in enumerate(records):
        by_b[r.B_mhz].append(i)
    bandwidth = {f"{B:g}MHz": group_metrics(gts, preds, idx) for B, idx in sorted(by_b.items())}

    # save per-sample predictions / GT
    ids = [r.sample_id for r in records]
    meta = {
        "split": "val",
        "checkpoint": info,
        "normalizer": norm_cfg,
        "split_manifest_sha256": __import__("hashlib").sha256(
            SPLIT_MANIFEST.read_bytes()
        ).hexdigest(),
        "config": json.loads((run_dir / "config.json").read_text())
        if (run_dir / "config.json").exists() else None,
        "n_samples": len(records),
        "n_views": vr.stats["n_views"],
        "expected_views": expected_views,
        "conf": args.conf, "iou": args.iou,
        "max_det": args.max_det, "max_det_view": args.max_det_view,
        "multi_label": args.multi_label,
        "drop_fine_classes": parse_class_set(args.drop_fine_classes),
    }
    npz = {}
    for name, arrs in [("pred", preds), ("gt", gts)]:
        boxes = np.concatenate([p.boxes.reshape(-1, 4) for p in arrs]) if arrs else np.zeros((0, 4))
        scores = np.concatenate([p.scores.reshape(-1) for p in arrs]) if arrs else np.zeros((0,))
        labels = np.concatenate([p.labels.reshape(-1) for p in arrs]) if arrs else np.zeros((0,))
        idx = np.concatenate([np.full(len(p.boxes), i, dtype=np.int64) for i, p in enumerate(arrs)])
        npz[f"{name}_boxes"] = boxes.astype(np.float32)
        npz[f"{name}_scores"] = scores.astype(np.float32)
        npz[f"{name}_labels"] = labels.astype(np.int64)
        npz[f"{name}_sample_index"] = idx
    np.savez_compressed(out_dir / "val_predictions.npz", **npz)
    (out_dir / "val_sample_ids.json").write_text(json.dumps(ids, indent=2))

    elapsed = time.perf_counter() - t
    summary = {
        "pass": True,
        "map50": vr.map50,
        "map50_agnostic": vr.map50_agnostic,
        "localization_recall": vr.localization_recall,
        "classification_accuracy": vr.classification_accuracy,
        "n_gt": vr.n_gt,
        "n_pred": vr.n_pred,
        "per_class": vr.per_class,
        "per_bandwidth": bandwidth,
        "per_bandwidth_duration": bt,
        "confusion_gt_x_pred": vr.confusion,
        "stats": vr.stats,
        "elapsed_s": elapsed,
        "peak_alloc_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
        "meta": meta,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "map50": vr.map50,
        "map50_agnostic": vr.map50_agnostic,
        "localization_recall": vr.localization_recall,
        "classification_accuracy": vr.classification_accuracy,
        "n_gt": vr.n_gt, "n_pred": vr.n_pred,
        "n_views": vr.stats["n_views"],
        "elapsed_s": elapsed,
        "peak_reserved_gib": summary["peak_reserved_gib"],
        "per_class_ap": {k: v["ap"] for k, v in vr.per_class.items()},
        "per_bandwidth_map50": {k: v.get("map50") for k, v in bandwidth.items()},
    }, indent=2))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
