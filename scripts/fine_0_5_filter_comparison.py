#!/usr/bin/env python3
"""Merge comparison: both vs both with fine-view classes 0-5 removed.

Protocol: docs/collaboration/README.md, commit 67a453a.
Fixed best_corrected.pt, full 1543-sample val, one inference. Before global
merge:
  both      = all coarse + all fine predictions
  filtered  = all coarse + fine predictions whose predicted class is 6-13
              (fine predictions of class 0-5 are removed)
Both use the same class-wise NMS (IoU 0.5) and 100-box cap.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.signals.data import SPLIT_MANIFEST, load_manifest
from src.signals.infer import (
    Prediction, _boundary_mask, _map_view_dets, _nms_view_batch,
    global_classwise_nms, gt_global,
)
from src.signals.metrics import ImagePredictions, ImageTargets, evaluate
from src.signals.model import build_model
from src.signals.validate import FullViewDataset, _collate_one
from scripts.view_source_attribution import (
    CLASS_GROUPS, GROUPS, cat_parts, evaluate_group,
)

ROOT = Path(__file__).resolve().parents[1]


def strip_examples(d):
    d = dict(d)
    d.pop("fp_examples", None)
    d.pop("fn_examples", None)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="yolo11s_c1_directreg_2048x512")
    ap.add_argument("--checkpoint", default="best_corrected.pt")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch-views", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--max-det-view", type=int, default=300)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run = ROOT / "runs" / args.run
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = run / ckpt
    norm_cfg = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm_cfg["lo"]), float(norm_cfg["hi"]))
    records = load_manifest(SPLIT_MANIFEST, split="val")
    if args.max_samples:
        records = records[: args.max_samples]
    sample_ids = [r.sample_id for r in records]
    model = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["ema"] if ck.get("ema") is not None else ck["model"])
    model.eval()

    ds = FullViewDataset(records, norm)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=args.workers,
        collate_fn=_collate_one, pin_memory=False,
        prefetch_factor=4 if args.workers else None,
        persistent_workers=args.workers > 0,
    )
    gts = []
    both_preds, filtered_preds = [], []
    removed_by_class = Counter()
    removed_by_group = Counter()
    boundary_total = 0
    started = time.perf_counter()

    for ni, item in enumerate(loader):
        rec = records[item["index"]]
        F = int(round(rec.B_mhz * 700))
        views = list(zip(item["tensors"], item["view_types"], item["starts"]))
        coarse_parts, fine_parts = [], []
        for begin in range(0, len(views), args.batch_views):
            chunk = views[begin:begin + args.batch_views]
            images = torch.stack([v[0] for v in chunk], 0).cuda()
            dets = _nms_view_batch(
                model, images, conf=args.conf, iou=args.iou,
                max_det=args.max_det_view, multi_label=False,
            )
            for (tensor, vt, st), det in zip(chunk, dets):
                pred = _map_view_dets(det, vt, st, F)
                bm = _boundary_mask(pred, vt, st, F)
                if bm.any():
                    boundary_total += int(bm.sum())
                post = Prediction(pred.boxes[~bm], pred.scores[~bm], pred.labels[~bm]) if bm.any() else pred
                (coarse_parts if vt == "coarse" else fine_parts).append(post)

        # both: all coarse + all fine
        both = global_classwise_nms(cat_parts(coarse_parts + fine_parts), iou=args.iou, max_det=args.max_det)
        both_preds.append(ImagePredictions(both.boxes.numpy(), both.scores.numpy(), both.labels.numpy()))

        # filtered: remove fine predictions with predicted class 0-5
        fine_kept = []
        for p in fine_parts:
            if len(p.labels) == 0:
                continue
            mask = p.labels > 5
            removed = int((~mask).sum())
            if removed:
                for c in p.labels[~mask].tolist():
                    removed_by_class[int(c)] += 1
                    for g, classes in CLASS_GROUPS.items():
                        if int(c) in classes:
                            removed_by_group[g] += 1
            if mask.any():
                fine_kept.append(Prediction(p.boxes[mask], p.scores[mask], p.labels[mask]))
        filtered = global_classwise_nms(cat_parts(coarse_parts + fine_kept), iou=args.iou, max_det=args.max_det)
        filtered_preds.append(ImagePredictions(
            filtered.boxes.numpy(), filtered.scores.numpy(), filtered.labels.numpy(),
        ))

        h = (rec.N + 510) // 511
        gt = gt_global(rec, h)
        gts.append(ImageTargets(gt.boxes.numpy(), gt.labels.numpy()))
        if (ni + 1) % 100 == 0 or ni + 1 == len(records):
            print(f"{ni+1}/{len(records)} samples, {time.perf_counter()-started:.0f}s", flush=True)

    evals = {}
    for name, preds in [("both", both_preds), ("filtered", filtered_preds)]:
        ev = evaluate(gts, preds, iou_thresholds=(0.5,), max_dets=args.max_det)
        evals[name] = ev
        pi = ev["per_iou"]["0.5"]
        print(f"{name}: mAP50={pi['map_class_aware']:.6f} agn={pi['ap_class_agnostic']:.6f} "
              f"R_agn={ev['classification']['localization_recall']:.6f} "
              f"n_pred={sum(len(p.boxes) for p in preds)}", flush=True)

    baseline = json.loads((run / "stage5_val" / "summary.json").read_text())
    reproduction = {
        "baseline_map50": baseline["map50"],
        "both_map50": evals["both"]["per_iou"]["0.5"]["map_class_aware"],
        "baseline_n_pred": baseline["n_pred"],
        "both_n_pred": sum(len(p.boxes) for p in both_preds),
        "map_delta": evals["both"]["per_iou"]["0.5"]["map_class_aware"] - baseline["map50"],
        "n_pred_delta": sum(len(p.boxes) for p in both_preds) - baseline["n_pred"],
    }
    print("reproduction", json.dumps(reproduction, indent=2), flush=True)

    out = {
        "protocol": {
            "run": args.run, "checkpoint": str(ckpt),
            "conf": args.conf, "iou": args.iou,
            "max_det_view": args.max_det_view, "max_det": args.max_det,
            "n_samples": len(records),
            "filter": "remove fine-view predictions whose predicted class is 0-5 before global merge; keep all coarse and fine 6-13",
        },
        "reproduction": reproduction,
        "boundary_total": boundary_total,
        "removed_fine_0_5": {
            "total": int(sum(removed_by_class.values())),
            "by_class": {str(k): int(v) for k, v in sorted(removed_by_class.items())},
            "by_group": {k: int(v) for k, v in removed_by_group.items()},
        },
        "groups": {},
    }
    for name in ["both", "filtered"]:
        pi = evals[name]["per_iou"]["0.5"]
        entry = {
            "mAP50": pi["map_class_aware"],
            "ap50_agnostic": pi["ap_class_agnostic"],
            "localization_recall": evals[name]["classification"]["localization_recall"],
            "classification_accuracy": evals[name]["classification"]["classification_accuracy"],
            "n_pred": sum(len(p.boxes) for p in (both_preds if name == "both" else filtered_preds)),
            "per_class": pi["per_class"],
            "groups": {},
        }
        for g in GROUPS:
            gm = evaluate_group(
                name, gts,
                both_preds if name == "both" else filtered_preds,
                CLASS_GROUPS[g], records, sample_ids,
            )
            aps = [pi["per_class"][str(c)]["ap"] for c in CLASS_GROUPS[g]
                   if str(c) in pi["per_class"] and pi["per_class"][str(c)]["ap"] is not None]
            gm["group_class_ap_mean"] = float(sum(aps) / len(aps)) if aps else None
            gm["class_ap"] = {str(c): (pi["per_class"].get(str(c), {}) or {}).get("ap")
                               for c in CLASS_GROUPS[g]}
            entry["groups"][g] = strip_examples(gm)
        # all-class totals for FP counts
        allg = evaluate_group(
            name, gts,
            both_preds if name == "both" else filtered_preds,
            list(range(14)), records, sample_ids,
        )
        entry["all"] = strip_examples(allg)
        out["groups"][name] = entry

    out["elapsed_s"] = time.perf_counter() - started
    out_path = Path(args.out) if args.out else (run / "stage5_val" / "fine_0_5_filter_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("wrote", out_path)

    def brief(e):
        return {
            "mAP50": e["mAP50"], "ap50_agnostic": e["ap50_agnostic"],
            "r_agn": e["localization_recall"], "n_pred": e["n_pred"],
            "tp/fp/fn": [e["all"]["tp"], e["all"]["fp"], e["all"]["fn"]],
            "fp_type": [e["all"]["fp_misclass"], e["all"]["fp_duplicate"],
                        e["all"]["fp_background"]],
            "fp_high": [e["all"]["fp_high_misclass"], e["all"]["fp_high_duplicate"],
                        e["all"]["fp_high_background"]],
            "groups": {g: {
                "ap": e["groups"][g]["group_class_ap_mean"],
                "r_agn": e["groups"][g]["r_agn"],
                "tp/fp/fn": [e["groups"][g]["tp"], e["groups"][g]["fp"], e["groups"][g]["fn"]],
            } for g in GROUPS},
        }
    print(json.dumps({
        "reproduction": out["reproduction"],
        "removed_fine_0_5": out["removed_fine_0_5"],
        "both": brief(out["groups"]["both"]),
        "filtered": brief(out["groups"]["filtered"]),
    }, indent=2))


if __name__ == "__main__":
    main()
