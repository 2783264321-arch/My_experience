#!/usr/bin/env python3
"""AP50/AP75/AP50:95 sweep on the saved full-val predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.signals.metrics import ImagePredictions, ImageTargets, evaluate

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="yolo11s_c1_directreg_2048x512")
    ap.add_argument("--val-dir", default="")
    ap.add_argument("--max-dets", type=int, default=100)
    args = ap.parse_args()
    vdir = Path(args.val_dir) if args.val_dir else ROOT / "runs" / args.run / "stage5_val"
    z = np.load(vdir / "val_predictions.npz")
    ids = json.loads((vdir / "val_sample_ids.json").read_text())
    n = len(ids)
    pred_idx = z["pred_sample_index"]
    gt_idx = z["gt_sample_index"]
    preds, gts = [], []
    for i in range(n):
        pm = pred_idx == i
        gm = gt_idx == i
        preds.append(ImagePredictions(
            z["pred_boxes"][pm], z["pred_scores"][pm], z["pred_labels"][pm],
        ))
        gts.append(ImageTargets(z["gt_boxes"][gm], z["gt_labels"][gm]))
    thresholds = tuple(round(float(x), 2) for x in np.arange(0.5, 0.96, 0.05))
    out = evaluate(gts, preds, iou_thresholds=thresholds, max_dets=args.max_dets)
    rows = []
    for t in thresholds:
        pi = out["per_iou"][str(t)]
        rows.append({
            "iou": t,
            "map_class_aware": pi["map_class_aware"],
            "ap_class_agnostic": pi["ap_class_agnostic"],
        })
    maps = [r["map_class_aware"] for r in rows if r["map_class_aware"] is not None]
    agns = [r["ap_class_agnostic"] for r in rows if r["ap_class_agnostic"] is not None]
    per_class = {}
    for c in out["per_iou"]["0.5"]["per_class"]:
        vals = []
        for t in thresholds:
            v = out["per_iou"][str(t)]["per_class"].get(c)
            if v and v["ap"] is not None:
                vals.append(v["ap"])
        per_class[c] = float(np.mean(vals)) if vals else None
    summary = {
        "thresholds": list(thresholds),
        "rows": rows,
        "ap50": rows[0]["map_class_aware"],
        "ap75": out["per_iou"]["0.75"]["map_class_aware"],
        "ap50_95": float(np.mean(maps)) if maps else None,
        "ap50_agnostic": rows[0]["ap_class_agnostic"],
        "ap75_agnostic": out["per_iou"]["0.75"]["ap_class_agnostic"],
        "ap50_95_agnostic": float(np.mean(agns)) if agns else None,
        "per_class_ap50_95": per_class,
        "n_samples": n,
        "max_dets": args.max_dets,
    }
    path = vdir / "iou_sweep.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "ap50": summary["ap50"], "ap75": summary["ap75"],
        "ap50_95": summary["ap50_95"],
        "ap50_agnostic": summary["ap50_agnostic"],
        "ap75_agnostic": summary["ap75_agnostic"],
        "ap50_95_agnostic": summary["ap50_95_agnostic"],
        "per_threshold": {str(r["iou"]): r["map_class_aware"] for r in rows},
        "per_class_ap50_95": per_class,
    }, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
