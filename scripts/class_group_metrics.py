#!/usr/bin/env python3
"""Class-group metrics from a validate_run.py val_predictions.npz.

Reports both conventions used in the project:
  - mean per-class AP over the group (matches the earlier attribution tables);
  - group-filtered AP / AP_agn / recall (GT and predictions restricted to the
    group's classes, then the same strict evaluator).

Usage:
  PYTHONPATH=. .venv/bin/python scripts/class_group_metrics.py \
    --val-dir runs/<run>/stage5_val --name A \
    --output reports/group_a_class_groups.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.signals.data import SPLIT_MANIFEST, load_manifest
from src.signals.metrics import ImagePredictions, ImageTargets, evaluate

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "wide_0_5": list(range(6)),
    "0/1/2": [0, 1, 2],
    "3/4/5": [3, 4, 5],
    "10/11": [10, 11],
    "9/12/13": [9, 12, 13],
}


def load_predictions(val_dir: Path):
    z = np.load(val_dir / "val_predictions.npz")
    ids = json.loads((val_dir / "val_sample_ids.json").read_text())
    records = load_manifest(SPLIT_MANIFEST, split="val")
    by_id = {r.sample_id: r for r in records}
    recs = [by_id[s] for s in ids]
    gts = []
    preds = []
    for i in range(len(recs)):
        gm = z["gt_sample_index"] == i
        pm = z["pred_sample_index"] == i
        gts.append(ImageTargets(z["gt_boxes"][gm], z["gt_labels"][gm]))
        preds.append(ImagePredictions(
            z["pred_boxes"][pm], z["pred_scores"][pm], z["pred_labels"][pm],
        ))
    for i, r in enumerate(recs):
        if int((z["gt_sample_index"] == i).sum()) != len(r.signals):
            raise SystemExit(f"sample {i} GT count mismatch")
    return recs, gts, preds


def group_result(gts, preds, classes):
    keep = set(int(c) for c in classes)
    g = [
        ImageTargets(
            gt.boxes[np.isin(gt.labels, list(keep))],
            gt.labels[np.isin(gt.labels, list(keep))],
        )
        for gt in gts
    ]
    p = [
        ImagePredictions(
            pr.boxes[np.isin(pr.labels, list(keep))],
            pr.scores[np.isin(pr.labels, list(keep))],
            pr.labels[np.isin(pr.labels, list(keep))],
        )
        for pr in preds
    ]
    r = evaluate(g, p, iou_thresholds=(0.5,), max_dets=100)
    pi = r["per_iou"]["0.5"]
    return {
        "n_gt": int(pi["n_gt"]),
        "n_pred": int(sum(len(x.boxes) for x in p)),
        "map50_group_filtered": pi["map_class_aware"],
        "ap50_agnostic_group_filtered": pi["ap_class_agnostic"],
        "localization_recall_group_filtered": r["classification"]["localization_recall"],
        "classification_accuracy_group_filtered": r["classification"]["classification_accuracy"],
        "per_class": pi["per_class"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--output", default="")
    ap.add_argument("--markdown", default="")
    args = ap.parse_args()

    val_dir = Path(args.val_dir)
    if not val_dir.is_absolute():
        val_dir = ROOT / val_dir
    recs, gts, preds = load_predictions(val_dir)

    overall = evaluate(gts, preds, iou_thresholds=(0.5,), max_dets=100)
    opi = overall["per_iou"]["0.5"]
    per_class = opi["per_class"]
    result = {
        "name": args.name or str(val_dir),
        "n_samples": len(recs),
        "overall": {
            "n_gt": int(opi["n_gt"]),
            "n_pred": int(sum(len(p.boxes) for p in preds)),
            "map50": opi["map_class_aware"],
            "ap50_agnostic": opi["ap_class_agnostic"],
            "localization_recall": overall["classification"]["localization_recall"],
            "classification_accuracy": overall["classification"]["classification_accuracy"],
            "per_class": per_class,
        },
        "class_groups": {},
    }
    for name, classes in GROUPS.items():
        gr = group_result(gts, preds, classes)
        aps = []
        for c in classes:
            entry = per_class.get(str(c))
            if entry and entry.get("ap") is not None and entry.get("n_gt", 0) > 0:
                aps.append(float(entry["ap"]))
        gr["mean_per_class_ap"] = float(np.mean(aps)) if aps else None
        gr["classes"] = classes
        result["class_groups"][name] = gr

    out = Path(args.output) if args.output else val_dir / "class_group_metrics.json"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)

    if args.markdown:
        md = Path(args.markdown)
        if not md.is_absolute():
            md = ROOT / md
        md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Class-group metrics: {result['name']}",
            "",
            "| group | n_gt | mean per-class AP | group AP | AP_agn | R_agn | cls acc |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, gr in result["class_groups"].items():
            lines.append(
                f"| {name} | {gr['n_gt']} | {gr['mean_per_class_ap']:.4f} | "
                f"{gr['map50_group_filtered']:.4f} | "
                f"{gr['ap50_agnostic_group_filtered']:.4f} | "
                f"{gr['localization_recall_group_filtered']:.4f} | "
                f"{gr['classification_accuracy_group_filtered']:.4f} |"
            )
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {md}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
