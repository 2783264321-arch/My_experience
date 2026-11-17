#!/usr/bin/env python3
"""Stage-5 error analysis from the saved full-val predictions.

Consumes runs/<run>/stage5_val/{val_predictions.npz,val_sample_ids.json}
and reports per-class FP/FN counts, top false positives, missed GT, and
same-location multi-class duplicates. No test data.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.signals.data import SPLIT_MANIFEST, load_manifest
from src.signals.metrics import ImageTargets, ImagePredictions, iou_matrix

ROOT = Path(__file__).resolve().parents[1]


def load_npz(path: Path):
    z = np.load(path)
    out = {}
    for name in ["pred_boxes", "pred_scores", "pred_labels", "pred_sample_index",
                 "gt_boxes", "gt_scores", "gt_labels", "gt_sample_index"]:
        out[name] = z[name] if name in z else None
    return out


def split_by_sample(boxes, scores, labels, index, n):
    out = []
    for i in range(n):
        m = index == i
        out.append((boxes[m], scores[m], labels[m]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="yolo11s_c1_directreg_2048x512")
    ap.add_argument("--val-dir", default="")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    vdir = Path(args.val_dir) if args.val_dir else ROOT / "runs" / args.run / "stage5_val"
    z = load_npz(vdir / "val_predictions.npz")
    ids = json.loads((vdir / "val_sample_ids.json").read_text())
    records = load_manifest(SPLIT_MANIFEST, split="val")
    by_id = {r.sample_id: r for r in records}
    n = len(ids)
    preds = split_by_sample(z["pred_boxes"], z["pred_scores"], z["pred_labels"], z["pred_sample_index"], n)
    gts = split_by_sample(z["gt_boxes"], z["gt_scores"], z["gt_labels"], z["gt_sample_index"], n)

    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0})
    fps = []
    fns = []
    multi_class_groups = 0
    duplicate_groups = 0
    for i, sid in enumerate(ids):
        r = by_id.get(sid)
        pb, ps, pl = preds[i]
        gb, _, gl = gts[i]
        for c in range(14):
            pm = np.where(pl == c)[0]
            gm = np.where(gl == c)[0]
            per_class[c]["gt"] += len(gm)
            per_class[c]["pred"] += len(pm)
            if len(pm) == 0 and len(gm) == 0:
                continue
            if len(pm) == 0:
                per_class[c]["fn"] += len(gm)
                for gi in gm:
                    fns.append({
                        "sample_id": sid, "class": int(c),
                        "gt": gb[gi].round(5).tolist(),
                        "B": r.B_mhz if r else None, "T": r.T_ms if r else None,
                        "reason": "no prediction of this class",
                    })
                continue
            if len(gm) == 0:
                per_class[c]["fp"] += len(pm)
                continue
            order = np.argsort(-ps[pm], kind="stable")
            pm = pm[order]
            ious = iou_matrix(gb[gm], pb[pm])
            used = set()
            matched = set()
            for k, pi in enumerate(pm):
                j = int(np.argmax(ious[:, k]))
                if ious[j, k] >= args.iou and j not in used:
                    used.add(j)
                    matched.add(int(pi))
                    per_class[c]["tp"] += 1
                else:
                    per_class[c]["fp"] += 1
                    jj = int(np.argmax(ious[:, k]))
                    fps.append({
                        "sample_id": sid, "class": int(c),
                        "score": float(ps[pi]), "box": pb[pi].round(5).tolist(),
                        "best_gt_iou": float(ious[jj, k]),
                        "best_gt_class": int(gl[gm[jj]]) if len(gm) else None,
                        "B": r.B_mhz if r else None, "T": r.T_ms if r else None,
                    })
            for j, gi in enumerate(gm):
                if j not in used:
                    per_class[c]["fn"] += 1
                    jj = int(np.argmax(ious[j]))
                    fns.append({
                        "sample_id": sid, "class": int(c),
                        "gt": gb[gi].round(5).tolist(),
                        "best_pred_iou": float(ious[j, jj]),
                        "best_pred_class": int(pl[pm[jj]]),
                        "best_pred_score": float(ps[pm[jj]]),
                        "B": r.B_mhz if r else None, "T": r.T_ms if r else None,
                        "reason": "unmatched GT",
                    })
        # same-location duplicate / multi-class predictions
        if len(pb) > 1:
            ious = iou_matrix(pb, pb)
            seen = set()
            for a in range(len(pb)):
                if a in seen:
                    continue
                group = [b for b in range(len(pb)) if ious[a, b] >= 0.9 and b not in seen]
                for b in group:
                    seen.add(b)
                if len(group) > 1:
                    classes = {int(pl[b]) for b in group}
                    if len(classes) > 1:
                        multi_class_groups += 1
                    else:
                        duplicate_groups += 1

    fps.sort(key=lambda x: -x["score"])
    fns.sort(key=lambda x: -float(x.get("best_pred_iou", -1.0)))
    summary = {
        "n_samples": n,
        "iou_threshold": args.iou,
        "per_class": {str(k): v for k, v in sorted(per_class.items())},
        "totals": {
            "gt": int(sum(v["gt"] for v in per_class.values())),
            "pred": int(sum(v["pred"] for v in per_class.values())),
            "tp": int(sum(v["tp"] for v in per_class.values())),
            "fp": int(sum(v["fp"] for v in per_class.values())),
            "fn": int(sum(v["fn"] for v in per_class.values())),
        },
        "same_location_groups": {
            "multi_class": multi_class_groups,
            "same_class_duplicate": duplicate_groups,
        },
        "top_false_positives": fps[: args.top],
        "top_missed_gt": fns[: args.top],
    }
    out = vdir / "error_analysis.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "totals": summary["totals"],
        "per_class": summary["per_class"],
        "same_location_groups": summary["same_location_groups"],
        "top_fp": summary["top_false_positives"][:5],
        "top_fn": summary["top_missed_gt"][:5],
    }, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
