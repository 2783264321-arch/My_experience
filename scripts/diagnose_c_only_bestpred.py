#!/usr/bin/env python3
"""Read-only: how does the frozen coarse-only model (B) behave on the narrow
GT instances that the all-view model (C) finds but B misses (c_only)?

For every c_only narrow GT, scan B's narrow predictions for the best-IoU box
and its score. Splitting the best-IoU distribution tells whether the miss is
score-ranking of correct boxes, poor localization, or no firing at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.signals.data import SPLIT_MANIFEST, load_manifest
from src.signals.metrics import ImagePredictions, ImageTargets, _match_single

ROOT = Path(__file__).resolve().parents[1]
C_ALL = Path("/root/autodl-tmp/workspace/baseline/runs/yolo11s_c1_directreg_2048x512/stage5_val")
C_COARSE = ROOT / "runs/regression/c_coarse"
NARROW = (9, 12, 13)


def load_preds(val_dir: Path):
    z = np.load(val_dir / "val_predictions.npz")
    ids = json.loads((val_dir / "val_sample_ids.json").read_text())
    per = {}
    for i, sid in enumerate(ids):
        pm = z["pred_sample_index"] == i
        gm = z["gt_sample_index"] == i
        per[sid] = {"pred": (z["pred_boxes"][pm], z["pred_scores"][pm], z["pred_labels"][pm]),
                    "gt_boxes": z["gt_boxes"][gm], "gt_labels": z["gt_labels"][gm]}
    return per


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    pred_c = load_preds(C_ALL)
    pred_b = load_preds(C_COARSE)
    rows = []
    for rec in load_manifest(SPLIT_MANIFEST, "val"):
        sid = rec.sample_id
        if sid not in pred_c or sid not in pred_b:
            continue
        pc, pb = pred_c[sid], pred_b[sid]
        gt = ImageTargets(pc["gt_boxes"], pc["gt_labels"])
        mc, _ = _match_single(gt, ImagePredictions(*pc["pred"]), 0.5, class_aware=False, max_dets=100)
        mb, _ = _match_single(gt, ImagePredictions(*pb["pred"]), 0.5, class_aware=False, max_dets=100)
        mc = {gi for pi, gi in mc}
        mb = {gi for pi, gi in mb}
        pb_boxes, pb_scores, pb_labels = pb["pred"]
        for gi, (box, lab) in enumerate(zip(pc["gt_boxes"], pc["gt_labels"])):
            if int(lab) not in NARROW:
                continue
            if gi in mc and gi not in mb:  # c_only
                best_iou, best_score = 0.0, None
                for b, s, l in zip(pb_boxes, pb_scores, pb_labels):
                    if int(l) not in NARROW:
                        continue
                    v = iou([float(x) for x in box], [float(x) for x in b])
                    if v > best_iou:
                        best_iou, best_score = v, float(s)
                rows.append({"sid": sid, "B": rec.B_mhz, "cls": int(lab), "best_iou": best_iou,
                             "best_score": best_score,
                             "h_rows": abs(float(box[3]) - float(box[1])) * 2048})

    (ROOT / "reports/coarse_c_only_bestpred.json").write_text(json.dumps(rows, indent=1))
    bands = [(0.5, 9.0, "IoU>=0.5 (matched: score/NMS cut)"),
             (0.3, 0.5, "IoU 0.3-0.5 (near miss)"),
             (0.1, 0.3, "IoU 0.1-0.3"),
             (0.0, 0.1, "IoU<0.1 (no fire)")]
    lines = [f"# c_only vs frozen coarse-only (B) predictions", "",
             f"c_only narrow GT: {len(rows)} (C finds them at IoU>=0.5, B does not).",
             "Best-IoU B narrow prediction per instance:", "",
             "| bucket | n | frac | score med | score p90 |", "| --- | ---: | ---: | ---: | ---: |"]
    for lo, hi, tag in bands:
        g = [r for r in rows if lo <= r["best_iou"] < hi]
        sc = [r["best_score"] for r in g if r["best_score"] is not None]
        lines.append(f"| {tag} | {len(g)} | {len(g)/len(rows):.1%} | "
                     f"{np.median(sc):.3f} | {np.percentile(sc,90):.3f} |" if sc else
                     f"| {tag} | {len(g)} | {len(g)/len(rows):.1%} | n/a | n/a |")
    h = [r["h_rows"] for r in rows]
    lines += ["", f"best-IoU median {np.median([r['best_iou'] for r in rows]):.3f}; "
              f"GT height rows med {np.median(h):.1f} (p25 {np.percentile(h,25):.1f}, "
              f"p75 {np.percentile(h,75):.1f})", ""]
    (ROOT / "reports/coarse_c_only_bestpred.md").write_text("\n".join(lines).rstrip("\n") + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
