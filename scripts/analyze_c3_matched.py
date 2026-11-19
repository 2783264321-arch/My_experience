#!/usr/bin/env python3
"""Matched-box 3/4/5 classification analysis for paired C3 checkpoints.

For each GT box in classes 3/4/5, match predictions class-agnostically at
IoU>=0.5 for both models. The "common" set contains GT boxes that both models
localized correctly; compare their predicted classes only on that set. This
separates classification from localization/ranking.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.signals.metrics import ImagePredictions, ImageTargets, _match_single, iou_matrix

ROOT = Path(__file__).resolve().parents[1]
CLASSES = [3, 4, 5]


def load_npz(path):
    z = np.load(Path(path))
    ids = None
    return z


def per_sample(z, name):
    boxes, scores, labels, idx = z[f"{name}_boxes"], z[f"{name}_scores"], z[f"{name}_labels"], z[f"{name}_sample_index"]
    out = []
    for i in range(int(idx.max()) + 1 if len(idx) else 0):
        m = idx == i
        out.append((boxes[m], scores[m], labels[m]))
    return out


def mcnemar(records):
    """Exact McNemar for dense-correct vs zero3-correct on common boxes."""
    a_only = sum(1 for r in records if r["pred_a"] == r["gt"] and r["pred_b"] != r["gt"])
    b_only = sum(1 for r in records if r["pred_a"] != r["gt"] and r["pred_b"] == r["gt"])
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        # two-sided exact binomial, p=0.5
        from math import comb
        k = min(a_only, b_only)
        p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
        p = min(1.0, p)
    return {"dense_only_correct": a_only, "zero3_only_correct": b_only, "p_two_sided": p}


def analyze_pair(a_npz, b_npz, a_name, b_name):
    a = load_npz(a_npz)
    b = load_npz(b_npz)
    ga = per_sample(a, "gt")
    # GT is identical in both files; use a.
    pa = per_sample(a, "pred")
    pb = per_sample(b, "pred")
    n_gt_345 = 0
    matched_a = matched_b = common = 0
    correct_a = correct_b = 0
    common_correct_a = common_correct_b = 0
    conf_a = Counter()
    conf_b = Counter()
    common_conf_a = Counter()
    common_conf_b = Counter()
    per_a = Counter()
    per_b = Counter()
    ious_common = []
    common_records = []
    for i, (gb, gs, gl) in enumerate(ga):
        pb_a = pa[i] if i < len(pa) else (np.zeros((0,4)), np.zeros((0,)), np.zeros((0,),dtype=int))
        pb_b = pb[i] if i < len(pb) else (np.zeros((0,4)), np.zeros((0,)), np.zeros((0,),dtype=int))
        gt = ImageTargets(gb, gl)
        pred_a = ImagePredictions(*pb_a)
        pred_b = ImagePredictions(*pb_b)
        ma, _ = _match_single(gt, pred_a, 0.5, class_aware=False, max_dets=100)
        mb, _ = _match_single(gt, pred_b, 0.5, class_aware=False, max_dets=100)
        amap = {gi: pi for pi, gi in ma}
        bmap = {gi: pi for pi, gi in mb}
        for gi, c in enumerate(gl):
            if int(c) not in CLASSES:
                continue
            n_gt_345 += 1
            has_a, has_b = gi in amap, gi in bmap
            if has_a:
                matched_a += 1
                pi = amap[gi]
                pc = int(pred_a.labels[pi])
                conf_a[(int(c), pc)] += 1
                per_a[int(c)] += 1
                if pc == int(c):
                    correct_a += 1
            if has_b:
                matched_b += 1
                pi = bmap[gi]
                pc = int(pred_b.labels[pi])
                conf_b[(int(c), pc)] += 1
                per_b[int(c)] += 1
                if pc == int(c):
                    correct_b += 1
            if has_a and has_b:
                common += 1
                ia, ib = amap[gi], bmap[gi]
                ca, cb = int(pred_a.labels[ia]), int(pred_b.labels[ib])
                common_conf_a[(int(c), ca)] += 1
                common_conf_b[(int(c), cb)] += 1
                if ca == int(c):
                    common_correct_a += 1
                if cb == int(c):
                    common_correct_b += 1
                iou_a = float(iou_matrix(gb[gi:gi+1], pred_a.boxes[ia:ia+1])[0,0])
                iou_b = float(iou_matrix(gb[gi:gi+1], pred_b.boxes[ib:ib+1])[0,0])
                ious_common.append(iou_a)
                common_records.append({"sample_index": i, "gt": int(c),
                                       "pred_a": ca, "pred_b": cb,
                                       "iou_a": iou_a, "iou_b": iou_b})

    def conf_table(counter):
        rows = {}
        for g in CLASSES:
            row = {str(p): int(counter.get((g,p), 0)) for p in CLASSES}
            row["other"] = int(sum(v for (gg,p),v in counter.items() if gg==g and p not in CLASSES))
            rows[str(g)] = row
        return rows
    def acc(correct, n):
        return float(correct / n) if n else None
    return {
        "pair": [a_name, b_name],
        "n_gt_345": n_gt_345,
        "matched_a": matched_a, "matched_b": matched_b, "common": common,
        "recall_a": acc(matched_a, n_gt_345),
        "recall_b": acc(matched_b, n_gt_345),
        "all_matched_accuracy_a": acc(correct_a, matched_a),
        "all_matched_accuracy_b": acc(correct_b, matched_b),
        "common_accuracy_a": acc(common_correct_a, common),
        "common_accuracy_b": acc(common_correct_b, common),
        "common_confusion_a": conf_table(common_conf_a),
        "common_confusion_b": conf_table(common_conf_b),
        "all_confusion_a": conf_table(conf_a),
        "all_confusion_b": conf_table(conf_b),
        "common_median_iou": float(np.median(ious_common)) if ious_common else None,
        "common_iou_p10": float(np.percentile(ious_common, 10)) if ious_common else None,
        "mcnemar": mcnemar(common_records),
        "common_records": common_records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    out = analyze_pair(args.a, args.b, args.a_name, args.b_name)
    print(json.dumps(out, indent=2))
    if args.output:
        p = Path(args.output)
        if not p.is_absolute():
            p = ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
