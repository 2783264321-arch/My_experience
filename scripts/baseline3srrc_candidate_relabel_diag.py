#!/usr/bin/env python3
"""Truth-based relabel diagnostic on frozen YOLO26l val candidates.

Uses the formal YOLO26l best.pt full-val predictions (filtered 0-5
protocol) and GT labels only. For SRRC 10/11 and Wi-Fi 20 MHz 0/1/2:
  - count predicted candidates, matched/unmatched/duplicate;
  - measure matched-class confusion;
  - compute the oracle AP change if every matched misclassified
    candidate were relabelled to its GT class (fixed boxes/scores/NMS).

This is an upper bound, not a classifier result. No training, no detector
inference, no test usage.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from src.signals.data import SPLIT_MANIFEST, load_manifest, N_TIME
from src.signals.infer import gt_global
from src.signals.metrics import ImageTargets, ImagePredictions, evaluate, iou_matrix

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRED = Path(
    "/root/autodl-tmp/worktrees/formal-yolo26/runs/"
    "yolo26l_c1_2048x512/val_full_best")
OUT = ROOT / "reports/baseline3_srrc_phase0"

GROUPS = {
    "srrc_10mhz": {
        "classes": (10, 11),
        "nominal_bw": 10.0,
        "bw_tol": 0.5,
        "gt_width_tol": 0.5,
    },
    "srrc_all_labels": {
        "classes": (10, 11),
        "nominal_bw": 10.0,
        "bw_tol": 0.5,
        "gt_width_tol": None,
    },
    "wifi20_20mhz": {
        "classes": (0, 1, 2),
        "nominal_bw": 20.0,
        "bw_tol": 0.5,
        "gt_width_tol": 0.5,
    },
    "wifi20_all_labels": {
        "classes": (0, 1, 2),
        "nominal_bw": 20.0,
        "bw_tol": 0.5,
        "gt_width_tol": None,
    },
}


def match_group(gt: ImageTargets, pred: ImagePredictions, classes, iou_thr=0.5,
                max_dets=100):
    """One-to-one, class-agnostic matching inside a target class group."""
    pidx = np.where(np.isin(pred.labels, classes))[0]
    gidx = np.where(np.isin(gt.labels, classes))[0]
    if len(pidx):
        pidx = pidx[np.argsort(-pred.scores[pidx], kind="stable")][:max_dets]
    if len(pidx) == 0:
        return [], [], [], [int(g) for g in gidx]
    if len(gidx) == 0:
        return [], [], [int(p) for p in pidx], []
    ious = iou_matrix(gt.boxes[gidx], pred.boxes[pidx])
    used = set()
    matches = []
    unmatched = []
    for k, pi in enumerate(pidx):
        best = -1.0; best_j = -1
        for j in range(len(gidx)):
            if j in used:
                continue
            if ious[j, k] > best:
                best = float(ious[j, k]); best_j = j
        if best >= iou_thr:
            used.add(best_j)
            matches.append((int(pi), int(gidx[best_j]), best))
        else:
            unmatched.append(int(pi))
    # classify unmatched as duplicate (IoU>=thr with an already matched GT)
    # or no-GT (max IoU below threshold)
    duplicates = []; no_gt = []
    for pi in unmatched:
        k = int(np.where(pidx == pi)[0][0])
        max_iou = float(ious[:, k].max()) if len(gidx) else 0.0
        if max_iou >= iou_thr:
            duplicates.append(pi)
        else:
            no_gt.append(pi)
    missed_gt = [int(g) for j, g in enumerate(gidx) if j not in used]
    return matches, duplicates, no_gt, missed_gt


def ap_summary(gts, preds, iou_thresholds):
    r = evaluate(gts, preds, iou_thresholds=iou_thresholds, max_dets=100)
    out = {}
    for thr in iou_thresholds:
        key = str(thr)
        d = r["per_iou"][key]
        out[key] = {
            "map50": float(d["map_class_aware"] or 0.0),
            "per_class": {k: v["ap"] for k, v in d["per_class"].items()},
        }
    out["classification"] = r["classification"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED)
    ap.add_argument("--out-json", type=Path,
                    default=OUT / "candidate_relabel_diag.json")
    ap.add_argument("--out-csv", type=Path,
                    default=OUT / "candidate_relabel_wrong.csv")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sample_ids = json.loads((args.pred_dir / "val_sample_ids.json").read_text())
    recs = load_manifest(SPLIT_MANIFEST, split="val")
    by_id = {r.sample_id: r for r in recs}
    if [r.sample_id for r in recs] != sample_ids:
        # keep the npz order, but require exact membership
        recs = [by_id[s] for s in sample_ids]
    z = np.load(args.pred_dir / "val_predictions_filtered_0_5.npz")
    boxes = z["pred_boxes"]; scores = z["pred_scores"]
    labels = z["pred_labels"]; sidx = z["pred_sample_index"]
    preds = []
    for i in range(len(recs)):
        m = sidx == i
        preds.append(ImagePredictions(boxes[m], scores[m], labels[m]))
    gts = []
    for r in recs:
        p = gt_global(r, math.ceil(r.N / (N_TIME - 1)))
        gts.append(ImageTargets(p.boxes.numpy(), p.labels.numpy()))

    baseline = ap_summary(gts, preds, (0.5,))
    out = {
        "pred_dir": str(args.pred_dir),
        "protocol": "YOLO26l best.pt, filtered_0_5, val 1,543 samples",
        "baseline_map50": baseline["0.5"]["map50"],
        "baseline_per_class": baseline["0.5"]["per_class"],
        "groups": {},
        "note": ("Oracle relabel uses GT labels for matched candidates only; "
                 "unmatched predictions cannot be verified and are left "
                 "unchanged. This is an upper bound, not a deployable rule."),
    }
    wrong_rows = []
    for gname, cfg in GROUPS.items():
        classes = cfg["classes"]
        matches_all = []
        dup_all = []; nogt_all = []; miss_all = []
        conf = Counter()
        gt_diag_all = []
        for i, (gt, pred) in enumerate(zip(gts, preds)):
            gmask = np.isin(gt.labels, classes)
            if cfg["gt_width_tol"] is not None:
                gw = (gt.boxes[:, 3] - gt.boxes[:, 1]) * recs[i].B_mhz
                gmask &= np.abs(gw - cfg["nominal_bw"]) <= cfg["gt_width_tol"]
            gt_diag = ImageTargets(gt.boxes[gmask], gt.labels[gmask])
            gt_diag_all.append(gt_diag)
            matches, dups, nogt, missed = match_group(gt_diag, pred, classes)
            for pi, gi, iou in matches:
                pcls = int(pred.labels[pi]); gcls = int(gt_diag.labels[gi])
                conf[(gcls, pcls)] += 1
                width = float((pred.boxes[pi, 3] - pred.boxes[pi, 1]) *
                              recs[i].B_mhz)
                gw = float((gt_diag.boxes[gi, 3] - gt_diag.boxes[gi, 1]) *
                           recs[i].B_mhz)
                if pcls != gcls:
                    wrong_rows.append({
                        "group": gname, "sample_id": recs[i].sample_id,
                        "pred_idx": int(pi), "gt_label": gcls,
                        "pred_label": pcls, "score": float(pred.scores[pi]),
                        "iou": float(iou), "pred_width_mhz": width,
                        "gt_width_mhz": gw,
                    })
                matches_all.append((i, pi, gi, iou))
            dup_all += [(i, pi) for pi in dups]
            nogt_all += [(i, pi) for pi in nogt]
            miss_all += [(i, gi) for gi in missed]
        n_pred = int(sum(int(np.isin(p.labels, classes).sum()) for p in preds))
        n_gt = int(sum(len(g.boxes) for g in gt_diag_all))
        matched = len(matches_all)
        correct = sum(v for (g, p), v in conf.items() if g == p)
        group = {
            "classes": list(classes),
            "n_gt": n_gt,
            "n_pred": n_pred,
            "n_matched": matched,
            "n_unmatched_duplicate": len(dup_all),
            "n_unmatched_no_gt": len(nogt_all),
            "n_missed_gt": len(miss_all),
            "gt_recall": matched / max(n_gt, 1),
            "matched_correct": correct,
            "matched_wrong": matched - correct,
            "matched_accuracy": correct / max(matched, 1),
            "confusion": {f"gt{g}_pred{p}": v for (g, p), v in sorted(conf.items())},
        }
        # oracle corrections at several score thresholds / width routing.
        # The prediction arrays are mutated only in local copies.
        base_labels = [p.labels.copy() for p in preds]
        for tag, score_thr, width_route in (
            ("all_matched_wrong", 0.0, False),
            ("score_ge_0.3", 0.3, False),
            ("score_ge_0.5", 0.5, False),
            ("score_ge_0.7", 0.7, False),
            ("routed_score_ge_0.3", 0.3, True),
            ("routed_score_ge_0.5", 0.5, True),
            ("routed_score_ge_0.7", 0.7, True),
        ):
            new_labels = [x.copy() for x in base_labels]
            changed = 0; changed_wrong = 0
            for (i, pi, gi, iou) in matches_all:
                pcls = int(preds[i].labels[pi])
                gcls = int(gt_diag_all[i].labels[gi])
                if pcls == gcls:
                    continue
                if float(preds[i].scores[pi]) < score_thr:
                    continue
                width = float((preds[i].boxes[pi, 3] - preds[i].boxes[pi, 1]) *
                              recs[i].B_mhz)
                if width_route and abs(width - cfg["nominal_bw"]) > cfg["bw_tol"]:
                    continue
                new_labels[i][pi] = gcls
                changed += 1
                if pcls != gcls:
                    changed_wrong += 1
            corrected_preds = [
                ImagePredictions(p.boxes, p.scores, lab)
                for p, lab in zip(preds, new_labels)
            ]
            a = ap_summary(gts, corrected_preds, (0.5,))["0.5"]
            group[tag] = {
                "n_changed": changed,
                "map50": a["map50"],
                "delta_map50_pp": (a["map50"] - baseline["0.5"]["map50"]) * 100.0,
                "per_class": a["per_class"],
            }
        out["groups"][gname] = group
    # CSV of wrong matched candidates
    fields = ["group", "sample_id", "pred_idx", "gt_label", "pred_label",
              "score", "iou", "pred_width_mhz", "gt_width_mhz"]
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in sorted(wrong_rows, key=lambda x: (x["group"], -x["score"])):
            w.writerow(r)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({
        "baseline_map50": out["baseline_map50"],
        "groups": {
            k: {x: v[x] for x in ("n_gt", "n_pred", "n_matched",
                                  "n_unmatched_duplicate", "n_unmatched_no_gt",
                                  "n_missed_gt", "matched_correct",
                                  "matched_wrong", "matched_accuracy")}
            for k, v in out["groups"].items()
        },
        "oracle": {
            k: {t: v[t]["delta_map50_pp"] for t in
                ("all_matched_wrong", "score_ge_0.5", "routed_score_ge_0.5")}
            for k, v in out["groups"].items()
        },
    }, indent=2))
    print("wrote", args.out_json, args.out_csv)


if __name__ == "__main__":
    main()
