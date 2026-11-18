#!/usr/bin/env python3
"""View-source attribution on the frozen baseline.

Protocol (docs/collaboration/README.md, commit 40e94aa):
- one all-view inference on the full 1543-sample val set with best_corrected.pt;
- keep per-view predictions, split before global merge into coarse-only,
  fine-only and both;
- apply the same class-wise global NMS (IoU 0.5) and 100-box cap to each;
- reproduce both = baseline mAP50 0.642701 and n_pred 65025;
- report wide groups 0/1/2, 3/4/5, 10/11: class AP, class-agnostic recall,
  TP/FP/FN, FP taxonomy (misclass / duplicate / background), high-score FPs,
  boundary drops, near-duplicate GT effects, and a few representative errors.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.signals.data import SPLIT_MANIFEST, load_manifest
from src.signals.infer import (
    Prediction, _boundary_mask, _map_view_dets, _nms_view_batch,
    global_classwise_nms, gt_global,
)
from src.signals.metrics import (
    ImagePredictions, ImageTargets, _match_single, evaluate, iou_matrix,
)
from src.signals.model import build_model
from src.signals.validate import FullViewDataset, _collate_one

ROOT = Path(__file__).resolve().parents[1]
CLASS_GROUPS = {
    "0/1/2": [0, 1, 2],
    "3/4/5": [3, 4, 5],
    "10/11": [10, 11],
    "narrow_9/12/13": [9, 12, 13],
}
# Aliases for the report
GROUPS = ["0/1/2", "3/4/5", "10/11", "narrow_9/12/13"]


def cat_parts(parts):
    if not parts:
        return Prediction(torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long))
    return Prediction(
        torch.cat([p.boxes for p in parts]),
        torch.cat([p.scores for p in parts]),
        torch.cat([p.labels for p in parts]),
    )


def to_physical(box, B, T, f_lo, f_hi, h):
    x1, y1, x2, y2 = [float(v) for v in box]
    fs = B * 1e6
    t1 = max(0.0, min(T, ((x1 * 512 - 0.5) * h / fs) * 1000.0))
    t2 = max(0.0, min(T, ((x2 * 512 - 0.5) * h / fs) * 1000.0))
    f1 = max(f_lo, min(f_hi, f_lo + y1 * B))
    f2 = max(f_lo, min(f_hi, f_lo + y2 * B))
    return t1, t2, f1, f2


def evaluate_group(src_name, gts, preds, classes, records, sample_ids):
    """Strict class-aware group metrics + FP/FN taxonomy + GT coverage."""
    cls_set = set(classes)
    tp = fp = fn = 0
    r_agn_matched = 0
    n_gt_group = 0
    fp_cat = Counter()
    fp_high = Counter()
    fn_cat = Counter()
    gt_correct = gt_wrong = gt_miss = 0
    fp_examples = defaultdict(list)
    fn_examples = defaultdict(list)

    for i, (gt, pred) in enumerate(zip(gts, preds)):
        gb, gl = gt.boxes, gt.labels
        pb, ps, pl = pred.boxes, pred.scores, pred.labels
        rec = records[i]
        sid = sample_ids[i]
        hh = (rec.N + 510) // 511
        B, T, f_lo, f_hi = rec.B_mhz, rec.T_ms, rec.f_lo, rec.f_hi
        if len(gb) == 0:
            continue
        ious = iou_matrix(gb, pb) if len(pb) else np.zeros((len(gb), 0))
        gidx_group = [j for j, c in enumerate(gl) if int(c) in cls_set]
        n_gt_group += len(gidx_group)
        # class-agnostic matching for group recall and GT coverage
        am, _ = _match_single(gt, pred, 0.5, class_aware=False, max_dets=100)
        amap = {gi: pi for pi, gi in am}
        for gi in gidx_group:
            if gi in amap:
                r_agn_matched += 1
                pi = amap[gi]
                if int(pl[pi]) == int(gl[gi]):
                    gt_correct += 1
                else:
                    gt_wrong += 1
            else:
                gt_miss += 1
        # strict class-aware per class
        for c in classes:
            gm = np.where(gl == c)[0]
            pm = np.where(pl == c)[0]
            if len(gm) == 0 and len(pm) == 0:
                continue
            matches, unmatched = _match_single(
                gt, pred, 0.5, class_aware=True, class_id=c, max_dets=100,
            )
            matched_g = {gi for pi, gi in matches}
            tp += len(matches)
            fn += len(gm) - len(matches)
            for pi in unmatched:
                s = float(ps[pi])
                if len(gb):
                    j = int(np.argmax(ious[:, pi]))
                    best = float(ious[j, pi])
                else:
                    j, best = -1, 0.0
                if best >= 0.5:
                    cat = "duplicate" if int(gl[j]) == c else "misclass"
                else:
                    cat = "background"
                fp += 1
                fp_cat[cat] += 1
                if s >= 0.5:
                    fp_high[cat] += 1
                if s >= 0.25:
                    bp = to_physical(pb[pi], B, T, f_lo, f_hi, hh)
                    rec = {
                        "source": src_name, "group": "/".join(map(str, classes)),
                        "category": cat, "sample_id": sid,
                        "pred_class": c, "score": s,
                        "gt_class": int(gl[j]) if j >= 0 else None,
                        "gt_iou": best,
                        "box_norm": [round(float(x), 6) for x in pb[pi]],
                        "box_phys": [round(float(x), 5) for x in bp],
                        "gt_box_norm": [round(float(x), 6) for x in gb[j]] if j >= 0 else None,
                        "gt_box_phys": [round(float(x), 5) for x in
                                        to_physical(gb[j], B, T, f_lo, f_hi, hh)] if j >= 0 else None,
                    }
                    fp_examples[cat].append(rec)
            for gi in gm:
                if gi in matched_g:
                    continue
                if len(pm):
                    same = float(ious[gi, pm].max())
                else:
                    same = 0.0
                if len(pb):
                    j = int(np.argmax(ious[gi]))
                    any_iou = float(ious[gi, j])
                    any_cls = int(pl[j])
                    any_score = float(ps[j])
                else:
                    j, any_iou, any_cls, any_score = -1, 0.0, None, None
                if same >= 0.5:
                    cat = "near_duplicate_gt"
                elif any_iou >= 0.5:
                    cat = "wrong_class_covered"
                else:
                    cat = "missed"
                fn_cat[cat] += 1
                if any_iou >= 0.25 or same >= 0.25:
                    pred_box_norm = [round(float(x), 6) for x in pb[j]] if j >= 0 else None
                    pred_box_phys = [round(float(x), 5) for x in
                                     to_physical(pb[j], B, T, f_lo, f_hi, hh)] if j >= 0 else None
                    rec = {
                        "source": src_name, "group": "/".join(map(str, classes)),
                        "category": cat, "sample_id": sid, "pred_class": any_cls,
                        "best_iou": max(same, any_iou), "pred_score": any_score,
                        "gt_class": int(gl[gi]),
                        "gt_box_norm": [round(float(x), 6) for x in gb[gi]],
                        "gt_box_phys": [round(float(x), 5) for x in
                                        to_physical(gb[gi], B, T, f_lo, f_hi, hh)],
                        "pred_box_norm": pred_box_norm,
                        "pred_box_phys": pred_box_phys,
                    }
                    fn_examples[cat].append(rec)

    # group class AP from the full evaluator is added by the caller
    return {
        "n_gt": int(n_gt_group),
        "r_agn": float(r_agn_matched / n_gt_group) if n_gt_group else None,
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "fp_misclass": int(fp_cat["misclass"]),
        "fp_duplicate": int(fp_cat["duplicate"]),
        "fp_background": int(fp_cat["background"]),
        "fp_high_misclass": int(fp_high["misclass"]),
        "fp_high_duplicate": int(fp_high["duplicate"]),
        "fp_high_background": int(fp_high["background"]),
        "fn_near_duplicate_gt": int(fn_cat["near_duplicate_gt"]),
        "fn_wrong_class_covered": int(fn_cat["wrong_class_covered"]),
        "fn_missed": int(fn_cat["missed"]),
        "gt_correct": int(gt_correct),
        "gt_wrong_class": int(gt_wrong),
        "gt_missed": int(gt_miss),
        "fp_examples": {k: sorted(v, key=lambda x: -x["score"])[:8] for k, v in fp_examples.items()},
        "fn_examples": {k: sorted(v, key=lambda x: -x["best_iou"])[:8] for k, v in fn_examples.items()},
    }


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
    sample_ids = [r.sample_id for r in records]
    preds = {s: [] for s in ["coarse", "fine", "both"]}
    raw_counts = {s: 0 for s in ["coarse", "fine", "both"]}
    boundary_total = 0
    boundary_by_class = Counter()
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
                    for c in pred.labels[bm].tolist():
                        boundary_by_class[int(c)] += 1
                post = Prediction(pred.boxes[~bm], pred.scores[~bm], pred.labels[~bm]) if bm.any() else pred
                if vt == "coarse":
                    coarse_parts.append(post)
                else:
                    fine_parts.append(post)
        raw_counts["coarse"] += len(cat_parts(coarse_parts).boxes)
        raw_counts["fine"] += len(cat_parts(fine_parts).boxes)
        raw_counts["both"] += len(cat_parts(coarse_parts + fine_parts).boxes)
        for src, parts in [("coarse", coarse_parts), ("fine", fine_parts), ("both", coarse_parts + fine_parts)]:
            merged = global_classwise_nms(cat_parts(parts), iou=args.iou, max_det=args.max_det)
            preds[src].append(ImagePredictions(
                merged.boxes.numpy(), merged.scores.numpy(), merged.labels.numpy(),
            ))
        h = (rec.N + 510) // 511
        gt = gt_global(rec, h)
        gts.append(ImageTargets(gt.boxes.numpy(), gt.labels.numpy()))
        if (ni + 1) % 100 == 0 or ni + 1 == len(records):
            print(f"{ni+1}/{len(records)} samples, {time.perf_counter()-started:.0f}s", flush=True)

    evals = {}
    for src in ["coarse", "fine", "both"]:
        ev = evaluate(gts, preds[src], iou_thresholds=(0.5,), max_dets=args.max_det)
        evals[src] = ev
        pi = ev["per_iou"]["0.5"]
        print(f"{src}: mAP50={pi['map_class_aware']:.6f} agn={pi['ap_class_agnostic']:.6f} "
              f"n_pred={sum(len(p.boxes) for p in preds[src])}", flush=True)

    baseline = json.loads((run / "stage5_val" / "summary.json").read_text())
    both = evals["both"]["per_iou"]["0.5"]
    both_n = sum(len(p.boxes) for p in preds["both"])
    reproduction = {
        "baseline_map50": baseline["map50"],
        "both_map50": both["map_class_aware"],
        "baseline_n_pred": baseline["n_pred"],
        "both_n_pred": both_n,
        "map_delta": both["map_class_aware"] - baseline["map50"],
        "n_pred_delta": both_n - baseline["n_pred"],
    }
    print("reproduction", json.dumps(reproduction, indent=2), flush=True)

    out = {
        "protocol": {
            "run": args.run, "checkpoint": str(ckpt),
            "conf": args.conf, "iou": args.iou,
            "max_det_view": args.max_det_view, "max_det": args.max_det,
            "n_samples": len(records), "max_samples": args.max_samples,
            "note": "per-view NMS -> boundary filter -> source split -> same global class-wise NMS + 100 cap",
        },
        "reproduction": reproduction,
        "sources": {},
        "boundary": {
            "total": boundary_total,
            "by_class": {str(k): int(v) for k, v in sorted(boundary_by_class.items())},
            "wide_groups": {
                g: int(sum(boundary_by_class[c] for c in CLASS_GROUPS[g]))
                for g in GROUPS
            },
        },
    }
    for src in ["coarse", "fine", "both"]:
        pi = evals[src]["per_iou"]["0.5"]
        cls = {c: pi["per_class"].get(str(c)) for c in range(14)}
        src_out = {
            "mAP50": pi["map_class_aware"],
            "ap50_agnostic": pi["ap_class_agnostic"],
            "localization_recall": evals[src]["classification"]["localization_recall"],
            "classification_accuracy": evals[src]["classification"]["classification_accuracy"],
            "n_pred": sum(len(p.boxes) for p in preds[src]),
            "per_class": cls,
            "groups": {},
        }
        for g in GROUPS:
            classes = CLASS_GROUPS[g]
            gm = evaluate_group(src, gts, preds[src], classes, records, sample_ids)
            aps = [pi["per_class"][str(c)]["ap"] for c in classes
                   if str(c) in pi["per_class"] and pi["per_class"][str(c)]["ap"] is not None]
            gm["group_class_ap_mean"] = float(np.mean(aps)) if aps else None
            gm["class_ap"] = {str(c): (pi["per_class"].get(str(c), {}) or {}).get("ap") for c in classes}
            src_out["groups"][g] = gm
        out["sources"][src] = src_out

    out["elapsed_s"] = time.perf_counter() - started
    out_path = Path(args.out) if args.out else (run / "stage5_val" / "view_source_attribution.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print("wrote", out_path)

    # concise console table
    print(json.dumps({
        "reproduction": reproduction,
        "sources": {
            s: {
                "map50": out["sources"][s]["mAP50"],
                "ap50_agnostic": out["sources"][s]["ap50_agnostic"],
                "r_agn": out["sources"][s]["localization_recall"],
                "n_pred": out["sources"][s]["n_pred"],
                "groups": {
                    g: {
                        "class_ap_mean": out["sources"][s]["groups"][g]["group_class_ap_mean"],
                        "r_agn": out["sources"][s]["groups"][g]["r_agn"],
                        "tp/fp/fn": [out["sources"][s]["groups"][g]["tp"],
                                     out["sources"][s]["groups"][g]["fp"],
                                     out["sources"][s]["groups"][g]["fn"]],
                        "fp_high": [out["sources"][s]["groups"][g]["fp_high_misclass"],
                                    out["sources"][s]["groups"][g]["fp_high_duplicate"],
                                    out["sources"][s]["groups"][g]["fp_high_background"]],
                    } for g in GROUPS
                },
            } for s in ["coarse", "fine", "both"]
        },
        "boundary": out["boundary"],
    }, indent=2))


if __name__ == "__main__":
    main()
