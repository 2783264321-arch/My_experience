#!/usr/bin/env python3
"""Diagnostic: oracle-class AP on the saved full-val predictions.

For each image, match predictions to GT class-agnostically (IoU 0.5, greedy,
max 100). Relabel each matched prediction with the matched GT class; keep
unmatched predictions unchanged. Re-evaluate class-aware AP. The gap between
actual class-aware AP and this oracle AP is the classification loss for the
matched boxes under the current detection/ranking.

This is a diagnostic ceiling, not a deployable model.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.signals.data import SPLIT_MANIFEST, load_manifest
from src.signals.metrics import (
    ImagePredictions, ImageTargets, _match_single, evaluate,
)
from scripts.analyze_signal_characteristics import (
    DIMS, summarize_group,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="yolo11s_c1_directreg_2048x512")
    ap.add_argument("--val-dir", default="")
    args = ap.parse_args()
    vdir = Path(args.val_dir) if args.val_dir else ROOT / "runs" / args.run / "stage5_val"
    z = np.load(vdir / "val_predictions.npz")
    ids = json.loads((vdir / "val_sample_ids.json").read_text())
    records = load_manifest(SPLIT_MANIFEST, split="val")
    by_id = {r.sample_id: r for r in records}
    recs = [by_id[s] for s in ids]
    gt_meta = []
    for r in recs:
        for s in r.signals:
            f1 = float(s["start_frequency"]); f2 = float(s["end_frequency"])
            t1 = float(s["start_time"]); t2 = float(s["end_time"])
            bw = f2 - f1
            gt_meta.append({
                "class": int(s["class"]), "bw_mhz": bw,
                "bw_rel": bw / r.B_mhz, "dur_ms": t2 - t1,
                "time_rel": (t2 - t1) / r.T_ms,
                "B": r.B_mhz, "T": r.T_ms,
            })
    gt_index = z["gt_sample_index"]
    gt_offset = np.zeros(len(recs), dtype=np.int64)
    for i in range(1, len(recs)):
        gt_offset[i] = gt_offset[i - 1] + len(recs[i - 1].signals)
    for i, r in enumerate(recs):
        if int((gt_index == i).sum()) != len(r.signals):
            raise SystemExit(f"GT alignment broken at sample {i}")

    pred_index = z["pred_sample_index"]
    pb_all, ps_all, pl_all = z["pred_boxes"], z["pred_scores"], z["pred_labels"]
    gb_all, gl_all = z["gt_boxes"], z["gt_labels"]

    gts_eval, actual_preds, oracle_preds = [], [], []
    tp_oracle = {d: defaultdict(lambda: defaultdict(list)) for d in DIMS}
    n_gt = {d: defaultdict(lambda: defaultdict(int)) for d in DIMS}
    fp_unmatched = defaultdict(list)
    n_matches = 0

    for i in range(len(recs)):
        pm = pred_index == i
        gm = gt_index == i
        pb, ps, pl = pb_all[pm], ps_all[pm], pl_all[pm]
        gb, gl = gb_all[gm], gl_all[gm]
        gts_eval.append(ImageTargets(gb, gl))
        actual_preds.append(ImagePredictions(pb, ps, pl))
        gt_obj = ImageTargets(gb, gl)
        pred_obj = ImagePredictions(pb, ps, pl)
        matches, unmatched = _match_single(
            gt_obj, pred_obj, 0.5, class_aware=False, max_dets=100,
        )
        olabels = pl.copy()
        for pi, gi in matches:
            olabels[pi] = gl[gi]
            gmeta = gt_meta[gt_offset[i] + gi]
            for d, fn in DIMS.items():
                g = fn(gmeta["bw_mhz"] if d == "abs_bw" else
                       gmeta["bw_rel"] if d == "rel_bw" else
                       gmeta["dur_ms"] if d == "duration" else gmeta["time_rel"])
                tp_oracle[d][g][int(gmeta["class"])].append(float(ps[pi]))
            n_matches += 1
        oracle_preds.append(ImagePredictions(pb, ps, olabels))
        for pi in unmatched:
            fp_unmatched[int(pl[pi])].append(float(ps[pi]))
        for gi, gmeta in enumerate(gt_meta[gt_offset[i]: gt_offset[i] + len(gb)]):
            c = int(gmeta["class"])
            for d, fn in DIMS.items():
                g = fn(gmeta["bw_mhz"] if d == "abs_bw" else
                       gmeta["bw_rel"] if d == "rel_bw" else
                       gmeta["dur_ms"] if d == "duration" else gmeta["time_rel"])
                n_gt[d][g][c] += 1

    actual = evaluate(gts_eval, actual_preds, iou_thresholds=(0.5,), max_dets=100)
    oracle = evaluate(gts_eval, oracle_preds, iou_thresholds=(0.5,), max_dets=100)
    ai = actual["per_iou"]["0.5"]
    oi = oracle["per_iou"]["0.5"]
    print("actual mAP50", ai["map_class_aware"], "agnostic", ai["ap_class_agnostic"])
    print("oracle mAP50", oi["map_class_aware"], "agnostic", oi["ap_class_agnostic"])
    print("matches used for oracle labels", n_matches)

    # actual group AP from signal_characteristics if available
    sc_path = vdir / "signal_characteristics.json"
    sc = json.loads(sc_path.read_text()) if sc_path.exists() else None
    per_group = {}
    for d in DIMS:
        per_group[d] = {}
        for g in sorted(n_gt[d]):
            oracle_g = summarize_group(tp_oracle[d][g], fp_unmatched, n_gt[d][g])
            actual_g = sc["dims"][d][g]["map50"] if sc else None
            per_group[d][g] = {
                "n_gt": oracle_g["n_gt"],
                "actual_map50": actual_g,
                "oracle_map50": oracle_g["map50"],
                "classification_gap": (oracle_g["map50"] - actual_g)
                if (oracle_g["map50"] is not None and actual_g is not None) else None,
                "oracle_per_class": oracle_g["per_class"],
            }

    out = {
        "actual": {
            "map50": ai["map_class_aware"],
            "ap50_agnostic": ai["ap_class_agnostic"],
            "per_class": ai["per_class"],
            "classification": actual["classification"],
        },
        "oracle_class": {
            "map50": oi["map_class_aware"],
            "ap50_agnostic": oi["ap_class_agnostic"],
            "per_class": oi["per_class"],
            "classification": oracle["classification"],
        },
        "classification_gap_overall": oi["map_class_aware"] - ai["map_class_aware"],
        "n_matches": n_matches,
        "per_group": per_group,
    }
    out_path = vdir / "oracle_class_ap.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "actual_map50": ai["map_class_aware"],
        "oracle_map50": oi["map_class_aware"],
        "gap": oi["map_class_aware"] - ai["map_class_aware"],
        "per_class_actual": {k: v["ap"] for k, v in ai["per_class"].items()},
        "per_class_oracle": {k: v["ap"] for k, v in oi["per_class"].items()},
        "per_group": {
            d: {g: {"n_gt": v["n_gt"], "actual": v["actual_map50"],
                    "oracle": v["oracle_map50"], "gap": v["classification_gap"]}
                for g, v in gs.items()}
            for d, gs in per_group.items()
        },
    }, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
