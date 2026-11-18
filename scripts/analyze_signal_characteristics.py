#!/usr/bin/env python3
"""Signal-level val analysis: narrowband vs wideband, duration, and B/T.

The earlier "per bandwidth" table grouped by the *observation* bandwidth
5/20/.../80 MHz. This script groups GT instances by the *signal's own*
bandwidth (MHz and fraction of the observation band), by signal duration,
and by observation B/T, then computes class-aware AP@0.5 with COCO-style
ignore of predictions matched to GT outside the group.

Inputs: runs/<run>/stage5_val/{val_predictions.npz,val_sample_ids.json}
Outputs: stage5_val/signal_characteristics.json, two CSV tables.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.signals.data import SPLIT_MANIFEST, load_manifest
from src.signals.metrics import (
    ImagePredictions, ImageTargets, _ap_101, _match_single, _precision_recall,
)

ROOT = Path(__file__).resolve().parents[1]


def bw_abs_group(bw: float) -> str:
    if bw <= 0.25 + 1e-9:
        return "narrow_<=0.25MHz"
    if bw <= 2.0 + 1e-9:
        return "medium_0.3-2MHz"
    if bw <= 10.0 + 1e-9:
        return "wide_2.1-10MHz"
    if bw <= 20.0 + 1e-9:
        return "verywide_10.1-20MHz"
    return "ultra_>20MHz"


def bw_rel_group(rel: float) -> str:
    if rel <= 0.005:
        return "rel_<=0.5%"
    if rel <= 0.02:
        return "rel_0.5-2%"
    if rel <= 0.05:
        return "rel_2-5%"
    if rel <= 0.20:
        return "rel_5-20%"
    return "rel_>20%"


def dur_group(d: float) -> str:
    if d <= 5:
        return "dur_<=5ms"
    if d <= 15:
        return "dur_5-15ms"
    if d <= 30:
        return "dur_15-30ms"
    if d <= 60:
        return "dur_30-60ms"
    return "dur_>60ms"


def time_ratio_group(x: float) -> str:
    if x <= 0.1:
        return "time_<=10%"
    if x <= 0.3:
        return "time_10-30%"
    if x <= 0.6:
        return "time_30-60%"
    if x <= 0.9:
        return "time_60-90%"
    return "time_>90%"


DIMS = {
    "abs_bw": bw_abs_group,
    "rel_bw": bw_rel_group,
    "duration": dur_group,
    "time_ratio": time_ratio_group,
}


def group_ap(tp_scores, fp_scores, n_gt):
    """AP for one group/class given TP scores and shared unmatched FP scores."""
    if n_gt == 0:
        return None, None
    entries = [(float(s), 1) for s in tp_scores] + [(float(s), 0) for s in fp_scores]
    entries.sort(key=lambda x: -x[0])
    tp = np.array([e[1] for e in entries], dtype=bool)
    fp = ~tp
    prec, rec = _precision_recall(tp, fp, n_gt)
    ap = _ap_101(prec, rec)
    return ap, prec, rec


def summarize_group(tp_by_class, fp_by_class, n_gt_by_class):
    per_class = {}
    for c, n in n_gt_by_class.items():
        if n <= 0:
            continue
        tp = tp_by_class.get(c, [])
        ap, _, _ = group_ap(tp, fp_by_class.get(c, []), n)
        per_class[str(c)] = {
            "ap": ap,
            "n_gt": int(n),
            "n_tp": int(len(tp)),
            "recall_matched": float(len(tp) / n),
            "recall@0.25": float(sum(1 for s in tp if s >= 0.25) / n),
            "mean_tp_score": float(np.mean(tp)) if tp else None,
        }
    aps = [v["ap"] for v in per_class.values() if v["ap"] is not None]
    n_gt = int(sum(v["n_gt"] for v in per_class.values()))
    n_tp = int(sum(v["n_tp"] for v in per_class.values()))
    return {
        "n_gt": n_gt,
        "n_classes": len(per_class),
        "map50": float(np.mean(aps)) if aps else None,
        "recall_matched": float(n_tp / n_gt) if n_gt else None,
        "recall@0.25": float(sum(v["n_tp"] * v["recall@0.25"] for v in per_class.values()) / n_gt)
        if n_gt else None,
        "per_class": per_class,
    }


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
                "class": int(s["class"]),
                "bw_mhz": bw,
                "bw_rel": bw / r.B_mhz,
                "dur_ms": t2 - t1,
                "time_rel": (t2 - t1) / r.T_ms,
                "B": r.B_mhz,
                "T": r.T_ms,
                "sample_id": r.sample_id,
            })
    if len(gt_meta) != len(z["gt_boxes"]):
        raise SystemExit(f"GT meta {len(gt_meta)} != gt boxes {len(z['gt_boxes'])}")
    gt_index = z["gt_sample_index"]
    gt_offset = np.zeros(len(recs), dtype=np.int64)
    for i in range(1, len(recs)):
        gt_offset[i] = gt_offset[i - 1] + len(recs[i - 1].signals)
    # verify alignment
    for i, r in enumerate(recs):
        n = int((gt_index == i).sum())
        if n != len(r.signals):
            raise SystemExit(f"sample {i} GT count {n} != {len(r.signals)}")

    pred_index = z["pred_sample_index"]
    pb_all, ps_all, pl_all = z["pred_boxes"], z["pred_scores"], z["pred_labels"]
    gb_all, gl_all = z["gt_boxes"], z["gt_labels"]
    preds = []
    gts = []
    for i in range(len(recs)):
        pm = pred_index == i
        gm = gt_index == i
        preds.append((pb_all[pm], ps_all[pm], pl_all[pm]))
        gts.append((gb_all[gm], gl_all[gm]))

    # accumulators: dim -> group -> class -> list / count
    tp = {d: defaultdict(lambda: defaultdict(list)) for d in DIMS}
    n_gt = {d: defaultdict(lambda: defaultdict(int)) for d in DIMS}
    fp_by_class = defaultdict(list)                 # global unmatched per class
    fp_by_class_B = defaultdict(lambda: defaultdict(list))
    tp_B = defaultdict(lambda: defaultdict(list))   # class -> observation B -> TP scores
    n_gt_B = defaultdict(lambda: defaultdict(int))  # class -> observation B -> GT
    fp_B = defaultdict(lambda: defaultdict(list))   # class -> observation B -> unmatched
    # class-agnostic per dimension/group
    agn_tp = {d: defaultdict(list) for d in DIMS}
    agn_fp = []
    agn_n_gt = {d: defaultdict(int) for d in DIMS}
    # B x relative-bandwidth accumulators
    tp_B_rel = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    n_gt_B_rel = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for i, r in enumerate(recs):
        pboxes, pscores, plabels = preds[i]
        gboxes, glabels = gts[i]
        if len(gboxes) == 0 and len(pboxes) == 0:
            continue
        gt_obj = ImageTargets(gboxes, glabels)
        pred_obj = ImagePredictions(pboxes, pscores, plabels)
        classes = sorted(set(glabels.tolist()) | set(plabels.tolist()))
        for c in classes:
            matches, unmatched = _match_single(
                gt_obj, pred_obj, 0.5, class_aware=True, class_id=int(c), max_dets=100,
            )
            for pi, gi in matches:
                gmeta = gt_meta[gt_offset[i] + gi]
                for d, fn in DIMS.items():
                    g = fn(gmeta["bw_mhz"] if d == "abs_bw" else
                           gmeta["bw_rel"] if d == "rel_bw" else
                           gmeta["dur_ms"] if d == "duration" else gmeta["time_rel"])
                    tp[d][g][int(c)].append(float(pscores[pi]))
                rg = bw_rel_group(gmeta["bw_rel"])
                tp_B_rel[r.B_mhz][rg][int(c)].append(float(pscores[pi]))
                tp_B[int(c)][r.B_mhz].append(float(pscores[pi]))
            for pi in unmatched:
                fp_by_class[int(c)].append(float(pscores[pi]))
                fp_by_class_B[int(c)][r.B_mhz].append(float(pscores[pi]))
                fp_B[int(c)][r.B_mhz].append(float(pscores[pi]))
        # class-agnostic matching for the same groups
        amatches, aunmatched = _match_single(
            gt_obj, pred_obj, 0.5, class_aware=False, max_dets=100,
        )
        for pi, gi in amatches:
            gmeta = gt_meta[gt_offset[i] + gi]
            for d, fn in DIMS.items():
                g = fn(gmeta["bw_mhz"] if d == "abs_bw" else
                       gmeta["bw_rel"] if d == "rel_bw" else
                       gmeta["dur_ms"] if d == "duration" else gmeta["time_rel"])
                agn_tp[d][g].append(float(pscores[pi]))
        for pi in aunmatched:
            agn_fp.append(float(pscores[pi]))
        for gi, gmeta in enumerate(gt_meta[gt_offset[i]: gt_offset[i] + len(gboxes)]):
            c = int(gmeta["class"])
            for d, fn in DIMS.items():
                g = fn(gmeta["bw_mhz"] if d == "abs_bw" else
                       gmeta["bw_rel"] if d == "rel_bw" else
                       gmeta["dur_ms"] if d == "duration" else gmeta["time_rel"])
                n_gt[d][g][c] += 1
                agn_n_gt[d][g] += 1
            n_gt_B_rel[r.B_mhz][bw_rel_group(gmeta["bw_rel"])][c] += 1
            n_gt_B[c][r.B_mhz] += 1

    out = {"n_samples": len(recs), "n_gt": len(gt_meta), "dims": {}}
    for d in DIMS:
        dout = {}
        for g in sorted(n_gt[d]):
            dout[g] = summarize_group(tp[d][g], fp_by_class, n_gt[d][g])
            ap_agn, _, _ = group_ap(agn_tp[d][g], agn_fp, agn_n_gt[d][g])
            dout[g]["ap50_agnostic"] = ap_agn
            dout[g]["n_gt_all_classes"] = int(agn_n_gt[d][g])
            dout[g]["recall_agnostic"] = (
                float(len(agn_tp[d][g]) / agn_n_gt[d][g])
                if agn_n_gt[d][g] else None
            )
        out["dims"][d] = dout

    # B x relative group
    bx = {}
    for B in sorted(n_gt_B_rel):
        bx[str(B)] = {}
        for g in sorted(n_gt_B_rel[B]):
            bx[str(B)][g] = summarize_group(
                tp_B_rel[B][g], fp_by_class_B[B], n_gt_B_rel[B][g]
            )
    out["observation_B_x_rel_bw"] = bx

    # class table
    cls = {}
    for c in range(14):
        vals = [m for m in gt_meta if m["class"] == c]
        if not vals:
            continue
        cls[str(c)] = {
            "n_gt": len(vals),
            "median_bw_mhz": float(np.median([m["bw_mhz"] for m in vals])),
            "min_bw_mhz": float(min(m["bw_mhz"] for m in vals)),
            "max_bw_mhz": float(max(m["bw_mhz"] for m in vals)),
            "median_bw_rel": float(np.median([m["bw_rel"] for m in vals])),
            "median_dur_ms": float(np.median([m["dur_ms"] for m in vals])),
        }
    out["class_bandwidth"] = cls

    # class x observation B
    cx = {}
    for c in range(14):
        cx[str(c)] = {}
        for B in sorted(n_gt_B[c]):
            n = n_gt_B[c][B]
            if n <= 0:
                continue
            ap, _, _ = group_ap(tp_B[c][B], fp_B[c][B], n)
            cx[str(c)][str(B)] = {
                "n_gt": int(n), "ap": ap,
                "recall_matched": float(len(tp_B[c][B]) / n),
            }
    out["class_x_observation_B"] = cx

    (vdir / "signal_characteristics.json").write_text(json.dumps(out, indent=2))

    # CSV: group summaries
    rows = []
    for d, groups in out["dims"].items():
        for g, v in groups.items():
            rows.append({
                "dimension": d, "group": g, "n_gt": v["n_gt"],
                "n_classes": v["n_classes"], "map50": v["map50"],
                "ap50_agnostic": v.get("ap50_agnostic"),
                "recall_agnostic": v.get("recall_agnostic"),
                "recall_matched": v["recall_matched"],
                "recall@0.25": v["recall@0.25"],
            })
    with (vdir / "signal_groups.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # CSV: B x rel group
    rows = []
    for B, groups in bx.items():
        for g, v in groups.items():
            rows.append({
                "observation_B_mhz": B, "rel_bw_group": g,
                "n_gt": v["n_gt"], "n_classes": v["n_classes"],
                "map50": v["map50"], "recall_matched": v["recall_matched"],
                "recall@0.25": v["recall@0.25"],
            })
    with (vdir / "observation_B_x_signal_bandwidth.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print(json.dumps({
        "n_gt": out["n_gt"],
        "abs_bw": {k: {kk: v[kk] for kk in ["n_gt", "n_classes", "map50", "recall_matched"]}
                   for k, v in out["dims"]["abs_bw"].items()},
        "rel_bw": {k: {kk: v[kk] for kk in ["n_gt", "n_classes", "map50", "recall_matched"]}
                   for k, v in out["dims"]["rel_bw"].items()},
        "duration": {k: {kk: v[kk] for kk in ["n_gt", "n_classes", "map50", "recall_matched"]}
                     for k, v in out["dims"]["duration"].items()},
        "class_bandwidth": out["class_bandwidth"],
    }, indent=2))
    print("wrote", vdir / "signal_characteristics.json")


if __name__ == "__main__":
    main()
