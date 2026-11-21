#!/usr/bin/env python3
"""CPU-only re-check of the Wi-Fi short-segment diagnostic (ca7e9b0).

The original `other_signal_risk` treated `gt_idx` (a GLOBAL GT index over
the whole val cache) as a sample-local list index.  It therefore usually
failed to exclude the target and reported the target itself as "another
signal".  This script:

1. maps the global GT index to the sample-local signal via
   `gt_idx - gt_offsets[sample_idx]`;
2. validates the target class and time-frequency coordinates against the
   JSON manifest and the cached GT box;
3. excludes the target by stable `signal_id` (not by a global index);
4. recomputes overlap only for the already-selected best short segment,
   reusing the existing expert outputs.

No training, no inference changes, no final-dev/test, no GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from src.signals.data import SPLIT_MANIFEST, load_manifest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports/baseline3_wifi_cascade"
OUT = ROOT / "reports/baseline3_wifi_sync_diag"
CACHE = ROOT / "runs_local/baseline3c"


def rect_intersection(a, b):
    iw = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    return iw * ih


def expected_gt_box(meta, sig):
    """Reconstruct the normalized YOLO GT box from JSON signal coordinates."""
    B = float(meta["B_mhz"])
    N = int(meta["N"])
    f_lo = float(meta["f_lo"])
    h = (N + 510) // 511
    x1 = ((float(sig["start_time"]) / 1000.0) * (B * 1e6) / h + 0.5) / 512.0
    x2 = ((float(sig["end_time"]) / 1000.0) * (B * 1e6) / h + 0.5) / 512.0
    y1 = (float(sig["start_frequency"]) - f_lo) / B
    y2 = (float(sig["end_frequency"]) - f_lo) / B
    return np.array([min(1, max(0, x1)), min(1, max(0, y1)),
                     min(1, max(0, x2)), min(1, max(0, y2))])


def other_risk(sigs, target_sid, f_center, bw, t1, t2):
    """Max overlap of the segment with signals other than target_sid."""
    a = (f_center - bw / 2.0, f_center + bw / 2.0, t1, t2)
    area = max((a[1] - a[0]) * (a[3] - a[2]), 1e-9)
    best = 0.0
    best_lab = None
    best_sid = None
    for s in sigs:
        sid = int(s["signal_id"])
        if sid == target_sid:
            continue
        b = (float(s["start_frequency"]), float(s["end_frequency"]),
             float(s["start_time"]), float(s["end_time"]))
        frac = rect_intersection(a, b) / area
        if frac > best:
            best, best_lab, best_sid = float(frac), int(s["class"]), sid
    return best, best_lab, best_sid


def as_int(v):
    return None if v is None or v == "" else int(v)


def as_float(v):
    return None if v is None or v == "" else float(v)


def summarize(rows):
    ent = [r for r in rows if r["expert_label"] != ""]
    return {
        "n": len(rows),
        "n_recovered": sum(1 for r in rows if as_int(r["n_pass_segments"]) > 0),
        "n_kept_total": int(sum(as_int(r["n_kept_total"]) or 0 for r in rows)),
        "n_blocks_total": int(sum(as_int(r["n_blocks_total"]) or 0 for r in rows)),
        "n_pass_segments": int(sum(as_int(r["n_pass_segments"]) or 0 for r in rows)),
        "n_target_identifiable": sum(1 for r in rows if r["target_identifiable"] == "1"),
        "n_high_other_overlap": sum(
            1 for r in rows if r["high_other_overlap"] == "1"),
        "n_expert_matches_other": sum(
            1 for r in rows if r["expert_matches_other"] == "1"),
        "expert_acc": (float(np.mean([int(r["expert_correct"]) for r in ent
                                      if r["expert_correct"] != ""]))
                       if any(r["expert_correct"] != "" for r in ent) else None),
        "yolo_acc": (float(np.mean([int(r["yolo_correct"]) for r in ent
                                    if r["yolo_correct"] != ""]))
                     if any(r["yolo_correct"] != "" for r in ent) else None),
        "corrected": int(sum(as_int(r["corrected"]) or 0 for r in ent)),
        "broken": int(sum(as_int(r["broken"]) or 0 for r in ent)),
        "net": int(sum((as_int(r["corrected"]) or 0) -
                       (as_int(r["broken"]) or 0) for r in ent)),
        "both_correct": sum(1 for r in ent
                            if r["expert_correct"] == "1" and
                            r["yolo_correct"] == "1"),
        "both_wrong": sum(1 for r in ent
                          if r["expert_correct"] == "0" and
                          r["yolo_correct"] == "0"),
        "confusion": [
            [sum(1 for r in ent if r["gt_label"] == str(a) and
                 r["expert_label"] == str(b)) for b in (3, 4, 5)]
            for a in (3, 4, 5)
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=OUT / "sync_segment_cases_v1.csv")
    ap.add_argument("--out-csv", type=Path,
                    default=OUT / "sync_segment_cases.csv")
    ap.add_argument("--out-json", type=Path,
                    default=OUT / "sync_segment_results.json")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.input.open()))
    pairs = json.loads((EXP / "adaptation_pairs.json").read_text())
    z = np.load(CACHE / "yolo_val_cache.npz", allow_pickle=True)
    z = {k: z[k] for k in z.files}
    gt_off = np.concatenate([[0], np.cumsum(z["gt_counts"])])
    metas = [json.loads(str(m)) for m in z["meta"]]
    recs = {r.sample_id: r for r in load_manifest(SPLIT_MANIFEST, split="val")}

    out = []
    for raw in rows:
        r = dict(raw)
        # Keep the original buggy values for the audit, then replace the
        # primary columns with corrected values.
        r["v1_other_frac"] = raw.get("other_frac", "")
        r["v1_other_label"] = raw.get("other_label", "")
        r["v1_high_other_overlap"] = raw.get("high_other_overlap", "")
        r["v1_expert_matches_other"] = raw.get("expert_matches_other", "")
        r["target_identifiable"] = "0"
        r["target_local_idx"] = ""
        r["target_signal_id"] = ""
        r["other_frac"] = ""
        r["other_label"] = ""
        r["other_signal_id"] = ""
        r["high_other_overlap"] = ""
        r["expert_matches_other"] = ""

        if r["kind"] != "matched":
            out.append(r)
            continue

        pair = pairs[int(r["pair_index"])]
        sid = r["sample_id"]
        sigs = recs[sid].signals
        s_idx = int(pair["sample_idx"])
        g_global = int(pair["gt_idx"])
        g_local = g_global - int(gt_off[s_idx])
        if not (0 <= g_local < len(sigs)):
            raise RuntimeError(f"{sid}: local GT out of range {g_local}")
        target = sigs[g_local]

        # Validate target identity and coordinates.
        if int(target["class"]) != int(r["gt_label"]):
            raise RuntimeError(f"{sid}: target class mismatch")
        if int(z["gt_labels"][g_global]) != int(r["gt_label"]):
            raise RuntimeError(f"{sid}: cache GT class mismatch")
        exp = expected_gt_box(metas[s_idx], target)
        if not np.allclose(exp, z["gt_boxes"][g_global], atol=2e-4, rtol=0):
            raise RuntimeError(f"{sid}: GT geometry mismatch")
        tgt_fc = 0.5 * (float(target["start_frequency"]) +
                        float(target["end_frequency"]))
        if abs(float(pair["gt_f_center"]) - tgt_fc) > 1e-4:
            raise RuntimeError(f"{sid}: pair/target center mismatch")
        if abs(float(pair["gt_t1"]) - float(target["start_time"])) > 1e-4 or \
           abs(float(pair["gt_t2"]) - float(target["end_time"])) > 1e-4:
            raise RuntimeError(f"{sid}: pair/target time mismatch")

        target_sid = int(target["signal_id"])
        r["target_identifiable"] = "1"
        r["target_local_idx"] = str(g_local)
        r["target_signal_id"] = str(target_sid)

        if r["best_t1"] == "" or r["best_t2"] == "":
            out.append(r)
            continue

        if r["box_type"] == "pred":
            seg_fc = float(pair["f_center"])
        else:
            seg_fc = float(pair["gt_f_center"])
        frac, lab, osid = other_risk(
            sigs, target_sid, seg_fc, int(pair["nominal_bw"]),
            float(r["best_t1"]), float(r["best_t2"]))
        r["other_frac"] = frac
        r["other_label"] = "" if lab is None else str(lab)
        r["other_signal_id"] = "" if osid is None else str(osid)
        r["high_other_overlap"] = str(int(frac >= 0.5))
        emo = bool(r["expert_label"] != "" and lab in (3, 4, 5) and
                   int(r["expert_label"]) == int(lab))
        r["expert_matches_other"] = str(int(emo))
        out.append(r)

    base_fields = [k for k in rows[0].keys()
                   if k not in ("other_frac", "other_label",
                                "high_other_overlap", "expert_matches_other")]
    fields = base_fields + [
        "target_identifiable", "target_local_idx", "target_signal_id",
        "other_frac", "other_label", "other_signal_id",
        "high_other_overlap", "expert_matches_other",
        "v1_other_frac", "v1_other_label",
        "v1_high_other_overlap", "v1_expert_matches_other",
    ]
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, "") for k in fields})

    matched = [r for r in out if r["kind"] == "matched"]
    unmatched = [r for r in out if r["kind"] == "unmatched"]

    v1 = json.loads((OUT / "sync_segment_results_v1.json").read_text())
    v1_pred = [r for r in rows if r["kind"] == "matched" and
               r["box_type"] == "pred" and r["expert_label"] != ""]
    summary = {
        "n_cases": len(out),
        "correction": {
            "issue": ("gt_idx is a global GT index; the v1 other_signal_risk "
                      "used it as a sample-local index, so the target was not "
                      "excluded and was reported as another signal"),
            "v1_pred_high_overlap": v1["matched"]["pred_all"]["n_high_other_overlap"],
            "v1_pred_expert_matches_other": v1["matched"]["pred_all"]["n_expert_matches_other"],
            "v1_pred_corrected": v1["matched"]["pred_all"]["corrected"],
            "v1_pred_broken": v1["matched"]["pred_all"]["broken"],
            "v1_pred_corrected_matching_other": sum(
                1 for r in v1_pred if r["corrected"] == "1" and
                r["expert_matches_other"] == "1"),
            "v1_pred_broken_matching_other": sum(
                1 for r in v1_pred if r["broken"] == "1" and
                r["expert_matches_other"] == "1"),
        },
        "matched": {},
        "unmatched": {},
    }

    for box in ("pred", "gt"):
        for cls in (3, 4, 5):
            sub = [r for r in matched if r["box_type"] == box and
                   r["gt_label"] == str(cls)]
            if sub:
                summary["matched"][f"{box}_class{cls}"] = summarize(sub)
        sub = [r for r in matched if r["box_type"] == box]
        if sub:
            summary["matched"][f"{box}_all"] = summarize(sub)

    for split in ("adaptation-train", "calibration-dev"):
        sub = [r for r in matched if r["split"] == split and
               r["box_type"] == "pred"]
        if sub:
            summary["matched"][f"pred_{split}"] = summarize(sub)

    pred = [r for r in matched if r["box_type"] == "pred" and
            r["expert_label"] != ""]
    cor = [r for r in pred if r["corrected"] == "1"]
    br = [r for r in pred if r["broken"] == "1"]
    summary["correction"].update({
        "new_pred_high_overlap": summary["matched"]["pred_all"]["n_high_other_overlap"],
        "new_pred_expert_matches_other": summary["matched"]["pred_all"]["n_expert_matches_other"],
        "new_pred_corrected": summary["matched"]["pred_all"]["corrected"],
        "new_pred_broken": summary["matched"]["pred_all"]["broken"],
        "new_pred_corrected_matching_other": sum(
            1 for r in cor if r["expert_matches_other"] == "1"),
        "new_pred_broken_matching_other": sum(
            1 for r in br if r["expert_matches_other"] == "1"),
        "pred_train_corrected": summary["matched"]["pred_adaptation-train"]["corrected"],
        "pred_train_broken": summary["matched"]["pred_adaptation-train"]["broken"],
        "pred_calibration_corrected": summary["matched"]["pred_calibration-dev"]["corrected"],
        "pred_calibration_broken": summary["matched"]["pred_calibration-dev"]["broken"],
    })

    summary["unmatched"] = {
        "n": len(unmatched),
        "n_recovered": sum(1 for r in unmatched
                           if as_int(r["n_pass_segments"]) > 0),
        "n_changed": sum(1 for r in unmatched if r["changed"] == "1"),
        "n_target_identifiable": 0,
        "n_high_other_overlap": 0,
        "n_expert_matches_other": 0,
        "expert_labels": dict(Counter(r["expert_label"] for r in unmatched
                                      if r["expert_label"] != "")),
        "n_kept_total": int(sum(as_int(r["n_kept_total"]) or 0
                                for r in unmatched)),
    }

    sig = {}
    for key in ("pred_all", "pred_class3", "pred_class4", "pred_class5",
                "pred_adaptation-train", "pred_calibration-dev", "gt_all"):
        v = summary["matched"].get(key)
        if v is None:
            continue
        b, c = v["corrected"], v["broken"]
        sig[key] = {
            "corrected": b,
            "broken": c,
            "net": b - c,
            "p_exact": float(binomtest(b, b + c, 0.5).pvalue)
            if b + c else 1.0,
        }
    summary["significance"] = sig

    args.out_json.write_text(json.dumps(summary, indent=2, default=float),
                             encoding="utf-8")
    print(json.dumps(summary["correction"], indent=2))
    print(json.dumps(summary["significance"], indent=2))
    print("wrote", args.out_csv, args.out_json)


if __name__ == "__main__":
    main()
