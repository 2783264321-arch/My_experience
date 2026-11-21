#!/usr/bin/env python3
"""Stage 0 data audit for SRRC classes 10/11 (no training).

Counts, geometry, overlap, duration, and existing C0 YOLO AP50 on fit/val.
The fit frames table only covers classes 0-5/10-11, so overlaps with all
classes are recomputed directly from the JSON annotations.

Output: reports/baseline3_srrc_stage0/srrc_data_audit.json
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.signals.metrics import ImagePredictions, ImageTargets, _match_single

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/root/autodl-fs/SpaceNet_Dataset/advanced/train")
MANIFEST = Path("/root/autodl-tmp/workspace/spacenet/analysis/outputs/split_manifest.csv")
OUT = ROOT / "reports/baseline3_srrc_stage0"
FRAMES = ROOT / "reports/local_iq_round1/frames.npz"
YOLO = ROOT / "runs_local/baseline3c/yolo_val_cache.npz"
CLASSES = (10, 11)


def load_manifest():
    rows = list(csv.DictReader(MANIFEST.open()))
    return rows, {r["sample_id"]: r for r in rows}


def parse_json(sid: str):
    stem = sid.split("/")[1]
    d = json.loads((DATA / f"{stem}.json").read_text())
    return d["signals"]


def rect_overlap(a, b):
    iw = max(0.0, min(a["f2"], b["f2"]) - max(a["f1"], b["f1"]))
    ih = max(0.0, min(a["t2"], b["t2"]) - max(a["t1"], b["t1"]))
    return iw * ih


def target_overlaps(sig, signals):
    a = {
        "f1": float(sig["start_frequency"]), "f2": float(sig["end_frequency"]),
        "t1": float(sig["start_time"]), "t2": float(sig["end_time"]),
    }
    area = max((a["f2"] - a["f1"]) * (a["t2"] - a["t1"]), 1e-9)
    any_frac = 0.0
    srrc_frac = 0.0
    srrc_cross = 0.0
    srrc_same = 0.0
    wifi_frac = 0.0
    for other in signals:
        if other is sig:
            continue
        b = {
            "f1": float(other["start_frequency"]),
            "f2": float(other["end_frequency"]),
            "t1": float(other["start_time"]),
            "t2": float(other["end_time"]),
        }
        inter = rect_overlap(a, b)
        if inter <= 0:
            continue
        frac = inter / area
        any_frac = max(any_frac, frac)
        oc = int(other["class"])
        if oc in CLASSES:
            srrc_frac = max(srrc_frac, frac)
            if oc == int(sig["class"]):
                srrc_same = max(srrc_same, frac)
            else:
                srrc_cross = max(srrc_cross, frac)
        if 0 <= oc <= 5:
            wifi_frac = max(wifi_frac, frac)
    return {
        "any_frac_max": any_frac,
        "srrc_frac_max": srrc_frac,
        "srrc_cross_frac_max": srrc_cross,
        "srrc_same_frac_max": srrc_same,
        "wifi_frac_max": wifi_frac,
        "any": int(any_frac > 0),
        "srrc": int(srrc_frac > 0),
        "srrc_cross": int(srrc_cross > 0),
        "srrc_same": int(srrc_same > 0),
        "wifi": int(wifi_frac > 0),
        "strong25": int(any_frac >= 0.25),
    }


def qstats(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    return {"n": int(len(x)), "mean": float(x.mean()),
            "q05": float(np.quantile(x, .05)),
            "median": float(np.median(x)),
            "q95": float(np.quantile(x, .95)),
            "min": float(x.min()), "max": float(x.max())}


def ap50_for_class(images, targets, cls):
    n_gt = sum(int((t.labels == cls).sum()) for t in targets)
    entries = []
    for img, (pred, gt) in enumerate(zip(images, targets)):
        matches, _ = _match_single(
            gt, pred, 0.5, class_aware=True, class_id=cls, max_dets=100)
        matched = {m[0] for m in matches}
        for pi in np.where(pred.labels == cls)[0]:
            entries.append((-float(pred.scores[pi]), img, int(pi),
                            int(pi) in matched))
    entries.sort()
    if n_gt == 0:
        return {"ap50": None, "n_gt": 0, "n_pred": len(entries)}
    tp = np.array([e[3] for e in entries], bool)
    if len(tp) == 0:
        return {"ap50": 0.0, "n_gt": n_gt, "n_pred": 0}
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(~tp)
    rec = tp_cum / n_gt
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = rec >= t
        ap += float(prec[mask].max()) if mask.any() else 0.0
    return {"ap50": ap / 101.0, "n_gt": n_gt, "n_pred": len(entries)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, mp = load_manifest()
    out = {"method": {}, "splits": {}}

    # ---------- fit: frames table for target list, JSON for all signals ----------
    fr = np.load(FRAMES, allow_pickle=True)
    sid = fr["sample_id"].astype(str)
    lab = fr["label"].astype(int)
    fit_targets = [(s, int(i)) for s, i in zip(sid, lab) if i in CLASSES]
    fit_by_file = defaultdict(list)
    for s, i in fit_targets:
        fit_by_file[s].append(i)

    # ---------- val: parse all val JSONs ----------
    val_sids = [r["sample_id"] for r in rows if r["split_new"] == "val"]
    val_by_file = {}
    for s in val_sids:
        sigs = parse_json(s)
        hits = [x for x in sigs if int(x["class"]) in CLASSES]
        if hits:
            val_by_file[s] = sigs

    out["method"] = {
        "fit_target_source": "reports/local_iq_round1/frames.npz (class 10/11 rows)",
        "val_target_source": "val JSON annotations parsed from advanced/train",
        "overlap_definition": "fraction of target box area intersected; any/srrc/wifi/cross/same; strong>=0.25",
        "ap_definition": "C0 YOLO labels, class-aware greedy matching IoU=0.5, 101-point AP",
    }

    for split, by_file in (("fit", fit_by_file), ("val", val_by_file)):
        recs = []
        for s, sigs_or_labels in by_file.items():
            if split == "fit":
                sigs = parse_json(s)
            else:
                sigs = sigs_or_labels
            for sig in sigs:
                if int(sig["class"]) not in CLASSES:
                    continue
                ov = target_overlaps(sig, sigs)
                recs.append({
                    "sample_id": s,
                    "signal_id": int(sig["signal_id"]),
                    "class": int(sig["class"]),
                    "f1": float(sig["start_frequency"]),
                    "f2": float(sig["end_frequency"]),
                    "t1": float(sig["start_time"]),
                    "t2": float(sig["end_time"]),
                    "dur_ms": float(sig["end_time"]) - float(sig["start_time"]),
                    "bw_mhz": float(sig["end_frequency"]) - float(sig["start_frequency"]),
                    **ov,
                })
        split_out = {"n_instances": len(recs),
                     "n_files": len({r["sample_id"] for r in recs}),
                     "n_instances_10mhz": sum(
                         1 for r in recs if 9.5 <= r["bw_mhz"] <= 10.5),
                     "per_class": {}}
        for c in CLASSES:
            sub_all = [r for r in recs if r["class"] == c]
            split_out["per_class"][str(c)] = {}
            for tag, sub in (
                    ("all_labels", sub_all),
                    ("10mhz", [r for r in sub_all
                               if 9.5 <= r["bw_mhz"] <= 10.5])):
                split_out["per_class"][str(c)][tag] = {
                    "n_instances": len(sub),
                    "n_files": len({r["sample_id"] for r in sub}),
                    "duration_ms": qstats([r["dur_ms"] for r in sub]),
                    "bandwidth_mhz": qstats([r["bw_mhz"] for r in sub]),
                    "overlap": {
                        "any": int(sum(r["any"] for r in sub)),
                        "srrc": int(sum(r["srrc"] for r in sub)),
                        "srrc_cross": int(sum(r["srrc_cross"] for r in sub)),
                        "srrc_same": int(sum(r["srrc_same"] for r in sub)),
                        "wifi": int(sum(r["wifi"] for r in sub)),
                        "strong25": int(sum(r["strong25"] for r in sub)),
                    },
                    "overlap_frac": {
                        "any": float(np.mean([r["any"] for r in sub]))
                        if sub else None,
                        "srrc": float(np.mean([r["srrc"] for r in sub]))
                        if sub else None,
                        "strong25": float(np.mean([r["strong25"] for r in sub]))
                        if sub else None,
                    },
                }
        out["splits"][split] = split_out
        out["splits"][split]["_records"] = recs

    # ---------- C0 YOLO AP for val classes 10/11 ----------
    z = np.load(YOLO, allow_pickle=True)
    z = {k: z[k] for k in z.files}
    c0_off = np.concatenate([[0], np.cumsum(z["c0_counts"])])
    gt_off = np.concatenate([[0], np.cumsum(z["gt_counts"])])
    images, targets = [], []
    for i in range(len(z["sample_ids"])):
        lo, hi = int(c0_off[i]), int(c0_off[i + 1])
        images.append(ImagePredictions(z["c0_boxes"][lo:hi],
                                       z["c0_scores"][lo:hi],
                                       z["c0_labels"][lo:hi]))
        glo, ghi = int(gt_off[i]), int(gt_off[i + 1])
        targets.append(ImageTargets(z["gt_boxes"][glo:ghi],
                                    z["gt_labels"][glo:ghi]))
    out["val_c0_yolo_ap50"] = {
        str(c): ap50_for_class(images, targets, c) for c in CLASSES}

    # 10 MHz subset AP: keep all non-SRRC GT, but drop class-10/11 GT whose
    # annotation bandwidth is outside 9.5-10.5 MHz. Predictions are unchanged.
    val_sigs = {}
    for s in [r["sample_id"] for r in rows if r["split_new"] == "val"]:
        val_sigs[s] = parse_json(s)
    targets_10 = []
    for i, t in enumerate(targets):
        sid = str(z["sample_ids"][i])
        sigs = val_sigs[sid]
        keep = []
        for j in range(len(t.labels)):
            cls = int(t.labels[j])
            if cls in CLASSES:
                bw = float(sigs[j]["end_frequency"]) - float(sigs[j]["start_frequency"])
                keep.append(9.5 <= bw <= 10.5)
            else:
                keep.append(True)
        keep = np.asarray(keep, bool)
        targets_10.append(ImageTargets(t.boxes[keep], t.labels[keep]))
    out["val_c0_yolo_ap50_10mhz"] = {
        str(c): ap50_for_class(images, targets_10, c) for c in CLASSES}
    out["val_10mhz_subset_note"] = (
        "val_c0_yolo_ap50 uses all class-10/11 GT; "
        "val_c0_yolo_ap50_10mhz drops class-10/11 GT with annotation "
        "bandwidth outside 9.5-10.5 MHz and keeps predictions unchanged.")
    out["val_c0_map50_reference"] = 0.6497250320877056
    out["note"] = ("Stage 0 only; no training. Fit has no C0 AP because the "
                   "YOLO cache is val. Test is sealed.")

    # keep JSON compact: drop per-instance records to a separate CSV
    csv_path = OUT / "srrc_instances.csv"
    fields = list(out["splits"]["fit"]["_records"][0]) + ["split"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for split in ("fit", "val"):
            for r in out["splits"][split]["_records"]:
                w.writerow({**r, "split": split})
    for split in ("fit", "val"):
        out["splits"][split].pop("_records")
    (OUT / "srrc_data_audit.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2)[:6000])
    print("wrote", OUT / "srrc_data_audit.json", csv_path)


if __name__ == "__main__":
    main()
