#!/usr/bin/env python3
"""Stage 0 data audit for SRRC classes 10/11 (no training).

Full split enumeration: every fit/val file in split_manifest.csv is parsed
from its JSON annotation; all class-10/11 instances are collected, then
the 9.5-10.5 MHz subset is selected. No file list is inferred from the
previously selected frames.npz.

Outputs:
  reports/baseline3_srrc_stage0/srrc_data_audit.json
  reports/baseline3_srrc_stage0/srrc_instances.csv

AP:
  - full-label C0 AP50 (all class-10/11 GT)
  - old "drop ignored GT, keep all predictions" AP50 (historical, biased)
  - ignore-aware AP50 (valid GT first; predictions that hit only ignored
    GT are excluded from TP/FP; unit tests in test_srrc_ignore_ap.py)
The JSON-vs-cache GT mapping is validated by class, order, and geometry
before any bandwidth filtering; no unverified sigs[j] assumption.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.srrc_ignore_ap import drop_gt_ap50, ignore_aware_ap50
from src.signals.metrics import ImagePredictions, ImageTargets, _match_single

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/root/autodl-fs/SpaceNet_Dataset/advanced/train")
MANIFEST = Path("/root/autodl-tmp/workspace/spacenet/analysis/outputs/split_manifest.csv")
OUT = ROOT / "reports/baseline3_srrc_stage0"
YOLO = ROOT / "runs_local/baseline3c/yolo_val_cache.npz"
CLASSES = (10, 11)
BAND_LO, BAND_HI = 9.5, 10.5
N_TIME = 512


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
    any_frac = srrc_frac = srrc_cross = srrc_same = wifi_frac = 0.0
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


def bw_category(bw: float) -> str:
    if bw < BAND_LO:
        return "<9.5"
    if bw <= BAND_HI:
        return "9.5-10.5"
    return ">10.5"


def ap50_for_class(images, targets, cls):
    """Reference full-label class AP50 (same protocol as production)."""
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


def build_cache_arrays():
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
    return z, images, targets


def expected_gt_boxes(meta: dict, signals):
    B = float(meta["B_mhz"]); N = int(meta["N"]); f_lo = float(meta["f_lo"])
    h = (N + 510) // 511
    boxes = []
    for s in signals:
        x1 = ((float(s["start_time"]) / 1000.0) * (B * 1e6) / h + 0.5) / N_TIME
        x2 = ((float(s["end_time"]) / 1000.0) * (B * 1e6) / h + 0.5) / N_TIME
        y1 = (float(s["start_frequency"]) - f_lo) / B
        y2 = (float(s["end_frequency"]) - f_lo) / B
        boxes.append([min(1.0, max(0.0, x1)), min(1.0, max(0.0, y1)),
                      min(1.0, max(0.0, x2)), min(1.0, max(0.0, y2))])
    return np.asarray(boxes, float)


def validate_cache_mapping(z, targets, sig_cache):
    """Validate class, order and geometry; return per-GT bandwidth arrays."""
    gt_off = np.concatenate([[0], np.cumsum(z["gt_counts"])])
    bw_per_gt = []
    n_checked = 0
    for i, (sid, meta_s) in enumerate(zip(z["sample_ids"], z["meta"])):
        sid = str(sid)
        sigs = sig_cache[sid]
        meta = json.loads(str(meta_s))
        exp = expected_gt_boxes(meta, sigs)
        lo, hi = int(gt_off[i]), int(gt_off[i + 1])
        got = targets[i].boxes
        glab = targets[i].labels
        if len(exp) != len(got):
            raise RuntimeError(f"{sid}: JSON signals={len(exp)} cache GT={len(got)}")
        if not np.allclose(exp, got, atol=2e-4, rtol=0.0):
            raise RuntimeError(f"{sid}: JSON/cache geometry mismatch")
        jlab = np.asarray([int(s["class"]) for s in sigs], int)
        if not np.array_equal(jlab, glab):
            raise RuntimeError(f"{sid}: JSON/cache class mismatch")
        bw = np.asarray([
            float(s["end_frequency"]) - float(s["start_frequency"])
            for s in sigs], float)
        bw_per_gt.append(bw)
        n_checked += 1
    return bw_per_gt, n_checked


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, mp = load_manifest()
    manifest_counts = Counter(r["split_new"] for r in rows)
    out = {"method": {}, "splits": {}}

    # Parse every fit/val JSON once. This is the full split, not a
    # frames.npz-derived subset.
    sig_cache = {}
    split_files = {}
    for split in ("fit", "val"):
        sids = sorted(r["sample_id"] for r in rows
                      if r["split_new"] == split)
        split_files[split] = sids
        for s in sids:
            sig_cache[s] = parse_json(s)
        print(f"{split}: parsed {len(sids)} files", flush=True)

    out["method"] = {
        "fit_source": "ALL split_manifest.csv split_new=fit files; JSON annotations parsed directly",
        "val_source": "ALL split_manifest.csv split_new=val files; JSON annotations parsed directly",
        "target_classes": list(CLASSES),
        "band_10mhz": [BAND_LO, BAND_HI],
        "overlap_definition": "fraction of target box area intersected; any/srrc/wifi/cross/same; strong>=0.25",
        "ap_full": "C0 YOLO labels, class-aware greedy matching IoU=0.5, 101-point AP",
        "ap_old_subset": "drop class-10/11 GT outside 9.5-10.5 MHz, keep all predictions (known biased)",
        "ap_ignore_aware": "valid GT first; unmatched predictions hitting ignored GT at IoU>=0.5 excluded from TP/FP",
        "manifest_counts": dict(manifest_counts),
    }

    for split in ("fit", "val"):
        recs = []
        n_files_with_target = 0
        n_files_with_10mhz = 0
        for s in split_files[split]:
            sigs = sig_cache[s]
            hits = [x for x in sigs if int(x["class"]) in CLASSES]
            if hits:
                n_files_with_target += 1
            if any(int(x["class"]) in CLASSES and
                   BAND_LO <= float(x["end_frequency"]) - float(x["start_frequency"]) <= BAND_HI
                   for x in sigs):
                n_files_with_10mhz += 1
            for sig in hits:
                ov = target_overlaps(sig, sigs)
                bw = float(sig["end_frequency"]) - float(sig["start_frequency"])
                recs.append({
                    "sample_id": s,
                    "signal_id": int(sig["signal_id"]),
                    "class": int(sig["class"]),
                    "f1": float(sig["start_frequency"]),
                    "f2": float(sig["end_frequency"]),
                    "t1": float(sig["start_time"]),
                    "t2": float(sig["end_time"]),
                    "dur_ms": float(sig["end_time"]) - float(sig["start_time"]),
                    "bw_mhz": bw,
                    "is_10mhz": int(BAND_LO <= bw <= BAND_HI),
                    "bandwidth_category": bw_category(bw),
                    **ov,
                })
        split_out = {
            "n_files_manifest": len(split_files[split]),
            "n_files_with_target": n_files_with_target,
            "n_files_10mhz_instances": n_files_with_10mhz,
            "n_instances": len(recs),
            "n_files": len({r["sample_id"] for r in recs}),
            "n_instances_10mhz": sum(r["is_10mhz"] for r in recs),
            "n_files_10mhz": len({r["sample_id"] for r in recs
                                  if r["is_10mhz"]}),
            "n_instances_narrow": sum(1 for r in recs if not r["is_10mhz"]),
            "n_files_narrow": len({r["sample_id"] for r in recs
                                   if not r["is_10mhz"]}),
            "bandwidth_categories": dict(Counter(
                r["bandwidth_category"] for r in recs)),
            "per_class": {},
        }
        for c in CLASSES:
            sub_all = [r for r in recs if r["class"] == c]
            split_out["per_class"][str(c)] = {}
            for tag, sub in (
                    ("all_labels", sub_all),
                    ("10mhz", [r for r in sub_all if r["is_10mhz"]])):
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
        print(f"{split}: {len(recs)} class10/11 instances, "
              f"{n_files_with_target} target files, "
              f"{split_out['n_instances_10mhz']} 10MHz instances", flush=True)

    # ---------- C0 YOLO AP, with validated JSON/cache mapping ----------
    z, images, targets = build_cache_arrays()
    bw_per_gt, n_checked = validate_cache_mapping(z, targets, sig_cache)
    out["val_gt_mapping_validation"] = {
        "n_samples_checked": n_checked,
        "n_samples_total": len(z["sample_ids"]),
        "method": "class + order + normalized geometry (atol=2e-4) vs JSON",
        "status": "exact" if n_checked == len(z["sample_ids"]) else "partial",
    }

    # full labels
    out["val_c0_yolo_ap50"] = {
        str(c): ap50_for_class(images, targets, c) for c in CLASSES}

    # build valid/ignored masks from the validated per-GT bandwidths
    valid_targets = []
    ignored_boxes = {str(c): [] for c in CLASSES}
    keep_masks = []
    for i, t in enumerate(targets):
        keep = np.ones(len(t.labels), bool)
        ign = {str(c): [] for c in CLASSES}
        for j, lab in enumerate(t.labels):
            lab = int(lab)
            if lab in CLASSES and not (BAND_LO <= bw_per_gt[i][j] <= BAND_HI):
                keep[j] = False
                ign[str(lab)].append(t.boxes[j])
        keep_masks.append(keep)
        valid_targets.append(ImageTargets(t.boxes[keep], t.labels[keep]))
        for c in CLASSES:
            ignored_boxes[str(c)].append(
                np.asarray(ign[str(c)], float).reshape(-1, 4))

    out["val_c0_yolo_ap50_10mhz_old_drop_gt"] = {
        str(c): drop_gt_ap50(images, targets, c, keep_masks) for c in CLASSES}
    out["val_c0_yolo_ap50_10mhz_ignore_aware"] = {}
    for c in CLASSES:
        r = ignore_aware_ap50(images, valid_targets, ignored_boxes[str(c)], c)
        out["val_c0_yolo_ap50_10mhz_ignore_aware"][str(c)] = {
            "ap50": r.ap50, "n_valid_gt": r.n_valid_gt,
            "n_ignored_gt": r.n_ignored_gt, "n_pred": r.n_pred,
            "n_tp": r.n_tp, "n_fp": r.n_fp,
            "n_ignored_pred": r.n_ignored_pred,
        }
    out["val_10mhz_ap_comparison"] = {}
    for c in CLASSES:
        old = out["val_c0_yolo_ap50_10mhz_old_drop_gt"][str(c)]["ap50"]
        new = out["val_c0_yolo_ap50_10mhz_ignore_aware"][str(c)]["ap50"]
        out["val_10mhz_ap_comparison"][str(c)] = {
            "old_drop_gt_ap50": old, "new_ignore_aware_ap50": new,
            "delta_new_minus_old_pp": None if old is None or new is None
            else 100.0 * (new - old),
        }
    out["val_c0_map50_reference"] = 0.6497250320877056
    out["note"] = ("Stage 0 only; no training. Fit has no C0 AP because the "
                   "YOLO cache is val. Test is sealed. The old drop-GT AP is "
                   "kept as a known-biased historical protocol; the "
                   "ignore-aware AP is the corrected 10 MHz subset result.")
    fit = out["splits"]["fit"]; val = out["splits"]["val"]
    out["revision_checks"] = {
        "old_fit_all_labels_instances": 2093,
        "new_fit_all_labels_instances": fit["n_instances"],
        "old_fit_all_labels_files": 1572,
        "new_fit_all_labels_files": fit["n_files"],
        "old_fit_10mhz_instances": 1824,
        "new_fit_10mhz_instances": fit["n_instances_10mhz"],
        "old_fit_10mhz_files": 1572,
        "new_fit_10mhz_files": fit["n_files_10mhz"],
        "old_val_all_labels_instances": 716,
        "new_val_all_labels_instances": val["n_instances"],
        "old_val_all_labels_files": 576,
        "new_val_all_labels_files": val["n_files"],
        "old_val_10mhz_instances": 469,
        "new_val_10mhz_instances": val["n_instances_10mhz"],
        "old_val_10mhz_files": 404,
        "new_val_10mhz_files": val["n_files_10mhz"],
        "narrow_share_fit": fit["n_instances_narrow"] / fit["n_instances"],
        "narrow_share_val": val["n_instances_narrow"] / val["n_instances"],
        "val_narrow_share_significantly_higher": False,
        "old_ap_10mhz_class10": 0.30454891288186015,
        "new_ap_10mhz_class10": out["val_10mhz_ap_comparison"]["10"]["new_ignore_aware_ap50"],
        "old_ap_10mhz_class11": 0.23769811095539448,
        "new_ap_10mhz_class11": out["val_10mhz_ap_comparison"]["11"]["new_ignore_aware_ap50"],
        "gt_mapping_status": out["val_gt_mapping_validation"]["status"],
    }

    # ---------- CSV ----------
    csv_path = OUT / "srrc_instances.csv"
    fields = list(out["splits"]["fit"]["_records"][0]) + ["split"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for split in ("fit", "val"):
            for r in out["splits"][split]["_records"]:
                w.writerow({**r, "split": split})
    for split in ("fit", "val"):
        out["splits"][split].pop("_records")
    (OUT / "srrc_data_audit.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("splits",)}, indent=2)[:5000])
    print("wrote", OUT / "srrc_data_audit.json", csv_path)


if __name__ == "__main__":
    main()
