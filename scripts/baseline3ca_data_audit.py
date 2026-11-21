#!/usr/bin/env python3
"""Phase C'-Adaptation stage 1: matched predicted-crop data audit.

For every val sample:
  1. class-agnostic one-to-one IoU>=0.5 matching of C0 predictions to GT
  2. keep only pairs where GT label is 0-5 (WiFi adaptation supervision)
  3. run IQ extraction + CP sync on the PREDICTED box geometry (not GT)
  4. record sync status, quality metrics, and geometric deltas
  5. stratify by class/family/overlap/IoU
  6. create file-level adaptation-train / calibration-dev / final-dev split

Output: adaptation_audit.json + adaptation_split.json + per-pair ledger.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from scripts.baseline3b_ofdm_frontend import (
    read_segment, demod_to_center, band_limit_fft, cp_scan_full,
    strip_symbols, filter_symbol_rows, legacy_masks,
)
from scripts.baseline3b_histogram import (
    consecutive_runs, block_histogram, histogram_for_training, BLOCK,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs_local/baseline3c"
EXP = ROOT / "reports/baseline3_wifi_cascade"
PASS_METRIC = 0.34
PASS_RATIO = 15.0
SCS_HZ = 312.5e3
MAX_SYMBOLS = 4000
NOMINAL_BW = {0: 20, 1: 20, 2: 20, 3: 40, 4: 40, 5: 40}


def box_to_physical(x1, y1, x2, y2, B_mhz, f_lo, N):
    fs = B_mhz * 1e6
    h = math.ceil(N / 511)
    t1 = (x1 * 512 - 0.5) * h / fs * 1000
    t2 = (x2 * 512 - 0.5) * h / fs * 1000
    f1 = f_lo + y1 * B_mhz
    f2 = f_lo + y2 * B_mhz
    return f1, f2, t1, t2


def iou_matrix(boxes1, boxes2):
    """IoU between [N,4] and [M,4] xyxy arrays."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    b1 = boxes1[:, None, :]
    b2 = boxes2[None, :, :]
    ix1 = np.maximum(b1[..., 0], b2[..., 0])
    iy1 = np.maximum(b1[..., 1], b2[..., 1])
    ix2 = np.minimum(b1[..., 2], b2[..., 2])
    iy2 = np.minimum(b1[..., 3], b2[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    a1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    a2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    return inter / np.maximum(a1 + a2 - inter, 1e-12)


def match_one_to_one(pred_boxes, gt_boxes, iou_thresh=0.5):
    """Class-agnostic greedy one-to-one matching. Returns (pred_idx, gt_idx, iou)."""
    iou = iou_matrix(pred_boxes, gt_boxes)
    if iou.size == 0:
        return []
    flat = iou.flatten()
    order = np.argsort(flat)[::-1]
    used_p, used_g = set(), set()
    pairs = []
    for f in order:
        p, g = divmod(int(f), iou.shape[1])
        if flat[f] < iou_thresh:
            break
        if p in used_p or g in used_g:
            continue
        used_p.add(p)
        used_g.add(g)
        pairs.append((p, g, float(flat[f])))
    return pairs


def sync_one_pred(args):
    """Run IQ extraction + CP sync on a predicted box."""
    (bin_path, f_lo, B_mhz, N,
     f_center, t1_ms, t2_ms, nominal_bw_mhz) = args
    try:
        fs = B_mhz * 1e6
        f_lo_hz = f_lo * 1e6
        fc = f_center * 1e6
        bw = nominal_bw_mhz * 1e6
        f1 = max(f_lo_hz, fc - bw / 2)
        f2 = min(f_lo_hz + B_mhz * 1e6, fc + bw / 2)
        if f2 - f1 < bw * 0.5:
            return {"status": "invalid_band"}
        ta = max(0.0, t1_ms)
        tb = min(t2_ms, 1e9)
        dur = tb - ta
        if dur < 0.5:
            return {"status": "invalid_duration"}
        cap_ms = min(dur, 20.0)
        ta_c = (ta + tb) / 2 - cap_ms / 2
        tb_c = (ta + tb) / 2 + cap_ms / 2
        i0 = max(0, int(round(ta_c / 1000 * fs)))
        i1 = min(int(N), int(round(tb_c / 1000 * fs)))
        if i1 - i0 < 64:
            return {"status": "too_short"}
        x = read_segment(bin_path, i0, i1)
        xb = demod_to_center(x, fs, fc, f_lo_hz)
        xb = band_limit_fft(xb, fs, bw * 1.02, edge_frac=0.05)
        k = int(0.5e-3 * fs)
        if len(xb) > 2 * k:
            xb = xb[k:-k]
        res = cp_scan_full(xb, fs)
        b = res["best"]
        if b is None:
            return {"status": "no_candidate"}
        out = {"metric": round(b["metric"], 4), "ratio": round(b["ratio"], 1),
               "tu_us": b["tu_us"], "cp_us": b["cp_us"]}
        if b["metric"] < PASS_METRIC or b["ratio"] < PASS_RATIO:
            out["status"] = "sync_fail"
            return out
        X = strip_symbols(xb, fs, b["tu_us"], b["cp_us"], b["phi"],
                          cfo_cycles=-b["cfo_cycles"], max_sym=MAX_SYMBOLS)
        n_fft_sig = int(round(nominal_bw_mhz * 1e6 / SCS_HZ))
        occ, pilots, data = legacy_masks(n_fft_sig)
        X, keep_idx, n_kept, _ = filter_symbol_rows(X, data)
        out["n_kept"] = n_kept
        if n_kept < BLOCK:
            out["status"] = "too_few_symbols"
            return out
        blocks = []
        for a, e in consecutive_runs(keep_idx):
            run = X[a:e]
            for kk in range(len(run) // BLOCK):
                H = block_histogram(run[kk * BLOCK:(kk + 1) * BLOCK], data)
                if H is not None:
                    Ht = histogram_for_training(H)
                    if Ht is not None:
                        blocks.append(Ht)
            if len(blocks) >= 4:
                break
        if not blocks:
            out["status"] = "no_histogram"
            return out
        out["status"] = "ok"
        out["n_blocks"] = len(blocks)
        out["blocks"] = np.stack(blocks).astype(np.float32)
        return out
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


def main():
    z = np.load(CACHE / "yolo_val_cache.npz", allow_pickle=True)
    sample_ids = [str(s) for s in z["sample_ids"]]
    metas = [json.loads(str(m)) for m in z["meta"]]
    c0_boxes = z["c0_boxes"]; c0_scores = z["c0_scores"]
    c0_labels = z["c0_labels"]; c0_counts = z["c0_counts"]
    gt_boxes = z["gt_boxes"]; gt_labels = z["gt_labels"]
    gt_counts = z["gt_counts"]
    n_samples = len(sample_ids)
    c0_offsets = np.concatenate([[0], np.cumsum(c0_counts)])
    gt_offsets = np.concatenate([[0], np.cumsum(gt_counts)])

    # ---- load manifest for overlap flags ----
    man_path = ROOT / "reports/baseline3_wifi_expert/dataset_manifest.json"
    man = json.loads(man_path.read_text())
    # build lookup: (sample_id, approximate freq) -> overlap info
    # we need overlap flags for GT instances; use the signal_id from GT box position
    overlap_by_sample = {}
    for inst in man["instances"]:
        overlap_by_sample.setdefault(inst["sample_id"], []).append(inst)

    print(f"samples={n_samples}")

    # ---- per-sample matching ----
    all_pairs = []  # matched WiFi pairs across all samples
    for si in range(n_samples):
        m = metas[si]
        c0_lo, c0_hi = int(c0_offsets[si]), int(c0_offsets[si + 1])
        gt_lo, gt_hi = int(gt_offsets[si]), int(gt_offsets[si + 1])
        pb = c0_boxes[c0_lo:c0_hi]
        gb = gt_boxes[gt_lo:gt_hi]
        pl = c0_labels[c0_lo:c0_hi]
        gl = gt_labels[gt_lo:gt_hi]
        ps = c0_scores[c0_lo:c0_hi]

        # class-agnostic matching
        pairs = match_one_to_one(pb, gb, iou_thresh=0.5)
        for p_idx, g_idx, iou in pairs:
            gt_label = int(gl[g_idx])
            if gt_label > 5:
                continue  # only WiFi GT for adaptation
            pred_label = int(pl[p_idx])
            x1, y1, x2, y2 = pb[p_idx]
            f1, f2, t1, t2 = box_to_physical(
                x1, y1, x2, y2, m["B_mhz"], m["f_lo"], m["N"])
            f_center = (f1 + f2) / 2
            nominal_bw = NOMINAL_BW.get(pred_label, 20)
            gt_x1, gt_y1, gt_x2, gt_y2 = gb[g_idx]
            gf1, gf2, gt_t1, gt_t2 = box_to_physical(
                gt_x1, gt_y1, gt_x2, gt_y2, m["B_mhz"], m["f_lo"], m["N"])

            # overlap flags: look up this GT instance in the manifest
            ov_any = 0; ov_wifi = 0; ov_frac = 0.0
            for inst in overlap_by_sample.get(sample_ids[si], []):
                if (abs(inst["f1_mhz"] - gf1) < 0.01 and
                        abs(inst["f2_mhz"] - gf2) < 0.01):
                    ov_any = inst["overlap_any"]
                    ov_wifi = inst["overlap_wifi"]
                    ov_frac = inst["overlap_wifi_frac_max"]
                    break

            bin_path = f"/root/autodl-fs/SpaceNet_Dataset/advanced/{sample_ids[si]}.bin"
            all_pairs.append({
                "sample_idx": si, "sample_id": sample_ids[si],
                "pred_idx": p_idx + c0_lo, "gt_idx": g_idx + gt_lo,
                "gt_label": gt_label, "pred_label": pred_label,
                "pred_score": float(ps[p_idx]), "iou": round(iou, 4),
                "f_center": round(f_center, 3),
                "t1_ms": round(t1, 3), "t2_ms": round(t2, 3),
                "nominal_bw": nominal_bw,
                "gt_f_center": round((gf1 + gf2) / 2, 3),
                "gt_t1": round(gt_t1, 3), "gt_t2": round(gt_t2, 3),
                "f_center_delta": round(f_center - (gf1 + gf2) / 2, 3),
                "t_center_delta": round((t1 + t2) / 2 - (gt_t1 + gt_t2) / 2, 3),
                "overlap_any": ov_any, "overlap_wifi": ov_wifi,
                "overlap_wifi_frac_max": ov_frac,
                "B_mhz": m["B_mhz"],
                "job": (bin_path, m["f_lo"], m["B_mhz"], m["N"],
                        f_center, t1, t2, nominal_bw),
            })

    print(f"matched WiFi pairs (IoU>=0.5, GT class 0-5): {len(all_pairs)}")

    # ---- run sync on matched predicted boxes ----
    t0 = time.perf_counter()
    sync_results = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        job_args = [p["job"] for p in all_pairs]
        for r in ex.map(sync_one_pred, job_args, chunksize=16):
            sync_results.append(r)
            if len(sync_results) % 500 == 0:
                print(f"  {len(sync_results)}/{len(all_pairs)} "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)
    sync_time = time.perf_counter() - t0
    print(f"sync done in {sync_time:.0f}s")

    # attach sync results
    for p, r in zip(all_pairs, sync_results):
        p["sync_status"] = r["status"]
        if r["status"] == "ok":
            p["sync_metric"] = r.get("metric")
            p["sync_ratio"] = r.get("ratio")
            p["n_blocks"] = r.get("n_blocks")
        elif "metric" in r:
            p["sync_metric"] = r.get("metric")
            p["sync_ratio"] = r.get("ratio")

    # ---- stratified counts ----
    status_by = defaultdict(Counter)
    for p in all_pairs:
        st = p["sync_status"]
        status_by["overall"][st] += 1
        status_by[f"class_{p['gt_label']}"][st] += 1
        status_by[f"family_{'W20' if p['gt_label'] <= 2 else 'W40'}"][st] += 1
        ov = "clean" if p["overlap_any"] == 0 else "overlap"
        status_by[f"overlap_{ov}"][st] += 1
        iou_b = "hi" if p["iou"] >= 0.75 else "mid" if p["iou"] >= 0.6 else "lo"
        status_by[f"iou_{iou_b}"][st] += 1

    # ---- file-level split (70/15/15) ----
    # group by sample_id; stratify by sync_ok presence
    sample_sync = defaultdict(lambda: [0, 0])  # sid -> [n_total, n_ok]
    for p in all_pairs:
        sample_sync[p["sample_id"]][0] += 1
        if p["sync_status"] == "ok":
            sample_sync[p["sample_id"]][1] += 1
    sids_with_wifi = sorted(sample_sync.keys())
    rng = np.random.default_rng(1729)
    shuffled = rng.permutation(len(sids_with_wifi))
    n_train = int(len(sids_with_wifi) * 0.70)
    n_cal = int(len(sids_with_wifi) * 0.15)
    split = {}
    for i, sid in enumerate(sids_with_wifi):
        if i < n_train:
            split[sid] = "adaptation-train"
        elif i < n_train + n_cal:
            split[sid] = "calibration-dev"
        else:
            split[sid] = "final-dev"

    # verify split balance
    split_stats = {}
    for sname in ("adaptation-train", "calibration-dev", "final-dev"):
        sids = [s for s, v in split.items() if v == sname]
        n_pairs = sum(1 for p in all_pairs if p["sample_id"] in sids)
        n_ok = sum(1 for p in all_pairs
                   if p["sample_id"] in sids and p["sync_status"] == "ok")
        cls_dist = Counter(p["gt_label"] for p in all_pairs
                          if p["sample_id"] in sids)
        split_stats[sname] = {
            "n_samples": len(sids), "n_pairs": n_pairs,
            "n_sync_ok": n_ok,
            "class_dist": {str(k): v for k, v in sorted(cls_dist.items())},
        }

    # ---- save ----
    audit = {
        "n_val_samples": n_samples,
        "n_samples_with_wifi_gt": len(sids_with_wifi),
        "n_matched_pairs": len(all_pairs),
        "sync_time_s": round(sync_time, 1),
        "status_by_stratum": {k: dict(v) for k, v in status_by.items()},
        "split": {k: v for k, v in split.items()},
        "split_stats": split_stats,
    }
    out_path = EXP / "adaptation_audit.json"
    out_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    # per-pair ledger (without raw IQ)
    def sanitize(o):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    ledger = [{k: sanitize(v) for k, v in p.items() if k != "job"}
              for p in all_pairs]
    (EXP / "adaptation_pairs.json").write_text(
        json.dumps(ledger, indent=2), encoding="utf-8")

    # summary print
    print(f"\n{'='*60}")
    print(f"WiFi matched pairs: {len(all_pairs)}")
    print(f"  sync ok: {sum(1 for p in all_pairs if p['sync_status']=='ok')}")
    print(f"  sync_fail: {sum(1 for p in all_pairs if p['sync_status']=='sync_fail')}")
    print(f"\nSplit:")
    for sname, ss in split_stats.items():
        print(f"  {sname}: {ss['n_samples']} samples, "
              f"{ss['n_pairs']} pairs, {ss['n_sync_ok']} sync_ok")
        print(f"    classes: {ss['class_dist']}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
