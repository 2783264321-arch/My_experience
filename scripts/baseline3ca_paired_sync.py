#!/usr/bin/env python3
"""Phase C'-Adaptation: paired GT vs Predicted-box sync comparison.

For every matched WiFi GT pair (1,394), run the SAME extraction recipe
twice:
  - GT box geometry   (GT center freq, GT-family nominal bandwidth,
                       GT time box)
  - Pred box geometry (already measured in v1; reused)
and tabulate the four-quadrant outcome, stratified by class, family,
overlap, IoU, predicted center offset and predicted width deviation.

Central question: is the W40 QPSK-vs-16/64QAM sync-rate gap (31.6% vs
~10%) already present with GT boxes (signal-level property), or is it
introduced by YOLO predicted-box geometry?
"""
from __future__ import annotations

import json
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
from scripts.baseline3ca_data_audit_v2 import overlap_fields
from src.signals.data import SPLIT_MANIFEST, load_manifest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports/baseline3_wifi_cascade"
PASS_METRIC = 0.34
PASS_RATIO = 15.0
SCS_HZ = 312.5e3
MAX_SYMBOLS = 4000
NOMINAL_BW = {0: 20, 1: 20, 2: 20, 3: 40, 4: 40, 5: 40}


def sync_box(args):
    """Sync one box (GT geometry) with the fixed recipe."""
    (bin_path, f_lo_mhz, B_mhz, N,
     f_center_mhz, t1_ms, t2_ms, nominal_bw_mhz) = args
    try:
        fs = B_mhz * 1e6
        f_lo_hz = f_lo_mhz * 1e6
        fc = f_center_mhz * 1e6
        bw = nominal_bw_mhz * 1e6
        f1 = max(f_lo_hz, fc - bw / 2)
        f2 = min(f_lo_hz + B_mhz * 1e6, fc + bw / 2)
        if f2 - f1 < bw * 0.5:
            return {"status": "invalid_band"}
        ta = max(0.0, t1_ms)
        tb = min(t2_ms, 1e9)
        if tb - ta < 0.5:
            return {"status": "invalid_duration"}
        cap_ms = min(tb - ta, 20.0)
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
        out = {"metric": round(b["metric"], 4),
               "ratio": round(b["ratio"], 1),
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
        n_blocks = 0
        for a, e in consecutive_runs(keep_idx):
            run = X[a:e]
            for kk in range(len(run) // BLOCK):
                H = block_histogram(run[kk * BLOCK:(kk + 1) * BLOCK], data)
                if H is not None and histogram_for_training(H) is not None:
                    n_blocks += 1
                if n_blocks >= 4:
                    break
            if n_blocks >= 4:
                break
        if n_blocks == 0:
            out["status"] = "no_histogram"
            return out
        out["status"] = "ok"
        out["n_blocks"] = n_blocks
        return out
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


def fam_of(c):
    return "W20" if c <= 2 else "W40"


def main():
    pairs = json.loads((EXP / "adaptation_pairs.json").read_text())
    recs = {r.sample_id: r for r in
            load_manifest(SPLIT_MANIFEST, split="val")}
    print(f"pairs: {len(pairs)}")

    # ---- resolve GT signal + overlap (same rule as audit v2) ----
    for p in pairs:
        rec = recs[p["sample_id"]]
        best, best_d = None, 1e9
        for gi, s in enumerate(rec.signals):
            if int(s["class"]) != p["gt_label"]:
                continue
            d = (abs(float(s["start_frequency"]) +
                     float(s["end_frequency"]) - 2 * p["gt_f_center"]) +
                 abs(float(s["start_time"]) - p["gt_t1"]) +
                 abs(float(s["end_time"]) - p["gt_t2"]))
            if d < best_d:
                best_d, best = d, gi
        if best is not None and best_d <= 1.0:
            p["gt_signal_index"] = int(best)
            p.update(overlap_fields(rec, best))
            sig = rec.signals[best]
            p["gt_f1"] = float(sig["start_frequency"])
            p["gt_f2"] = float(sig["end_frequency"])
            p["gt_width_mhz"] = p["gt_f2"] - p["gt_f1"]
        else:
            p["overlap_any"] = None
            p["gt_width_mhz"] = None

    # ---- run GT-box sync ----
    t0 = time.perf_counter()
    gt_results = []
    jobs = []
    for p in pairs:
        rec = recs[p["sample_id"]]
        if p.get("gt_signal_index") is None:
            gt_results.append({"status": "unresolved"})
            jobs.append(None)
            continue
        gt_fam_bw = NOMINAL_BW[p["gt_label"]]
        jobs.append((f"/root/autodl-fs/SpaceNet_Dataset/advanced/"
                     f"{p['sample_id']}.bin",
                     rec.f_lo, rec.B_mhz, rec.N,
                     p["gt_f_center"], p["gt_t1"], p["gt_t2"],
                     gt_fam_bw))
    valid = [j for j in jobs if j is not None]
    with ProcessPoolExecutor(max_workers=12) as ex:
        out_iter = iter(ex.map(sync_box, valid, chunksize=16))
        for j in jobs:
            gt_results.append(next(out_iter) if j is not None
                              else {"status": "unresolved"})
    dt = time.perf_counter() - t0
    print(f"GT sync done in {dt:.0f}s")

    # ---- four-quadrant tabulation ----
    rows = []
    for p, g in zip(pairs, gt_results):
        pred_ok = p["sync_status"] == "ok"
        gt_ok = g["status"] == "ok"
        quad = ("gt_ok_pred_ok" if gt_ok and pred_ok else
                "gt_ok_pred_fail" if gt_ok and not pred_ok else
                "gt_fail_pred_ok" if (not gt_ok) and pred_ok else
                "gt_fail_pred_fail")
        rows.append({
            "sample_id": p["sample_id"],
            "gt_label": p["gt_label"],
            "pred_label": p["pred_label"],
            "family": fam_of(p["gt_label"]),
            "iou": p["iou"],
            "gt_sync": g["status"],
            "pred_sync": p["sync_status"],
            "gt_metric": g.get("metric"),
            "gt_ratio": g.get("ratio"),
            "pred_metric": p.get("sync_metric"),
            "pred_ratio": p.get("sync_ratio"),
            "gt_blocks": g.get("n_blocks"),
            "pred_blocks": min(int(p.get("n_blocks") or 0), 4),
            "quadrant": quad,
            "overlap_any": p.get("overlap_any"),
            "overlap_wifi": p.get("overlap_wifi"),
            "strong_overlap": p.get("strong_overlap"),
            "f_center_delta": p.get("f_center_delta"),
            "gt_width_mhz": p.get("gt_width_mhz"),
            "nominal_bw": p.get("nominal_bw"),
        })

    n = len(rows)
    quads = Counter(r["quadrant"] for r in rows)
    summary = {"n": n, "quadrants": dict(quads),
               "gt_sync_rate": round(sum(1 for r in rows
                                         if r["gt_sync"] == "ok") / n, 4),
               "pred_sync_rate": round(sum(1 for r in rows
                                           if r["pred_sync"] == "ok") / n, 4)}

    def strat(field, val_fn):
        out = defaultdict(lambda: [0, 0, 0])  # n, gt_ok, pred_ok
        for r in rows:
            k = val_fn(r)
            if k is None:
                continue
            out[k][0] += 1
            out[k][1] += int(r["gt_sync"] == "ok")
            out[k][2] += int(r["pred_sync"] == "ok")
        return {str(k): {"n": v[0], "gt_ok": v[1], "pred_ok": v[2],
                         "gt_rate": round(v[1] / v[0], 4),
                         "pred_rate": round(v[2] / v[0], 4)}
                for k, v in sorted(out.items())}

    summary["by_class"] = strat("class", lambda r: r["gt_label"])
    summary["by_overlap"] = strat("ov", lambda r: r["overlap_any"])
    summary["by_strong_overlap"] = strat("sov", lambda r: r["strong_overlap"])
    summary["by_iou_bin"] = strat(
        "iou", lambda r: ("hi>=0.75" if r["iou"] >= 0.75 else
                          "mid0.6-0.75" if r["iou"] >= 0.6 else "lo0.5-0.6"))
    summary["by_center_offset"] = strat(
        "off", lambda r: ("<0.5MHz" if abs(r["f_center_delta"]) < 0.5 else
                          "0.5-2MHz" if abs(r["f_center_delta"]) < 2.0 else
                          ">=2MHz"))
    summary["by_width_dev"] = strat(
        "wd", lambda r: (None if r["gt_width_mhz"] is None else
                         "<0.5MHz" if abs(r["gt_width_mhz"] -
                                          r["nominal_bw"]) < 0.5 else
                         "0.5-2MHz" if abs(r["gt_width_mhz"] -
                                           r["nominal_bw"]) < 2.0 else
                         ">=2MHz"))

    # W40 specific: GT vs Pred sync by class
    w40 = [r for r in rows if r["family"] == "W40"]
    summary["w40_detail"] = {}
    for c in (3, 4, 5):
        sel = [r for r in w40 if r["gt_label"] == c]
        if not sel:
            continue
        summary["w40_detail"][f"c{c}"] = {
            "n": len(sel),
            "gt_ok": sum(1 for r in sel if r["gt_sync"] == "ok"),
            "pred_ok": sum(1 for r in sel if r["pred_sync"] == "ok"),
            "gt_rate": round(sum(1 for r in sel if r["gt_sync"] == "ok")
                             / len(sel), 4),
            "pred_rate": round(sum(1 for r in sel if r["pred_sync"] == "ok")
                               / len(sel), 4),
        }
    # clean subset only
    clean = [r for r in rows if r["overlap_any"] == 0]
    summary["clean_subset"] = {
        "n": len(clean),
        "gt_rate": round(sum(1 for r in clean if r["gt_sync"] == "ok") /
                         max(len(clean), 1), 4),
        "pred_rate": round(sum(1 for r in clean if r["pred_sync"] == "ok") /
                           max(len(clean), 1), 4),
        "by_class": strat("clean_class",
                          lambda r: (r["gt_label"] if r["overlap_any"] == 0
                                     else None)),
    }

    out = {"summary": summary,
           "gt_sync_seconds": round(dt, 1),
           "pairs": rows}
    (EXP / "paired_gt_pred_sync.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote paired_gt_pred_sync.json")


if __name__ == "__main__":
    main()
