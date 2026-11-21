#!/usr/bin/env python3
"""Phase C'-Adaptation stage 2: build paired GT/Pred histogram caches.

One pass over the frozen C0 predictions:
  - every WiFi prediction (label 0-5): sync + histogram blocks on the
    PREDICTED box geometry (nominal bandwidth by predicted family);
  - every matched WiFi pair (1,394): sync + histogram blocks on the GT
    box geometry (nominal bandwidth by GT family).

Saved locally (never committed):
  pred_hist_cache.npz  per global c0 prediction index
  gt_hist_cache.npz    per adaptation_pairs.json ledger order

Blocks are float32 [1,50,50] per-instance max-normalized, capped at 4.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from concurrent.futures import ProcessPoolExecutor

from scripts.baseline3b_ofdm_frontend import (
    read_segment, demod_to_center, band_limit_fft, cp_scan_full,
    strip_symbols, filter_symbol_rows, legacy_masks,
)
from scripts.baseline3b_histogram import (
    consecutive_runs, block_histogram, histogram_for_training, BLOCK,
)
from src.signals.data import SPLIT_MANIFEST, load_manifest

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs_local/baseline3c"
EXP = ROOT / "reports/baseline3_wifi_cascade"
PASS_METRIC = 0.34
PASS_RATIO = 15.0
SCS_HZ = 312.5e3
MAX_SYMBOLS = 4000
NOMINAL_BW = {0: 20, 1: 20, 2: 20, 3: 40, 4: 40, 5: 40}


def sync_box(args):
    (bin_path, f_lo_mhz, B_mhz, N, f_center_mhz, t1_ms, t2_ms,
     nominal_bw_mhz) = args
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
            if len(blocks) >= 4:
                break
        if not blocks:
            out["status"] = "no_histogram"
            return out
        out["status"] = "ok"
        out["blocks"] = np.stack(blocks).astype(np.float32)
        return out
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


def pack(records, prefix):
    """Pack variable-length block lists into flat arrays + offsets."""
    blocks, offs, status, metric, ratio, nb = [], [0], [], [], [], []
    for r in records:
        if r["status"] == "ok":
            blocks.append(r["blocks"])
            offs.append(offs[-1] + len(r["blocks"]))
        else:
            offs.append(offs[-1])
        status.append(r["status"])
        metric.append(r.get("metric", -1))
        ratio.append(r.get("ratio", -1))
        nb.append(len(r["blocks"]) if r["status"] == "ok" else 0)
    flat = (np.concatenate(blocks) if blocks else
            np.zeros((0, 1, 50, 50), np.float32))
    return {f"{prefix}_blocks": flat.astype(np.float32),
            f"{prefix}_offsets": np.array(offs, np.int64),
            f"{prefix}_status": np.array(status),
            f"{prefix}_metric": np.array(metric, np.float32),
            f"{prefix}_ratio": np.array(ratio, np.float32),
            f"{prefix}_n_blocks": np.array(nb, np.int64)}


def main():
    z = np.load(CACHE / "yolo_val_cache.npz", allow_pickle=True)
    sample_ids = [str(s) for s in z["sample_ids"]]
    metas = [json.loads(str(m)) for m in z["meta"]]
    c0_boxes = z["c0_boxes"]; c0_scores = z["c0_scores"]
    c0_labels = z["c0_labels"]; c0_counts = z["c0_counts"]
    c0_offsets = np.concatenate([[0], np.cumsum(c0_counts)])
    pairs = json.loads((EXP / "adaptation_pairs.json").read_text())
    recs = {r.sample_id: r for r in load_manifest(SPLIT_MANIFEST, split="val")}

    # ---- WiFi prediction jobs ----
    pred_jobs = []
    pred_idx = []
    for si in range(len(sample_ids)):
        lo, hi = int(c0_offsets[si]), int(c0_offsets[si + 1])
        for pi in range(lo, hi):
            if c0_labels[pi] not in (0, 1, 2, 3, 4, 5):
                continue
            m = metas[si]
            x1, y1, x2, y2 = c0_boxes[pi]
            h = (m["N"] + 510) // 511
            t1 = (x1 * 512 - 0.5) * h / (m["B_mhz"] * 1e6) * 1000
            t2 = (x2 * 512 - 0.5) * h / (m["B_mhz"] * 1e6) * 1000
            f1 = m["f_lo"] + y1 * m["B_mhz"]
            f2 = m["f_lo"] + y2 * m["B_mhz"]
            fc = (f1 + f2) / 2
            bw = NOMINAL_BW[int(c0_labels[pi])]
            pred_jobs.append(
                (f"/root/autodl-fs/SpaceNet_Dataset/advanced/"
                 f"{sample_ids[si]}.bin",
                 m["f_lo"], m["B_mhz"], m["N"], fc, t1, t2, bw))
            pred_idx.append(pi)
    print(f"WiFi prediction jobs: {len(pred_jobs)}")

    t0 = time.perf_counter()
    pred_results = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for r in ex.map(sync_box, pred_jobs, chunksize=16):
            pred_results.append(r)
            if len(pred_results) % 1000 == 0:
                print(f"  pred {len(pred_results)}/{len(pred_jobs)} "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)
    print(f"pred sync done {time.perf_counter()-t0:.0f}s")

    # ---- GT jobs for matched pairs ----
    gt_jobs = []
    for p in pairs:
        rec = recs[p["sample_id"]]
        bw = NOMINAL_BW[p["gt_label"]]
        gt_jobs.append(
            (f"/root/autodl-fs/SpaceNet_Dataset/advanced/"
             f"{p['sample_id']}.bin",
             rec.f_lo, rec.B_mhz, rec.N, p["gt_f_center"],
             p["gt_t1"], p["gt_t2"], bw))
    print(f"GT jobs: {len(gt_jobs)}")
    t1 = time.perf_counter()
    gt_results = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for r in ex.map(sync_box, gt_jobs, chunksize=16):
            gt_results.append(r)
    print(f"gt sync done {time.perf_counter()-t1:.0f}s")

    # ---- pack and save ----
    out = {}
    out.update(pack(pred_results, "pred"))
    out["pred_idx"] = np.array(pred_idx, np.int64)
    out["pred_sample_idx"] = np.array(
        [int(np.searchsorted(c0_offsets, i, side="right") - 1)
         for i in pred_idx], np.int64)
    out["pred_labels"] = c0_labels[pred_idx].astype(np.int64)
    out["pred_scores"] = c0_scores[pred_idx].astype(np.float32)
    out["pred_boxes"] = c0_boxes[pred_idx].astype(np.float32)
    out["pred_sample_ids"] = np.array([sample_ids[s] for s in
                                       out["pred_sample_idx"]])

    out.update(pack(gt_results, "gt"))
    out["gt_pair_count"] = np.array(len(pairs))
    np.savez_compressed(CACHE / "hist_cache.npz", **out)
    n_pred_ok = sum(1 for r in pred_results if r["status"] == "ok")
    n_gt_ok = sum(1 for r in gt_results if r["status"] == "ok")
    print(f"\nwrote {CACHE/'hist_cache.npz'}")
    print(f"pred ok {n_pred_ok}/{len(pred_results)}, "
          f"gt ok {n_gt_ok}/{len(gt_results)}")


if __name__ == "__main__":
    main()
