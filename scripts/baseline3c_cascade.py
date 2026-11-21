#!/usr/bin/env python3
"""Phase C' cascade: apply WiFi modulation experts to YOLO predicted boxes.

Loads the cached YOLO val predictions (baseline3c_yolo_pass.py output),
converts each WiFi-class predicted box to physical IQ coordinates, runs
the OFDM frontend + histogram + expert CNN, and replaces the class label
when sync succeeds. Produces C0 (baseline), identity control, C1
(W-Shared), and C2 (W20/W40) evaluation results.

Protocol (frozen):
- Only class labels change; boxes, scores, candidate set, and ordering
  are identical across all variants.
- Family routing from YOLO's ORIGINAL predicted class (0-2 -> W20,
  3-5 -> W40); analysis bandwidth = nominal 20/40 MHz centered on the
  predicted box's frequency center (not the predicted width).
- Sync gate: metric > 0.34 AND ratio > 15 (Phase A' calibration).
- Sync failure: keep original YOLO class.
- GT is never used during inference; only for evaluation and analysis.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.baseline3b_ofdm_frontend import (
    read_segment, demod_to_center, band_limit_fft, cp_scan_full,
    strip_symbols, filter_symbol_rows, legacy_masks,
)
from scripts.baseline3b_histogram import block_histogram, histogram_for_training
from scripts.baseline3b_train_experts import HistCNN

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "runs_local/baseline3c"
WEIGHTS = ROOT / "runs_local/baseline3b/weights"
EXP_DIR = ROOT / "reports/baseline3_wifi_cascade"
PASS_METRIC = 0.34
PASS_RATIO = 15.0
SCS_HZ = 312.5e3
MAX_SYMBOLS = 4000
BLOCK = 128

# from scripts.train_local_representation_classifiers / metrics
import sys
sys.path.insert(0, str(ROOT))


def box_to_physical(x1, y1, x2, y2, B_mhz, f_lo, N):
    """Normalized global xyxy -> physical (f1, f2, t1_ms, t2_ms)."""
    fs = B_mhz * 1e6
    h = math.ceil(N / 511)
    t1 = (x1 * 512 - 0.5) * h / fs * 1000
    t2 = (x2 * 512 - 0.5) * h / fs * 1000
    f1 = f_lo + y1 * B_mhz
    f2 = f_lo + y2 * B_mhz
    return f1, f2, t1, t2


def process_one_wifi_pred(args):
    """Extract IQ, sync, histogram for one WiFi predicted box.
    Runs in a worker process; returns sync status + expert features.
    """
    (bin_path, f_lo, B_mhz, N, f_center, t1_ms, t2_ms,
     nominal_bw_mhz) = args
    try:
        fs = B_mhz * 1e6
        f_lo_hz = f_lo * 1e6
        B_hz = B_mhz * 1e6
        fc = f_center * 1e6
        bw = nominal_bw_mhz * 1e6
        # clip to observation
        f1 = max(f_lo_hz, fc - bw / 2)
        f2 = min(f_lo_hz + B_hz, fc + bw / 2)
        if f2 - f1 < bw * 0.5:
            return {"status": "invalid_band"}
        # time: use predicted box time range, clipped
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
        xb = demod_to_center(x, fs, fc, f_lo * 1e6)
        xb = band_limit_fft(xb, fs, bw * 1.02, edge_frac=0.05)
        # trim edges
        k = int(0.5e-3 * fs)
        if len(xb) > 2 * k:
            xb = xb[k:-k]
        # sync
        res = cp_scan_full(xb, fs)
        b = res["best"]
        if b is None:
            return {"status": "no_candidate"}
        if b["metric"] < PASS_METRIC or b["ratio"] < PASS_RATIO:
            return {"status": "sync_fail",
                    "metric": b["metric"], "ratio": b["ratio"],
                    "tu_us": b["tu_us"], "cp_us": b["cp_us"]}
        # strip
        X = strip_symbols(xb, fs, b["tu_us"], b["cp_us"], b["phi"],
                           cfo_cycles=-b["cfo_cycles"],
                           max_sym=MAX_SYMBOLS)
        n_fft_sig = int(round(nominal_bw_mhz * 1e6 / SCS_HZ))
        occ, pilots, data = legacy_masks(n_fft_sig)
        X, keep_idx, n_kept, _ = filter_symbol_rows(X, data)
        if n_kept < BLOCK:
            return {"status": "too_few_symbols", "n_kept": n_kept}
        # histogram blocks
        from scripts.baseline3b_histogram import consecutive_runs
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
            return {"status": "no_histogram"}
        return {
            "status": "ok",
            "metric": b["metric"], "ratio": b["ratio"],
            "tu_us": b["tu_us"], "cp_us": b["cp_us"],
            "n_kept": n_kept, "n_blocks": len(blocks),
            "blocks": np.stack(blocks).astype(np.float32),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE_DIR / "yolo_val_cache.npz"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-dir", default=str(EXP_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load cached YOLO predictions ----
    z = np.load(args.cache, allow_pickle=True)
    sample_ids = [str(s) for s in z["sample_ids"]]
    metas = [json.loads(str(m)) for m in z["meta"]]
    c0_boxes = z["c0_boxes"]; c0_scores = z["c0_scores"]
    c0_labels = z["c0_labels"]; c0_counts = z["c0_counts"]
    gt_boxes = z["gt_boxes"]; gt_labels = z["gt_labels"]
    gt_counts = z["gt_counts"]
    n_samples = len(sample_ids)
    print(f"samples={n_samples}, total c0 preds={len(c0_boxes)}")

    # ---- build per-sample index ranges ----
    c0_offsets = np.concatenate([[0], np.cumsum(c0_counts)])
    gt_offsets = np.concatenate([[0], np.cumsum(gt_counts)])

    # ---- collect WiFi predictions to process ----
    wifi_jobs = []  # (sample_idx, pred_idx_within_sample, job_args)
    for si in range(n_samples):
        m = metas[si]
        lo, hi = c0_offsets[si], c0_offsets[si + 1]
        for pi in range(lo, hi):
            if c0_labels[pi] not in (0, 1, 2, 3, 4, 5):
                continue
            x1, y1, x2, y2 = c0_boxes[pi]
            f1, f2, t1, t2 = box_to_physical(
                x1, y1, x2, y2, m["B_mhz"], m["f_lo"], m["N"])
            f_center = (f1 + f2) / 2
            nominal_bw = 20 if c0_labels[pi] <= 2 else 40
            bin_path = f"/root/autodl-fs/SpaceNet_Dataset/advanced/{sample_ids[si]}.bin"
            wifi_jobs.append((si, pi, (
                bin_path, m["f_lo"], m["B_mhz"], m["N"],
                f_center, t1, t2, nominal_bw)))
    print(f"WiFi predictions to process: {len(wifi_jobs)}")

    # ---- run IQ extraction + sync in parallel ----
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        job_args = [j[2] for j in wifi_jobs]
        for r in ex.map(process_one_wifi_pred, job_args, chunksize=16):
            results.append(r)
            if len(results) % 500 == 0:
                print(f"  {len(results)}/{len(wifi_jobs)} "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)
    sync_time = time.perf_counter() - t0
    print(f"sync done in {sync_time:.0f}s")

    # ---- load expert models ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experts = {}
    for arm in ("W20", "W40", "SHARED"):
        for seed in (1729, 1730, 1731):
            model = HistCNN(3).to(device)
            sd = torch.load(WEIGHTS / f"{arm}_seed{seed}.pt",
                            map_location=device)
            model.load_state_dict(sd)
            model.eval()
            experts[(arm, seed)] = model

    # ---- classify histograms ----
    t1 = time.perf_counter()
    wifi_results = []
    for (si, pi, _), r in zip(wifi_jobs, results):
        entry = {"sample_idx": si, "pred_idx": pi,
                 "status": r["status"]}
        if r["status"] == "ok":
            blocks = torch.from_numpy(r["blocks"]).to(device)
            for arm in ("W20", "W40", "SHARED"):
                # routing: W20 only for original class 0-2, etc.
                orig_class = int(c0_labels[pi])
                if arm == "W20" and orig_class > 2:
                    continue
                if arm == "W40" and orig_class <= 2:
                    continue
                for seed in (1729, 1730, 1731):
                    model = experts[(arm, seed)]
                    with torch.no_grad():
                        probs = F.softmax(model(blocks), dim=1)
                        mean_prob = probs.mean(dim=0).cpu().numpy()
                    entry[f"{arm}_seed{seed}_prob"] = mean_prob.tolist()
                    entry[f"{arm}_seed{seed}_pred"] = int(np.argmax(mean_prob))
        wifi_results.append(entry)
    cnn_time = time.perf_counter() - t1
    print(f"CNN forward done in {cnn_time:.1f}s")

    # ---- build variant prediction sets ----
    from src.signals.metrics import ImagePredictions, ImageTargets, evaluate

    def build_eval(labels_override=None):
        """Build per-sample prediction/target lists for evaluation."""
        preds = []
        cur = 0
        pred_map = {}  # (si, pi) -> new_label
        if labels_override:
            for e in wifi_results:
                key = (e["sample_idx"], e["pred_idx"])
                pred_map[key] = labels_override(e)
        for si in range(n_samples):
            lo, hi = c0_offsets[si], c0_offsets[si + 1]
            b = c0_boxes[lo:hi]
            s = c0_scores[lo:hi]
            l = c0_labels[lo:hi].copy()
            if labels_override:
                for pi in range(lo, hi):
                    key = (si, pi)
                    if key in pred_map:
                        l[pi - lo] = pred_map[key]
            preds.append(ImagePredictions(b, s, l))
        targets = []
        for si in range(n_samples):
            lo, hi = gt_offsets[si], gt_offsets[si + 1]
            targets.append(ImageTargets(gt_boxes[lo:hi], gt_labels[lo:hi]))
        return preds, targets

    # C0: original labels (no override)
    print("\n--- C0 (baseline, no expert) ---")
    preds_c0, targets = build_eval(None)
    ev_c0 = evaluate(targets, preds_c0, iou_thresholds=(0.5,), max_dets=100)
    m_c0 = ev_c0["per_iou"]["0.5"]["map_class_aware"]
    n_c0 = sum(len(p.boxes) for p in preds_c0)
    print(f"  mAP50={m_c0:.6f} n_pred={n_c0}")

    # identity control: force all predictions to keep original labels
    print("\n--- identity control ---")
    def identity(e):
        pi = e["pred_idx"]
        return int(c0_labels[pi])
    preds_id, _ = build_eval(identity)
    ev_id = evaluate(targets, preds_id, iou_thresholds=(0.5,), max_dets=100)
    m_id = ev_id["per_iou"]["0.5"]["map_class_aware"]
    print(f"  mAP50={m_id:.6f} (should equal C0)")
    identity_ok = abs(m_id - m_c0) < 1e-8

    # C1: W-Shared expert
    print("\n--- C1 (W-Shared) ---")
    def shared_label(e, seed=1729):
        pi = e["pred_idx"]
        orig = int(c0_labels[pi])
        key = f"SHARED_seed{seed}_pred"
        if e["status"] == "ok" and key in e:
            expert_pred = e[key]
            if orig <= 2:
                return expert_pred  # 0/1/2
            else:
                return expert_pred + 3  # 3/4/5
        return orig
    for seed in (1729, 1730, 1731):
        fn = lambda e, s=seed: shared_label(e, s)
        preds_c1, _ = build_eval(fn)
        ev_c1 = evaluate(targets, preds_c1, iou_thresholds=(0.5,), max_dets=100)
        m_c1 = ev_c1["per_iou"]["0.5"]["map_class_aware"]
        print(f"  seed {seed}: mAP50={m_c1:.6f} "
              f"delta={100*(m_c1-m_c0):+.3f}pp")

    # C2: W20/W40 independent experts
    print("\n--- C2 (W20/W40) ---")
    def dual_label(e, seed=1729):
        pi = e["pred_idx"]
        orig = int(c0_labels[pi])
        arm = "W20" if orig <= 2 else "W40"
        key = f"{arm}_seed{seed}_pred"
        if e["status"] == "ok" and key in e:
            expert_pred = e[key]
            if orig <= 2:
                return expert_pred
            else:
                return expert_pred + 3
        return orig
    for seed in (1729, 1730, 1731):
        fn = lambda e, s=seed: dual_label(e, s)
        preds_c2, _ = build_eval(fn)
        ev_c2 = evaluate(targets, preds_c2, iou_thresholds=(0.5,), max_dets=100)
        m_c2 = ev_c2["per_iou"]["0.5"]["map_class_aware"]
        print(f"  seed {seed}: mAP50={m_c2:.6f} "
              f"delta={100*(m_c2-m_c0):+.3f}pp")

    # ---- save results ----
    summary = {
        "identity_ok": identity_ok,
        "c0_map50": m_c0, "c0_n_pred": n_c0,
        "sync_time_s": round(sync_time, 1),
        "cnn_time_s": round(cnn_time, 1),
        "n_wifi_preds": len(wifi_jobs),
    }
    # per-status counts
    from collections import Counter
    status_counts = Counter(r["status"] for r in results)
    summary["status_counts"] = dict(status_counts)
    (out_dir / "cascade_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir / 'cascade_summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
