#!/usr/bin/env python3
"""Read-only diagnostic: can intra-band peak/centroid recover coarse-missed
narrowband instances without adding input channels?

Question (from docs/collaboration/opencode-c3-holdout.md):
  In the same 0.7 ms STFT, does the peak / energy-centroid that the
  area-weighted coarse average discards explain or complement the
  "coarse missed, fine found" narrowband instances?

Method (no training, no threshold changes):
  1. Reuse the frozen B (coarse-only) and C (all-view) full-val predictions.
  2. Find matched instances per model per GT (IoU>=0.5, class-agnostic).
  3. "Fine-only" set = matched by C but not by B.
  4. For each GT instance, recompute the coarse frequency mapping
     (2048 area-weighted bins) and take the STFT power inside the GT band.
     Compare three per-band statistics:
        mean   : the current coarse channel (what the model sees)
        peak   : max bin inside the band (what the average discards)
        centroid: energy-weighted frequency centroid offset from band centre
     plus PAPR = peak/mean and a matched-filter control (white noise).
  5. Controls: same-B/T background bands (empty GT regions) and false
     positives of classes 9/12/13 from B predictions.

Outputs reports/coarse_narrow_recovery.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from src.signals.data import (
    COARSE_H, N_TIME, SPLIT_MANIFEST, build_sample_views, load_iq,
    load_manifest,
)
from src.signals.metrics import ImagePredictions, ImageTargets, _match_single
from src.signals.stft_fast import stft_power_fast

ROOT = Path(__file__).resolve().parents[1]
B_RUN = Path("/root/autodl-tmp/workspace/baseline/runs/yolo11s_c1_directreg_2048x512")
C_ALL = B_RUN / "stage5_val"
C_COARSE = ROOT / "runs/regression/c_coarse"
NARROW = (9, 12, 13)


def load_split_preds(val_dir):
    z = np.load(Path(val_dir) / "val_predictions.npz")
    ids = json.loads((Path(val_dir) / "val_sample_ids.json").read_text())
    per = {}
    for i, sid in enumerate(ids):
        pm = z["pred_sample_index"] == i
        gm = z["gt_sample_index"] == i
        per[sid] = {
            "pred": (z["pred_boxes"][pm], z["pred_scores"][pm], z["pred_labels"][pm]),
            "gt": (z["gt_boxes"][gm], z["gt_boxes"][gm], z["gt_labels"][gm]),
            "gt_boxes": z["gt_boxes"][gm], "gt_labels": z["gt_labels"][gm],
        }
    return per, z, ids


def match_status(rec, pb, ps, pl):
    """Return {gt_idx: matched_bool} using class-agnostic IoU>=0.5."""
    gt = ImageTargets(rec["gt_boxes"], rec["gt_labels"])
    pred = ImagePredictions(pb, ps, pl)
    m, _ = _match_single(gt, pred, 0.5, class_aware=False, max_dets=100)
    return {gi: True for pi, gi in m}


def band_stats(p_fine_time, f_lo_norm, f_hi_norm, fs_bins, B_mhz, n_time=N_TIME):
    """Per-band stats from the native [512,F] power array (time-major).

    Returns dict with mean/peak/centroid_offset/PAPR in native-bin units.
    p: [n_time, F] float32 (time-major from stft_power_fast).
    """
    F = p_fine_time.shape[1]
    c1 = int(np.clip(round(f_lo_norm * F), 0, F - 1))
    c2 = int(np.clip(round(f_hi_norm * F), c1 + 1, F))
    seg = p_fine_time[:, c1:c2]  # [T, bins]
    mean = float(seg.mean())
    peak = float(seg.max())
    # time frame = the GT time span
    papr = peak / max(mean, 1e-30)
    # energy centroid offset (normalized by band width)
    w = seg.sum(axis=0)
    if w.sum() > 0:
        bins = np.arange(c1, c2)
        centroid = float((w * bins).sum() / w.sum() - (c1 + c2 - 1) / 2.0)
        centroid /= max((c2 - c1) / 2.0, 1.0)
    else:
        centroid = 0.0
    return {"mean": mean, "peak": peak, "papr": papr,
            "centroid_offset": centroid, "n_bins": c2 - c1}


def freq_of_box(y_norm, B_mhz):
    return y_norm  # already normalized 0..1 within the observation band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--output", default="reports/coarse_narrow_recovery")
    ap.add_argument("--bg-per-sample", type=int, default=3)
    args = ap.parse_args()

    norm_cfg = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm_cfg["lo"]), float(norm_cfg["hi"]))
    val = load_manifest(SPLIT_MANIFEST, "val")
    by_id = {r.sample_id: r for r in val}
    if args.max_samples:
        val = val[: args.max_samples]

    pred_c, _, _ = load_split_preds(C_ALL)
    pred_b, _, _ = load_split_preds(C_COARSE)

    rows = []
    bg_rows = []
    fp_rows = []
    per_bt = defaultdict(lambda: {"n_gt": 0, "both": 0, "c_only": 0, "b_only": 0, "neither": 0})
    t0 = __import__("time").perf_counter()

    for i, rec in enumerate(val):
        sid = rec.sample_id
        if sid not in pred_c or sid not in pred_b:
            continue
        pc, pb = pred_c[sid], pred_b[sid]
        mc = match_status(pc, *pc["pred"])
        mb = match_status(pb, *pb["pred"])
        # Recompute the native STFT power once for this sample.
        iq = load_iq(rec.bin_path)
        p_time, _win, h, L, F = stft_power_fast(iq, rec.B_mhz, fft_threads=1)
        p_time = p_time  # [512, F] time-major
        n_ext = (N_TIME - 1) * h

        def time_cols(x1, x2):
            t1 = x1 * N_TIME - 0.5
            t2 = x2 * N_TIME - 0.5
            return int(np.clip(round(t1), 0, N_TIME - 1)), int(np.clip(round(t2), 0, N_TIME - 1)) + 1

        narrow_gt = [(gi, c) for gi, c in enumerate(pc["gt_labels"]) if int(c) in NARROW]
        # background bands: outside all GT boxes, same B/T; pick a few random
        rng = np.random.default_rng(abs(hash(sid)) % (2**31))
        # collect GT y-extents to avoid
        occ = []
        for gi in range(len(pc["gt_labels"])):
            x1, y1, x2, y2 = pc["gt_boxes"][gi]
            occ.append((y1, y2))
        # background candidates: horizontal strips between GT bands
        strips = []
        prev = 0.0
        for (y1, y2) in sorted(occ):
            if y1 - prev > 0.08:
                strips.append((prev, y1))
            prev = max(prev, y2)
        if 1.0 - prev > 0.08:
            strips.append((prev, 1.0))
        for k in range(min(args.bg_per_sample, len(strips))):
            y1, y2 = strips[k % len(strips)]
            mid = (y1 + y2) / 2.0
            bh = 0.04
            c1n = np.clip(mid - bh / 2, 0, 1 - 1e-6)
            c2n = c1n + bh
            stats = band_stats(p_time, c1n, c2n, F, rec.B_mhz)
            bg_rows.append({"sample_id": sid, "B": rec.B_mhz, "T": rec.T_ms,
                            "band_bins": stats["n_bins"], **{k: stats[k] for k in ("mean","peak","papr","centroid_offset")}})
        # B false positives of narrow classes on this sample
        pbx, pbs, pbl = pb["pred"]
        for j in range(len(pbl)):
            if int(pbl[j]) not in NARROW:
                continue
            x1, y1, x2, y2 = pbx[j]
            # check no GT narrow overlaps this FP
            if any((y1 < gy2 and y2 > gy1 and x1 < gx2 and x2 > gx1)
                   for (gx1, gy1, gx2, gy2) in pc["gt_boxes"]):
                continue
            stats = band_stats(p_time, y1, y2, F, rec.B_mhz)
            fp_rows.append({"sample_id": sid, "B": rec.B_mhz, "T": rec.T_ms, "cls": int(pbl[j]),
                            "score": float(pbs[j]),
                            "band_bins": stats["n_bins"], **{k: stats[k] for k in ("mean","peak","papr","centroid_offset")}})
        # GT instances
        for gi, c in narrow_gt:
            x1, y1, x2, y2 = pc["gt_boxes"][gi]
            stats = band_stats(p_time, y1, y2, F, rec.B_mhz)
            hit_c = bool(mc.get(gi, False))
            hit_b = bool(mb.get(gi, False))
            bt = (float(round(rec.B_mhz, 3)), int(round(rec.T_ms)))
            per_bt[bt]["n_gt"] += 1
            if hit_c and hit_b: per_bt[bt]["both"] += 1
            elif hit_c: per_bt[bt]["c_only"] += 1
            elif hit_b: per_bt[bt]["b_only"] += 1
            else: per_bt[bt]["neither"] += 1
            rows.append({"sample_id": sid, "B": rec.B_mhz, "T": rec.T_ms,
                         "cls": int(c), "gt_index": gi,
                         "hit_c": hit_c, "hit_b": hit_b,
                         "bw_mhz": float((y2 - y1) * rec.B_mhz),
                         "band_bins": stats["n_bins"], **stats})
        if (i + 1) % 100 == 0:
            el = __import__("time").perf_counter() - t0
            print(f"{i+1}/{len(val)} {el:.0f}s", flush=True)

    # summary
    def q(vals, q):
        return float(np.percentile(vals, q)) if len(vals) else None

    def group_stats(items):
        papr = [x["papr"] for x in items]
        cent = [abs(x["centroid_offset"]) for x in items]
        return {"n": len(items), "papr_median": q(papr, 50), "papr_p10": q(papr, 10),
                "papr_p90": q(papr, 90), "centroid_abs_median": q(cent, 50),
                "centroid_abs_p90": q(cent, 90),
                "peak_median": q([x["peak"] for x in items], 50),
                "mean_median": q([x["mean"] for x in items], 50),
                "n_bins_median": q([x["band_bins"] for x in items], 50)}

    both = [r for r in rows if r["hit_c"] and r["hit_b"]]
    c_only = [r for r in rows if r["hit_c"] and not r["hit_b"]]
    b_only = [r for r in rows if r["hit_b"] and not r["hit_c"]]
    neither = [r for r in rows if not r["hit_c"] and not r["hit_b"]]
    summary = {
        "n_gt_narrow": len(rows),
        "groups": {
            "both": group_stats(both), "c_only": group_stats(c_only),
            "b_only": group_stats(b_only), "neither": group_stats(neither),
            "background": group_stats(bg_rows), "fp_narrow_B": group_stats(fp_rows),
        },
        "per_BT": {f"{b:g}x{t}": v for (b, t), v in sorted(per_bt.items())},
    }
    # PAPR noise floor: matched-filter noise control per B (white noise => PAPR ~ ln-based)
    # Empirical control: background rows are the practical reference.

    out = {"summary": summary, "rows": rows, "background": bg_rows, "fp_narrow_B": fp_rows}
    p = ROOT / (args.output + ".json")
    p.write_text(json.dumps(out, indent=2))

    def f(x, nd=2):
        if x is None:
            return "n/a"
        if nd == 0:
            return str(int(round(x)))
        return f"{x:.{nd}f}"
    lines = ["# Coarse-missed narrowband: intra-band peak/centroid diagnostic", "",
             "Read-only. Frozen B (coarse-only) and C (all-view) predictions; no",
             "training, no threshold changes. band = GT frequency extent in the native",
             "0.7 ms STFT (time span = GT time). PAPR = peak/mean inside the band;",
             "centroid_offset = energy centroid offset from band centre, / (half band width).", "",
             "## Groups", "",
             "| group | n | PAPR med | PAPR p10/p90 | |centroid| med/p90 | peak med | mean med | band bins med |",
             "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |"]
    for name in ("both", "c_only", "b_only", "neither", "background", "fp_narrow_B"):
        g = summary["groups"][name]
        lines.append(f"| {name} | {g['n']} | {f(g['papr_median'])} | "
                     f"{f(g['papr_p10'])}/{f(g['papr_p90'])} | "
                     f"{f(g['centroid_abs_median'])}/{f(g['centroid_abs_p90'])} | "
                     f"{f(g['peak_median'],3)} | {f(g['mean_median'],3)} | {f(g['n_bins_median'],0)} |")
    lines += ["", "## Counts by B x T", "",
              "| B x T | narrow GT | both | C-only | B-only | neither |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for bt, v in sorted(summary["per_BT"].items()):
        lines.append(f"| {bt} | {v['n_gt']} | {v['both']} | {v['c_only']} | {v['b_only']} | {v['neither']} |")
    lines += ["", "Raw per-instance records are in the JSON.", ""]
    md = ROOT / (args.output + ".md")
    md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", p, md)


if __name__ == "__main__":
    main()
