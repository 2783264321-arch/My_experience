#!/usr/bin/env python3
"""Grid-level diagnostic: does intra-band peak add information beyond the
coarse mean on the actual 2048x512 grid the model sees?

Read-only. Reuses frozen B (coarse-only) and C (all-view) full-val
predictions; no training, no threshold changes. For every val sample the same
0.7 ms STFT is mapped to the existing 2048-row coarse segmentation, and three
per-cell maps are computed:

  mean     : area-weighted mean power per coarse row/time column (today's C1)
  peak     : max power inside the same row's native frequency support
  centroid : energy-weighted frequency offset from the row centre

Evaluation regions are the annotated time span of each narrow (9/12/13) GT.
Controls are same-shape background blocks in the same B/T sample and B's own
narrow false-positive boxes. Scores are the max cell value inside a region
(log10 power). Percentile calibration is per B/T against the background
blocks, so mean / peak / joint are compared at the same 5% background
false-positive budget.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.signals.data import COARSE_H, N_TIME, SPLIT_MANIFEST, load_iq, load_manifest
from src.signals.metrics import ImagePredictions, ImageTargets, _match_single
from src.signals.stft_fast import stft_power_fast

ROOT = Path(__file__).resolve().parents[1]
B_MAIN = Path("/root/autodl-tmp/workspace/baseline/runs/yolo11s_c1_directreg_2048x512")
C_ALL = B_MAIN / "stage5_val"
C_COARSE = ROOT / "runs/regression/c_coarse"
NARROW = (9, 12, 13)
TINY = float(np.finfo(np.float32).tiny)


def load_preds(val_dir: Path):
    z = np.load(val_dir / "val_predictions.npz")
    ids = json.loads((val_dir / "val_sample_ids.json").read_text())
    per = {}
    for i, sid in enumerate(ids):
        pm = z["pred_sample_index"] == i
        gm = z["gt_sample_index"] == i
        per[sid] = {
            "pred": (z["pred_boxes"][pm], z["pred_scores"][pm], z["pred_labels"][pm]),
            "gt_boxes": z["gt_boxes"][gm],
            "gt_labels": z["gt_labels"][gm],
        }
    return per


def match_map(gt_boxes, gt_labels, pb, ps, pl):
    gt = ImageTargets(gt_boxes, gt_labels)
    pred = ImagePredictions(pb, ps, pl)
    m, _ = _match_single(gt, pred, 0.5, class_aware=False, max_dets=100)
    return {gi for pi, gi in m}


def coarse_bounds(F: int, out_h: int = COARSE_H) -> np.ndarray:
    bounds = np.linspace(0, F, out_h + 1).round().astype(np.int64)
    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i - 1]:
            bounds[i] = bounds[i - 1] + 1
    return np.clip(bounds, 0, F)


def grids_from_power(p_time: np.ndarray, F: int):
    """Return mean/peak/centroid maps in [512, 2048] and row geometry."""
    bounds = coarse_bounds(F)
    counts = (bounds[1:] - bounds[:-1]).astype(np.float32)
    mean = np.add.reduceat(p_time, bounds[:-1], axis=1) / counts[None, :]
    peak = np.maximum.reduceat(p_time, bounds[:-1], axis=1)
    # centroid: energy-weighted bin index inside each segment
    bin_idx = np.arange(F, dtype=np.float32)
    weighted = np.add.reduceat(p_time * bin_idx[None, :], bounds[:-1], axis=1)
    total = np.add.reduceat(p_time, bounds[:-1], axis=1)
    centroid_bin = weighted / np.maximum(total, 1e-30)
    centers = (bounds[:-1] + bounds[1:] - 1) / 2.0
    half = np.maximum((bounds[1:] - bounds[:-1]) / 2.0, 1.0)
    centroid_off = (centroid_bin - centers[None, :]) / half[None, :]
    row_center_norm = centers / F
    half_norm = half / F
    return mean, peak, centroid_off, row_center_norm, half_norm, bounds


def log10_power(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, TINY))


def bbox_cells(box, n_rows=COARSE_H, n_cols=N_TIME):
    x1, y1, x2, y2 = [float(v) for v in box]
    c0 = int(np.clip(round(x1 * n_cols), 0, n_cols - 1))
    c1 = int(np.clip(round(x2 * n_cols), 0, n_cols - 1))
    r0 = int(np.clip(round(y1 * n_rows), 0, n_rows - 1))
    r1 = int(np.clip(round(y2 * n_rows), 0, n_rows - 1))
    if c1 < c0:
        c1 = c0
    if r1 < r0:
        r1 = r0
    return r0, r1, c0, c1


def region_scores(log_mean, log_peak, r0, r1, c0, c1):
    lm = log_mean[c0:c1 + 1, r0:r1 + 1]
    lp = log_peak[c0:c1 + 1, r0:r1 + 1]
    return float(lm.max()), float(lp.max())


def overlaps(r0, r1, c0, c1, boxes_cells, pad=2):
    for br0, br1, bc0, bc1 in boxes_cells:
        if r0 <= br1 + pad and r1 + pad >= br0 and c0 <= bc1 + pad and c1 + pad >= bc0:
            return True
    return False


def background_blocks(gt_boxes, pred_boxes, pred_labels, r0, r1, c0, c1, rng, k=3, tries=60):
    """Same-shape empty blocks, avoiding all GT and all B narrow predictions."""
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    occ = [bbox_cells(b) for b in gt_boxes]
    occ += [bbox_cells(b) for b, l in zip(pred_boxes, pred_labels) if int(l) in NARROW]
    out = []
    for _ in range(k):
        for _ in range(tries):
            rr0 = int(rng.integers(0, COARSE_H - h + 1))
            cc0 = int(rng.integers(0, N_TIME - w + 1))
            if not overlaps(rr0, rr0 + h - 1, cc0, cc0 + w - 1, occ):
                out.append((rr0, rr0 + h - 1, cc0, cc0 + w - 1))
                break
    return out


def percentile_rank(value, ref):
    if len(ref) == 0:
        return None
    return float(np.searchsorted(np.sort(ref), value, side="right") / len(ref))


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(allv)
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    return float((ranks[lab == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if nd == 0:
        return str(int(round(x)))
    return f"{x:.{nd}f}"


def save_example(sid, rec, log_mean, log_peak, gt_boxes, gt_labels, pred_b, tag, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, img, title in ((axes[0], log_mean, "coarse mean (C1)"), (axes[1], log_peak, "coarse intra-band peak")):
        vmin, vmax = np.percentile(img, [1, 99.5])
        # arrays are [time, freq-row]; transpose so x = time, y = frequency,
        # matching the box convention (x = time norm, y = frequency norm)
        ax.imshow(img.T, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                  extent=[0, 1, 0, 1])
        for b, l in zip(gt_boxes, gt_labels):
            if int(l) not in NARROW:
                continue
            x1, y1, x2, y2 = [float(v) for v in b]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", lw=1.5))
        for b, s, l in zip(*pred_b):
            if int(l) not in NARROW:
                continue
            x1, y1, x2, y2 = [float(v) for v in b]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="cyan", lw=1.0, ls="--"))
        ax.set_title(f"{sid} B={rec.B_mhz:g} T={rec.T_ms:g}\n{title} (red=narrow GT, cyan=B narrow pred)")
        ax.set_xlabel("time (norm)")
        ax.set_ylabel("frequency (norm)")
    fig.tight_layout()
    p = outdir / f"{tag}_{sid.replace('/', '_')}.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--bg-per-target", type=int, default=3)
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--output", default="reports/coarse_grid_peak")
    args = ap.parse_args()

    val = load_manifest(SPLIT_MANIFEST, "val")
    if args.max_samples:
        val = val[: args.max_samples]
    pred_c = load_preds(C_ALL)
    pred_b = load_preds(C_COARSE)
    outdir = ROOT / "reports/coarse_grid_peak_examples"
    outdir.mkdir(parents=True, exist_ok=True)

    targets, bgs, fps, cents = [], [], [], []
    example_paths = []
    n_examples = 0
    t0 = time.perf_counter()
    rng = np.random.default_rng(1729)

    for i, rec in enumerate(val):
        sid = rec.sample_id
        if sid not in pred_c or sid not in pred_b:
            continue
        pc, pb = pred_c[sid], pred_b[sid]
        mc = match_map(pc["gt_boxes"], pc["gt_labels"], *pc["pred"])
        mb = match_map(pc["gt_boxes"], pc["gt_labels"], *pb["pred"])
        p_time, _win, h, L, F = stft_power_fast(load_iq(rec.bin_path), rec.B_mhz, fft_threads=1)
        mean, peak, centroid, row_center, half_norm, bounds = grids_from_power(p_time, F)
        log_mean = log10_power(mean)
        log_peak = log10_power(peak)

        # B narrow false positives (no GT overlap)
        pb_boxes, pb_scores, pb_labels = pb["pred"]
        for b, s, l in zip(pb_boxes, pb_scores, pb_labels):
            if int(l) not in NARROW:
                continue
            x1, y1, x2, y2 = [float(v) for v in b]
            if any((y1 < gy2 and y2 > gy1 and x1 < gx2 and x2 > gx1)
                   for (gx1, gy1, gx2, gy2) in pc["gt_boxes"]):
                continue
            r0, r1, c0, c1 = bbox_cells(b)
            sm, sp = region_scores(log_mean, log_peak, r0, r1, c0, c1)
            fps.append({"sid": sid, "B": rec.B_mhz, "T": rec.T_ms, "cls": int(l),
                        "score": float(s), "mean": sm, "peak": sp,
                        "h": r1 - r0 + 1, "w": c1 - c0 + 1})

        # narrow GT instances
        for gi, (box, lab) in enumerate(zip(pc["gt_boxes"], pc["gt_labels"])):
            if int(lab) not in NARROW:
                continue
            r0, r1, c0, c1 = bbox_cells(box)
            sm, sp = region_scores(log_mean, log_peak, r0, r1, c0, c1)
            hit_c, hit_b = gi in mc, gi in mb
            group = "both" if hit_c and hit_b else "c_only" if hit_c else "b_only" if hit_b else "neither"
            targets.append({"sid": sid, "B": rec.B_mhz, "T": rec.T_ms, "cls": int(lab),
                            "group": group, "mean": sm, "peak": sp,
                            "h": r1 - r0 + 1, "w": c1 - c0 + 1,
                            "y1": float(box[1]), "y2": float(box[3])})
            # background blocks with same shape
            for br0, br1, bc0, bc1 in background_blocks(
                pc["gt_boxes"], pb_boxes, pb_labels, r0, r1, c0, c1, rng, k=args.bg_per_target
            ):
                bm, bp = region_scores(log_mean, log_peak, br0, br1, bc0, bc1)
                bgs.append({"sid": sid, "B": rec.B_mhz, "T": rec.T_ms,
                            "mean": bm, "peak": bp, "h": br1 - br0 + 1, "w": bc1 - bc0 + 1})
            # centroid error row (energy-weighted over the GT time columns)
            if c1 >= c0 and r1 >= r0:
                seg_mean = mean[c0:c1 + 1, r0:r1 + 1]
                seg_cent = centroid[c0:c1 + 1, r0:r1 + 1]
                w = seg_mean
                if w.sum() > 0:
                    energy_center = float((w * row_center[None, r0:r1 + 1]).sum() / w.sum())
                    cent_adjust = float((w * (row_center[None, r0:r1 + 1] +
                                               seg_cent * half_norm[None, r0:r1 + 1])).sum() / w.sum())
                    # row with max total energy
                    ridx = int(np.argmax(seg_mean.sum(axis=0)))
                    row_err = abs(float(row_center[r0 + ridx]) - (float(box[1]) + float(box[3])) / 2.0)
                    energy_err = abs(energy_center - (float(box[1]) + float(box[3])) / 2.0)
                    cent_err = abs(cent_adjust - (float(box[1]) + float(box[3])) / 2.0)
                    cents.append({"sid": sid, "B": rec.B_mhz, "T": rec.T_ms, "cls": int(lab),
                                  "group": group, "row_err": row_err * rec.B_mhz,
                                  "energy_err": energy_err * rec.B_mhz,
                                  "cent_err": cent_err * rec.B_mhz})
        # example plots: prefer 80 MHz c_only first, then 20 MHz
        if n_examples < args.examples:
            has_c_only = any(int(l) in NARROW and gi in mc and gi not in mb
                             for gi, l in enumerate(pc["gt_labels"]))
            if has_c_only and (rec.B_mhz == 80.0 or n_examples >= 2):
                example_paths.append(save_example(sid, rec, log_mean, log_peak,
                                                  pc["gt_boxes"], pc["gt_labels"], pb["pred"],
                                                  f"example{n_examples + 1}", outdir))
                n_examples += 1
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(val)} {time.perf_counter()-t0:.0f}s", flush=True)

    # per-B/T percentile calibration against background blocks
    by_bt_bg = defaultdict(list)
    for b in bgs:
        by_bt_bg[(float(round(b["B"], 3)), int(round(b["T"])))].append(b)
    def pct_scores(row):
        key = (float(round(row["B"], 3)), int(round(row["T"])))
        ref = by_bt_bg.get(key, [])
        if not ref:
            return None, None
        ref_mean = np.array([x["mean"] for x in ref])
        ref_peak = np.array([x["peak"] for x in ref])
        return percentile_rank(row["mean"], ref_mean), percentile_rank(row["peak"], ref_peak)
    for row in targets + fps + bgs:
        pm, pp = pct_scores(row)
        row["p_mean"], row["p_peak"] = pm, pp
        row["joint"] = None if pm is None or pp is None else pm + pp
    # thresholds per B/T at the requested background FPR
    thresholds = {}
    for key, ref in by_bt_bg.items():
        if not ref:
            continue
        tm = float(np.percentile([x["mean"] for x in ref], 100 * (1 - args.fpr)))
        tp = float(np.percentile([x["peak"] for x in ref], 100 * (1 - args.fpr)))
        tj = float(np.percentile([x["p_mean"] + x["p_peak"] for x in ref], 100 * (1 - args.fpr)))
        thresholds[key] = (tm, tp, tj)

    def passes(row, feature):
        key = (float(round(row["B"], 3)), int(round(row["T"])))
        if key not in thresholds:
            return None
        tm, tp, tj = thresholds[key]
        if feature == "mean":
            return row["mean"] >= tm
        if feature == "peak":
            return row["peak"] >= tp
        return row["joint"] >= tj

    # summary by B and group
    summary = {}
    for B in sorted({float(round(r["B"], 3)) for r in targets}):
        bt = [r for r in targets if float(round(r["B"], 3)) == B]
        bg = [r for r in bgs if float(round(r["B"], 3)) == B]
        fp = [r for r in fps if float(round(r["B"], 3)) == B]
        cen = [r for r in cents if float(round(r["B"], 3)) == B]
        entry = {"n_gt_narrow": len(bt), "n_bg": len(bg), "n_fp": len(fp)}
        entry["fp_pass"] = {
            f: ((sum(1 for r in fp if passes(r, f)) / len(fp)) if fp else None)
            for f in ("mean", "peak", "joint")
        }
        for group in ("c_only", "both"):
            g = [r for r in bt if r["group"] == group]
            entry[group] = {
                "n": len(g),
                "auc_mean_vs_bg": auc([r["mean"] for r in g], [r["mean"] for r in bg]),
                "auc_peak_vs_bg": auc([r["peak"] for r in g], [r["peak"] for r in bg]),
                "auc_joint_vs_bg": auc([r["joint"] for r in g], [r["joint"] for r in bg]),
                "recall_mean": (sum(1 for r in g if passes(r, "mean")) / len(g)) if g else None,
                "recall_peak": (sum(1 for r in g if passes(r, "peak")) / len(g)) if g else None,
                "recall_joint": (sum(1 for r in g if passes(r, "joint")) / len(g)) if g else None,
                "auc_peak_vs_fp": auc([r["peak"] for r in g], [r["peak"] for r in fp]),
                "h_med": float(np.median([r["h"] for r in g])) if g else None,
                "w_med": float(np.median([r["w"] for r in g])) if g else None,
                "n_pmean_eq_1": sum(1 for r in g if r.get("p_mean") is not None and r["p_mean"] >= 1.0),
            }
        if cen:
            entry["centroid"] = {
                "n": len(cen),
                "row_err_med_mhz": float(np.median([r["row_err"] for r in cen])),
                "energy_err_med_mhz": float(np.median([r["energy_err"] for r in cen])),
                "cent_err_med_mhz": float(np.median([r["cent_err"] for r in cen])),
                "row_err_p90_mhz": float(np.percentile([r["row_err"] for r in cen], 90)),
                "cent_err_p90_mhz": float(np.percentile([r["cent_err"] for r in cen], 90)),
                "cent_better_than_row": float(np.mean([r["cent_err"] < r["row_err"] for r in cen])),
                "cent_better_than_energy": float(np.mean([r["cent_err"] < r["energy_err"] for r in cen])),
            }
        summary[f"{B:g}"] = entry

    # mean-matched peak analysis (all B and 80 MHz separately)
    def mean_matched(items_targets, items_bg, bins=(0.0, 0.25, 0.5, 0.75, 1.0)):
        out = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            tg = [r for r in items_targets if r.get("p_mean") is not None and lo <= r["p_mean"] < hi]
            bg = [r for r in items_bg if r.get("p_mean") is not None and lo <= r["p_mean"] < hi]
            if not tg or not bg:
                out.append({"bin": [lo, hi], "n_target": len(tg), "n_bg": len(bg)})
                continue
            # threshold for peak at the same 5% FPR within this mean bin
            thr = float(np.percentile([r["peak"] for r in bg], 95))
            out.append({
                "bin": [lo, hi], "n_target": len(tg), "n_bg": len(bg),
                "peak_auc": auc([r["peak"] for r in tg], [r["peak"] for r in bg]),
                "peak_recall_at_5pct_bg": float(np.mean([r["peak"] >= thr for r in tg])),
                "mean_auc_in_bin": auc([r["mean"] for r in tg], [r["mean"] for r in bg]),
            })
        return out

    c_only_all = [r for r in targets if r["group"] == "c_only"]
    both_all = [r for r in targets if r["group"] == "both"]
    c_only_80 = [r for r in c_only_all if float(round(r["B"], 3)) == 80.0]
    bg_80 = [r for r in bgs if float(round(r["B"], 3)) == 80.0]
    fp_80 = [r for r in fps if float(round(r["B"], 3)) == 80.0]

    def pmean_hist(rows):
        edges = [0.0, 0.5, 0.75, 0.9, 0.97, 1.0]
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            out.append({"bin": [lo, hi],
                        "n": sum(1 for r in rows if r.get("p_mean") is not None and lo <= r["p_mean"] < hi)})
        out.append({"bin": ">=1.0", "n": sum(1 for r in rows if r.get("p_mean") is not None and r["p_mean"] >= 1.0)})
        return out

    def count_by(rows, key):
        out = {}
        for r in rows:
            k = f"{float(round(r[key], 3)):g}" if key == "B" else str(r[key])
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: float(kv[0]) if key == "B" else int(kv[0])))

    hard = [r for r in c_only_all if r.get("p_mean") is not None and r["p_mean"] < 1.0]

    def shape_of(group):
        g = [r for r in targets if r["group"] == group]
        return {"n": len(g),
                "h_med": float(np.median([r["h"] for r in g])) if g else None,
                "w_med": float(np.median([r["w"] for r in g])) if g else None}

    def cent_stats(rows):
        if not rows:
            return None
        return {
            "n": len(rows),
            "row_err_med_mhz": float(np.median([r["row_err"] for r in rows])),
            "energy_err_med_mhz": float(np.median([r["energy_err"] for r in rows])),
            "cent_err_med_mhz": float(np.median([r["cent_err"] for r in rows])),
            "row_err_p90_mhz": float(np.percentile([r["row_err"] for r in rows], 90)),
            "cent_err_p90_mhz": float(np.percentile([r["cent_err"] for r in rows], 90)),
            "cent_better_than_row": float(np.mean([r["cent_err"] < r["row_err"] for r in rows])),
            "cent_better_than_energy": float(np.mean([r["cent_err"] < r["energy_err"] for r in rows])),
        }

    fp_pass_overall = {
        f: ((sum(1 for r in fps if passes(r, f)) / len(fps)) if fps else None)
        for f in ("mean", "peak", "joint")
    }

    result = {
        "config": {"fpr": args.fpr, "bg_per_target": args.bg_per_target,
                   "n_val_samples": len(val), "examples": [str(p) for p in example_paths]},
        "summary_by_B": summary,
        "overall": {
            "n_targets": len(targets), "n_bg": len(bgs), "n_fp": len(fps),
            "groups": {g: sum(1 for r in targets if r["group"] == g)
                       for g in ("both", "c_only", "b_only", "neither")},
            "c_only_auc_mean_vs_bg": auc([r["mean"] for r in c_only_all], [r["mean"] for r in bgs]),
            "c_only_auc_peak_vs_bg": auc([r["peak"] for r in c_only_all], [r["peak"] for r in bgs]),
            "c_only_auc_joint_vs_bg": auc([r["joint"] for r in c_only_all], [r["joint"] for r in bgs]),
            "c_only_auc_peak_vs_fp": auc([r["peak"] for r in c_only_all], [r["peak"] for r in fps]),
            "both_auc_peak_vs_bg": auc([r["peak"] for r in both_all], [r["peak"] for r in bgs]),
            "c_only_pmean_eq_1": sum(1 for r in c_only_all if r.get("p_mean") is not None and r["p_mean"] >= 1.0),
            "both_pmean_eq_1": sum(1 for r in both_all if r.get("p_mean") is not None and r["p_mean"] >= 1.0),
            "fp_pass": fp_pass_overall,
        },
        "c_only_pmean_hist": pmean_hist(c_only_all),
        "both_pmean_hist": pmean_hist(both_all),
        "shape": {g: shape_of(g) for g in ("both", "c_only", "neither")},
        "centroid_overall": cent_stats(cents),
        "centroid_c_only": cent_stats([r for r in cents if r["group"] == "c_only"]),
        "hard_c_only": {
            "definition": "c_only whose region-mean does not exceed every same-shape bg block (p_mean < 1.0)",
            "n": len(hard),
            "by_B": count_by(hard, "B"),
            "by_cls": count_by(hard, "cls"),
            "peak_above_all_bg": sum(1 for r in hard if r.get("p_peak") is not None and r["p_peak"] >= 1.0),
            "peak_auc_vs_bg": auc([r["peak"] for r in hard], [r["peak"] for r in bgs]),
            "mean_auc_vs_bg": auc([r["mean"] for r in hard], [r["mean"] for r in bgs]),
            "h_med": float(np.median([r["h"] for r in hard])) if hard else None,
        },
        "mean_matched_all": mean_matched(c_only_all, bgs),
        "mean_matched_80": mean_matched(c_only_80, bg_80),
        "80MHz": {
            "n_c_only": len(c_only_80), "n_bg": len(bg_80), "n_fp": len(fp_80),
            "auc_mean": auc([r["mean"] for r in c_only_80], [r["mean"] for r in bg_80]),
            "auc_peak": auc([r["peak"] for r in c_only_80], [r["peak"] for r in bg_80]),
            "auc_joint": auc([r["joint"] for r in c_only_80], [r["joint"] for r in bg_80]),
            "auc_peak_vs_fp": auc([r["peak"] for r in c_only_80], [r["peak"] for r in fp_80]),
        },
    }

    # scatter plots
    def scatter(items_t, items_bg, items_fp, title, path):
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter([r["p_mean"] for r in items_bg], [r["p_peak"] for r in items_bg],
                   s=4, alpha=0.25, label="background", color="tab:gray")
        if items_fp:
            ax.scatter([r["p_mean"] for r in items_fp], [r["p_peak"] for r in items_fp],
                       s=6, alpha=0.35, label="B narrow FP", color="tab:orange")
        for group, color in (("both", "tab:blue"), ("c_only", "tab:red")):
            g = [r for r in items_t if r["group"] == group]
            ax.scatter([r["p_mean"] for r in g], [r["p_peak"] for r in g],
                       s=8, alpha=0.6, label=group, color=color)
        ax.set_xlabel("mean percentile (vs same-B/T background)")
        ax.set_ylabel("peak percentile (vs same-B/T background)")
        ax.set_title(title)
        ax.legend(markerscale=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)

    if c_only_all:
        scatter(targets, bgs, fps, "All B: mean vs peak percentiles", ROOT / "reports/coarse_grid_peak_scatter_all.png")
    if c_only_80:
        scatter([r for r in targets if float(round(r["B"], 3)) == 80.0], bg_80, fp_80,
                "80 MHz: mean vs peak percentiles", ROOT / "reports/coarse_grid_peak_scatter_80.png")

    out_json = ROOT / (args.output + ".json")
    out_json.write_text(json.dumps(result, indent=2))
    raw_path = ROOT / (args.output + "_raw.json")
    raw_path.write_text(json.dumps({"targets": targets, "bgs": bgs, "fps": fps, "cents": cents},
                                   separators=(",", ":")))

    # markdown
    lines = ["# Grid-level coarse mean vs intra-band peak (2048x512)", "",
             "Read-only. Same 0.7 ms STFT and the exact C1 coarse segmentation",
             "(linspace row bounds, reduceat mean; the training pipeline only adds a fixed",
             "global monotone dB scaling, so `mean` below is information-identical to the",
             "channel the model sees). The three grids are computed first (images), then GT",
             "boxes define evaluation regions restricted to each target's annotated time span.",
             "Region score = max cell value inside the annotated extent (log10 power).",
             "Background blocks are same-shape, same-B/T, empty of GT and B narrow predictions;",
             "B's own narrow false positives are the second control.",
             "Percentile calibration is per B/T against background; the equal-FP threshold is the",
             f"{100*args.fpr:g}% background false-positive budget. joint = p_mean + p_peak.", "",
             "## Overall", ""]
    o = result["overall"]
    sh = result["shape"]
    lines += [
        f"- narrow GT: {o['n_targets']} (both {o['groups']['both']}, c_only {o['groups']['c_only']}, b_only {o['groups']['b_only']}, neither {o['groups']['neither']})",
        f"- background blocks: {o['n_bg']}; B narrow FPs: {o['n_fp']}",
        f"- c_only AUC vs background: mean {fmt(o['c_only_auc_mean_vs_bg'])}, peak **{fmt(o['c_only_auc_peak_vs_bg'])}**, joint {fmt(o['c_only_auc_joint_vs_bg'])}",
        f"- c_only peak AUC vs B narrow FPs: **{fmt(o['c_only_auc_peak_vs_fp'])}**",
        f"- c_only whose region-mean beats **every** same-shape bg block (p_mean = 1.0): **{o['c_only_pmean_eq_1']}/{o['groups']['c_only']}**; both: {o['both_pmean_eq_1']}/{o['groups']['both']}",
        f"- at the 5% bg threshold, fraction of B narrow FPs passing: mean {fmt(o['fp_pass']['mean'])}, peak {fmt(o['fp_pass']['peak'])}, joint {fmt(o['fp_pass']['joint'])}",
        f"- box height (rows): both h_med {fmt(sh['both']['h_med'], 1)}, c_only h_med {fmt(sh['c_only']['h_med'], 1)}, neither h_med {fmt(sh['neither']['h_med'], 1)}; width (cols) both {fmt(sh['both']['w_med'], 1)} vs c_only {fmt(sh['c_only']['w_med'], 1)}",
        "", "## c_only mean-percentile histogram (vs same-B/T same-shape background)", "",
        "| p_mean bin | c_only n | both n |", "| --- | ---: | ---: |",
    ]
    for hc, hb in zip(result["c_only_pmean_hist"], result["both_pmean_hist"]):
        b = hc["bin"]
        bl = f"{b[0]:.2f}-{b[1]:.2f}" if isinstance(b, list) else str(b)
        lines.append(f"| {bl} | {hc['n']} | {hb['n']} |")
    hard = result["hard_c_only"]
    lines += [
        "", f"## Hard c_only (p_mean < 1.0): n = {hard['n']}", "",
        f"- by B: {hard['by_B']}; by class: {hard['by_cls']}",
        f"- peak AUC vs background for this subset: {fmt(hard['peak_auc_vs_bg'])} (mean AUC {fmt(hard['mean_auc_vs_bg'])}); peak beats every bg block for {hard['peak_above_all_bg']}/{hard['n']}",
        f"- h_med {fmt(hard['h_med'], 1)} rows",
        "", "## By observation bandwidth", "",
        "| B | narrow GT | c_only | mean AUC | peak AUC | joint AUC | recall mean | recall peak | recall joint | peak AUC vs FP | c_only h_med | centroid row→cent err (MHz) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for B, e in summary.items():
        c = e["c_only"]
        cent = e.get("centroid")
        cent_txt = "n/a" if not cent else (
            f"{fmt(cent['row_err_med_mhz'], 3)} → {fmt(cent['cent_err_med_mhz'], 3)} "
            f"({fmt(cent['cent_better_than_row'], 2)} better than row)")
        lines.append(
            f"| {B} | {e['n_gt_narrow']} | {c['n']} | {fmt(c['auc_mean_vs_bg'])} | "
            f"**{fmt(c['auc_peak_vs_bg'])}** | {fmt(c['auc_joint_vs_bg'])} | "
            f"{fmt(c['recall_mean'])} | {fmt(c['recall_peak'])} | {fmt(c['recall_joint'])} | "
            f"{fmt(c['auc_peak_vs_fp'])} | {fmt(c['h_med'], 1)} | {cent_txt} |"
        )
    lines += ["", "## B narrow FPs passing the 5% background threshold", "",
              "| B | n FP | pass mean | pass peak | pass joint |", "| --- | ---: | ---: | ---: | ---: |"]
    for B, e in summary.items():
        fp_ = e["fp_pass"]
        lines.append(f"| {B} | {e['n_fp']} | {fmt(fp_['mean'])} | {fmt(fp_['peak'])} | {fmt(fp_['joint'])} |")
    lines += ["", "## Mean-matched peak separation (all B)", "",
              "| mean percentile bin | c_only n | bg n | peak AUC | peak recall @5% bg FP |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in result["mean_matched_all"]:
        lines.append(f"| {row['bin'][0]:.2f}–{row['bin'][1]:.2f} | {row['n_target']} | {row['n_bg']} | "
                     f"{fmt(row.get('peak_auc'))} | {fmt(row.get('peak_recall_at_5pct_bg'))} |")
    lines += ["", "## 80 MHz", "",
              f"- c_only {result['80MHz']['n_c_only']}, background {result['80MHz']['n_bg']}, B narrow FP {result['80MHz']['n_fp']}",
              f"- AUC vs background: mean {fmt(result['80MHz']['auc_mean'])}, peak **{fmt(result['80MHz']['auc_peak'])}**, joint {fmt(result['80MHz']['auc_joint'])}",
              f"- peak AUC vs B narrow FP: **{fmt(result['80MHz']['auc_peak_vs_fp'])}**",
              "", "Mean-matched 80 MHz:", "",
              "| mean percentile bin | c_only n | bg n | peak AUC | peak recall @5% bg FP |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in result["mean_matched_80"]:
        lines.append(f"| {row['bin'][0]:.2f}–{row['bin'][1]:.2f} | {row['n_target']} | {row['n_bg']} | "
                     f"{fmt(row.get('peak_auc'))} | {fmt(row.get('peak_recall_at_5pct_bg'))} |")
    lines += ["", "## Centroid validation (frequency error vs coarse row centre, MHz)", ""]
    co = result["centroid_overall"]
    cc = result["centroid_c_only"]
    if co:
        lines += [
            "Using the max-energy row centre as the frequency estimate vs adding the intra-band",
            "energy centroid (energy-weighted, restricted to the target's annotated time columns):", "",
            "| set | n | row err med | energy-weighted err med | centroid err med | centroid err p90 | centroid better than row |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| all narrow | {co['n']} | {fmt(co['row_err_med_mhz'], 3)} | {fmt(co['energy_err_med_mhz'], 3)} | {fmt(co['cent_err_med_mhz'], 3)} | {fmt(co['cent_err_p90_mhz'], 3)} | {fmt(co['cent_better_than_row'], 2)} |",
        ]
        if cc:
            lines.append(
                f"| c_only | {cc['n']} | {fmt(cc['row_err_med_mhz'], 3)} | {fmt(cc['energy_err_med_mhz'], 3)} | "
                f"{fmt(cc['cent_err_med_mhz'], 3)} | {fmt(cc['cent_err_p90_mhz'], 3)} | {fmt(cc['cent_better_than_row'], 2)} |")
        lines += ["", "For reference, one coarse row spans B/2048 MHz (e.g. 0.039 MHz at 80 MHz).", ""]
    lines += ["", "## Example images", ""]
    for p in example_paths:
        lines.append(f"- `{p}`")
    lines += ["", "Per-B/T thresholds are in the JSON; raw per-record values in",
              f"`{args.output}_raw.json`.", ""]
    md = ROOT / (args.output + ".md")
    md.write_text("\n".join(lines).rstrip("\n") + "\n")
    print("\n".join(lines))
    print("wrote", out_json, md)


if __name__ == "__main__":
    main()
