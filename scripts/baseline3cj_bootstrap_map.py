#!/usr/bin/env python3
"""Paired IQ-file cluster bootstrap for final-dev mAP differences.

Compares, on the same 140 final-dev IQ files:
  E0 - C0          (frozen W-Shared minus YOLO labels)
  E2 - E0          (mixed fine-tune minus frozen W-Shared)
  E2 - C0          (kept separate; never reuse the E2-E0 interval)

E0 and E2 are the mean of the three seed mAPs, matching the reported
tables. The bootstrap resamples IQ files with replacement (2,000 draws),
keeping all predictions/targets of a file together. mAP is class-aware
at IoU 0.5, 14 classes, 101-point AP, max_dets=100.

The per-image TP/FP flags are precomputed once; each bootstrap draw only
re-aggregates them. This is exact for the greedy per-image matching and
makes 2,000 paired draws practical.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.baseline3b_train_experts import HistCNN
from scripts.baseline3cb_e0_eval import predict
from scripts.baseline3ce_evaluate import load_all
from src.signals.metrics import (
    ImagePredictions, ImageTargets, _match_single, evaluate,
)

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports/baseline3_wifi_cascade"
SEEDS = (1729, 1730, 1731)
N_BOOT = 2000
SHARED_MAP = {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2}


def ap101_from_flags(tp_sorted: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return 0.0
    if len(tp_sorted) == 0:
        return 0.0
    tp = tp_sorted.astype(np.float64)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1.0 - tp)
    rec = tp_cum / n_gt
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    prec_suf = np.maximum.accumulate(prec[::-1])[::-1]
    ts = np.linspace(0.0, 1.0, 101)
    idx = np.searchsorted(rec, ts, side="left")
    vals = np.where(idx < len(rec), prec_suf[np.minimum(idx, len(rec) - 1)], 0.0)
    return float(vals.mean())


def precompute(images, targets):
    """Per-class global entries: img, score, pred_idx, tp flag, n_gt."""
    n_img = len(images)
    n_gt = np.zeros((n_img, 14), np.int64)
    for i, t in enumerate(targets):
        for c in t.labels:
            n_gt[i, int(c)] += 1
    entries = {}
    for c in range(14):
        e_img, e_score, e_pi, e_tp = [], [], [], []
        for i, (pred, gt) in enumerate(zip(images, targets)):
            pidx = np.where(pred.labels == c)[0]
            if len(pidx) == 0:
                continue
            matches, _ = _match_single(
                gt, pred, 0.5, class_aware=True, class_id=c, max_dets=100)
            matched = {m[0] for m in matches}
            for pi in pidx:
                e_img.append(i)
                e_score.append(float(pred.scores[pi]))
                e_pi.append(int(pi))
                e_tp.append(int(pi) in matched)
        entries[c] = {
            "img": np.asarray(e_img, np.int64),
            "score": np.asarray(e_score, np.float64),
            "pi": np.asarray(e_pi, np.int64),
            "tp": np.asarray(e_tp, bool),
            "n_gt": n_gt[:, c],
        }
    return entries


def map_from_entries(entries, counts):
    """mAP for one bootstrap draw; counts = multiplicity per image."""
    aps = []
    for c in range(14):
        e = entries[c]
        ng = int(np.dot(counts, e["n_gt"]))
        if ng == 0:
            continue
        mult = counts[e["img"]]
        mask = mult > 0
        if not mask.any():
            aps.append(0.0)
            continue
        img = np.repeat(e["img"][mask], mult[mask])
        sc = np.repeat(e["score"][mask], mult[mask])
        pi = np.repeat(e["pi"][mask], mult[mask])
        tp = np.repeat(e["tp"][mask], mult[mask])
        order = np.lexsort((pi, img, -sc))
        aps.append(ap101_from_flags(tp[order], ng))
    return float(np.mean(aps)) if aps else 0.0


def build_model_images(model, hz, yz, c0_off, gt_off, final_si, device):
    sample_ids = [str(s) for s in yz["sample_ids"]]
    c0_boxes = yz["c0_boxes"]; c0_scores = yz["c0_scores"]
    c0_labels = yz["c0_labels"]; gt_boxes = yz["gt_boxes"]
    gt_labels = yz["gt_labels"]
    h_pred_pos = {int(pi): k for k, pi in enumerate(hz["pred_idx"])}
    pred_status = hz["pred_status"]; pred_off = hz["pred_offsets"]
    label_map = {}
    if model is not None:
        for pi, pos in h_pred_pos.items():
            if pred_status[pos] != "ok":
                continue
            b = hz["pred_blocks"][pred_off[pos]:pred_off[pos + 1]]
            p = int(np.argmax(predict(model, b, device)))
            orig = int(c0_labels[pi])
            label_map[pi] = p if orig <= 2 else p + 3
    preds, targets = [], []
    used = set()
    for si in final_si:
        lo, hi = int(c0_off[si]), int(c0_off[si + 1])
        l = c0_labels[lo:hi].copy()
        for pi in range(lo, hi):
            if pi in label_map:
                l[pi - lo] = label_map[pi]
                used.add(pi)
        preds.append(ImagePredictions(c0_boxes[lo:hi], c0_scores[lo:hi], l))
        glo, ghi = int(gt_off[si]), int(gt_off[si + 1])
        targets.append(ImageTargets(gt_boxes[glo:ghi], gt_labels[glo:ghi]))
    return preds, targets, len(label_map)


def point_map(preds, targets):
    ev = evaluate(targets, preds, iou_thresholds=(0.5,), max_dets=100)
    return float(ev["per_iou"]["0.5"]["map_class_aware"])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    hz, yz, pairs, split = load_all()
    assign = split["assignment"]
    final_sids = sorted({s for s, v in assign.items() if v == "final-dev"})
    sample_ids = [str(s) for s in yz["sample_ids"]]
    si_of = {s: i for i, s in enumerate(sample_ids)}
    final_si = [si_of[s] for s in final_sids]
    c0_off = np.concatenate([[0], np.cumsum(yz["c0_counts"])])
    gt_off = np.concatenate([[0], np.cumsum(yz["gt_counts"])])
    print(f"final-dev IQ files: {len(final_si)}")

    models = {}
    # C0 = unchanged YOLO labels
    c0_preds, targets, _ = build_model_images(
        None, hz, yz, c0_off, gt_off, final_si, device)
    models["C0"] = (c0_preds, targets, 0)
    for arm in ("E0", "E2"):
        for s in SEEDS:
            m = HistCNN(3).to(device)
            path = (ROOT / f"runs_local/baseline3b/weights/SHARED_seed{s}.pt"
                    if arm == "E0" else
                    ROOT / f"runs_local/baseline3ca/weights/{arm}_seed{s}.pt")
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            p, t, n = build_model_images(
                m, hz, yz, c0_off, gt_off, final_si, device)
            models[f"{arm}_{s}"] = (p, t, n)
            print(f"{arm} seed{s}: relabeled={n}")

    # point estimates and fast-exact validation
    points = {}
    entries = {}
    for name, (preds, tgts, nrel) in models.items():
        fast = map_from_entries(precompute(preds, tgts),
                                np.ones(len(preds), np.int64))
        ref = point_map(preds, tgts)
        points[name] = {"map50_fast": fast, "map50_ref": ref,
                        "n_relabeled_sync_ok": nrel}
        entries[name] = precompute(preds, tgts)
        print(f"{name}: fast={fast:.9f} ref={ref:.9f} "
              f"diff={fast-ref:+.2e}")

    e0 = np.mean([points[f"E0_{s}"]["map50_ref"] for s in SEEDS])
    e2 = np.mean([points[f"E2_{s}"]["map50_ref"] for s in SEEDS])
    c0 = points["C0"]["map50_ref"]
    print(f"point: C0={c0:.9f} E0mean={e0:.9f} E2mean={e2:.9f}")
    print(f"point deltas: E0-C0={100*(e0-c0):+.4f}pp "
          f"E2-E0={100*(e2-e0):+.4f}pp E2-C0={100*(e2-c0):+.4f}pp")

    rng = np.random.default_rng(20261121)
    n = len(final_si)
    diffs = defaultdict(list)
    maps = defaultdict(list)
    for b in range(N_BOOT):
        pick = rng.integers(0, n, size=n)
        counts = np.bincount(pick, minlength=n)
        for name in models:
            maps[name].append(map_from_entries(entries[name], counts))
        c0b = maps["C0"][-1]
        e0b = np.mean([maps[f"E0_{s}"][-1] for s in SEEDS])
        e2b = np.mean([maps[f"E2_{s}"][-1] for s in SEEDS])
        diffs["E0-C0"].append(100 * (e0b - c0b))
        diffs["E2-E0"].append(100 * (e2b - e0b))
        diffs["E2-C0"].append(100 * (e2b - c0b))
        for s in SEEDS:
            diffs[f"E0_{s}-C0"].append(
                100 * (maps[f"E0_{s}"][-1] - c0b))
            diffs[f"E2_{s}-E0_{s}"].append(
                100 * (maps[f"E2_{s}"][-1] - maps[f"E0_{s}"][-1]))
        if (b + 1) % 250 == 0:
            print(f"  bootstrap {b+1}/{N_BOOT} "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    def ci(v):
        v = np.asarray(v)
        return {"mean": float(v.mean()), "lo": float(np.quantile(v, .025)),
                "hi": float(np.quantile(v, .975)), "sd": float(v.std())}

    out = {
        "n_iq_files": n,
        "n_boot": N_BOOT,
        "bootstrap_seed": 20261121,
        "point_estimates": {
            "C0": c0, "E0_mean_of_seeds": e0, "E2_mean_of_seeds": e2,
            "E0-C0_pp": 100 * (e0 - c0),
            "E2-E0_pp": 100 * (e2 - e0),
            "E2-C0_pp": 100 * (e2 - c0),
            "per_seed": {k: v["map50_ref"] for k, v in points.items()},
        },
        "paired_bootstrap_pp": {k: ci(v) for k, v in diffs.items()},
        "note": ("E0/E2 are the mean of the three seed mAPs, matching the "
                 "reported tables. E2-C0 and E2-E0 use the same draws but "
                 "are separate estimands. IQ files resampled with "
                 "replacement; all predictions of a file stay together."),
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    (EXP / "paired_bootstrap_map.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["paired_bootstrap_pp"], indent=2))
    print(f"wrote paired_bootstrap_map.json ({out['runtime_seconds']}s)")


if __name__ == "__main__":
    main()
