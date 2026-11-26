#!/usr/bin/env python3
"""Compare SAHI-style post-processing on the fixed e25 candidate pool.

No model inference: uses the verified 742,414-candidate val cache. A is the
current class-wise NMS 0.5 + global top100. Variants test whether hard NMS /
the global cap is a merge short-board.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import nms as tv_nms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/detector_upgrade"))
from src.signals.data import N_TIME, SPLIT_MANIFEST, load_manifest
from src.signals.infer import Prediction, global_classwise_nms, gt_global
from src.signals.metrics import ImagePredictions
from validate_dual import _eval_set

CACHE = Path("/root/autodl-fs/runs_local/dfine_local_iq/val_candidates.npz")


def iou_np(a, b):
    # a [N,4], b [4]; normalized xyxy
    x1 = np.maximum(a[:, 0], b[0]); y1 = np.maximum(a[:, 1], b[1])
    x2 = np.minimum(a[:, 2], b[2]); y2 = np.minimum(a[:, 3], b[3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    aa = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / np.maximum(aa + bb - inter, 1e-12)


def nms_class_torch(boxes, scores, labels, iou, max_det):
    if len(boxes) == 0:
        return boxes, scores, labels
    keep = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        k = tv_nms(torch.from_numpy(boxes[idx]).float(),
                   torch.from_numpy(scores[idx]).float(), iou).numpy()
        keep.extend(idx[k].tolist())
    keep = np.asarray(keep, dtype=np.int64)
    order = np.argsort(-scores[keep])[:max_det]
    keep = keep[order]
    return boxes[keep], scores[keep], labels[keep]


def soft_nms(boxes, scores, labels, iou_thr=0.5, sigma=0.5,
             method="gaussian", score_thr=1e-3, max_det=100):
    kb, ks, kl = [], [], []
    for cls in np.unique(labels):
        b = boxes[labels == cls].copy()
        s = scores[labels == cls].copy()
        while len(b):
            j = int(np.argmax(s))
            kb.append(b[j]); ks.append(float(s[j])); kl.append(int(cls))
            bj = b[j].copy(); sj = float(s[j])
            b = np.delete(b, j, axis=0); s = np.delete(s, j)
            if not len(b):
                break
            ov = iou_np(b, bj)
            if method == "linear":
                s = s * np.where(ov > iou_thr, 1.0 - ov, 1.0)
            else:
                s = s * np.exp(-(ov ** 2) / max(sigma, 1e-6))
            m = s >= score_thr
            b, s = b[m], s[m]
    kb = np.asarray(kb, dtype=np.float32).reshape(-1, 4)
    ks = np.asarray(ks, dtype=np.float32)
    kl = np.asarray(kl, dtype=np.int64)
    order = np.argsort(-ks)[:max_det]
    return kb[order], ks[order], kl[order]


def wbf(boxes, scores, labels, iou_thr=0.55, max_det=100):
    kb, ks, kl = [], [], []
    for cls in np.unique(labels):
        b = boxes[labels == cls].copy()
        s = scores[labels == cls].copy()
        order = np.argsort(-s)
        clusters = []  # each [box, score_sum, weight_sum, n]
        for i in order:
            bi, si = b[i], float(s[i])
            best, best_iou = -1, iou_thr
            for c, cl in enumerate(clusters):
                ov = float(iou_np(bi[None], cl[0])[0])
                if ov > best_iou:
                    best, best_iou = c, ov
            if best >= 0:
                cl = clusters[best]
                w = cl[2] + si
                cl[0] = (cl[0] * cl[2] + bi * si) / max(w, 1e-9)
                cl[1] += si; cl[2] = w; cl[3] += 1
            else:
                clusters.append([bi.copy(), si, si, 1])
        for cl in clusters:
            kb.append(cl[0]); ks.append(cl[1] / max(cl[3], 1)); kl.append(int(cls))
    kb = np.asarray(kb, dtype=np.float32).reshape(-1, 4)
    ks = np.asarray(ks, dtype=np.float32)
    kl = np.asarray(kl, dtype=np.int64)
    order = np.argsort(-ks)[:max_det]
    return kb[order], ks[order], kl[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with np.load(args.cache, allow_pickle=True) as zf:
        c = {k: zf[k] for k in zf.files}
    meta = json.loads(args.cache.with_suffix(".json").read_text())
    n = int(meta["samples"])
    records = load_manifest(SPLIT_MANIFEST, split="val")[:n]
    gts = [gt_global(r, math.ceil(r.N / (N_TIME - 1))) for r in records]
    offsets = c["sample_offsets"]

    def preds_for(fn):
        out = []
        for i in range(n):
            a, b = int(offsets[i]), int(offsets[i + 1])
            boxes = c["boxes"][a:b].astype(np.float32)
            scores = c["scores"][a:b].astype(np.float32)
            labels = c["labels"][a:b].astype(np.int64)
            bb, ss, ll = fn(boxes, scores, labels)
            out.append(ImagePredictions(bb, ss, ll))
        return out

    variants = {
        "A_nms050": lambda b, s, l: nms_class_torch(b, s, l, 0.5, 100),
        "nms060": lambda b, s, l: nms_class_torch(b, s, l, 0.6, 100),
        "nms070": lambda b, s, l: nms_class_torch(b, s, l, 0.7, 100),
        "soft_linear": lambda b, s, l: soft_nms(b, s, l, 0.5, 0.5, "linear"),
        "soft_gauss": lambda b, s, l: soft_nms(b, s, l, 0.5, 0.5, "gaussian"),
        "wbf055": lambda b, s, l: wbf(b, s, l, 0.55),
        "wbf060": lambda b, s, l: wbf(b, s, l, 0.60),
    }
    results = {}
    for name, fn in variants.items():
        preds = preds_for(fn)
        r = _eval_set(gts, preds)
        r["n_pred"] = int(sum(len(p.boxes) for p in preds))
        results[name] = r
        print(name, f"{r['map50']:.6f}", f"{r['map50_95']:.6f}",
              "n", r["n_pred"], flush=True)
    args.out.write_text(json.dumps({"cache": str(args.cache), "results": results}, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
