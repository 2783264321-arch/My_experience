#!/usr/bin/env python3
"""Reconstruct A from a fixed-candidate cache and verify the original AP."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/detector_upgrade"))
from src.signals.data import N_TIME, SPLIT_MANIFEST, load_manifest
from src.signals.infer import Prediction, global_classwise_nms, gt_global
from src.signals.metrics import ImagePredictions
from validate_dual import _eval_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path)
    ap.add_argument("--split", choices=("fit", "val"), default="val")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    z = np.load(args.cache)
    meta = json.loads(args.cache.with_suffix(".json").read_text())
    n = int(meta["samples"])
    records = load_manifest(SPLIT_MANIFEST, split=args.split)[:n]
    gts = [gt_global(r, math.ceil(r.N / (N_TIME - 1))) for r in records]
    offsets = z["sample_offsets"]
    preds = []
    for i in range(n):
        a, b = int(offsets[i]), int(offsets[i + 1])
        pred = Prediction(torch.from_numpy(z["boxes"][a:b]).float(),
                          torch.from_numpy(z["scores"][a:b]).float(),
                          torch.from_numpy(z["labels"][a:b]).long())
        final = global_classwise_nms(pred, iou=0.5, max_det=100)
        preds.append(ImagePredictions(final.boxes.numpy(), final.scores.numpy(),
                                      final.labels.numpy()))
    res = _eval_set(gts, preds)
    out = {"split": args.split, "samples": n, "candidates": int(len(z["scores"])),
           "n_pred": int(sum(len(p.boxes) for p in preds)),
           "map50": res["map50"], "map50_95": res["map50_95"],
           "localization_recall": res["localization_recall"],
           "per_class": res["per_class"], "groups": res["groups"]}
    print(json.dumps({k: out[k] for k in ("split", "samples", "candidates", "n_pred",
                                          "map50", "map50_95",
                                          "localization_recall")}, indent=2))
    if args.out:
        args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
