#!/usr/bin/env python3
"""Evaluate one C3 checkpoint on the fixed 384-sample val subset and save
per-sample GT/predictions for matched-box analysis."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.signals.data import SPLIT_MANIFEST, load_manifest
from src.signals.model import build_model
from scripts.train_c3_subset import C3Dataset, run_validation

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--frontend", required=True,
                    choices=["power", "zero3", "overlap", "sparse", "dense"])
    ap.add_argument("--complex-dir", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run_dir = ROOT / "runs" / args.run
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = run_dir / ckpt_path
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    channels = 1 if args.frontend == "power" else 3
    model = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"),
                        device="cuda", ch=channels)
    sd = ck["ema"] if ck.get("ema") is not None else ck["model"]
    model.load_state_dict(sd)
    model.eval()

    subset = json.loads((ROOT / "cache/c3_subset.json").read_text())
    val_all = load_manifest(SPLIT_MANIFEST, "val")
    by_val = {r.sample_id: r for r in val_all}
    val_ids = subset["val_ids"]
    val_recs = [by_val[s] for s in val_ids]
    complex_dir = Path(args.complex_dir) if args.complex_dir else None
    if complex_dir and not complex_dir.is_absolute():
        complex_dir = ROOT / complex_dir
    val_complex = complex_dir / "val_complex.npy" if complex_dir else None
    ds = C3Dataset("val", val_ids, by_val,
                   ROOT / "cache/c3_subset_val_power_bf16.npy",
                   val_complex, channels,
                   zero_complex=(args.frontend == "zero3"))
    out, gts, preds = run_validation(model, ds, val_recs, channels,
                                     batch_sources=8, return_preds=True)
    out["checkpoint"] = str(ckpt_path)
    out["run"] = args.run
    out["frontend"] = args.frontend
    out["channels"] = channels
    out["best_fitness"] = ck.get("best_fitness")
    out["epoch"] = ck.get("epoch")

    if args.out:
        stem = Path(args.out)
        if not stem.is_absolute():
            stem = ROOT / stem
    else:
        stem = run_dir / f"eval_{Path(args.checkpoint).stem}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    npz = {}
    for name, arrs in [("pred", preds), ("gt", gts)]:
        boxes = np.concatenate([p.boxes.reshape(-1, 4) for p in arrs]) if arrs else np.zeros((0, 4))
        scores = np.concatenate([p.scores.reshape(-1) for p in arrs]) if arrs else np.zeros((0,))
        labels = np.concatenate([p.labels.reshape(-1) for p in arrs]) if arrs else np.zeros((0,))
        idx = np.concatenate([np.full(len(p.boxes), i, dtype=np.int64) for i, p in enumerate(arrs)])
        npz[f"{name}_boxes"] = boxes.astype(np.float32)
        npz[f"{name}_scores"] = scores.astype(np.float32)
        npz[f"{name}_labels"] = labels.astype(np.int64)
        npz[f"{name}_sample_index"] = idx
    np.savez_compressed(stem.with_suffix(".npz"), **npz)
    stem.with_suffix(".json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ["run", "frontend", "checkpoint", "mAP50",
                                         "map50_agnostic", "localization_recall",
                                         "classification_accuracy", "n_pred"]}, indent=2))
    print("wrote", stem.with_suffix(".npz"))


if __name__ == "__main__":
    main()
