#!/usr/bin/env python3
"""Stage-5 inference speed benchmark: full-sample STFT + all views + NMS."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.signals.data import SPLIT_MANIFEST, build_sample_views, load_manifest, frequency_starts
from src.signals.infer import gt_global, parse_class_set, predict_sample
from src.signals.model import build_model

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="yolo11s_c1_directreg_2048x512")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--split", choices=["fit", "val"], default="val")
    ap.add_argument("--ids", default="")
    ap.add_argument("--batch-views", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--drop-fine-classes", default="")
    args = ap.parse_args()
    run = ROOT / "runs" / args.run
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = run / ckpt_path
    norm_cfg = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm_cfg["lo"]), float(norm_cfg["hi"]))
    records = load_manifest(SPLIT_MANIFEST, split=args.split)
    by_id = {r.sample_id: r for r in records}
    if args.ids:
        picks = [by_id[s.strip()] for s in args.ids.split(",") if s.strip()]
    else:
        picks = [
            next(r for r in records if r.B_mhz == 5.0 and abs(r.T_ms - 20.0) < 1e-6),
            next(r for r in records if r.B_mhz == 40.0 and abs(r.T_ms - 80.0) < 1e-6),
            next(r for r in records if r.B_mhz == 80.0 and abs(r.T_ms - 150.0) < 1e-6),
        ]
    model = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["ema"] if ckpt.get("ema") is not None else ckpt["model"])
    model.eval()

    out = []
    for r in picks:
        views = len(frequency_starts(int(round(r.B_mhz * 700)))) + 1
        # warm
        predict_sample(model, r, norm, batch_views=args.batch_views, conf=args.conf,
                       drop_fine_classes=parse_class_set(args.drop_fine_classes))
        times = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            t = time.perf_counter()
            pred, stats = predict_sample(
                model, r, norm, batch_views=args.batch_views, conf=args.conf,
                drop_fine_classes=parse_class_set(args.drop_fine_classes),
            )
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        t = time.perf_counter()
        build_sample_views(r, norm, list(range(views - 1)))
        stft_s = time.perf_counter() - t
        gt = gt_global(r, (r.N + 510) // 511)
        out.append({
            "sample_id": r.sample_id, "B": r.B_mhz, "T": r.T_ms,
            "views": views, "n_gt": len(gt.boxes), "n_pred": len(pred.boxes),
            "total_s_median": sorted(times)[len(times) // 2],
            "total_s_all": times,
            "stft_plus_views_s": stft_s,
            "stats": stats,
        })
        print(out[-1], flush=True)
    result = {
        "checkpoint": str(ckpt_path),
        "batch_views": args.batch_views,
        "peak_alloc_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
        "samples": out,
    }
    path = run / "stage5_inference.json"
    path.write_text(json.dumps(result, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
