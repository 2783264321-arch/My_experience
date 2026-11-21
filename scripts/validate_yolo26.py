#!/usr/bin/env python3
"""Formal YOLO26l validation entry: rebuild from checkpoint, one-to-many inference,
dual protocol (unfiltered + fine-0-5 filtered), mAP50/50:95/groups, npz artifacts."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/detector_upgrade"))
from adapters import build_yolo26
from src.signals.data import SPLIT_MANIFEST, load_manifest
from validate_dual import run_dual

def stratified(records, n_per_b):
    by = {}
    for r in records:
        by.setdefault(round(float(r.B_mhz)), []).append(r)
    out = []
    for b in sorted(by):
        out += by[b][:n_per_b]
    return out

def save_npz(path, preds):
    np.savez_compressed(path,
        pred_boxes=np.concatenate([p.boxes.reshape(-1, 4) for p in preds]).astype(np.float32),
        pred_scores=np.concatenate([p.scores.reshape(-1) for p in preds]).astype(np.float32),
        pred_labels=np.concatenate([p.labels.reshape(-1) for p in preds]).astype(np.int64),
        pred_sample_index=np.concatenate([np.full(len(p.boxes), i, dtype=np.int64) for i, p in enumerate(preds)]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="yolo26l_c1_2048x512")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--weights", choices=["ema", "model"], default="ema")
    ap.add_argument("--stratified", type=int, default=0, help="N per bandwidth; 0 = full val")
    ap.add_argument("--batch-views", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    run_dir = ROOT / "runs" / args.run
    ck_path = Path(args.checkpoint)
    if not ck_path.is_absolute(): ck_path = run_dir / ck_path
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    sd = ck["ema"] if args.weights == "ema" and ck.get("ema") is not None else ck["model"]
    model, info = build_yolo26()
    model.criterion = model.init_criterion()
    model.load_state_dict(sd)
    model = model.cuda().eval()
    model.model[-1].end2end = False
    assert tuple(model.model[0].conv.weight.shape) == (64, 1, 3, 3)
    assert int(model.model[-1].nc) == 14

    norm_d = json.loads((ROOT / "configs/normalizer.json").read_text()); norm = (norm_d["lo"], norm_d["hi"])
    records = load_manifest(SPLIT_MANIFEST, split="val")
    if args.stratified:
        records = stratified(records, args.stratified)
    out_dir = Path(args.output) if args.output else run_dir / ("val_subset" if args.stratified else "val_full")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    res = run_dual(model, records, norm, batch_views=args.batch_views, workers=args.workers)
    elapsed = time.perf_counter() - t0
    for name in ("unfiltered", "filtered_0_5"):
        save_npz(out_dir / f"val_predictions_{name}.npz", res[f"preds_{name}"])
    (out_dir / "val_sample_ids.json").write_text(json.dumps([r.sample_id for r in records], indent=2))
    summary = {"run": args.run, "checkpoint": str(ck_path), "weights": args.weights,
               "split": "val", "n_samples": len(records), "n_views": res["n_views"],
               "n_boundary_dropped": res["n_boundary_dropped"],
               "conf": 0.001, "iou": 0.5, "max_det": 100, "max_det_view": 300,
               "settings": {k: v for k, v in res["settings"].items()},
               "elapsed_s": elapsed, "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
               "model": info, "checkpoint_epoch": ck.get("epoch"),
               "checkpoint_best_fitness": ck.get("best_fitness")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    f = summary["settings"]["filtered_0_5"]; u = summary["settings"]["unfiltered"]
    print(json.dumps({"samples": len(records), "views": res["n_views"],
                      "filtered_map50": f["map50"], "filtered_map50_95": f["map50_95"],
                      "unfiltered_map50": u["map50"], "unfiltered_map50_95": u["map50_95"],
                      "filtered_recall": f["localization_recall"],
                      "elapsed_s": elapsed}, ensure_ascii=False))

if __name__ == "__main__":
    main()
