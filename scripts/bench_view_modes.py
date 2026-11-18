#!/usr/bin/env python3
"""Latency benchmark: Group A (coarse-only) vs C (coarse + all fine views).

Both modes use the same stratified val samples, the same per-view NMS and
merge parameters. Build time (STFT + view construction) is measured once per
sample/mode; model time (forward + NMS + merge) is the median of `--repeats`.
End-to-end = build + model median.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/bench_view_modes.py \
    --run-a yolo11s_c1_directreg_coarseonly_2048x512 \
    --checkpoint-a last.pt \
    --run-c yolo11s_c1_directreg_2048x512 \
    --checkpoint-c /path/to/best_corrected.pt \
    --samples 128 --repeats 3 --output reports/group_a_latency.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.signals.data import (
    SPLIT_MANIFEST, build_sample_views, frequency_starts, load_manifest,
)
from src.signals.infer import predict_views
from src.signals.model import build_model

ROOT = Path(__file__).resolve().parents[1]


def load_model(config_path: Path, checkpoint: Path):
    model = build_model(str(config_path), device="cuda")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["ema"] if ckpt.get("ema") is not None else ckpt["model"]
    model.load_state_dict(sd)
    model.eval()
    return model


def pick_stratified(records, n: int, seed: int):
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[(round(r.B_mhz, 4), round(r.T_ms, 4))].append(i)
    rng = random.Random(seed)
    keys = sorted(groups)
    for k in keys:
        rng.shuffle(groups[k])
    picks = []
    while len(picks) < min(n, len(records)):
        progressed = False
        for k in keys:
            if groups[k] and len(picks) < n:
                picks.append(groups[k].pop())
                progressed = True
        if not progressed:
            break
    return picks


def summarize(values):
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def timed_model(model, views, fbins: int, repeats: int, batch_views: int):
    times = []
    pred = None
    stats = None
    # one warmup per mode/sample is too expensive; the caller warms up on the
    # first two samples, so here we just repeat the timed run.
    for _ in range(repeats):
        torch.cuda.synchronize()
        t = time.perf_counter()
        pred, stats = predict_views(
            model, views, fbins, batch_views=batch_views,
            conf=0.001, iou=0.5, max_det=100, max_det_view=300,
        )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return statistics.median(times), times, len(pred.boxes), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--checkpoint-a", default="last.pt")
    ap.add_argument("--run-c", required=True)
    ap.add_argument("--checkpoint-c", required=True)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch-views", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--output", default="reports/group_a_latency.json")
    args = ap.parse_args()

    def resolve(run: str, ckpt: str) -> tuple[Path, Path]:
        run_dir = ROOT / "runs" / run
        ck = Path(ckpt)
        if not ck.is_absolute():
            ck = run_dir / ck
        return run_dir, ck

    run_a, ckpt_a = resolve(args.run_a, args.checkpoint_a)
    run_c, ckpt_c = resolve(args.run_c, args.checkpoint_c)
    for p in (ckpt_a, ckpt_c):
        if not p.exists():
            raise SystemExit(f"checkpoint not found: {p}")

    norm_cfg = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm_cfg["lo"]), float(norm_cfg["hi"]))
    records = load_manifest(SPLIT_MANIFEST, split="val")
    picks = pick_stratified(records, args.samples, args.seed)

    model_a = load_model(ROOT / "configs/yolo11s_c1_directreg.yaml", ckpt_a)
    model_c = load_model(ROOT / "configs/yolo11s_c1_directreg.yaml", ckpt_c)
    torch.backends.cudnn.benchmark = False

    # Warm up both models on the first two selected samples.
    for idx in picks[:2]:
        r = records[idx]
        fbins = int(round(r.B_mhz * 700))
        coarse, _, _ = build_sample_views(r, norm, [])
        predict_views(model_a, [coarse], fbins, batch_views=args.batch_views)
        starts = frequency_starts(fbins)
        coarse, fines, _ = build_sample_views(r, norm, list(range(len(starts))))
        predict_views(model_c, [coarse] + fines, fbins, batch_views=args.batch_views)
    torch.cuda.empty_cache()

    rows = []
    for n, idx in enumerate(picks):
        r = records[idx]
        fbins = int(round(r.B_mhz * 700))
        starts = frequency_starts(fbins)

        t = time.perf_counter()
        coarse_a, _, _ = build_sample_views(r, norm, [])
        build_a = time.perf_counter() - t
        model_a_s, model_a_all, n_pred_a, stats_a = timed_model(
            model_a, [coarse_a], fbins, args.repeats, args.batch_views,
        )

        t = time.perf_counter()
        coarse_c, fines_c, _ = build_sample_views(r, norm, list(range(len(starts))))
        build_c = time.perf_counter() - t
        views_c = [coarse_c] + fines_c
        model_c_s, model_c_all, n_pred_c, stats_c = timed_model(
            model_c, views_c, fbins, args.repeats, args.batch_views,
        )

        rows.append({
            "sample_id": r.sample_id,
            "B_mhz": r.B_mhz,
            "T_ms": r.T_ms,
            "views_a": 1,
            "views_c": len(views_c),
            "build_a_s": build_a,
            "build_c_s": build_c,
            "model_a_s": model_a_s,
            "model_c_s": model_c_s,
            "total_a_s": build_a + model_a_s,
            "total_c_s": build_c + model_c_s,
            "model_a_all": model_a_all,
            "model_c_all": model_c_all,
            "n_pred_a": n_pred_a,
            "n_pred_c": n_pred_c,
            "stats_a": stats_a,
            "stats_c": stats_c,
        })
        if (n + 1) % 10 == 0 or n + 1 == len(picks):
            print(f"  {n + 1}/{len(picks)} samples", flush=True)

    def mode_summary(key_build, key_model, key_total, key_views):
        return {
            "views_per_sample": summarize([x[key_views] for x in rows]),
            "build_s": summarize([x[key_build] for x in rows]),
            "model_s": summarize([x[key_model] for x in rows]),
            "total_s": summarize([x[key_total] for x in rows]),
        }

    by_b = {}
    for B in sorted({x["B_mhz"] for x in rows}):
        rs = [x for x in rows if x["B_mhz"] == B]
        by_b[f"{B:g}MHz"] = {
            "n": len(rs),
            "views_a": summarize([x["views_a"] for x in rs]),
            "views_c": summarize([x["views_c"] for x in rs]),
            "total_a_s": summarize([x["total_a_s"] for x in rs]),
            "total_c_s": summarize([x["total_c_s"] for x in rs]),
            "model_a_s": summarize([x["model_a_s"] for x in rs]),
            "model_c_s": summarize([x["model_c_s"] for x in rs]),
        }

    summary = {
        "A_coarse": mode_summary("build_a_s", "model_a_s", "total_a_s", "views_a"),
        "C_all": mode_summary("build_c_s", "model_c_s", "total_c_s", "views_c"),
    }
    total_a_p50 = summary["A_coarse"]["total_s"]["p50"]
    total_c_p50 = summary["C_all"]["total_s"]["p50"]
    model_a_p50 = summary["A_coarse"]["model_s"]["p50"]
    model_c_p50 = summary["C_all"]["model_s"]["p50"]
    result = {
        "run_a": args.run_a,
        "checkpoint_a": str(ckpt_a),
        "run_c": args.run_c,
        "checkpoint_c": str(ckpt_c),
        "n_samples": len(rows),
        "repeats": args.repeats,
        "batch_views": args.batch_views,
        "conf": 0.001, "iou": 0.5, "max_det": 100, "max_det_view": 300,
        "summary": summary,
        "by_B": by_b,
        "ratios": {
            "views_c_over_a": (
                summary["C_all"]["views_per_sample"]["mean"]
                / summary["A_coarse"]["views_per_sample"]["mean"]
            ),
            "total_p50_c_over_a": total_c_p50 / max(total_a_p50, 1e-9),
            "model_p50_c_over_a": model_c_p50 / max(model_a_p50, 1e-9),
        },
        "samples": rows,
    }
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "n_samples": len(rows),
        "summary": summary,
        "ratios": result["ratios"],
        "by_B": by_b,
    }, indent=2), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
