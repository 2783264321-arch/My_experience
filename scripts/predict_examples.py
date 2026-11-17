#!/usr/bin/env python3
"""Qualitative val examples: full-sample predictions over the coarse view."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from src.signals.data import (
    COARSE_H, N_TIME, SPLIT_MANIFEST, build_sample_views, load_manifest,
)
from src.signals.infer import gt_global, predict_sample
from src.signals.model import build_model

ROOT = Path(__file__).resolve().parents[1]
COLORS = plt.cm.tab20(np.linspace(0, 1, 20))


def pick_examples(records, limit=10):
    picks = []
    used = set()

    def add(r):
        if r.sample_id not in used and len(picks) < limit:
            used.add(r.sample_id)
            picks.append(r)

    # one per bandwidth, favoring many classes
    for B in [5.0, 20.0, 30.0, 40.0, 50.0, 80.0]:
        pool = [r for r in records if r.B_mhz == B]
        pool.sort(key=lambda r: (-len({int(s["class"]) for s in r.signals}), len(r.signals)))
        if pool:
            add(pool[0])
    # ensure class 12 and classes 3/4/5 appear
    for cls in [12, 3, 4, 5, 0, 2]:
        pool = [r for r in records if any(int(s["class"]) == cls for s in r.signals)]
        pool.sort(key=lambda r: (len(r.signals), abs(r.T_ms - 60)))
        for r in pool:
            if r.sample_id not in used:
                add(r)
                break
    return picks


def draw(record, tensor, gt, pred, out_path: Path):
    img = tensor[0].numpy()
    # downsample time by 2 for a 1024x512 figure
    img = img[:, ::2]
    fig, ax = plt.subplots(figsize=(12, 8), dpi=120)
    ax.imshow(img, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    for b, lab in zip(gt.boxes.numpy(), gt.labels.numpy()):
        x1, y1, x2, y2 = b
        ax.add_patch(Rectangle((x1 * N_TIME / 2, y1 * COARSE_H),
                               (x2 - x1) * N_TIME / 2, (y2 - y1) * COARSE_H,
                               fill=False, edgecolor="lime", lw=1.2, alpha=0.9))
        ax.text(x1 * N_TIME / 2, y1 * COARSE_H, f"gt{int(lab)}", color="lime",
                fontsize=6, va="bottom")
    for b, sc, lab in zip(pred.boxes.numpy(), pred.scores.numpy(), pred.labels.numpy()):
        x1, y1, x2, y2 = b
        c = COLORS[int(lab) % len(COLORS)]
        ax.add_patch(Rectangle((x1 * N_TIME / 2, y1 * COARSE_H),
                               (x2 - x1) * N_TIME / 2, (y2 - y1) * COARSE_H,
                               fill=False, edgecolor=c, lw=1.0, ls="--", alpha=0.9))
        ax.text(x2 * N_TIME / 2, y2 * COARSE_H, f"{int(lab)}:{sc:.2f}",
                color=c, fontsize=5, va="top")
    ax.set_title(f"{record.sample_id}  B={record.B_mhz:g} MHz  T={record.T_ms:g} ms  "
                 f"GT={len(gt.boxes)}  pred={len(pred.boxes)}")
    ax.set_xlabel("time (downsampled 2x)")
    ax.set_ylabel("frequency (coarse 2048)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="yolo11s_c1_directreg_2048x512")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--split", choices=["fit", "val"], default="val")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--ids", default="")
    ap.add_argument("--batch-views", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    run_dir = ROOT / "runs" / args.run
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = run_dir / ckpt_path
    out_dir = run_dir / "stage5_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_cfg = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm_cfg["lo"]), float(norm_cfg["hi"]))
    records = load_manifest(SPLIT_MANIFEST, split=args.split)
    by_id = {r.sample_id: r for r in records}
    if args.ids:
        ids = [s.strip() for s in args.ids.split(",") if s.strip()]
        picks = [by_id[s] for s in ids]
    else:
        picks = pick_examples(records, args.limit)

    model = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["ema"] if ckpt.get("ema") is not None else ckpt["model"]
    model.load_state_dict(sd)
    model.eval()

    summary = []
    for r in picks:
        pred, stats = predict_sample(
            model, r, norm, batch_views=args.batch_views,
            conf=args.conf, iou=args.iou, max_det=100,
        )
        gt = gt_global(r, (r.N + 510) // 511)
        coarse, _, _ = build_sample_views(r, norm, [0])
        draw(r, coarse.tensor, gt, pred, out_dir / f"{r.sample_id.replace('/', '_')}.png")
        summary.append({
            "sample_id": r.sample_id, "B": r.B_mhz, "T": r.T_ms,
            "signals": len(r.signals), "gt_classes": sorted({int(s["class"]) for s in r.signals}),
            "stats": stats, "n_gt": len(gt.boxes), "n_pred": len(pred.boxes),
            "pred_classes": dict(Counter(pred.labels.tolist())),
            "pred_scores": {
                "min": float(pred.scores.min()) if len(pred.scores) else None,
                "max": float(pred.scores.max()) if len(pred.scores) else None,
                "mean": float(pred.scores.mean()) if len(pred.scores) else None,
            },
        })
        print(f"{r.sample_id}: gt={len(gt.boxes)} pred={len(pred.boxes)} "
              f"views={stats['n_views']} boundary_dropped={stats['n_boundary_dropped']}", flush=True)
    (out_dir / "examples_summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
