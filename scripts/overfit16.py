#!/usr/bin/env python3
"""Stage-3 16-source overfit gate.

16 fit sources covering all 14 classes and B/T extremes, the same
coarse + one rotating fine view training definition, and full-sample
(all fine + coarse) strict mAP@0.5 evaluation. Pass target: mAP@0.5 >= 0.8
within at most 2000 optimizer updates.

The selection is frozen here (manifest SHA is recorded by the training run);
it was chosen greedily for class coverage, B/T diversity and absence of
near-duplicate same-class GT boxes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.signals.data import (
    SPLIT_MANIFEST, SpaceNetViews, collate_views, load_manifest,
)
from src.signals.model import build_model, transfer_pretrained
from src.signals.train import build_param_groups, lr_at, sha256_file, worker_init
from src.signals.validate import run_validation

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IDS = [
    "train/27", "train/625", "train/4956", "train/495", "train/117",
    "train/1897", "train/479", "train/1617", "train/692", "train/2106",
    "train/6848", "train/4211", "train/2238", "train/941", "train/2077",
    "train/446",
]


class SharedEpochViews(SpaceNetViews):
    """Persistent workers read the current epoch from shared memory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_epoch = mp.Value("i", 0, lock=False)

    def __getitem__(self, idx: int):
        self.epoch = int(self.shared_epoch.value)
        return super().__getitem__(idx)


def build_loader(records, norm, batch_sources, workers, seed):
    ds = SharedEpochViews(records, norm, seed=seed, epoch=0)
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(
        ds, batch_size=batch_sources, shuffle=True, generator=gen,
        num_workers=workers, collate_fn=collate_views, pin_memory=True,
        drop_last=True, prefetch_factor=2, persistent_workers=workers > 0,
        worker_init_fn=worker_init if workers > 0 else None,
    )
    return ds, loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-updates", type=int, default=2000)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--batch-sources", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--weight-decay", type=float, default=0.0005)
    ap.add_argument("--warmup-updates", type=int, default=100)
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--run", default="overfit16")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-ema", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    run_dir = ROOT / "runs" / args.run
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{run_dir} exists; pass --overwrite or use another --run")
    run_dir.mkdir(parents=True, exist_ok=True)

    norm_cfg = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm_cfg["lo"]), float(norm_cfg["hi"]))
    fit = load_manifest(SPLIT_MANIFEST, split="fit")
    by_id = {r.sample_id: r for r in fit}
    missing = [s for s in SAMPLE_IDS if s not in by_id]
    if missing:
        raise SystemExit(f"missing sample ids: {missing}")
    records = [by_id[s] for s in SAMPLE_IDS]
    covered = sorted({int(s["class"]) for r in records for s in r.signals})
    if covered != list(range(14)):
        raise SystemExit(f"class coverage broken: {covered}")
    info = {
        "sample_ids": SAMPLE_IDS,
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
        "normalizer_sha256": sha256_file(ROOT / "configs/normalizer.json"),
        "classes_covered": covered,
        "samples": [
            {"id": r.sample_id, "B": r.B_mhz, "T": r.T_ms,
             "signals": len(r.signals),
             "classes": sorted({int(s["class"]) for s in r.signals})}
            for r in records
        ],
        "max_updates": args.max_updates,
        "target": args.target,
        "batch_sources": args.batch_sources,
        "precision": "bf16",
    }
    (run_dir / "overfit_config.json").write_text(json.dumps(info, indent=2))

    model = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"))
    transfer_pretrained(model, ROOT / "yolo11s.pt", verbose=True)
    model.train()
    opt = torch.optim.AdamW(build_param_groups(model, args.weight_decay), lr=args.lr0)
    ema = None
    if not args.no_ema:
        from ultralytics.utils.torch_utils import ModelEMA
        ema = ModelEMA(model, decay=0.99, tau=100)

    ds, loader = build_loader(records, norm, args.batch_sources, args.workers, args.seed)
    it = iter(loader)
    epoch = 0
    history = []
    best = -1.0
    t_start = time.perf_counter()
    losses = []
    print(f"overfit samples={len(records)} batch_sources={args.batch_sources} "
          f"views/step={args.batch_sources * 2} max_updates={args.max_updates}", flush=True)

    def validate(update: int):
        nonlocal best
        vmodel = ema.ema if ema is not None else model
        vr, gts, preds = run_validation(
            vmodel, records, norm, batch_views=8, workers=4,
            conf=0.001, iou=0.5, max_det=100, max_det_view=300,
            multi_label=False, log_interval_s=60,
        )
        entry = {
            "update": update,
            "map50": vr.map50,
            "map50_agnostic": vr.map50_agnostic,
            "localization_recall": vr.localization_recall,
            "classification_accuracy": vr.classification_accuracy,
            "n_pred": vr.n_pred,
            "n_gt": vr.n_gt,
            "per_class": vr.per_class,
            "stats": vr.stats,
            "val_elapsed_s": vr.elapsed_s,
        }
        history.append(entry)
        (run_dir / "overfit_history.json").write_text(json.dumps(history, indent=2))
        print(f"[update {update}] mAP50={vr.map50:.4f} agnostic={vr.map50_agnostic:.4f} "
              f"recall={vr.localization_recall} pred={vr.n_pred} views={vr.stats['n_views']} "
              f"val={vr.elapsed_s:.0f}s", flush=True)
        if vr.map50 > best:
            best = vr.map50
            torch.save({
                "model": model.state_dict(),
                "ema": ema.ema.state_dict() if ema is not None else None,
                "update": update,
                "map50": vr.map50,
                "config": info,
                "history": history,
            }, run_dir / "best_overfit.pt")
            print(f"  new best {best:.4f} -> best_overfit.pt", flush=True)
        return vr

    update = 0
    while update < args.max_updates:
        try:
            batch = next(it)
        except StopIteration:
            epoch += 1
            ds.shared_epoch.value = epoch
            it = iter(loader)
            batch = next(it)
        opt.zero_grad(set_to_none=True)
        images = batch["img"].cuda(non_blocking=True)
        target = {
            "img": images,
            "batch_idx": batch["batch_idx"].cuda(non_blocking=True),
            "cls": batch["cls"].cuda(non_blocking=True),
            "bboxes": batch["bboxes"].cuda(non_blocking=True),
        }
        lr = lr_at(update, args.max_updates, args.lr0, args.warmup_updates, 0.01)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds = model(images)
            loss_parts, items = model.loss(target, preds=preds)
            loss = loss_parts.sum()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at update {update}")
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if ema is not None:
            ema.update(model)
        update += 1
        losses.append(float(loss.detach()))
        if update % 25 == 0:
            recent = losses[-25:]
            print(f"update {update}/{args.max_updates} lr {lr:.2e} "
                  f"loss {sum(recent) / len(recent):.3f} "
                  f"box {float(loss_parts[0]):.3f} cls {float(loss_parts[1]):.3f} "
                  f"l1 {float(loss_parts[2]):.4f} gnorm {float(gn):.1f} "
                  f"t {time.perf_counter() - t_start:.0f}s "
                  f"mem {torch.cuda.memory_reserved() / 1024 ** 3:.1f}G", flush=True)
        if update % args.val_every == 0 or update == args.max_updates:
            vr = validate(update)
            if vr.map50 >= args.target:
                print(f"PASS: mAP50 {vr.map50:.4f} >= {args.target} at update {update}", flush=True)
                break

    elapsed = time.perf_counter() - t_start
    summary = {
        "pass": best >= args.target,
        "best_map50": best,
        "updates": update,
        "epochs_seen": epoch,
        "elapsed_s": elapsed,
        "loss_first50_mean": float(np.mean(losses[:50])) if losses else None,
        "loss_last50_mean": float(np.mean(losses[-50:])) if losses else None,
        "history": history,
        "config": info,
    }
    (run_dir / "overfit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ["pass", "best_map50", "updates", "epochs_seen",
                       "loss_first50_mean", "loss_last50_mean", "elapsed_s"]},
                     indent=2), flush=True)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
