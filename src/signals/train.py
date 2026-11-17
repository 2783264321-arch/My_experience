"""Training entry point for YOLO11s-C1-DirectReg on 2048x512 STFT views.

Design notes (see docs/Baseline执行计划.md):
- one coarse + one rotating fine view per fit source per epoch
- effective batch target 16 views (physical sources x 2 views x accumulation)
- AdamW lr0=1e-3, wd=5e-4, cosine to 1%, 3-epoch warmup
- AMP after the stage-2 acceptance test; EMA weights are validated
- best checkpoint is selected by full-sample strict class-aware mAP@0.5
- only best/last checkpoints are retained
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from ultralytics.utils.torch_utils import ModelEMA

from .data import (
    SPLIT_MANIFEST, SampleRecord, SpaceNetViews, collate_views, load_manifest,
)
from .model import build_model, transfer_pretrained
from .validate import run_validation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = "yolo11s_c1_directreg_2048x512"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init(worker_id: int):
    # STFT is CPU-heavy; one thread per worker avoids oversubscribing 16 vCPUs
    torch.set_num_threads(1)
    random.seed(1729 + worker_id)
    np.random.seed(1729 + worker_id)


class SharedEpochViews(SpaceNetViews):
    """Persistent workers read the current epoch from shared memory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_epoch = mp.Value("i", 0, lock=False)

    def __getitem__(self, idx: int):
        self.epoch = int(self.shared_epoch.value)
        return super().__getitem__(idx)


def make_loader(
    ds: SpaceNetViews,
    batch_sources: int,
    workers: int,
    seed: int,
    shuffle: bool = True,
) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_sources,
        shuffle=shuffle,
        generator=gen,
        num_workers=workers,
        collate_fn=collate_views,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=4 if workers > 0 else None,
        persistent_workers=workers > 0,
        worker_init_fn=worker_init if workers > 0 else None,
    )


def lr_at(epoch_float: float, epochs: int, lr0: float, warmup: float, final_frac: float) -> float:
    if warmup > 0 and epoch_float < warmup:
        return lr0 * (0.1 + 0.9 * epoch_float / warmup)
    progress = (epoch_float - warmup) / max(epochs - warmup, 1e-9)
    progress = min(1.0, max(0.0, progress))
    return lr0 * (final_frac + (1.0 - final_frac) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def build_param_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "bn" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def save_checkpoint(
    path: Path,
    model,
    ema: Optional[ModelEMA],
    optimizer,
    scaler,
    epoch: int,
    best_fitness: float,
    args: dict,
    metrics: dict,
    include_optimizer: bool = True,
):
    payload = {
        "model": model.state_dict(),
        "ema": ema.ema.state_dict() if ema is not None else None,
        "epoch": epoch,
        "best_fitness": best_fitness,
        "args": args,
        "metrics": metrics,
        "reg_max": int(model.model[-1].reg_max),
        "end2end": bool(model.model[-1].end2end),
        "nc": int(model.model[-1].nc),
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
        payload["scaler"] = scaler.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _validate_gt_guard(batch: dict):
    if batch["bboxes"].numel():
        if not torch.isfinite(batch["bboxes"]).all():
            raise RuntimeError("non-finite GT boxes")
        if (batch["bboxes"] < -1e-4).any() or (batch["bboxes"] > 1 + 1e-4).any():
            raise RuntimeError("GT boxes out of [0,1]")
        if batch["cls"].numel() and ((batch["cls"] < 0).any() or (batch["cls"] > 13).any()):
            raise RuntimeError("class label out of range")
        if (batch["bboxes"][:, 2:] <= 0).any():
            raise RuntimeError("non-positive GT width/height")


def train(args):
    seed_everything(args.seed)
    # The last batch of each epoch has 10 views instead of 16 (5957 % 8 == 5).
    # cudnn.benchmark=True autotunes the new shape for ~100 s every epoch; the
    # heuristic planner is fast and the fixed-shape throughput loss is small.
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.workers > 0:
        torch.set_num_threads(4)
    else:
        torch.set_num_threads(args.cpu_threads)

    run_dir = ROOT / "runs" / args.run
    if (run_dir / "config.json").exists() and not args.resume and not args.overwrite:
        raise SystemExit(f"run {run_dir} already exists; use --resume or a new --run/--overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)
    normalizer = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(normalizer["lo"]), float(normalizer["hi"]))

    fit = load_manifest(SPLIT_MANIFEST, split="fit")
    val = load_manifest(SPLIT_MANIFEST, split="val")
    if args.limit_sources:
        fit = fit[: args.limit_sources]
        val = val[: max(8, args.limit_sources // 4)]
    effective_views = args.batch_sources * 2 * args.accum
    if not args.allow_any_effective_batch and effective_views != args.target_views:
        raise SystemExit(
            f"effective views {effective_views} != target {args.target_views}; "
            "use --allow-any-effective-batch only for diagnostics"
        )

    config = {
        "baseline_id": DEFAULT_RUN,
        "model": "configs/yolo11s_c1_directreg.yaml",
        "epochs": args.epochs,
        "batch_sources": args.batch_sources,
        "accum": args.accum,
        "effective_views": effective_views,
        "workers": args.workers,
        "optimizer": "AdamW",
        "lr0": args.lr0,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup,
        "final_lr_frac": args.final_lr_frac,
        "precision": args.precision,
        "amp": args.precision != "fp32",
        "seed": args.seed,
        "val_every": args.val_every,
        "val_batch_views": args.val_batch_views,
        "val_workers": args.val_workers,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "max_det_view": args.max_det_view,
        "target_views": args.target_views,
        "split_manifest": str(SPLIT_MANIFEST),
        "split_sha256": sha256_file(SPLIT_MANIFEST),
        "normalizer_sha256": sha256_file(ROOT / "configs/normalizer.json"),
        "model_yaml_sha256": sha256_file(ROOT / "configs/yolo11s_c1_directreg.yaml"),
        "fit_samples": len(fit),
        "val_samples": len(val),
        "pretrained": str(ROOT / "yolo11s.pt"),
        "pretrained_sha256": sha256_file(ROOT / "yolo11s.pt"),
        "stft_backend": __import__("src.signals.data", fromlist=["STFT_BACKEND"]).STFT_BACKEND,
        "fft_threads": int(__import__("src.signals.data", fromlist=["FFT_THREADS"]).FFT_THREADS),
        "window_ms": 0.7,
        "n_time": 512,
        "fine_h": 2048,
        "fine_step": 1639,
        "coarse_h": 2048,
        "normalizer_lo": norm[0],
        "normalizer_hi": norm[1],
        "torch": torch.__version__,
        "ultralytics": __import__("ultralytics").__version__,
        "cudnn_benchmark": False,
        "cudnn_benchmark_note": "disabled because 5957%8=5 makes the last batch 10 views; benchmark autotuned it for ~100 s per epoch",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "normalizer.json").write_text((ROOT / "configs/normalizer.json").read_text())

    model = build_model(str(ROOT / args.model), device=args.device)
    head = model.model[-1]
    assert int(head.nc) == 14 and int(head.reg_max) == 1 and bool(head.end2end) is False
    assert tuple(model.model[0].conv.weight.shape) == (32, 1, 3, 3)
    if not args.resume:
        stats = transfer_pretrained(model, ROOT / "yolo11s.pt", verbose=True)
        (run_dir / "pretrained_transfer.json").write_text(json.dumps(stats, indent=2))
    optimizer = torch.optim.AdamW(
        build_param_groups(model, args.weight_decay), lr=args.lr0,
        betas=(0.9, 0.999), eps=1e-8,
    )
    # FP16 autocast overflows on this 21504-anchor head (millions of non-finite
    # grads in stage-2 probes); BF16 keeps FP32 range and is accepted instead.
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    use_scaler = args.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    ema = ModelEMA(model, decay=0.9999, tau=2000) if not args.no_ema else None

    start_epoch = 0
    best_fitness = -1.0
    if args.resume:
        ckpt_path = run_dir / "last.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"resume requested but {ckpt_path} does not exist")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if ema is not None and ckpt.get("ema") is not None:
            ema.ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt.get("scaler", {}))
        start_epoch = int(ckpt["epoch"]) + 1
        best_fitness = float(ckpt.get("best_fitness", -1.0))
        print(f"resumed from {ckpt_path} at epoch {start_epoch}", flush=True)

    log_path = run_dir / "train_log.csv"
    log_exists = log_path.exists() and log_path.stat().st_size > 0
    log_f = log_path.open("a", newline="")
    log = csv.writer(log_f)
    if not log_exists:
        log.writerow([
            "epoch", "step", "lr", "loss", "box", "cls", "l1",
            "step_s", "epoch_s", "gpu_alloc_gib", "gpu_reserved_gib", "time",
        ])
        log_f.flush()

    val_history = []
    hist_path = run_dir / "val_history.json"
    if args.resume and hist_path.exists():
        val_history = json.loads(hist_path.read_text())

    model.train()
    print(f"run={args.run} fit={len(fit)} val={len(val)} "
          f"batch_sources={args.batch_sources} accum={args.accum} "
          f"effective_views={effective_views} precision={args.precision}", flush=True)
    micro_per_epoch = math.ceil(len(fit) / args.batch_sources)
    steps_per_epoch = micro_per_epoch // args.accum
    print(f"micro/epoch={micro_per_epoch} opt_steps/epoch={steps_per_epoch} "
          f"total_steps~{steps_per_epoch * args.epochs}", flush=True)
    train_ds = SharedEpochViews(fit, norm, seed=args.seed, epoch=start_epoch)
    train_loader = make_loader(train_ds, args.batch_sources, args.workers, args.seed)

    t_run = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.perf_counter()
        train_ds.shared_epoch.value = epoch
        loader = train_loader
        optimizer.zero_grad(set_to_none=True)
        accum_loss = torch.zeros(3, device=args.device)
        n_micro = 0
        step = 0
        last_print = time.perf_counter()
        for micro, batch in enumerate(loader):
            _validate_gt_guard(batch)
            images = batch["img"].to(args.device, non_blocking=True)
            target = {
                "batch_idx": batch["batch_idx"].to(args.device, non_blocking=True),
                "cls": batch["cls"].to(args.device, non_blocking=True),
                "bboxes": batch["bboxes"].to(args.device, non_blocking=True),
                "img": images,
            }
            epoch_float = epoch + micro / max(len(loader), 1)
            lr = lr_at(epoch_float, args.epochs, args.lr0, args.warmup, args.final_lr_frac)
            for group in optimizer.param_groups:
                group["lr"] = lr
            with torch.autocast("cuda", dtype=amp_dtype, enabled=args.precision != "fp32"):
                preds = model(images)
                loss_parts, loss_items = model.loss(target, preds=preds)
                loss = loss_parts.sum()
            if not torch.isfinite(loss):
                save_checkpoint(
                    run_dir / "nan_stop.pt", model, ema, optimizer, scaler,
                    epoch, best_fitness, config,
                    {"micro": micro, "loss": float(loss) if torch.isfinite(loss) else "nonfinite"},
                    include_optimizer=True,
                )
                raise RuntimeError(f"non-finite loss at epoch {epoch} micro {micro}")
            scaler.scale(loss / args.accum).backward()
            accum_loss += loss_parts.detach().float()
            n_micro += 1
            if n_micro % args.accum == 0:
                if args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                step += 1
                mean = (accum_loss / args.accum)
                log.writerow([
                    epoch, step, f"{lr:.8f}", f"{float(mean.sum()):.6f}",
                    f"{float(mean[0]):.6f}", f"{float(mean[1]):.6f}", f"{float(mean[2]):.6f}",
                    "", "", round(torch.cuda.memory_allocated() / 1024 ** 3, 3),
                    round(torch.cuda.memory_reserved() / 1024 ** 3, 3),
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                ])
                log_f.flush()
                accum_loss.zero_()
                now = time.perf_counter()
                if now - last_print > args.print_every_s:
                    print(f"ep {epoch} step {step}/{steps_per_epoch} "
                          f"lr {lr:.2e} loss {float(mean.sum()):.4f} "
                          f"({float(mean[0]):.3f}/{float(mean[1]):.3f}/{float(mean[2]):.3f}) "
                          f"elapsed {now - t_epoch:.0f}s "
                          f"mem {torch.cuda.memory_reserved() / 1024 ** 3:.1f}G", flush=True)
                    last_print = now
        # flush any partial accumulation at epoch end (drop_last makes this rare)
        if n_micro % args.accum:
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
        epoch_s = time.perf_counter() - t_epoch
        print(f"epoch {epoch} done in {epoch_s:.0f}s", flush=True)

        free_gib = shutil.disk_usage(ROOT).free / 1024 ** 3
        if free_gib < args.min_free_gib:
            raise RuntimeError(f"disk free {free_gib:.1f} GiB below {args.min_free_gib}")

        do_val = (epoch + 1) % args.val_every == 0 or (epoch + 1) == args.epochs
        if do_val:
            vmodel = ema.ema if ema is not None else model
            t_val = time.perf_counter()
            vr, gts, preds = run_validation(
                vmodel, val, norm,
                batch_views=args.val_batch_views, workers=args.val_workers,
                conf=args.conf, iou=args.iou, max_det=args.max_det,
                max_det_view=args.max_det_view,
            )
            entry = {
                "epoch": epoch,
                "map50": vr.map50,
                "map50_agnostic": vr.map50_agnostic,
                "localization_recall": vr.localization_recall,
                "classification_accuracy": vr.classification_accuracy,
                "n_gt": vr.n_gt,
                "n_pred": vr.n_pred,
                "per_class": vr.per_class,
                "confusion": vr.confusion,
                "stats": vr.stats,
                "val_elapsed_s": vr.elapsed_s,
                "wall_val_s": time.perf_counter() - t_val,
            }
            val_history.append(entry)
            hist_path.write_text(json.dumps(val_history, indent=2))
            print(f"val epoch {epoch}: mAP50={vr.map50:.4f} "
                  f"agnostic={vr.map50_agnostic:.4f} loc_recall={vr.localization_recall} "
                  f"pred={vr.n_pred} views={vr.stats['n_views']} "
                  f"time={vr.elapsed_s:.0f}s", flush=True)
            if vr.map50 > best_fitness:
                best_fitness = vr.map50
                save_checkpoint(
                    run_dir / "best.pt", model, ema, optimizer, scaler,
                    epoch, best_fitness, config, entry, include_optimizer=False,
                )
                print(f"new best mAP50={best_fitness:.4f} -> best.pt", flush=True)
        save_checkpoint(
            run_dir / "last.pt", model, ema, optimizer, scaler,
            epoch, best_fitness, config, val_history[-1] if val_history else {},
            include_optimizer=True,
        )
        log_f.flush()

    total_s = time.perf_counter() - t_run
    summary = {
        "run": args.run,
        "epochs": args.epochs,
        "best_fitness": best_fitness,
        "total_s": total_s,
        "total_h": total_s / 3600,
        "val_history": val_history,
        "config": config,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"training complete: best mAP50={best_fitness:.4f} total={total_s / 3600:.2f}h", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=DEFAULT_RUN)
    p.add_argument("--model", default="configs/yolo11s_c1_directreg.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-sources", type=int, default=4)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--target-views", type=int, default=16)
    p.add_argument("--allow-any-effective-batch", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--val-workers", type=int, default=4)
    p.add_argument("--val-batch-views", type=int, default=8)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--warmup", type=float, default=3.0)
    p.add_argument("--final-lr-frac", type=float, default=0.01)
    p.add_argument("--clip-grad", type=float, default=10.0)
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--limit-sources", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--max-det", type=int, default=100)
    p.add_argument("--max-det-view", type=int, default=300)
    p.add_argument("--min-free-gib", type=float, default=20.0)
    p.add_argument("--print-every-s", type=float, default=30.0)
    p.add_argument("--cpu-threads", type=int, default=8)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
