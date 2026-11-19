#!/usr/bin/env python3
"""Matched-step training for the three C3 front-ends and the C1 control.

The power channel is always the original Baseline-0 coarse power. C3 arms add
the two complex channels from one of the three caches. All arms use the same
1536 fit / 384 val subset, 20,000 optimizer steps, effective batch 16, model
body and paired initialization.

This is a screening budget, not the full Baseline-0 37,250-step budget; the
8-hour constraint and the four arms make a full-budget sweep impossible.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from ultralytics.utils.torch_utils import ModelEMA

from src.signals.data import (
    COARSE_H, N_TIME, SPLIT_MANIFEST, coarse_targets, collate_coarse,
    load_manifest,
)
from src.signals.infer import (
    Prediction, _map_view_dets, _nms_view_batch, global_classwise_nms, gt_global,
)
from src.signals.metrics import ImagePredictions, ImageTargets, evaluate
from src.signals.model import build_model, transfer_pretrained
from src.signals.train import (
    _validate_gt_guard, build_param_groups, save_checkpoint, seed_everything,
    should_early_stop,
)

ROOT = Path(__file__).resolve().parents[1]
CLASS_GROUPS = {"0/1/2": [0, 1, 2], "3/4/5": [3, 4, 5],
                "10/11": [10, 11], "9/12/13": [9, 12, 13]}


def lr_at_step(step: int, total: int, lr0: float, warmup: int, final_frac: float) -> float:
    if warmup > 0 and step < warmup:
        return lr0 * (0.1 + 0.9 * step / warmup)
    p = (step - warmup) / max(total - warmup, 1)
    p = min(1.0, max(0.0, p))
    return lr0 * (final_frac + (1.0 - final_frac) * 0.5 * (1.0 + math.cos(math.pi * p)))


class C3Dataset(Dataset):
    def __init__(self, split, ids, records_by_id, power_path, complex_path,
                 channels: int, power_indices=None, zero_complex: bool = False):
        self.split = split
        self.ids = list(ids)
        self.records = [records_by_id[s] for s in self.ids]
        self.channels = int(channels)
        self.zero_complex = bool(zero_complex)
        self.power_indices = list(power_indices) if power_indices is not None else None
        self.power = np.load(power_path, mmap_mode="r")
        self.complex = (np.load(complex_path, mmap_mode="r")
                        if channels == 3 and not self.zero_complex else None)
        if self.power_indices is not None and len(self.power_indices) != len(self.ids):
            raise ValueError("power index count does not match ids")
        if self.power_indices is None and len(self.power) != len(self.ids):
            raise ValueError(f"power rows {len(self.power)} != ids {len(self.ids)}")
        if channels == 3 and not self.zero_complex and len(self.complex) != len(self.ids):
            raise ValueError(f"complex rows {len(self.complex)} != ids {len(self.ids)}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        pi = self.power_indices[i] if self.power_indices is not None else i
        bits = np.asarray(self.power[pi]).copy()
        p = torch.from_numpy(bits).view(torch.bfloat16).float()
        if self.channels == 3:
            if self.zero_complex:
                c = torch.zeros((2, COARSE_H, N_TIME), dtype=torch.float32)
            else:
                c = torch.from_numpy(np.asarray(self.complex[i]).copy()).float()
            img = torch.cat([p, c], dim=0)
        else:
            img = p
        boxes, labels, _, _ = coarse_targets(self.records[i])
        return {"tensor": img, "boxes": boxes, "labels": labels,
                "meta": {"sample_id": self.records[i].sample_id}}


def class_group_eval(gts, preds, classes):
    keep = set(int(c) for c in classes)
    g = [ImageTargets(gt.boxes[np.isin(gt.labels, list(keep))],
                      gt.labels[np.isin(gt.labels, list(keep))]) for gt in gts]
    p = [ImagePredictions(pr.boxes[np.isin(pr.labels, list(keep))],
                          pr.scores[np.isin(pr.labels, list(keep))],
                          pr.labels[np.isin(pr.labels, list(keep))]) for pr in preds]
    r = evaluate(g, p, iou_thresholds=(0.5,), max_dets=100)
    pi = r["per_iou"]["0.5"]
    return {
        "n_gt": int(pi["n_gt"]),
        "n_pred": int(sum(len(x.boxes) for x in p)),
        "map50": pi["map_class_aware"],
        "ap50_agnostic": pi["ap_class_agnostic"],
        "localization_recall": r["classification"]["localization_recall"],
        "classification_accuracy": r["classification"]["classification_accuracy"],
    }


def run_validation(model, dataset, records, channels, batch_sources=8, return_preds=False):
    was_training = model.training
    model.eval()
    dl = DataLoader(dataset, batch_size=batch_sources, shuffle=False,
                    num_workers=4, collate_fn=collate_coarse, pin_memory=True,
                    persistent_workers=False)
    gts = [gt_global(r, math.ceil(r.N / (N_TIME - 1))) for r in records]
    preds = [[] for _ in records]
    n_pred = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            images = batch["img"].cuda(non_blocking=True)
            dets = _nms_view_batch(model, images, conf=0.001, iou=0.5,
                                   max_det=300, multi_label=False)
            for k, det in enumerate(dets):
                idx = bi * batch_sources + k
                if idx >= len(records):
                    continue
                r = records[idx]
                fb = int(round(r.B_mhz * 700))
                pred = _map_view_dets(det, "coarse", 0, fb)
                preds[idx].append(pred)
    final = []
    for parts in preds:
        if parts:
            cat = Prediction(torch.cat([p.boxes for p in parts]),
                             torch.cat([p.scores for p in parts]),
                             torch.cat([p.labels for p in parts]))
            f = global_classwise_nms(cat, iou=0.5, max_det=100)
        else:
            f = Prediction(torch.zeros((0, 4)), torch.zeros((0,)),
                           torch.zeros((0,), dtype=torch.long))
        final.append(ImagePredictions(f.boxes.numpy(), f.scores.numpy(),
                                      f.labels.numpy()))
    ev = evaluate(gts, final, iou_thresholds=(0.5,), max_dets=100)
    pi = ev["per_iou"]["0.5"]
    out = {
        "mAP50": pi["map_class_aware"],
        "map50_agnostic": pi["ap_class_agnostic"],
        "localization_recall": ev["classification"]["localization_recall"],
        "classification_accuracy": ev["classification"]["classification_accuracy"],
        "n_gt": int(pi["n_gt"]),
        "n_pred": int(sum(len(p.boxes) for p in final)),
        "per_class": pi["per_class"],
        "confusion": ev["classification"]["confusion_gt_x_pred"],
        "groups": {g: class_group_eval(gts, final, cls)
                   for g, cls in CLASS_GROUPS.items()},
        "val_elapsed_s": time.perf_counter() - t0,
    }
    if was_training:
        model.train()
    if return_preds:
        return out, gts, final
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--frontend", choices=["power", "zero3", "overlap", "sparse", "dense"],
                    required=True)
    ap.add_argument("--channels", type=int, choices=[1, 3], default=0)
    ap.add_argument("--subset", default="cache/c3_subset.json")
    ap.add_argument("--complex-dir", default="")
    ap.add_argument("--power-cache", default="cache/coarse_fit_bf16.npy")
    ap.add_argument("--val-power-cache", default="cache/c3_subset_val_power_bf16.npy")
    ap.add_argument("--total-steps", type=int, default=20000)
    ap.add_argument("--batch-sources", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-every-steps", type=int, default=1280)
    ap.add_argument("--early-stop-evals", type=int, default=4)
    ap.add_argument("--early-stop-min-delta", type=float, default=0.001)
    ap.add_argument("--early-stop-min-evals", type=int, default=3)
    ap.add_argument("--lr0", type=float, default=1e-3)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--final-lr-frac", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--clip-grad", type=float, default=10.0)
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    zero_complex = args.frontend == "zero3"
    channels = args.channels or (1 if args.frontend == "power" else 3)
    if channels == 1 and args.frontend != "power":
        raise SystemExit("3-channel front-end cannot run with --channels 1")
    if channels == 3 and args.frontend == "power":
        raise SystemExit("power control must use --channels 1")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(4)

    subset = json.loads((ROOT / args.subset).read_text())
    fit_all = load_manifest(SPLIT_MANIFEST, "fit")
    val_all = load_manifest(SPLIT_MANIFEST, "val")
    by_fit = {r.sample_id: r for r in fit_all}
    by_val = {r.sample_id: r for r in val_all}
    fit_ids = subset["fit_ids"]
    val_ids = subset["val_ids"]
    fit_recs = [by_fit[s] for s in fit_ids]
    val_recs = [by_val[s] for s in val_ids]

    power_fit = ROOT / args.power_cache
    power_val = ROOT / args.val_power_cache
    if args.frontend in ("power", "zero3"):
        complex_dir = None
        fit_complex = val_complex = None
    else:
        complex_dir = Path(args.complex_dir) if args.complex_dir else ROOT / f"cache/c3_{args.frontend}"
        if not complex_dir.is_absolute():
            complex_dir = ROOT / complex_dir
        fit_complex = complex_dir / "fit_complex.npy"
        val_complex = complex_dir / "val_complex.npy"
        for p in (fit_complex, val_complex):
            if not p.exists():
                raise SystemExit(f"missing complex cache: {p}")

    fit_ds = C3Dataset("fit", fit_ids, by_fit, power_fit, fit_complex, channels,
                       power_indices=subset["fit_power_index"],
                       zero_complex=zero_complex)
    val_ds = C3Dataset("val", val_ids, by_val, power_val, val_complex, channels,
                       zero_complex=zero_complex)
    steps_per_epoch = math.ceil(len(fit_ds) / args.batch_sources) // args.accum
    total_steps = args.total_steps
    epochs = math.ceil(total_steps / steps_per_epoch)

    run_dir = ROOT / "runs" / args.run
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{run_dir} exists; pass --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run": args.run, "frontend": args.frontend, "channels": channels, "zero_complex": zero_complex,
        "subset": args.subset, "fit_samples": len(fit_ds),
        "val_samples": len(val_ds), "batch_sources": args.batch_sources,
        "accum": args.accum, "effective_views": args.batch_sources * args.accum,
        "steps_per_epoch": steps_per_epoch, "epochs_planned": epochs,
        "total_steps": total_steps, "warmup_steps": args.warmup_steps,
        "val_every_steps": args.val_every_steps,
        "early_stop_evals": args.early_stop_evals,
        "early_stop_min_delta": args.early_stop_min_delta,
        "early_stop_min_evals": args.early_stop_min_evals,
        "lr0": args.lr0, "weight_decay": args.weight_decay,
        "precision": args.precision, "seed": args.seed,
        "power_cache": str(power_fit), "val_power_cache": str(power_val),
        "complex_dir": str(complex_dir) if complex_dir else None,
        "init": "yolo11s.pt paired transfer; C3 power stem=RGB sum, Re/Im=0",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    model = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"),
                        device="cuda", ch=channels)
    assert tuple(model.model[0].conv.weight.shape)[1] == channels
    transfer_pretrained(model, ROOT / "yolo11s.pt", ch=channels)
    optimizer = torch.optim.AdamW(build_param_groups(model, args.weight_decay),
                                  lr=args.lr0, betas=(0.9, 0.999), eps=1e-8)
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    ema = ModelEMA(model, decay=0.9999, tau=2000)

    gen = torch.Generator(); gen.manual_seed(args.seed)
    train_dl = DataLoader(fit_ds, batch_size=args.batch_sources, shuffle=True,
                          generator=gen, num_workers=args.workers,
                          collate_fn=collate_coarse, pin_memory=True,
                          persistent_workers=args.workers > 0,
                          prefetch_factor=4 if args.workers else None)
    log_f = (run_dir / "train_log.csv").open("w", newline="")
    log = csv.writer(log_f)
    log.writerow(["epoch", "global_step", "lr", "loss", "box", "cls", "l1",
                  "gpu_alloc_gib", "gpu_reserved_gib", "time"])
    log_f.flush()

    val_history = []
    best_fitness = -1.0
    best_epoch = -1
    patience_best = -1.0
    last_improve_eval = -1
    evals_since = 0
    global_step = 0
    epoch = 0
    stop = False
    t_run = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    while global_step < total_steps and not stop:
        model.train()
        for batch in train_dl:
            _validate_gt_guard(batch)
            images = batch["img"].cuda(non_blocking=True)
            target = {"batch_idx": batch["batch_idx"].cuda(non_blocking=True),
                      "cls": batch["cls"].cuda(non_blocking=True),
                      "bboxes": batch["bboxes"].cuda(non_blocking=True),
                      "img": images}
            lr = lr_at_step(global_step, total_steps, args.lr0,
                            args.warmup_steps, args.final_lr_frac)
            for g in optimizer.param_groups:
                g["lr"] = lr
            with torch.autocast("cuda", dtype=amp_dtype,
                                enabled=args.precision == "bf16"):
                preds = model(images)
                parts, _ = model.loss(target, preds=preds)
                loss = parts.sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at step {global_step}")
            scaler.scale(loss / args.accum).backward()
            if (global_step + 1) % args.accum == 0:
                if args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            global_step += 1
            log.writerow([epoch, global_step, f"{lr:.8f}", f"{float(loss.detach()):.6f}",
                          f"{float(parts[0].detach()):.6f}", f"{float(parts[1].detach()):.6f}",
                          f"{float(parts[2].detach()):.6f}", "",
                          round(torch.cuda.memory_allocated() / 1024 ** 3, 3),
                          round(torch.cuda.memory_reserved() / 1024 ** 3, 3),
                          time.strftime("%Y-%m-%d %H:%M:%S")])
            log_f.flush()
            do_val = (global_step % args.val_every_steps == 0 or
                      global_step >= total_steps)
            if do_val:
                vr = run_validation(ema.ema, val_ds, val_recs, channels)
                entry = {"global_step": global_step, "epoch": epoch, **vr}
                val_history.append(entry)
                (run_dir / "val_history.json").write_text(json.dumps(val_history, indent=2))
                print(f"step {global_step} mAP {vr['mAP50']:.4f} "
                      f"agn {vr['map50_agnostic']:.4f} "
                      f"R {vr['localization_recall']:.4f} "
                      f"pred {vr['n_pred']} val {vr['val_elapsed_s']:.0f}s", flush=True)
                improved = vr["mAP50"] > best_fitness
                if improved:
                    best_fitness = vr["mAP50"]
                    best_epoch = global_step
                if vr["mAP50"] > patience_best + args.early_stop_min_delta:
                    patience_best = vr["mAP50"]
                    last_improve_eval = len(val_history) - 1
                    evals_since = 0
                else:
                    evals_since += 1
                if improved:
                    save_checkpoint(run_dir / "best.pt", model, ema, optimizer,
                                    scaler, epoch, best_fitness, config, entry,
                                    include_optimizer=False, best_epoch=best_epoch,
                                    patience_best=patience_best,
                                    last_improve_eval=last_improve_eval,
                                    evals_since_improve=evals_since)
                    print(f"  new best {best_fitness:.4f}", flush=True)
                save_checkpoint(run_dir / "last.pt", model, ema, optimizer, scaler,
                                epoch, best_fitness, config, entry,
                                include_optimizer=True, best_epoch=best_epoch,
                                patience_best=patience_best,
                                last_improve_eval=last_improve_eval,
                                evals_since_improve=evals_since)
                stop = should_early_stop(evals_since, len(val_history),
                                         args.early_stop_evals,
                                         args.early_stop_min_evals)
                if stop:
                    print(f"  early stop at step {global_step}, best {best_fitness:.4f}", flush=True)
                    break
            if global_step >= total_steps:
                break
        epoch += 1
        free = shutil.disk_usage(ROOT).free / 1024 ** 3
        if free < 5:
            raise RuntimeError(f"disk free {free:.1f} GiB below 5")
    log_f.close()
    total_s = time.perf_counter() - t_run
    out = {
        "run": args.run, "frontend": args.frontend, "channels": channels,
        "config": config, "best_fitness": best_fitness, "best_step": best_epoch,
        "actual_steps": global_step, "early_stopped": bool(stop),
        "total_s": total_s, "total_h": total_s / 3600,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "val_history": val_history,
    }
    (run_dir / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"done {args.run}: best {best_fitness:.4f} steps {global_step} "
          f"total {total_s/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
