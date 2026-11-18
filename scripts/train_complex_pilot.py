#!/usr/bin/env python3
"""Short paired C1/C3 end-to-end pilot on the complex coarse/fine cache.

C1 uses channel 0 (power) only. C3 uses [power, Re, Im] from the same cache.
Both arms use the same split, fixed views, labels, model body and training
budget. C3's extra input channels are zero-initialized so the power path starts
identical to C1.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from ultralytics.utils.torch_utils import ModelEMA

from src.signals.data import N_TIME, SPLIT_MANIFEST, load_manifest
from src.signals.infer import (
    Prediction, _nms_view_batch, global_classwise_nms, gt_global,
)
from src.signals.metrics import ImagePredictions, ImageTargets, evaluate
from src.signals.model import build_model, transfer_pretrained
from src.signals.validate import _run_view_chunk
from scripts.view_source_attribution import CLASS_GROUPS, GROUPS, evaluate_group

ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); os.environ["PYTHONHASHSEED"] = str(seed)


def build_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "bn" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": weight_decay}]


def lr_at(epoch_float, epochs, lr0, warmup, final_frac):
    if warmup > 0 and epoch_float < warmup:
        return lr0 * (0.1 + 0.9 * epoch_float / warmup)
    progress = (epoch_float - warmup) / max(epochs - warmup, 1e-9)
    progress = min(1.0, max(0.0, progress))
    return lr0 * (final_frac + (1.0 - final_frac) * 0.5 *
                  (1.0 + math.cos(math.pi * progress)))


class CachedPairs(Dataset):
    def __init__(self, cache, split, channels):
        self.dir = Path(cache)
        self.meta = json.loads((self.dir / f"{split}_meta.json").read_text())
        self.coarse = np.load(self.dir / f"{split}_coarse.npy", mmap_mode="r")
        self.fine = np.load(self.dir / f"{split}_fine.npy", mmap_mode="r")
        self.channels = int(channels)

    def __len__(self):
        return len(self.meta["samples"])

    def __getitem__(self, idx):
        m = self.meta["samples"][idx]
        c = torch.from_numpy(np.asarray(self.coarse[idx, :self.channels]).copy()).float()
        f = torch.from_numpy(np.asarray(self.fine[idx, :self.channels]).copy()).float()
        return {"coarse": c, "fine": f, "meta": m}


def collate_pairs(items):
    images, boxes, labels, batch_idx, metas, view_types, starts = [], [], [], [], [], [], []
    for si, item in enumerate(items):
        for vtype, tensor, key in [("coarse", item["coarse"], "coarse"),
                                   ("fine", item["fine"], "fine")]:
            vi = len(images)
            images.append(tensor)
            m = item["meta"]
            b = torch.tensor(m[f"{key}_boxes"], dtype=torch.float32).reshape(-1, 4)
            l = torch.tensor(m[f"{key}_labels"], dtype=torch.long)
            if len(b):
                boxes.append(b); labels.append(l)
                batch_idx.append(torch.full((len(b),), vi, dtype=torch.long))
            view_types.append(vtype)
            starts.append(0 if vtype == "coarse" else int(m["fine_start"]))
            metas.append(m)
    images = torch.stack(images, 0)
    if boxes:
        b = torch.cat(boxes, 0)
        b = torch.cat([(b[:, :2] + b[:, 2:]) / 2, b[:, 2:] - b[:, :2]], 1)
        labels = torch.cat(labels, 0)
        batch_idx = torch.cat(batch_idx, 0)
    else:
        b = torch.zeros((0, 4)); labels = torch.zeros((0,), dtype=torch.long)
        batch_idx = torch.zeros((0,), dtype=torch.long)
    return {"img": images, "batch_idx": batch_idx, "cls": labels[:, None],
            "bboxes": b, "meta": metas, "view_types": view_types, "starts": starts}


def run_validation(model, cache, split, channels, records, batch_views=8):
    ds_meta = json.loads((Path(cache) / f"{split}_meta.json").read_text())
    by_id = {r.sample_id: r for r in records}
    recs = [by_id[m["sample_id"]] for m in ds_meta["samples"]]
    coarse = np.load(Path(cache) / f"{split}_coarse.npy", mmap_mode="r")
    fine = np.load(Path(cache) / f"{split}_fine.npy", mmap_mode="r")
    gts = [gt_global(r, math.ceil(r.N / (N_TIME - 1))) for r in recs]
    parts = [[] for _ in recs]
    buffer = []
    was_training = model.training
    model.eval()
    t0 = time.perf_counter()

    def flush(chunk):
        merged, _ = _run_view_chunk(model, chunk, batch_views, 0.001, 0.5, 300, False)
        for idx, pred in merged.items():
            parts[idx].append(pred)

    for i, m in enumerate(ds_meta["samples"]):
        for vtype, arr, start in [("coarse", coarse[i], 0),
                                  ("fine", fine[i], int(m["fine_start"]))]:
            buffer.append({
                "index": i, "tensor": torch.from_numpy(np.asarray(arr[:channels]).copy()).float(),
                "view_type": vtype, "start": start, "F": int(m["F"]),
            })
            if len(buffer) >= batch_views:
                chunk, buffer = buffer[:batch_views], buffer[batch_views:]
                flush(chunk)
    if buffer:
        flush(buffer)
    preds = []
    for plist in parts:
        if plist:
            cat = Prediction(torch.cat([p.boxes for p in plist]),
                             torch.cat([p.scores for p in plist]),
                             torch.cat([p.labels for p in plist]))
            final = global_classwise_nms(cat, iou=0.5, max_det=100)
        else:
            final = Prediction(torch.zeros((0, 4)), torch.zeros((0,)),
                               torch.zeros((0,), dtype=torch.long))
        preds.append(ImagePredictions(final.boxes.numpy(), final.scores.numpy(),
                                      final.labels.numpy()))
    ev = evaluate(gts, preds, iou_thresholds=(0.5,), max_dets=100)
    pi = ev["per_iou"]["0.5"]
    out = {
        "mAP50": pi["map_class_aware"],
        "map50_agnostic": pi["ap_class_agnostic"],
        "localization_recall": ev["classification"]["localization_recall"],
        "classification_accuracy": ev["classification"]["classification_accuracy"],
        "n_pred": sum(len(p.boxes) for p in preds),
        "per_class": pi["per_class"],
        "confusion": ev["classification"]["confusion_gt_x_pred"],
        "groups": {},
        "val_elapsed_s": time.perf_counter() - t0,
    }
    for g in GROUPS:
        gm = evaluate_group(g, gts, preds, CLASS_GROUPS[g], recs,
                            [r.sample_id for r in recs])
        aps = [pi["per_class"].get(str(c), {}).get("ap") for c in CLASS_GROUPS[g]]
        gm["group_class_ap_mean"] = float(np.mean([a for a in aps if a is not None])) if any(a is not None for a in aps) else None
        out["groups"][g] = gm
    if was_training:
        model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="reports/complex_pilot_cache")
    ap.add_argument("--run", required=True)
    ap.add_argument("--channels", type=int, choices=[1, 3], required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-sources", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-batch-views", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--lr0", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--final-lr-frac", type=float, default=0.01)
    ap.add_argument("--clip-grad", type=float, default=10.0)
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--init-checkpoint", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(4)
    cache = ROOT / args.cache
    subset = json.loads((cache / "subset.json").read_text())
    fit_all = load_manifest(SPLIT_MANIFEST, "fit")
    val_all = load_manifest(SPLIT_MANIFEST, "val")
    by_fit = {r.sample_id: r for r in fit_all}
    fit = [by_fit[s] for s in subset["fit"]]
    train_ds = CachedPairs(cache, "fit", args.channels)
    gen = torch.Generator(); gen.manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_sources, shuffle=True,
                          generator=gen, num_workers=args.workers,
                          collate_fn=collate_pairs, pin_memory=True,
                          persistent_workers=args.workers > 0,
                          prefetch_factor=2 if args.workers else None)
    model = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"),
                        device="cuda", ch=args.channels)
    assert tuple(model.model[0].conv.weight.shape)[1] == args.channels
    if args.init_checkpoint:
        ck = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        print(f"init from {args.init_checkpoint}", flush=True)
    else:
        transfer_pretrained(model, ROOT / "yolo11s.pt", ch=args.channels)
    optimizer = torch.optim.AdamW(build_param_groups(model, args.weight_decay),
                                  lr=args.lr0, betas=(0.9, 0.999), eps=1e-8)
    run_dir = ROOT / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = math.ceil(len(fit) / args.batch_sources) // args.accum
    total_steps = steps_per_epoch * args.epochs
    config = {"run": args.run, "channels": args.channels, "cache": str(cache),
              "epochs": args.epochs, "batch_sources": args.batch_sources,
              "accum": args.accum, "steps_per_epoch": steps_per_epoch,
              "total_steps": total_steps, "lr0": args.lr0,
              "weight_decay": args.weight_decay, "seed": args.seed,
              "fit_samples": len(subset["fit"]), "val_samples": len(subset["val"]),
              "cache_scale_coarse": train_ds.meta.get("scale_coarse"),
              "cache_scale_fine": train_ds.meta.get("scale_fine"),
              "init_checkpoint": args.init_checkpoint}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)
    history = []
    t_train = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        total = 0.0; nb = 0
        accum = 0; opt_step = 0
        optimizer.zero_grad(set_to_none=True)
        for batch in train_dl:
            images = batch["img"].cuda(non_blocking=True)
            target = {"batch_idx": batch["batch_idx"].cuda(non_blocking=True),
                      "cls": batch["cls"].cuda(non_blocking=True),
                      "bboxes": batch["bboxes"].cuda(non_blocking=True),
                      "img": images}
            epoch_float = epoch + min(1.0, accum / max(1, math.ceil(len(fit)/args.batch_sources)))
            lr = lr_at(epoch_float, args.epochs, args.lr0, args.warmup, args.final_lr_frac)
            for g in optimizer.param_groups: g["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=args.precision == "bf16"):
                preds = model(images)
                parts, _ = model.loss(target, preds=preds)
                loss = parts.sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss epoch {epoch}")
            (loss / args.accum).backward()
            total += float(loss.detach()); nb += 1; accum += 1
            if accum >= args.accum:
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); accum = 0; opt_step += 1
        if accum:
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        entry = {"epoch": epoch, "loss": total / max(nb, 1),
                 "train_s": time.perf_counter() - t0,
                 "gpu_alloc_gib": torch.cuda.max_memory_allocated() / 1024**3,
                 "gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
        do_val = (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs
        if do_val:
            entry["val"] = run_validation(model, cache, "val", args.channels,
                                           val_all, args.val_batch_views)
            print(f"epoch {epoch} loss {entry['loss']:.4f} "
                  f"mAP {entry['val']['mAP50']:.4f} "
                  f"loc {entry['val']['localization_recall']:.4f} "
                  f"time {entry['train_s']:.0f}s", flush=True)
        history.append(entry)
    total_s = time.perf_counter() - t_train
    last = history[-1]
    if "val" not in last:
        last["val"] = run_validation(model, cache, "val", args.channels,
                                     val_all, args.val_batch_views)
    out = {"config": config, "history": history, "total_s": total_s,
           "final": last["val"], "peak_gpu_gib": torch.cuda.max_memory_allocated()/1024**3}
    out_path = Path(args.out) if args.out else (run_dir / "pilot_result.json")
    out_path.write_text(json.dumps(out, indent=2))
    torch.save({"model": model.state_dict(), "config": config}, run_dir / "last.pt")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
