#!/usr/bin/env python3
"""Verify the BF16 coarse cache against the on-the-fly path.

Checks, on the same weights and the same source batch:
  1. image bit patterns: cached float32 (expanded from bf16 bits) vs online
     float32 -> bf16;
  2. targets: boxes / labels / batch indices are identical;
  3. forward / loss / gradients: online run repeated twice (to estimate GPU
     nondeterminism) vs one cached run.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/verify_coarse_cache.py \
    --cache cache/coarse_fit_bf16.npy \
    --output reports/group_a_cache_verification.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from src.signals.data import (
    SPLIT_MANIFEST, CoarseCacheDataset, SpaceNetCoarseViews, collate_coarse,
    load_manifest,
)
from src.signals.model import build_model, transfer_pretrained

ROOT = Path(__file__).resolve().parents[1]


def flatten_tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(flatten_tensors(v))
        return out
    if isinstance(value, dict):
        out = []
        for v in value.values():
            out.extend(flatten_tensors(v))
        return out
    return []


def compare_tensors(a, b):
    if len(a) != len(b):
        return {"same_count": False, "n_a": len(a), "n_b": len(b)}
    max_abs = 0.0
    mismatched = 0
    for x, y in zip(a, b):
        if x.shape != y.shape:
            mismatched += 1
            continue
        d = (x.float() - y.float()).abs().max().item()
        max_abs = max(max_abs, d)
        if d != 0:
            mismatched += 1
    return {"n_tensors": len(a), "max_abs_diff": max_abs, "mismatched_tensors": mismatched}


def run_batch(model, batch, device, precision):
    model.zero_grad(set_to_none=True)
    images = batch["img"].to(device, non_blocking=True)
    target = {
        "batch_idx": batch["batch_idx"].to(device, non_blocking=True),
        "cls": batch["cls"].to(device, non_blocking=True),
        "bboxes": batch["bboxes"].to(device, non_blocking=True),
        "img": images,
    }
    torch.cuda.synchronize()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
        preds = model(images)
        loss_parts, _ = model.loss(target, preds=preds)
        loss = loss_parts.sum()
    loss.backward()
    torch.cuda.synchronize()
    grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.detach().float().cpu().clone()
    return {
        "loss": float(loss.detach().float().item()),
        "preds": [t.detach().float().cpu().clone() for t in flatten_tensors(preds)],
        "grads": grads,
    }


def compare_grads(a, b):
    keys = sorted(set(a) | set(b))
    max_abs = 0.0
    diff_sq = 0.0
    base_sq = 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    n_diff = 0
    for k in keys:
        if k not in a or k not in b:
            n_diff += 1
            continue
        ga, gb = a[k], b[k]
        d = ga - gb
        max_abs = max(max_abs, d.abs().max().item())
        diff_sq += d.pow(2).sum().item()
        base_sq += ga.pow(2).sum().item()
        dot += (ga * gb).sum().item()
        norm_a += ga.pow(2).sum().item()
        norm_b += gb.pow(2).sum().item()
        if not torch.equal(ga, gb):
            n_diff += 1
    rel_l2 = math.sqrt(diff_sq / max(base_sq, 1e-30))
    cosine = dot / max(math.sqrt(norm_a) * math.sqrt(norm_b), 1e-30)
    return {
        "n_params": len(keys),
        "n_different": n_diff,
        "max_abs_diff": max_abs,
        "rel_l2": rel_l2,
        "cosine": cosine,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/coarse_fit_bf16.npy")
    ap.add_argument("--model", default="configs/yolo11s_c1_directreg.yaml")
    ap.add_argument("--pretrained", default="yolo11s.pt")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--output", default="reports/group_a_cache_verification.json")
    args = ap.parse_args()

    cache = Path(args.cache)
    if not cache.is_absolute():
        cache = ROOT / cache
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    norm_cfg = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm_cfg["lo"]), float(norm_cfg["hi"]))
    records = load_manifest(SPLIT_MANIFEST, split="fit")
    rng = random.Random(args.seed)
    picks = rng.sample(range(len(records)), args.samples)

    online_ds = SpaceNetCoarseViews(records, norm)
    cached_ds = CoarseCacheDataset(records, norm, cache)
    online = collate_coarse([online_ds[i] for i in picks])
    cached = collate_coarse([cached_ds[i] for i in picks])

    image_check = {
        "max_abs_diff": float((online["img"] - cached["img"]).abs().max()),
        "bf16_bit_exact": bool(torch.equal(
            online["img"].to(torch.bfloat16), cached["img"].to(torch.bfloat16)
        )),
        "shapes": [list(online["img"].shape), list(cached["img"].shape)],
    }
    target_check = {
        "batch_idx_exact": bool(torch.equal(online["batch_idx"], cached["batch_idx"])),
        "cls_exact": bool(torch.equal(online["cls"], cached["cls"])),
        "bboxes_max_abs_diff": float((online["bboxes"] - cached["bboxes"]).abs().max())
        if online["bboxes"].numel() else 0.0,
        "bboxes_exact": bool(torch.equal(online["bboxes"], cached["bboxes"])),
    }

    device = "cuda"
    model = build_model(str(ROOT / args.model), device=device)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = ROOT / ckpt_path
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["ema"] if ckpt.get("ema") is not None else ckpt["model"]
        model.load_state_dict(sd)
        init = {"checkpoint": str(ckpt_path)}
    else:
        init = transfer_pretrained(model, ROOT / args.pretrained, verbose=False)
        init = {"pretrained": str(ROOT / args.pretrained), **init}
    model.train()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    t0 = time.perf_counter()
    online_a = run_batch(model, online, device, "bf16")
    online_b = run_batch(model, online, device, "bf16")
    cached_run = run_batch(model, cached, device, "bf16")

    repeat = {
        "loss_abs_diff": abs(online_a["loss"] - online_b["loss"]),
        "preds": compare_tensors(online_a["preds"], online_b["preds"]),
        "grads": compare_grads(online_a["grads"], online_b["grads"]),
    }
    cache_vs_online = {
        "loss_abs_diff": abs(online_a["loss"] - cached_run["loss"]),
        "loss_online": online_a["loss"],
        "loss_cached": cached_run["loss"],
        "preds": compare_tensors(online_a["preds"], cached_run["preds"]),
        "grads": compare_grads(online_a["grads"], cached_run["grads"]),
    }

    loss_tol = max(1e-4, repeat["loss_abs_diff"] * 2 + 1e-7)
    pred_tol = max(1e-3, repeat["preds"]["max_abs_diff"] * 2 + 1e-6)
    grad_tol = max(1e-4, repeat["grads"]["rel_l2"] * 2 + 1e-7)
    passed = (
        image_check["bf16_bit_exact"]
        and target_check["bboxes_exact"]
        and target_check["batch_idx_exact"]
        and target_check["cls_exact"]
        and cache_vs_online["loss_abs_diff"] <= loss_tol
        and cache_vs_online["preds"]["max_abs_diff"] <= pred_tol
        and cache_vs_online["grads"]["rel_l2"] <= grad_tol
    )

    result = {
        "pass": bool(passed),
        "cache": str(cache),
        "init": init,
        "sample_indices": picks,
        "sample_ids": [records[i].sample_id for i in picks],
        "precision": "bf16",
        "images": image_check,
        "targets": target_check,
        "online_repeat_noise": repeat,
        "cache_vs_online": cache_vs_online,
        "tolerances": {
            "loss_abs_diff": loss_tol,
            "preds_max_abs_diff": pred_tol,
            "grad_rel_l2": grad_tol,
        },
        "elapsed_s": time.perf_counter() - t0,
    }
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    if not passed:
        raise SystemExit("cache verification failed: inspect the JSON before training")


if __name__ == "__main__":
    main()
