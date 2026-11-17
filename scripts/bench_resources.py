#!/usr/bin/env python3
"""Stage-2 resource benchmark on real 2048x512 views.

Measures optimizer-step time and reserved GPU memory for physical source
batches (2 views per source), with AMP and EMA, and reports the data pipeline
throughput separately.
"""
import argparse
import json
import random
import time

import torch
from torch.utils.data import DataLoader

import src.signals.data as D
from src.signals.model import build_model, transfer_pretrained
from src.signals.train import build_param_groups, worker_init
from ultralytics.utils.torch_utils import ModelEMA


def make_real_batch(records, normalizer, batch_sources, workers, seed=1729):
    ds = D.SpaceNetViews(records, normalizer, epoch=0)
    dl = DataLoader(
        ds, batch_size=batch_sources, shuffle=False, num_workers=workers,
        collate_fn=D.collate_views, pin_memory=True, prefetch_factor=2,
        worker_init_fn=worker_init, drop_last=True,
    )
    batch = next(iter(dl))
    return batch


def subset_batch(batch, n_sources):
    n_views = n_sources * 2
    img = batch["img"][:n_views].cuda()
    keep = batch["batch_idx"] < n_views
    target = {
        "img": img,
        "batch_idx": batch["batch_idx"][keep].cuda(),
        "cls": batch["cls"][keep].cuda(),
        "bboxes": batch["bboxes"][keep].cuda(),
    }
    return target, n_views


def bench_step(model, opt, scaler, ema, target, amp, accum=1):
    images = target["img"]
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(accum):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            preds = model(images)
            loss_parts, items = model.loss(target, preds=preds)
            loss = loss_parts.sum()
        scaler.scale(loss / accum).backward()
    if True:
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)
    torch.cuda.synchronize()
    return time.perf_counter() - t, float(loss.detach()), float(gn), items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/stage2_resources.json")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sizes", default="2,4,6,8,12")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    n = json.load(open("configs/normalizer.json"))
    norm = (n["lo"], n["hi"])
    recs = D.load_manifest(split="fit")
    rng = random.Random(1729)
    pool = rng.sample(recs, 96)

    # data pipeline throughput (cold-ish, 96 sources)
    data_stats = {}
    for workers in [4, 8, 12, 16]:
        for fft_threads in ([1, 2] if D.STFT_BACKEND == "scipy" else [1]):
            D.FFT_THREADS = fft_threads
            ds = D.SpaceNetViews(pool[:64], norm, epoch=0)
            dl = DataLoader(
                ds, batch_size=8, shuffle=False, num_workers=workers,
                collate_fn=D.collate_views, pin_memory=True, prefetch_factor=2,
                worker_init_fn=worker_init, drop_last=True,
            )
            t = time.perf_counter()
            nsrc = 0
            for b in dl:
                nsrc += b["img"].shape[0] // 2
            dt = time.perf_counter() - t
            data_stats[f"workers{workers}_fft{fft_threads}"] = {
                "sources": nsrc, "seconds": dt, "sources_per_s": nsrc / dt,
                "epoch_min_5957": 5957 / (nsrc / dt) / 60,
            }
            print("data", workers, fft_threads, round(nsrc / dt, 2), "src/s", flush=True)
    D.FFT_THREADS = 2

    batch = make_real_batch(pool, norm, 16, workers=args.workers)
    print("real batch views", batch["img"].shape, "boxes", batch["bboxes"].shape, flush=True)
    results = {}
    sizes = [int(x) for x in args.sizes.split(",")]
    for size in sizes:
        if size * 2 > batch["img"].shape[0]:
            continue
        target, n_views = subset_batch(batch, size)
        model = build_model()
        transfer_pretrained(model, "yolo11s.pt", verbose=False)
        opt = torch.optim.AdamW(build_param_groups(model, 5e-4), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        ema = ModelEMA(model)
        model.train()
        for _ in range(2):
            bench_step(model, opt, scaler, ema, target, amp=True)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.repeats):
            dt, loss, gn, items = bench_step(model, opt, scaler, ema, target, amp=True)
            times.append(dt)
        peak_alloc = torch.cuda.max_memory_allocated() / 1024 ** 3
        peak_res = torch.cuda.max_memory_reserved() / 1024 ** 3
        results[str(size)] = {
            "views": n_views,
            "step_s_median": sorted(times)[len(times) // 2],
            "step_s_all": times,
            "loss": loss,
            "grad_norm": gn,
            "loss_items": {k: float(v) for k, v in items.items()} if isinstance(items, dict) else None,
            "peak_alloc_gib": peak_alloc,
            "peak_reserved_gib": peak_res,
            "views_per_s": n_views / sorted(times)[len(times) // 2],
        }
        print(size, results[str(size)], flush=True)
        del model, opt, scaler, ema, target
        torch.cuda.empty_cache()

    out = {"amp": True, "data": data_stats, "steps": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
