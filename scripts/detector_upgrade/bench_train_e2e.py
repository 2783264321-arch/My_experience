"""End-to-end training-step benchmark with the project's real data loader."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/detector_upgrade"))
from src.signals.data import (SPLIT_MANIFEST, SpaceNetViews, collate_views, load_manifest)

OUT = ROOT / "reports/detector_upgrade"; OUT.mkdir(parents=True, exist_ok=True)


def winit(_):
    torch.set_num_threads(1)


def make_loader(ds, bs, workers):
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers,
                      collate_fn=collate_views, pin_memory=True, drop_last=True,
                      prefetch_factor=4 if workers else None,
                      persistent_workers=workers > 0, worker_init_fn=winit if workers else None)


def micro_batch(batch, a, b):
    img = batch["img"][a:b].cuda(non_blocking=True)
    bidx = batch["batch_idx"]
    m = (bidx >= a) & (bidx < b)
    boxes = batch["bboxes"][m].cuda(non_blocking=True)      # normalized cxcywh
    labels = batch["cls"][m, 0].long().cuda(non_blocking=True)
    return img, boxes, labels, bidx[m].cuda(non_blocking=True) - a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["yolo26", "dfine"], required=True)
    ap.add_argument("--sources", type=int, default=8)
    ap.add_argument("--micro", type=int, default=4, help="sources per micro-batch (4 -> 8 views)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()
    import adapters

    norm_d = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm_d["lo"]), float(norm_d["hi"]))
    fit = load_manifest(SPLIT_MANIFEST, split="fit")
    ds = SpaceNetViews(fit, norm, epoch=0)
    loader = make_loader(ds, args.sources, args.workers)

    if args.kind == "yolo26":
        model, info = adapters.build_yolo26(); model = model.cuda().train()
        model.criterion = model.init_criterion()
    else:
        model, info = adapters.build_dfine(); model = model.cuda().train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    n_micro = args.sources // args.micro
    it = iter(loader)
    times, data_t, comp_t = [], 0.0, 0.0
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    for step in range(args.warmup + args.steps):
        t0 = time.perf_counter()
        batch = next(it)
        t1 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for k in range(n_micro):
                a, b = k * args.micro * 2, (k + 1) * args.micro * 2   # 2 views per source
                img, boxes, labels, bidx = micro_batch(batch, a, b)
                if args.kind == "yolo26":
                    loss, _ = adapters.yolo26_loss_batch(model, img, boxes, labels, bidx)
                else:
                    from adapters import dfine_loss_batch, dfine_targets
                    tg = dfine_targets(boxes, labels, bidx, img.shape[0])
                    loss, _ = dfine_loss_batch(model, img, tg)
                (loss / n_micro).backward()
            opt.step()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if step >= args.warmup:
            times.append(t2 - t1); comp_t += t2 - t1; data_t += t1 - t0

    mean = sum(times) / len(times)
    res = {"kind": args.kind, "sources_per_step": args.sources,
           "views_per_step": args.sources * 2, "micro_sources": args.micro,
           "micro_views": args.micro * 2, "accum": n_micro,
           "workers": args.workers, "step_s_mean": mean,
           "data_s_mean": data_t / len(times), "compute_s_mean": comp_t / len(times),
           "steps_per_epoch": 745, "epoch_s": mean * 745,
           "train_50ep_s": mean * 745 * 50,
           "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
           "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
           "build": info}
    print(json.dumps(res, ensure_ascii=False))
    (OUT / f"bench_train_e2e_{args.kind}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
