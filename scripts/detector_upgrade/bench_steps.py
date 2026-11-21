"""Model-only step benchmark on real coarse+fine views (BF16, same 4090)."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/detector_upgrade"))

from src.signals.data import SPLIT_MANIFEST, build_sample_views, load_manifest, frequency_starts

OUT = ROOT / "reports/detector_upgrade"
OUT.mkdir(parents=True, exist_ok=True)


def pick_sources(n):
    fit = load_manifest(SPLIT_MANIFEST, split="fit")
    by_b = {}
    for r in fit:
        by_b.setdefault(round(float(r.B_mhz)), []).append(r)
    pick = []
    keys = sorted(by_b)
    i = 0
    while len(pick) < n:
        b = keys[i % len(keys)]
        if by_b[b]:
            pick.append(by_b[b].pop(0))
        i += 1
    return pick


def build_views(n_sources):
    import json as _json
    norm_d = _json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm_d["lo"]), float(norm_d["hi"]))
    recs = pick_sources(n_sources)
    views = []
    for i, r in enumerate(recs):
        F = round(r.B_mhz * 700)
        starts = frequency_starts(F)
        coarse, fines, _ = build_sample_views(r, norm, [i % len(starts)])
        views += [coarse, fines[0]]
    images = torch.stack([v.tensor for v in views], 0)
    boxes = torch.cat([v.boxes for v in views]) if any(len(v.boxes) for v in views) else torch.zeros(0, 4)
    labels = torch.cat([v.labels for v in views]) if any(len(v.labels) for v in views) else torch.zeros(0, dtype=torch.long)
    bidx = torch.cat([torch.full((len(v.boxes),), k, dtype=torch.long) for k, v in enumerate(views)])
    meta = [(r.sample_id, float(r.B_mhz), float(r.T_ms), len(starts) + 1) for r in recs]
    return images, boxes, labels, bidx, meta, len(views)


def step_yolo(model, opt, images, boxes_norm, labels, bidx):
    loss, items = __import__("adapters").yolo26_loss_batch(model, images, boxes_norm, labels, bidx)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.detach()


def step_dfine(model, opt, images, boxes_cxcywh, labels, bidx):
    from adapters import dfine_targets, dfine_loss_batch
    targets = dfine_targets(boxes_cxcywh, labels, bidx, images.shape[0])
    loss, parts = dfine_loss_batch(model, images, targets)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["yolo26", "dfine"], required=True)
    ap.add_argument("--views", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "scripts/detector_upgrade"))
    import adapters

    results = []
    for nv in args.views:
        if nv % 2:
            raise SystemExit("views must be even")
        images, boxes, labels, bidx, meta, n = build_views(nv // 2)
        images = images.cuda()
        boxes = boxes.cuda(); labels = labels.cuda(); bidx = bidx.cuda()
        cxcywh = torch.cat([(boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]], 1)
        if args.kind == "yolo26":
            model, info = adapters.build_yolo26()
            model = model.cuda().train()
            model.criterion = model.init_criterion()
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            fn = lambda: step_yolo(model, opt, images, boxes, labels, bidx)
        else:
            model, info = adapters.build_dfine()
            model = model.cuda().train()
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            fn = lambda: step_dfine(model, opt, images, cxcywh, labels, bidx)

        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        times = []
        oom = False
        try:
            for i in range(args.warmup + args.iters):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = fn()
                torch.cuda.synchronize(); dt = time.perf_counter() - t0
                if i >= args.warmup:
                    times.append(dt)
        except torch.OutOfMemoryError:
            oom = True
            torch.cuda.empty_cache()
        peak_alloc = torch.cuda.max_memory_allocated() / 2**30
        peak_resv = torch.cuda.max_memory_reserved() / 2**30
        r = {"kind": args.kind, "views": n, "sources": n // 2, "oom": oom,
             "step_s_mean": (sum(times) / len(times)) if times else None,
             "step_s_min": min(times) if times else None,
             "step_s_max": max(times) if times else None,
             "loss": float(loss) if not oom and 'loss' in dir() else None,
             "peak_alloc_gib": peak_alloc, "peak_reserved_gib": peak_resv,
             "build": info, "samples": meta}
        print(json.dumps(r, ensure_ascii=False))
        results.append(r)
        (OUT / f"bench_steps_{args.kind}.json").write_text(json.dumps(results, indent=1))
        del model, opt, images, boxes, labels, bidx, cxcywh
        torch.cuda.empty_cache()
    (OUT / f"bench_steps_{args.kind}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
