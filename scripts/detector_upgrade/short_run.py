"""Short pipeline-sanity run: 200 optimizer steps, checkpoint, reload, inference."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/detector_upgrade"))
from src.signals.data import (SPLIT_MANIFEST, SpaceNetViews, build_sample_views,
                              collate_views, frequency_starts, load_manifest)
from src.signals.infer import _boundary_mask, _map_view_dets, global_classwise_nms, Prediction
from ultralytics.utils.nms import non_max_suppression

OUT = ROOT / "reports/detector_upgrade"; OUT.mkdir(parents=True, exist_ok=True)


def winit(_): torch.set_num_threads(1)


def make_loader(ds, bs, workers):
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers,
                      collate_fn=collate_views, pin_memory=True, drop_last=True,
                      prefetch_factor=4, persistent_workers=True, worker_init_fn=winit)


def micro(batch, a, b):
    img = batch["img"][a:b].cuda(non_blocking=True)
    m = (batch["batch_idx"] >= a) & (batch["batch_idx"] < b)
    boxes = batch["bboxes"][m].cuda(non_blocking=True)
    labels = batch["cls"][m, 0].long().cuda(non_blocking=True)
    return img, boxes, labels, batch["batch_idx"][m].cuda(non_blocking=True) - a


def step(model, opt, kind, batch, n_micro, timed):
    import adapters
    opt.zero_grad(set_to_none=True)
    tf = tl = tb = 0.0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for k in range(n_micro):
            a, b = k * 8, (k + 1) * 8
            img, boxes, labels, bidx = micro(batch, a, b)
            t0 = time.perf_counter()
            if kind == "yolo26":
                loss, _ = adapters.yolo26_loss_batch(model, img, boxes, labels, bidx)
                t1 = time.perf_counter()
            else:
                from adapters import dfine_targets
                tg = dfine_targets(boxes, labels, bidx, img.shape[0])
                out = model(img, targets=tg)
                t1 = time.perf_counter()
                loss = sum(model.criterion(out, tg).values())
            t2 = time.perf_counter()
            (loss / n_micro).backward()
            t3 = time.perf_counter()
            tf += t1 - t0; tl += t2 - t1; tb += t3 - t2
    opt.step()
    return float(loss.detach()), tf, tl, tb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["yolo26", "dfine"], required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    import adapters

    norm = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm["lo"]), float(norm["hi"]))
    fit = load_manifest(SPLIT_MANIFEST, split="fit")
    loader = make_loader(SpaceNetViews(fit, norm, epoch=0), 8, args.workers)

    if args.kind == "yolo26":
        model, info = adapters.build_yolo26(); model = model.cuda().train()
        model.criterion = model.init_criterion()
    else:
        model, info = adapters.build_dfine(); model = model.cuda().train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    it = iter(loader); losses = []; tf = tl = tb = 0.0
    t_start = time.perf_counter()
    for s in range(args.steps):
        try: batch = next(it)
        except StopIteration: it = iter(loader); batch = next(it)
        loss, a, b, c = step(model, opt, args.kind, batch, 2, True)
        losses.append(loss); tf += a; tl += b; tb += c
        if (s + 1) % 50 == 0:
            print(f"step {s+1} loss {sum(losses[-50:])/50:.4f} "
                  f"fwd {tf/50:.3f}s loss {tl/50:.3f}s bwd {tb/50:.3f}s", flush=True)
            tf = tl = tb = 0.0
    wall = time.perf_counter() - t_start

    ckpt = {"model": {k: v.cpu() for k, v in model.state_dict().items()},
            "kind": args.kind, "steps": args.steps, "lr": args.lr,
            "loss_first50": sum(losses[:50]) / 50, "loss_last50": sum(losses[-50:]) / 50,
            "build": info}
    if args.kind == "yolo26":
        ckpt["criterion_updates"] = int(model.criterion.updates)
        ckpt["o2m"] = float(model.criterion.o2m)
    path = OUT / f"short_{args.kind}.pt"
    torch.save(ckpt, path)

    # rebuild + load + inference on a few val samples
    if args.kind == "yolo26":
        model2, _ = adapters.build_yolo26(); model2.criterion = model2.init_criterion()
        model2.load_state_dict(ckpt["model"], strict=True)
        model2 = model2.cuda().eval(); model2.model[-1].end2end = False
        def infer(imgs):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                return [d.float() for d in non_max_suppression(model2(imgs), 0.001, 0.5, nc=14, max_det=300, max_nms=30000, max_time_img=20)]
    else:
        model2, _ = adapters.build_dfine(); model2.load_state_dict(ckpt["model"], strict=True)
        model2 = model2.cuda().eval(); post = model2.postprocessor
        def infer(imgs):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                res = post(model2(imgs), torch.tensor([[512, 2048]] * imgs.shape[0], device=imgs.device))
            return [torch.cat([r["boxes"], r["scores"][:, None], r["labels"][:, None].float()], 1).float() for r in res]

    val = load_manifest(SPLIT_MANIFEST, split="val")[:4]
    counts = []
    for r in val:
        F = round(r.B_mhz * 700); starts = frequency_starts(F)
        coarse, fines, _ = build_sample_views(r, norm, list(range(len(starts))))
        imgs = torch.stack([v.tensor for v in [coarse] + fines]).cuda()
        dets = infer(imgs)
        counts.append(int(sum(len(d) for d in dets)))
    print(json.dumps({"kind": args.kind, "wall_s": wall, "steps": args.steps,
                      "loss_first50": sum(losses[:50])/50, "loss_last50": sum(losses[-50:])/50,
                      "checkpoint": str(path), "infer_dets_4_samples": counts,
                      "criterion_updates": ckpt.get("criterion_updates")}, ensure_ascii=False))
    (OUT / f"short_{args.kind}.json").write_text(json.dumps({"kind": args.kind, "wall_s": wall,
        "loss_first50": sum(losses[:50])/50, "loss_last50": sum(losses[-50:])/50,
        "infer_dets_4_samples": counts, "checkpoint": str(path)}, indent=1))


if __name__ == "__main__":
    main()
