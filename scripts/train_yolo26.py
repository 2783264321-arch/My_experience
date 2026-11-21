#!/usr/bin/env python3
"""Formal YOLO26l training on the frozen fit split (coarse + rotating fine views).

Protocol: BF16, 4 sources = 8 views per micro-batch, accum 2 (effective 16),
AdamW 1e-3 / wd 5e-4, 3-epoch warmup, cosine to 1%, EMA 0.9999/tau2000,
50 epochs, full dual-protocol validation every 5 epochs.
"""
from __future__ import annotations

import argparse, csv, json, math, shutil, sys, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/detector_upgrade"))

from adapters import build_yolo26
from src.signals.data import SPLIT_MANIFEST, load_manifest
from src.signals.train import (SharedEpochViews, build_param_groups, lr_at,
                               make_loader)
from ultralytics.utils.torch_utils import ModelEMA
from validate_dual import FINE_0_5, run_dual

RUN = "yolo26l_c1_2048x512"


def save(path: Path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=RUN)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-sources", type=int, default=4)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--val-workers", type=int, default=8)
    p.add_argument("--val-batch-views", type=int, default=8)
    p.add_argument("--lr0", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup", type=float, default=3.0)
    p.add_argument("--final-lr-frac", type=float, default=0.01)
    p.add_argument("--clip-grad", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-steps", type=int, default=0, help="smoke: stop after N optimizer steps")
    p.add_argument("--no-val", action="store_true")
    p.add_argument("--min-free-gib", type=float, default=4.0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True

    run_dir = ROOT / "runs" / args.run
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume and not args.overwrite:
        raise SystemExit(f"{run_dir} exists; use --resume/--overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)

    norm_d = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm_d["lo"]), float(norm_d["hi"]))
    fit = load_manifest(SPLIT_MANIFEST, split="fit")
    val = load_manifest(SPLIT_MANIFEST, split="val")
    ds = SharedEpochViews(fit, norm, epoch=0)
    loader = make_loader(ds, args.batch_sources, args.workers, args.seed)
    steps_per_epoch = math.ceil(len(fit) / args.batch_sources / args.accum)

    model, info = build_yolo26()
    model = model.cuda()
    model.model[-1].end2end = False   # inference uses one-to-many; training unchanged
    criterion = model.init_criterion()
    model.criterion = criterion
    optimizer = torch.optim.AdamW(build_param_groups(model, args.weight_decay),
                                  lr=args.lr0, betas=(0.9, 0.999), eps=1e-8)
    ema = ModelEMA(model, decay=0.9999, tau=2000)

    start_epoch, global_step, best_fit, best_epoch = 0, 0, -1.0, -1
    if args.resume:
        ck = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.ema.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["optimizer"])
        criterion.updates = int(ck["criterion"]["updates"])
        criterion.o2m = float(ck["criterion"]["o2m"])
        criterion.o2o = float(ck["criterion"]["o2o"])
        start_epoch = int(ck["epoch"]) + 1
        global_step = int(ck["global_step"])
        best_fit = float(ck["best_fitness"]); best_epoch = int(ck["best_epoch"])
        print(f"resumed epoch {start_epoch} step {global_step} crit_updates {criterion.updates}", flush=True)

    config = vars(args) | {"run": args.run, "fit_samples": len(fit), "val_samples": len(val),
                           "steps_per_epoch": steps_per_epoch, "effective_views": 16,
                           "model": info, "task": "yolo26l detection",
                           "checkpoint_init": "yolo26l.pt (RGB stem summed, cls heads re-init)"}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "normalizer.json").write_text(json.dumps(norm_d, indent=2))
    log_path = run_dir / "train_log.csv"
    if not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(["epoch", "step", "lr", "loss", "box", "cls", "dfl", "step_s", "time"])
    val_history = []
    hist_path = run_dir / "val_history.json"
    if args.resume and hist_path.exists():
        val_history = json.loads(hist_path.read_text())

    t_all = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        ds.shared_epoch.value = epoch
        it = iter(loader)
        t_ep = time.perf_counter()
        for step in range(steps_per_epoch):
            lr = lr_at(epoch + step / steps_per_epoch, args.epochs, args.lr0,
                       args.warmup, args.final_lr_frac)
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            accum_items = torch.zeros(3)
            t0 = time.perf_counter()
            for _ in range(args.accum):
                batch = next(it)
                img = batch["img"].cuda(non_blocking=True)
                b = {"batch_idx": batch["batch_idx"].cuda(non_blocking=True),
                     "cls": batch["cls"].float().cuda(non_blocking=True),
                     "bboxes": batch["bboxes"].float().cuda(non_blocking=True)}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    preds = model(img)
                    loss_vec, items = criterion(preds, b)
                    loss = loss_vec.sum()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss epoch {epoch} step {step}")
                (loss / args.accum).backward()
                accum_items += torch.stack([items.get("box_loss", items.get("l1_loss", 0.0)),
                                        items["cls_loss"],
                                        items.get("l1_loss", 0.0)]).detach().float().cpu()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            ema.update(model)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            global_step += 1
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow([epoch, global_step, f"{lr:.6g}", f"{float(loss.detach()):.4f}",
                                        f"{float(accum_items[0]):.4f}", f"{float(accum_items[1]):.4f}",
                                        f"{float(accum_items[2]):.4f}", f"{dt:.3f}",
                                        time.strftime("%Y-%m-%d %H:%M:%S")])
            if step % 50 == 0 or step == steps_per_epoch - 1:
                print(f"ep {epoch} step {step}/{steps_per_epoch} loss {float(loss):.3f} "
                      f"box {float(accum_items[0]):.2f} cls {float(accum_items[1]):.2f} "
                      f"dfl {float(accum_items[2]):.2f} {dt:.3f}s", flush=True)
            if args.max_steps and global_step >= args.max_steps:
                break
        criterion.update()
        ep_s = time.perf_counter() - t_ep
        print(f"epoch {epoch} done {ep_s:.0f}s crit_updates {criterion.updates} o2m {criterion.o2m:.4f}", flush=True)

        do_val = (not args.no_val) and ((epoch + 1) % args.val_every == 0 or (epoch + 1) == args.epochs)
        entry = {"epoch": epoch, "train_s": ep_s, "global_step": global_step,
                 "criterion_updates": criterion.updates, "o2m": criterion.o2m}
        if do_val:
            t_val = time.perf_counter()
            res = run_dual(ema.ema, val, norm, batch_views=args.val_batch_views,
                           workers=args.val_workers)
            entry.update({k: v for k, v in res.items() if k not in ("preds_unfiltered", "preds_filtered_0_5")})
            entry["wall_val_s"] = time.perf_counter() - t_val
            val_history.append(entry)
            hist_path.write_text(json.dumps(val_history, indent=2))
            f = res["settings"]["filtered_0_5"]; u = res["settings"]["unfiltered"]
            print(f"val ep {epoch}: filtered mAP50={f['map50']:.4f} AP50:95={f['map50_95']:.4f} "
                  f"| unfiltered mAP50={u['map50']:.4f} AP50:95={u['map50_95']:.4f} "
                  f"| walls {entry['wall_val_s']:.0f}s", flush=True)
            if f["map50"] > best_fit:
                best_fit, best_epoch = f["map50"], epoch
                save(run_dir / "best.pt", {"model": model.state_dict(),
                                           "ema": ema.ema.state_dict(),
                                           "epoch": epoch, "global_step": global_step,
                                           "best_fitness": best_fit, "metrics": entry,
                                           "config": config})
                print(f"new best filtered mAP50={best_fit:.4f} -> best.pt", flush=True)
        free = shutil.disk_usage(run_dir).free / 2**30
        if free < args.min_free_gib:
            raise RuntimeError(f"disk free {free:.1f} GiB")
        save(run_dir / "last.pt", {"model": model.state_dict(), "ema": ema.ema.state_dict(),
                                   "optimizer": optimizer.state_dict(), "epoch": epoch,
                                   "global_step": global_step,
                                   "criterion": {"updates": criterion.updates, "o2m": criterion.o2m,
                                                 "o2o": criterion.o2o},
                                   "best_fitness": best_fit, "best_epoch": best_epoch,
                                   "metrics": entry, "config": config})
        if args.max_steps and global_step >= args.max_steps:
            break

    total_s = time.perf_counter() - t_all
    summary = {"run": args.run, "epochs": args.epochs, "stopped_epoch": epoch,
               "global_step": global_step, "best_filtered_map50": best_fit,
               "best_epoch": best_epoch, "total_s": total_s, "total_h": total_s / 3600,
               "val_history": val_history, "config": config}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"training done best filtered mAP50={best_fit:.4f} epoch {best_epoch} "
          f"total {total_s/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
