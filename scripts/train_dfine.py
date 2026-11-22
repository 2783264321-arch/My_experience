#!/usr/bin/env python3
"""Formal D-FINE-L training on the frozen fit split (coarse + rotating fine views)."""
from __future__ import annotations
import argparse, csv, json, math, shutil, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/detector_upgrade"))
from adapters import build_dfine, dfine_targets
from src.signals.data import SPLIT_MANIFEST, load_manifest
from src.signals.train import SharedEpochViews, lr_at, make_loader
from ultralytics.utils.torch_utils import ModelEMA
from validate_dual import run_dual

RUN = "dfine_l_c1_2048x512"


def save(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp"); torch.save(payload, tmp); tmp.replace(path)


def param_groups(model, base_lr, backbone_lr, wd):
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        lr = backbone_lr if n.startswith("backbone") else base_lr
        no_decay = p.ndim <= 1 or n.endswith(".bias") or "norm" in n or "bn" in n
        key = (lr, 0.0 if no_decay else wd)
        groups.setdefault(key, []).append(p)
    return [{"params": ps, "lr": lr, "base_lr": lr, "weight_decay": w} for (lr, w), ps in groups.items()]


def make_infer(model):
    post = model.postprocessor
    def infer(images):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(images)
            res = post(out, torch.tensor([[512, 2048]] * images.shape[0], device=images.device))
        return [torch.cat([r["boxes"], r["scores"][:, None], r["labels"][:, None].float()], 1).float() for r in res]
    return infer


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
    p.add_argument("--lr0", type=float, default=2.5e-4)
    p.add_argument("--backbone-lr", type=float, default=1.25e-5)
    p.add_argument("--weight-decay", type=float, default=1.25e-4)
    p.add_argument("--warmup", type=float, default=3.0)
    p.add_argument("--final-lr-frac", type=float, default=0.01)
    p.add_argument("--clip-grad", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--no-val", action="store_true")
    p.add_argument("--min-free-gib", type=float, default=4.0)
    a = p.parse_args()

    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    run_dir = ROOT / "runs" / a.run
    if run_dir.exists() and any(run_dir.iterdir()) and not a.resume and not a.overwrite:
        raise SystemExit(f"{run_dir} exists; use --resume/--overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)

    norm_d = json.loads((ROOT / "configs/normalizer.json").read_text()); norm = (norm_d["lo"], norm_d["hi"])
    fit = load_manifest(SPLIT_MANIFEST, split="fit"); val = load_manifest(SPLIT_MANIFEST, split="val")
    ds = SharedEpochViews(fit, norm, epoch=0)
    loader = make_loader(ds, a.batch_sources, a.workers, a.seed)
    steps_per_epoch = math.ceil(len(fit) / a.batch_sources / a.accum)

    model, info = build_dfine()
    model = model.cuda(); criterion = model.criterion
    optimizer = torch.optim.AdamW(param_groups(model, a.lr0, a.backbone_lr, a.weight_decay),
                                  betas=(0.9, 0.999), eps=1e-8)
    ema = ModelEMA(model, decay=0.9999, tau=2000)

    start_epoch, global_step, best_fit, best_epoch = 0, 0, -1.0, -1
    if a.resume:
        ck = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); ema.ema.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1; global_step = ck["global_step"]
        best_fit = ck["best_fitness"]; best_epoch = ck["best_epoch"]
        print(f"resumed epoch {start_epoch} step {global_step}", flush=True)

    config = vars(a) | {"run": a.run, "fit_samples": len(fit), "val_samples": len(val),
                        "steps_per_epoch": steps_per_epoch, "effective_views": 16,
                        "model": info, "freeze": "upstream freeze_stem_only; new 1ch stem conv trainable",
                        "init": "dfine_l_obj2coco_e25.pth (RGB stem summed, heads re-init)"}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "normalizer.json").write_text(json.dumps(norm_d, indent=2))
    log = run_dir / "train_log.csv"
    if not log.exists():
        with log.open("w", newline="") as f:
            csv.writer(f).writerow(["epoch", "step", "lr", "loss", "vfl", "bbox", "giou", "step_s", "time"])
    hist_path = run_dir / "val_history.json"
    val_history = json.loads(hist_path.read_text()) if a.resume and hist_path.exists() else []
    t_all = time.perf_counter()

    for epoch in range(start_epoch, a.epochs):
        ds.shared_epoch.value = epoch
        it = iter(loader); t_ep = time.perf_counter()
        for step in range(steps_per_epoch):
            mult = lr_at(epoch + step / steps_per_epoch, a.epochs, 1.0, a.warmup, a.final_lr_frac)
            for g in optimizer.param_groups: g["lr"] = g["base_lr"] * mult
            optimizer.zero_grad(set_to_none=True)
            acc = torch.zeros(3); t0 = time.perf_counter()
            for _ in range(a.accum):
                batch = next(it)
                img = batch["img"].cuda(non_blocking=True)
                tg = dfine_targets(batch["bboxes"].cuda(non_blocking=True),
                                   batch["cls"][:, 0].long().cuda(non_blocking=True),
                                   batch["batch_idx"].cuda(non_blocking=True), img.shape[0])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(img, targets=tg)
                    ld = criterion(out, tg)
                    loss = sum(ld.values())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss ep {epoch} step {step}")
                (loss / a.accum).backward()
                acc += torch.stack([ld.get("loss_vfl", ld.get("loss_bbox", 0.0)),
                                    ld.get("loss_bbox", 0.0), ld.get("loss_giou", 0.0)]).detach().float().cpu()
            if a.clip_grad > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad)
            optimizer.step(); ema.update(model); torch.cuda.synchronize(); dt = time.perf_counter() - t0
            global_step += 1
            with log.open("a", newline="") as f:
                csv.writer(f).writerow([epoch, global_step, f"{optimizer.param_groups[0]['lr']:.6g}",
                                        f"{float(loss.detach()):.4f}", f"{float(acc[0]):.4f}",
                                        f"{float(acc[1]):.4f}", f"{float(acc[2]):.4f}",
                                        f"{dt:.3f}", time.strftime("%Y-%m-%d %H:%M:%S")])
            if step % 50 == 0 or step == steps_per_epoch - 1:
                print(f"ep {epoch} step {step}/{steps_per_epoch} loss {float(loss.detach()):.3f} "
                      f"vfl {float(acc[0]):.2f} bbox {float(acc[1]):.2f} giou {float(acc[2]):.2f} {dt:.3f}s", flush=True)
            if a.max_steps and global_step >= a.max_steps: break
        ep_s = time.perf_counter() - t_ep
        print(f"epoch {epoch} done {ep_s:.0f}s", flush=True)
        do_val = (not a.no_val) and ((epoch + 1) % a.val_every == 0 or (epoch + 1) == a.epochs)
        entry = {"epoch": epoch, "train_s": ep_s, "global_step": global_step}
        if do_val:
            tv = time.perf_counter()
            res = run_dual(ema.ema, val, norm, batch_views=a.val_batch_views,
                           workers=a.val_workers, infer_fn=make_infer(ema.ema))
            entry.update({k: v for k, v in res.items() if k not in ("preds_unfiltered", "preds_filtered_0_5")})
            entry["wall_val_s"] = time.perf_counter() - tv
            val_history.append(entry); hist_path.write_text(json.dumps(val_history, indent=2))
            f = res["settings"]["filtered_0_5"]; u = res["settings"]["unfiltered"]
            print(f"val ep {epoch}: filtered {f['map50']:.4f}/{f['map50_95']:.4f} "
                  f"unfiltered {u['map50']:.4f}/{u['map50_95']:.4f} walls {entry['wall_val_s']:.0f}s", flush=True)
            if f["map50"] > best_fit:
                best_fit, best_epoch = f["map50"], epoch
                save(run_dir / "best.pt", {"model": model.state_dict(), "ema": ema.ema.state_dict(),
                                           "epoch": epoch, "global_step": global_step,
                                           "best_fitness": best_fit, "metrics": entry, "config": config})
                print(f"new best filtered mAP50={best_fit:.4f} -> best.pt", flush=True)
        free = shutil.disk_usage(run_dir).free / 2**30
        if free < a.min_free_gib: raise RuntimeError(f"disk free {free:.1f} GiB")
        save(run_dir / "last.pt", {"model": model.state_dict(), "ema": ema.ema.state_dict(),
                                   "optimizer": optimizer.state_dict(), "epoch": epoch,
                                   "global_step": global_step, "best_fitness": best_fit,
                                   "best_epoch": best_epoch, "metrics": entry, "config": config})
        if a.max_steps and global_step >= a.max_steps: break

    total = time.perf_counter() - t_all
    peak = torch.cuda.max_memory_reserved() / 2**30
    print(f"peak reserved {peak:.2f} GiB", flush=True)
    (run_dir / "summary.json").write_text(json.dumps({"run": a.run, "epochs": a.epochs, "peak_reserved_gib": peak,
        "stopped_epoch": epoch, "global_step": global_step, "best_filtered_map50": best_fit,
        "best_epoch": best_epoch, "total_h": total / 3600, "val_history": val_history,
        "config": config}, indent=2))
    print(f"done best {best_fit:.4f} epoch {best_epoch} total {total/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
