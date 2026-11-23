#!/usr/bin/env python3
"""D-FINE-L classification-head short fine-tune (control vs Wi-Fi20 group CE).

Only decoder classification heads are trainable. Backbone/encoder, query
selection, box heads and merge remain frozen. Two arms start from round25:
  control  : original D-FINE criterion only;
  experiment: original criterion + 0/1/2 group CE on queries matched to
              Wi-Fi20 GT (IoU>=0.5).
No test use.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import box_convert, box_iou

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/detector_upgrade"))
from adapters import build_dfine, dfine_targets
from src.signals.data import SPLIT_MANIFEST, load_manifest
from src.signals.train import SharedEpochViews, lr_at, make_loader

CKPT = Path("/root/autodl-tmp/worktrees/formal-dfine/runs/dfine_l_c1_2048x512/best.pt")
OUT = ROOT / "runs/dfine_head_finetune"
WIFI20 = (0, 1, 2)


def load_round25(device):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ck["ema"] if ck.get("ema") is not None else ck["model"]
    model, info = build_dfine()
    model.load_state_dict(state)
    return model.to(device), ck, info


def freeze_except_class_head(model):
    for p in model.parameters():
        p.requires_grad_(False)
    for head in model.decoder.dec_score_head:
        for p in head.parameters():
            p.requires_grad_(True)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()


def group_loss(out, targets, device):
    total = 0.0
    count = 0
    for b, t in enumerate(targets):
        labels = t["labels"]
        if len(labels) == 0:
            continue
        mask = torch.isin(labels, torch.tensor(WIFI20, device=device))
        if not mask.any():
            continue
        gb = box_convert(t["boxes"][mask].float(), "cxcywh", "xyxy")
        pb = box_convert(out["pred_boxes"][b].float(), "cxcywh", "xyxy")
        ious = box_iou(pb, gb)
        pairs = torch.nonzero(ious >= 0.5, as_tuple=False)
        if len(pairs) == 0:
            continue
        order = torch.argsort(ious[pairs[:, 0], pairs[:, 1]], descending=True)
        pairs = pairs[order]
        used_q = set(); used_g = set(); sel_q = []; sel_y = []
        for q, g in pairs.tolist():
            if q in used_q or g in used_g:
                continue
            used_q.add(q); used_g.add(g)
            sel_q.append(q); sel_y.append(int(labels[mask][g].item()))
        if sel_q:
            logits = out["pred_logits"][b][sel_q][:, :3].float()
            y = torch.tensor(sel_y, device=device, dtype=torch.long)
            total = total + F.cross_entropy(logits, y)
            count += len(sel_q)
    return (total / max(count, 1), count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("control", "group"), required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-sources", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = args.out or (OUT / args.arm)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model, ck, info = load_round25(args.device)
    model.train()
    freeze_except_class_head(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params {sum(p.numel() for p in trainable):,}", flush=True)

    norm_d = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (norm_d["lo"], norm_d["hi"])
    fit = load_manifest(SPLIT_MANIFEST, split="fit")
    ds = SharedEpochViews(fit, norm, epoch=0)
    loader = make_loader(ds, args.batch_sources, args.workers, args.seed)
    steps = math.ceil(len(fit) / args.batch_sources / args.accum)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    criterion = model.criterion

    # Fixed inference batch for box-invariance check.
    probe = next(iter(loader))
    probe_img = probe["img"].to(args.device)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        base_boxes = model(probe_img)["pred_boxes"].float().cpu()
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()

    log_path = out / "train_log.csv"
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "lr", "loss", "group_loss", "group_n", "step_s", "time"])
    t0 = time.perf_counter(); global_step = 0
    for epoch in range(args.epochs):
        ds.shared_epoch.value = epoch
        it = iter(loader)
        for step in range(steps):
            if args.max_steps and global_step >= args.max_steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(epoch + step / steps, args.epochs, args.lr, 0.0, 0.1)
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0; group_sum = 0.0; group_n = 0
            ts = time.perf_counter()
            for _ in range(args.accum):
                batch = next(it)
                img = batch["img"].to(args.device, non_blocking=True)
                tg = dfine_targets(batch["bboxes"].to(args.device, non_blocking=True),
                                   batch["cls"][:, 0].long().to(args.device, non_blocking=True),
                                   batch["batch_idx"].to(args.device, non_blocking=True),
                                   img.shape[0])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out_d = model(img, targets=tg)
                    ld = criterion(out_d, tg)
                    loss = sum(ld.values())
                    gl, gn = (0.0, 0)
                    if args.arm == "group":
                        gl, gn = group_loss(out_d, tg, args.device)
                        loss = loss + gl
                loss_sum += float(loss.detach()); group_sum += float(gl); group_n += gn
                (loss / args.accum).backward()
            torch.nn.utils.clip_grad_norm_(trainable, 0.1)
            opt.step()
            global_step += 1
            dt = time.perf_counter() - ts
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow([epoch, global_step, f"{opt.param_groups[0]['lr']:.6g}",
                                        f"{loss_sum/args.accum:.4f}", f"{group_sum/args.accum:.4f}",
                                        group_n, f"{dt:.3f}", time.strftime("%Y-%m-%d %H:%M:%S")])
            if step % 50 == 0 or step == steps - 1:
                print(f"{args.arm} ep {epoch} step {step}/{steps} loss {loss_sum/args.accum:.4f} "
                      f"group {group_sum/args.accum:.4f} n {group_n} {dt:.3f}s", flush=True)
        if args.max_steps and global_step >= args.max_steps:
            break
        print(f"{args.arm} epoch {epoch} done {time.perf_counter()-t0:.0f}s", flush=True)

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        new_boxes = model(probe_img)["pred_boxes"].float().cpu()
    box_diff = float((new_boxes - base_boxes).abs().max())
    state = {"model": model.state_dict(), "ema": model.state_dict(),
             "epoch": epoch, "global_step": global_step,
             "best_fitness": float(ck.get("best_fitness", 0.0)),
             "config": {"arm": args.arm, "init": str(CKPT), "epochs": args.epochs,
                        "lr": args.lr, "box_max_abs_diff": box_diff}}
    torch.save(state, out / "last.pt")
    summary = {"arm": args.arm, "init": str(CKPT), "epochs": args.epochs,
               "steps": global_step, "seconds": time.perf_counter() - t0,
               "trainable_params": int(sum(p.numel() for p in trainable)),
               "box_max_abs_diff": box_diff, "model": info,
               "final_loss": float(loss_sum / args.accum),
               "final_group_loss": float(group_sum / args.accum),
               "final_group_n": int(group_n)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
