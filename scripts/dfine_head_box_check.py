#!/usr/bin/env python3
"""Verify frozen boxes/queries: raw pred_boxes must match round25 bitwise."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/detector_upgrade"))
from adapters import build_dfine
from src.signals.data import SPLIT_MANIFEST, load_manifest
from src.signals.validate import FullViewDataset, _collate_one

BASE = Path("/root/autodl-tmp/worktrees/formal-dfine/runs/dfine_l_c1_2048x512/best.pt")


def load(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["ema"] if ck.get("ema") is not None else ck["model"]
    model, _ = build_dfine()
    model.load_state_dict(sd)
    return model.to(device).eval(), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("control", "group", "group_normfix"))
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--batch-views", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    ck_path = ROOT / f"runs/dfine_head_finetune/{args.arm}/last.pt"
    torch.manual_seed(0)
    base, _ = load(BASE, args.device)
    arm, ck = load(ck_path, args.device)

    norm_d = json.loads((ROOT / "configs/normalizer.json").read_text())
    records = load_manifest(SPLIT_MANIFEST, split="val")[:args.samples]
    ds = FullViewDataset(records, (norm_d["lo"], norm_d["hi"]), view_mode="all")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers,
                        collate_fn=_collate_one, pin_memory=False,
                        prefetch_factor=4 if args.workers else None,
                        persistent_workers=args.workers > 0)
    buf = []
    n_views = 0
    max_box = 0.0
    equal_boxes = True
    max_logit = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for item in loader:
            buf.extend(item["tensors"])
            while len(buf) >= args.batch_views:
                chunk, buf = buf[:args.batch_views], buf[args.batch_views:]
                img = torch.stack(chunk).to(args.device)
                o1, o2 = base(img), arm(img)
                b1, b2 = o1["pred_boxes"].float(), o2["pred_boxes"].float()
                max_box = max(max_box, float((b1 - b2).abs().max()))
                equal_boxes &= bool(torch.equal(b1, b2))
                max_logit = max(max_logit, float((o1["pred_logits"].float()
                                                  - o2["pred_logits"].float()).abs().max()))
                n_views += len(chunk)
        if buf:
            img = torch.stack(buf).to(args.device)
            o1, o2 = base(img), arm(img)
            b1, b2 = o1["pred_boxes"].float(), o2["pred_boxes"].float()
            max_box = max(max_box, float((b1 - b2).abs().max()))
            equal_boxes &= bool(torch.equal(b1, b2))
            max_logit = max(max_logit, float((o1["pred_logits"].float()
                                              - o2["pred_logits"].float()).abs().max()))
            n_views += len(buf)
    out = {"arm": args.arm, "samples": len(records), "views": n_views,
           "boxes_bitwise_equal": equal_boxes, "box_max_abs_diff": max_box,
           "logit_max_abs_diff": max_logit, "checkpoint": str(ck_path),
           "epoch": ck.get("epoch")}
    path = ROOT / f"runs/dfine_head_finetune/{args.arm}_box_check.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
