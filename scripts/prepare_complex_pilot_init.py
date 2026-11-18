#!/usr/bin/env python3
"""Create C1/C3 pilot initializations with an identical non-stem state.

C1: RGB stem summed to one power channel.
C3: power channel gets the same stem; Re/Im stem channels are zero.
Every non-stem parameter is copied from the same C1 construction, so the two
arms start with identical power-path outputs and losses.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from src.signals.model import build_model, transfer_pretrained
from scripts.train_complex_pilot import collate_pairs, CachedPairs

ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    out = ROOT / "runs/complex_pilot_init"
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(1729)
    m1 = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"), device="cuda", ch=1)
    transfer_pretrained(m1, ROOT / "yolo11s.pt", ch=1)
    seed_everything(1729)
    m3 = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"), device="cuda", ch=3)
    transfer_pretrained(m3, ROOT / "yolo11s.pt", ch=3)
    s1 = m1.state_dict(); s3 = m3.state_dict()
    copied = 0
    for k, v in s3.items():
        if k == "model.0.conv.weight":
            v.zero_()
            v[:, 0] = s1[k].sum(dim=1)
            copied += 1
        elif k in s1 and s1[k].shape == v.shape:
            v.copy_(s1[k]); copied += 1
    c1_path = out / "c1_init.pt"; c3_path = out / "c3_init.pt"
    torch.save({"model": m1.state_dict()}, c1_path)
    torch.save({"model": m3.state_dict()}, c3_path)
    # Verify on cached power-only batch; C3 extra channels are zero.
    ds = CachedPairs(ROOT / "reports/complex_pilot_cache", "fit", 3)
    items = [ds[i] for i in range(min(4, len(ds)))]
    batch = collate_pairs(items)
    x1 = batch["img"][:, :1].cuda()
    x3 = batch["img"].cuda()
    m1.eval(); m3.eval()
    with torch.no_grad():
        p1, _ = m1(x1); p3, _ = m3(x3)
    diff = float((p1 - p3).abs().max())
    target = {"batch_idx": batch["batch_idx"].cuda(), "cls": batch["cls"].cuda(),
              "bboxes": batch["bboxes"].cuda(), "img": x1}
    m1.train(); m3.train()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        l1 = m1.loss(target, preds=m1(x1))[0].sum().item()
        target["img"] = x3
        l3 = m3.loss(target, preds=m3(x3))[0].sum().item()
    report = {"copied": copied, "init_logit_max_abs_diff": diff,
              "init_loss_c1": float(l1), "init_loss_c3": float(l3),
              "init_loss_abs_diff": abs(float(l1)-float(l3)),
              "c1": str(c1_path), "c3": str(c3_path)}
    (out / "init_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
