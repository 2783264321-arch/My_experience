#!/usr/bin/env python3
"""Stage-2 regression/AMP probes on the real 2048x512 head.

Checks:
- FP32 forward/loss/backward for a normal box and a full-height box
- finite gradients and nonzero final box-head gradients
- FP16 autocast overflow count (documented rejection)
- BF16 autocast finite gradients
- 200-step single-sample full-height learning check (decoded height grows)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from ultralytics.utils.tal import dist2bbox, make_anchors

from src.signals.model import build_model, transfer_pretrained

ROOT = Path(__file__).resolve().parents[1]


def make_target(boxes, cls):
    return {
        "img": torch.zeros(1, 1, 2048, 512, device="cuda"),
        "batch_idx": torch.zeros(len(boxes), device="cuda"),
        "cls": torch.tensor(cls, device="cuda", dtype=torch.float32)[:, None],
        "bboxes": torch.tensor(boxes, device="cuda", dtype=torch.float32),
    }


def probe(precision: str):
    model = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"))
    transfer_pretrained(model, ROOT / "yolo11s.pt", verbose=False)
    model.train()
    out = {}
    for name, boxes, cls in [
        ("normal", [[0.70, 0.10, 0.15, 0.20]], [[7]]),
        ("full_height", [[0.25, 0.50, 0.20, 1.00]], [[3]]),
        ("normal_and_full", [[0.25, 0.50, 0.20, 1.00], [0.75, 0.10, 0.15, 0.20]], [[3], [7]]),
    ]:
        target = make_target(boxes, cls)
        model.zero_grad(set_to_none=True)
        dtype = None if precision == "fp32" else (torch.float16 if precision == "fp16" else torch.bfloat16)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            preds = model(target["img"])
            loss_parts, items = model.loss(target, preds=preds)
            loss = loss_parts.sum()
        loss.backward()
        head = model.model[-1]
        g = head.cv2[-1][-1].weight.grad
        nf = int(sum(int((~torch.isfinite(p.grad)).sum()) for p in model.parameters() if p.grad is not None))
        out[name] = {
            "loss": float(loss.detach()),
            "loss_parts": [float(x) for x in loss_parts.detach()],
            "loss_items": {k: float(v) for k, v in items.items()},
            "grad_nonfinite": nf,
            "box_final_grad_norm": float(g.norm()) if g is not None else None,
            "box_final_grad_channels": [round(float(g[i].norm()), 8) for i in range(4)] if g is not None else None,
        }
        del target, preds, loss_parts, items, loss
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    return out


def full_height_learning_check(steps=200):
    model = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"))
    transfer_pretrained(model, ROOT / "yolo11s.pt", verbose=False)
    model.train()
    opt = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": 5e-4}],
        lr=1e-3,
    )
    target = make_target([[0.25, 0.50, 0.20, 1.00]], [[3]])
    head = model.model[-1]
    trace = []
    max_h = 0.0
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(target["img"])
            loss_parts, items = model.loss(target, preds=pred)
            loss = loss_parts.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step % 50 == 0 or step == steps:
            with torch.no_grad():
                feats = [f.float() for f in pred["feats"]]
                anchors, strides = make_anchors(feats, head.stride, 0.5)
                raw = pred["boxes"][0].permute(1, 0).float()
                boxes = dist2bbox(raw, anchors, xywh=False) * strides
                cx = (boxes[:, 0] + boxes[:, 2]) / 2
                sel = (cx > 76) & (cx < 180)
                h = (boxes[:, 3] - boxes[:, 1])[sel]
                max_h = float(h.max()) if len(h) else 0.0
            trace.append({
                "step": step, "loss": float(loss.detach()),
                "box": float(loss_parts[0]), "cls": float(loss_parts[1]),
                "l1": float(loss_parts[2]), "max_height_px": max_h,
            })
    del model, opt, target
    torch.cuda.empty_cache()
    return trace


def main():
    report = {"probes": {}, "full_height_trace": None}
    report["probes"]["fp32"] = probe("fp32")
    report["probes"]["fp16"] = probe("fp16")
    report["probes"]["bf16"] = probe("bf16")
    report["full_height_trace"] = full_height_learning_check(200)
    path = ROOT / "reports/stage2_grad_probe.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "fp32_nonfinite": [v["grad_nonfinite"] for v in report["probes"]["fp32"].values()],
        "fp16_nonfinite": [v["grad_nonfinite"] for v in report["probes"]["fp16"].values()],
        "bf16_nonfinite": [v["grad_nonfinite"] for v in report["probes"]["bf16"].values()],
        "full_height_steps": len(report["full_height_trace"]),
        "first": report["full_height_trace"][0],
        "last": report["full_height_trace"][-1],
    }, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
