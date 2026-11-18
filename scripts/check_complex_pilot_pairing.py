#!/usr/bin/env python3
"""Check C1/C3 power identity, complex scaling, and identical initialization."""
from __future__ import annotations

import json
import random
import os
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
    cache = ROOT / "reports/complex_pilot_cache"
    meta = json.loads((cache / "fit_meta.json").read_text())
    ds = CachedPairs(cache, "fit", 3)
    items = [ds[i] for i in range(min(4, len(ds)))]
    c1 = torch.stack([it["coarse"][:1] for it in items])
    c3 = torch.stack([it["coarse"] for it in items])
    f1 = torch.stack([it["fine"][:1] for it in items])
    f3 = torch.stack([it["fine"] for it in items])
    power_identical = bool(torch.equal(c1, c3[:, :1]) and torch.equal(f1, f3[:, :1]))
    stats = {}
    for name, x in [("coarse_power", c3[:, 0]), ("fine_power", f3[:, 0]),
                    ("coarse_re", c3[:, 1]), ("coarse_im", c3[:, 2]),
                    ("fine_re", f3[:, 1]), ("fine_im", f3[:, 2])]:
        a = x.float().numpy()
        stats[name] = {"mean": float(a.mean()), "std": float(a.std()),
                       "min": float(a.min()), "max": float(a.max()),
                       "nan": int(np.isnan(a).sum()), "inf": int(np.isinf(a).sum())}
    # Initialization identity from the prepared paired checkpoints.
    m1 = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"), device="cuda", ch=1)
    c1ck = torch.load(ROOT / "runs/complex_pilot_init/c1_init.pt", map_location="cpu", weights_only=False)
    m1.load_state_dict(c1ck["model"])
    m3 = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"), device="cuda", ch=3)
    c3ck = torch.load(ROOT / "runs/complex_pilot_init/c3_init.pt", map_location="cpu", weights_only=False)
    m3.load_state_dict(c3ck["model"])
    batch = collate_pairs(items)
    images_c1 = batch["img"][:, :1].cuda()
    images_c3 = batch["img"].cuda()
    m1.eval(); m3.eval()
    with torch.no_grad():
        p1, _ = m1(images_c1)
        p3, _ = m3(images_c3)
    diff = float((p1 - p3).abs().max())
    # Same target/loss on the same coarse+fine batch.
    target = {"batch_idx": batch["batch_idx"].cuda(), "cls": batch["cls"].cuda(),
              "bboxes": batch["bboxes"].cuda(), "img": images_c1}
    m1.train(); m3.train()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        l1 = m1.loss(target, preds=m1(images_c1))[0].sum().item()
        target["img"] = images_c3
        l3 = m3.loss(target, preds=m3(images_c3))[0].sum().item()
    out = {
        "power_identical": power_identical,
        "input_shapes": {"c1": list(c1.shape), "c3": list(c3.shape),
                         "f1": list(f1.shape), "f3": list(f3.shape)},
        "complex_stats": stats,
        "init_logit_max_abs_diff": diff,
        "init_loss_c1": float(l1), "init_loss_c3": float(l3),
        "init_loss_abs_diff": abs(float(l1) - float(l3)),
        "scales": {"coarse": meta.get("scale_coarse"), "fine": meta.get("scale_fine")},
        "clip_fractions": {
            "coarse_power": [s["coarse_power_clip"] for s in meta["samples"]],
            "fine_power": [s["fine_power_clip"] for s in meta["samples"]],
            "coarse_complex": [s["coarse_complex_clip"] for s in meta["samples"]],
            "fine_complex": [s["fine_complex_clip"] for s in meta["samples"]],
        },
    }
    out["clip_fraction_means"] = {k: float(np.mean(v)) for k, v in out["clip_fractions"].items()}
    p = ROOT / "reports/complex_pilot_pairing_check.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
