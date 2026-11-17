"""Model construction and pretrained weight transfer for YOLO11s-C1-DirectReg."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from ultralytics.nn.tasks import DetectionModel


def build_model(config: str = "configs/yolo11s_c1_directreg.yaml", device: str = "cuda") -> DetectionModel:
    model = DetectionModel(config, ch=1, nc=14, verbose=False)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    model = model.to(device)
    model.names = {i: str(i) for i in range(14)}
    return model


def transfer_pretrained(model: DetectionModel, weights: str | Path, verbose: bool = True) -> dict:
    """Transfer body weights; C1 RGB stem is summed, head shape mismatches are skipped."""
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    src = ckpt.get("model", ckpt)
    if hasattr(src, "float"):
        src = src.float()
    src_sd = src.state_dict()
    tgt_sd = model.state_dict()
    copied, skipped, stem = 0, 0, 0
    for k, tgt in tgt_sd.items():
        if k not in src_sd:
            skipped += 1
            continue
        s = src_sd[k]
        if k == "model.0.conv.weight" and s.shape != tgt.shape:
            if s.ndim == 4 and tgt.ndim == 4 and s.shape[0] == tgt.shape[0] and s.shape[1] == 3 and tgt.shape[1] == 1:
                tgt.copy_(s.sum(dim=1, keepdim=True))
                copied += 1
                stem += 1
                continue
            skipped += 1
            continue
        if s.shape == tgt.shape:
            tgt.copy_(s)
            copied += 1
        else:
            skipped += 1
    if verbose:
        print(f"transferred={copied} skipped={skipped} c1_stem_summed={stem}")
    return {"transferred": copied, "skipped": skipped, "stem_summed": stem}
