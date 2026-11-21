"""Adapters for YOLO26l and D-FINE-L on the project's 1x2048x512 views."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

DFINE_ROOT = Path("/root/autodl-tmp/third_party/D-FINE-956d170")
WEIGHTS = Path("/root/autodl-tmp/weights")


# ------------------------------------------------------------------ YOLO26 ---
def build_yolo26(weights: Path = WEIGHTS / "yolo26l.pt",
                 config: Path | None = None, ch: int = 1, nc: int = 14):
    from ultralytics.nn.tasks import DetectionModel
    config = config or Path(__file__).resolve().parents[2] / "configs/yolo26l_c1.yaml"
    model = DetectionModel(str(config), ch=ch, nc=nc, verbose=False)
    ck = torch.load(str(weights), map_location="cpu", weights_only=False)
    src = ck["model"].float().state_dict()
    tgt = model.state_dict()
    copied = skipped = stem = 0
    for k, t in tgt.items():
        if k not in src:
            skipped += 1
            continue
        s = src[k]
        if k == "model.0.conv.weight" and s.ndim == 4 and s.shape[1] == 3 and t.shape[1] == 1:
            t.copy_(s.sum(1, keepdim=True)); copied += 1; stem += 1; continue
        if s.shape == t.shape:
            t.copy_(s); copied += 1
        else:
            skipped += 1
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=50)
    model.names = {i: str(i) for i in range(nc)}
    model.nc = nc
    return model, {"copied": copied, "skipped": skipped, "stem_summed": stem,
                   "head": type(model.model[-1]).__name__}


def yolo26_loss_batch(model, images, boxes_cxcywh_norm, labels, batch_idx):
    """Ultralytics v8DetectionLoss contract: boxes are normalized cxcywh."""
    batch = {
        "batch_idx": batch_idx.long(),
        "cls": labels.float().view(-1, 1),
        "bboxes": boxes_cxcywh_norm.float(),
    }
    preds = model(images)
    loss_vec, items = model.criterion(preds, batch)
    return loss_vec.sum(), items


# ------------------------------------------------------------------- D-FINE ---
def _patch_pos_embed():
    from dfine_src.zoo.dfine.hybrid_encoder import HybridEncoder
    if getattr(HybridEncoder, "_rect_patched", False):
        return
    orig = HybridEncoder.build_2d_sincos_position_embedding

    def fixed(w, h, embed_dim=256, temperature=10000.0):
        if w == h:
            return orig(w, h, embed_dim, temperature)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w = torch.arange(int(w), dtype=torch.float32)
        gh, gw = torch.meshgrid(grid_h, grid_w, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)
        out_w = gw.flatten()[..., None] @ omega[None]
        out_h = gh.flatten()[..., None] @ omega[None]
        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None]

    HybridEncoder.build_2d_sincos_position_embedding = staticmethod(fixed)
    HybridEncoder._rect_patched = True


def build_dfine(weights: Path = WEIGHTS / "dfine_l_obj2coco_e25.pth",
                nc: int = 14, eval_hw=(2048, 512)):
    if str(DFINE_ROOT) not in sys.path:
        sys.path.insert(0, str(DFINE_ROOT))
    _patch_pos_embed()
    from dfine_src.core import YAMLConfig

    cfg = YAMLConfig(
        str(DFINE_ROOT / "configs/dfine/dfine_hgnetv2_l_coco.yml"),
        num_classes=nc, remap_mscoco_category=False,
        eval_spatial_size=[eval_hw[0], eval_hw[1]],
        HGNetv2={"pretrained": False},
    )
    model = cfg.model
    stem1 = model.backbone.stem.stem1
    old = stem1.conv
    new = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding,
                    bias=old.bias is not None)
    stem1.conv = new

    ck = torch.load(str(weights), map_location="cpu", weights_only=False)
    src = ck["model"] if "model" in ck else ck
    tgt = model.state_dict()
    copied = skipped = stem = 0
    for k, t in tgt.items():
        if k not in src:
            skipped += 1
            continue
        s = src[k]
        if k == "backbone.stem.stem1.conv.weight" and s.ndim == 4 and s.shape[1] == 3 and t.shape[1] == 1:
            t.copy_(s.sum(1, keepdim=True)); copied += 1; stem += 1; continue
        if s.shape == t.shape:
            t.copy_(s); copied += 1
        else:
            skipped += 1
    # new stem conv: trainable (old stem was frozen); keep rest of stem as upstream
    model.criterion = cfg.criterion
    model.postprocessor = cfg.postprocessor
    return model, {"copied": copied, "skipped": skipped, "stem_summed": stem,
                   "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)}


def dfine_targets(boxes_cxcywh_norm, labels, batch_idx, n_views):
    """Split normalized cxcywh targets by batch_idx into per-view dicts."""
    out = []
    for i in range(n_views):
        m = batch_idx == i
        out.append({"labels": labels[m].long(), "boxes": boxes_cxcywh_norm[m].float()})
    return out


def dfine_loss_batch(model, images, targets):
    outputs = model(images, targets=targets)
    loss_dict = model.criterion(outputs, targets)
    return sum(loss_dict.values()), loss_dict
