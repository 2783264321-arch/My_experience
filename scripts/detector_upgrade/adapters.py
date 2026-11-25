"""Adapters for YOLO26l and D-FINE-L on the project's 1x2048x512 views."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, box_iou

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
        out_h = gh.flatten()[..., None] @ omega[None]
        out_w = gw.flatten()[..., None] @ omega[None]
        # square convention: first half = vertical (h), second half = horizontal (w)
        return torch.concat([out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos()], dim=1)[None]

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


def dfine_targets(boxes_cxcywh_norm, labels, batch_idx, n_views,
                  view_types=None, truncated=None):
    """Split normalized cxcywh targets by batch_idx into per-view dicts.

    Optional view_types/truncated add a per-target cls_ignore mask for fine
    views whose Wi-Fi label is cut by the slice boundary. Box regression and
    coarse supervision are unaffected.
    """
    out = []
    for i in range(n_views):
        m = batch_idx == i
        ign = torch.zeros((int(m.sum()),), dtype=torch.bool)
        if view_types is not None and truncated is not None and i < len(view_types):
            if view_types[i] == "fine":
                ign = truncated[m].bool() & torch.isin(
                    labels[m].long(), labels.new_tensor(list(range(6))))
        out.append({"labels": labels[m].long(), "boxes": boxes_cxcywh_norm[m].float(),
                    "cls_ignore": ign})
    return out


def dfine_loss_batch(model, images, targets):
    outputs = model(images, targets=targets)
    loss_dict = model.criterion(outputs, targets)
    return sum(loss_dict.values()), loss_dict


def install_cls_ignore_patch(criterion):
    """Mask VFL classification loss for matched targets flagged cls_ignore.

    The D-FINE matcher still uses these targets for box regression. Only the
    classification loss at the matched query is zeroed; all background
    queries and all box terms remain unchanged.
    """
    from types import MethodType

    def loss_labels_vfl_masked(self, outputs, targets, indices, num_boxes, values=None):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            iou_res = box_iou(box_convert(src_boxes, "cxcywh", "xyxy"),
                              box_convert(target_boxes, "cxcywh", "xyxy"))
            ious = iou_res[0] if isinstance(iou_res, (tuple, list)) else iou_res
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )

        ignore = torch.cat([
            (t["cls_ignore"].to(src_logits.device)[J] if "cls_ignore" in t
             else t["labels"].new_zeros((len(J),), dtype=torch.bool))
            for t, (_, J) in zip(targets, indices)
        ])
        ignore_q = torch.zeros(src_logits.shape[:2], dtype=torch.bool,
                               device=src_logits.device)
        ignore_q[idx] = ignore.to(src_logits.device)
        loss = loss * (~ignore_q).unsqueeze(-1)
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    criterion.loss_labels_vfl = MethodType(loss_labels_vfl_masked, criterion)
    return criterion
