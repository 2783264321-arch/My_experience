#!/usr/bin/env python3
"""Query-Residual Evidence (QRE-lite) head for D-FINE.

A zero-initialized local-C1 RoI head adds a residual to each query's class
logits. The base D-FINE forward is frozen and run under no_grad; only the QRE
head receives gradients. Boxes and candidate identity are unchanged.
"""
from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, roi_align


class QREHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 14,
                 hidden: int = 128, roi_size: int = 7):
        super().__init__()
        self.roi_size = roi_size
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden), nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(inplace=True),
                                nn.Linear(hidden, num_classes))
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, feat: torch.Tensor, boxes_cxcywh: torch.Tensor) -> torch.Tensor:
        # feat: [B,C,Hf,Wf]; boxes: [B,Q,4] normalized cxcywh.
        b, q, _ = boxes_cxcywh.shape
        boxes = box_convert(boxes_cxcywh.float(), "cxcywh", "xyxy")
        hf, wf = feat.shape[-2:]
        scale = boxes.new_tensor([wf, hf, wf, hf])
        roi = boxes * scale
        idx = torch.arange(b, device=feat.device).view(b, 1, 1).expand(b, q, 1)
        rois = torch.cat([idx.float(), roi], dim=-1).reshape(-1, 5)
        pooled = roi_align(feat, rois, output_size=self.roi_size,
                           spatial_scale=1.0, aligned=True)
        pooled = self.body(pooled).flatten(1)
        return self.fc(pooled).view(b, q, -1)


def _qre_forward(self, x, targets=None):
    was_training = self.training
    # Frozen base: no gradients and no BN/dropout drift. QRE remains trainable.
    self.backbone.eval()
    self.encoder.eval()
    # Training needs decoder aux outputs for the criterion; evaluation must
    # stay in eval mode or the decoder tries denoising with targets=None.
    self.decoder.train(was_training)
    if was_training:
        for m in self.decoder.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm,
                              nn.Dropout, nn.Dropout2d)):
                m.eval()
    with torch.no_grad():
        feats = self.encoder(self.backbone(x))
        out = self.decoder(feats, targets)
    feats = [f.detach() for f in feats]

    def add(o):
        if "pred_logits" not in o or "pred_boxes" not in o:
            return
        o["pred_logits"] = o["pred_logits"].detach() + self.qre(feats[0], o["pred_boxes"])

    add(out)
    for aux in out.get("aux_outputs", []):
        add(aux)
    if was_training:
        self.qre.train()
    return out


def attach_qre(model, channels: int | None = None, device: str | torch.device = "cuda"):
    """Attach QREHead and patch DFINE.forward.

    channels is inferred from one dummy encoder pass when omitted. The dummy
    pass is done once at attach time, not per training step.
    """
    if channels is None:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 512, 2048, device=device)
            feats = model.encoder(model.backbone(dummy))
        channels = int(feats[0].shape[1])
        model.train(was_training)
    nc = int(getattr(model.decoder, "num_classes", 14))
    model.qre = QREHead(channels, nc).to(device)
    model.forward = types.MethodType(_qre_forward, model)
    return model


def freeze_base(model):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.qre.parameters():
        p.requires_grad_(True)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm,
                          nn.Dropout, nn.Dropout2d)):
            m.eval()
    model.qre.train()
