#!/usr/bin/env python3
"""Audit what the E1/E2/E3 fine-tune actually froze.

Reads the original SHARED_seed*.pt and the fine-tuned arm weights. Reports
whether each parameter/buffer changed. This is a post-hoc audit only; it
does not change any weight or metric.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / "runs_local/baseline3b/weights"
FT = ROOT / "runs_local/baseline3ca/weights"
EXP = ROOT / "reports/baseline3_wifi_cascade"
SEEDS = (1729, 1730, 1731)
ARMS = ("E1", "E2", "E3")
KEYS = [
    "features.0.weight", "features.0.bias",
    "features.1.running_mean", "features.1.running_var",
    "features.1.num_batches_tracked",
    "features.4.weight", "features.4.bias",
    "features.5.running_mean", "features.5.running_var",
    "features.5.num_batches_tracked",
    "features.8.weight", "features.8.bias",
    "features.9.running_mean", "features.9.running_var",
    "features.9.num_batches_tracked",
    "fc1.weight", "fc2.weight",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = {"original": {}, "runs": {}}
    for s in SEEDS:
        p = ORIG / f"SHARED_seed{s}.pt"
        out["original"][f"SHARED_seed{s}"] = {
            "path": str(p.relative_to(ROOT)), "sha256": sha256(p),
            "bytes": p.stat().st_size}
    for arm in ARMS:
        for s in SEEDS:
            op = ORIG / f"SHARED_seed{s}.pt"
            fp = FT / f"{arm}_seed{s}.pt"
            o = torch.load(op, map_location="cpu")
            f = torch.load(fp, map_location="cpu")
            rec = {}
            for k in KEYS:
                if k not in o:
                    continue
                a, b = o[k], f[k]
                if a.dtype.is_floating_point:
                    diff = float((a.float() - b.float()).abs().max())
                else:
                    diff = float((a - b).abs().max())
                rec[k] = {
                    "changed": bool(not torch.equal(a, b)),
                    "max_abs_diff": diff,
                }
            out["runs"][f"{arm}_seed{s}"] = {
                "path": str(fp.relative_to(ROOT)), "sha256": sha256(fp),
                "bytes": fp.stat().st_size,
                "state": rec,
                "conv1_conv2_weights_frozen": all(
                    not rec[k]["changed"] for k in (
                        "features.0.weight", "features.0.bias",
                        "features.4.weight", "features.4.bias")),
                "bn1_bn2_running_stats_changed": any(
                    rec[k]["changed"] for k in (
                        "features.1.running_mean", "features.1.running_var",
                        "features.5.running_mean", "features.5.running_var")),
            }
    (EXP / "finetune_freeze_audit.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({
        "original_sha256": {k: v["sha256"] for k, v in out["original"].items()},
        "conv1_conv2_weights_frozen": {
            k: v["conv1_conv2_weights_frozen"] for k, v in out["runs"].items()},
        "bn1_bn2_running_stats_changed": {
            k: v["bn1_bn2_running_stats_changed"] for k, v in out["runs"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
