#!/usr/bin/env python3
"""CPU-only pipeline smoke/benchmark for the SRRC GT direct-IQ baseline.

Builds the 20 MS/s, 8 x 1024-point window cache for a few real fit files,
then runs one forward/backward of a small 1D ResNet on CPU. Used to estimate
cache size and training throughput while the GPU is busy with another job.
No training, no val/test, no detector inference.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.signals.data import SPLIT_MANIFEST, load_manifest
from scripts.baseline3srrc_phase0_verify import (
    crop_and_resample, windows_from_crop, WIN, N_WIN,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/baseline3_srrc_phase0"


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 5, padding=2)
        self.b1 = nn.BatchNorm1d(c)
        self.c2 = nn.Conv1d(c, c, 5, padding=2)
        self.b2 = nn.BatchNorm1d(c)

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(x + y)


class TinyResNet(nn.Module):
    def __init__(self, width=32, *, in_channels=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width, 9, padding=4),
            nn.BatchNorm1d(width), nn.ReLU())
        self.blocks = nn.Sequential(Block(width), Block(width), Block(width))
        self.head = nn.Linear(width, 2)

    def forward(self, x):
        y = self.blocks(self.stem(x))
        return self.head(y.mean(dim=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-files", type=int, default=8)
    ap.add_argument("--out", default=str(OUT / "pipeline_smoke.json"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    recs = load_manifest(SPLIT_MANIFEST, split="fit")
    picked = []
    seen = set()
    for r in recs:
        if r.sample_id in seen:
            continue
        for s in r.signals:
            bw = float(s["end_frequency"]) - float(s["start_frequency"])
            if int(s["class"]) in (10, 11) and 9.5 <= bw <= 10.5:
                picked.append((r, s))
                seen.add(r.sample_id)
                break
        if len(picked) >= args.n_files:
            break

    t0 = time.perf_counter()
    xs = []
    ys = []
    per_instance = []
    for r, s in picked:
        a = time.perf_counter()
        crop = crop_and_resample(r, s)
        w = windows_from_crop(crop, s)
        a = time.perf_counter() - a
        wins = []
        for x in w["windows"]:
            z = crop["x"][x["start"]:x["stop"]]
            if len(z) < WIN:
                z = np.pad(z, (0, WIN - len(z)))
            wins.append(z)
        arr = np.stack(wins)
        # complex [W, N] -> real [W, 2, N]
        ten = np.stack([arr.real, arr.imag], axis=1).astype(np.float32)
        xs.append(ten)
        ys.append(0 if int(s["class"]) == 10 else 1)
        per_instance.append({
            "sample_id": r.sample_id, "class": int(s["class"]),
            "B_obs_mhz": float(r.B_mhz),
            "duration_ms": float(s["end_time"]) - float(s["start_time"]),
            "raw_points": crop["raw_points"],
            "out_points_20msps": crop["out_points"],
            "windows": len(w["windows"]), "coverage": w["coverage"],
            "elapsed_s": a,
            "shape": list(ten.shape),
            "finite": bool(np.isfinite(ten).all()),
        })
    build_s = time.perf_counter() - t0
    X = torch.from_numpy(np.concatenate(xs, axis=0))
    Y = torch.tensor(ys, dtype=torch.long).repeat_interleave(N_WIN)
    # one batch of 16 windows
    model = TinyResNet()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    batch = min(16, len(X))
    xb = X[:batch]; yb = Y[:batch]
    times = []
    for _ in range(3):
        opt.zero_grad()
        a = time.perf_counter()
        out = model(xb)
        loss = lossf(out, yb)
        loss.backward()
        opt.step()
        times.append(time.perf_counter() - a)
    # 1,824 fit instances at 10 MHz (not all 5,957 fit files)
    cache_bytes = int(1824 * N_WIN * 2 * WIN * 4)
    result = {
        "status": "ok",
        "n_files": len(picked),
        "n_windows": int(len(X)),
        "tensor_shape": list(X.shape),
        "labels": ys,
        "build_s": build_s,
        "build_s_per_file": build_s / max(len(picked), 1),
        "estimated_full_fit_cache_s": build_s / max(len(picked), 1) *
        1572 / 8,  # 8 parallel workers, rough
        "estimated_full_fit_cache_bytes": cache_bytes,
        "estimated_full_fit_cache_mib": cache_bytes / 2 ** 20,
        "batch": batch,
        "step_s": times,
        "step_s_median": float(np.median(times)),
        "examples": per_instance,
        "note": ("CPU-only smoke; GPU training skipped because the 4090 is "
                 "busy with the D-FINE formal run."),
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in
                      ("n_files", "n_windows", "tensor_shape", "build_s",
                       "build_s_per_file", "estimated_full_fit_cache_s",
                       "estimated_full_fit_cache_mib", "step_s_median")},
                     indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
