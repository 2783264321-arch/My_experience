#!/usr/bin/env python3
"""Evaluate QRE-full v1 on the fixed e25 candidate pool with Soft-NMS."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/detector_upgrade"))
from src.signals.data import N_TIME, SPLIT_MANIFEST, load_manifest
from src.signals.infer import (Prediction, global_classwise_nms, gt_global,
                               soft_classwise_nms)
from src.signals.metrics import ImagePredictions
from eval_local_iq_abc import load_model, predict_group
from validate_dual import _eval_set

CAND = Path("/root/autodl-fs/runs_local/dfine_local_iq/val_candidates.npz")
CROPS = Path("/root/autodl-fs/runs_local/dfine_local_iq/val_local_crops.npz")
MODELS = Path("/root/autodl-fs/runs_local/dfine_local_iq/models_qre_full")
GROUPS = ("012", "345", "1011")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, default=CAND)
    ap.add_argument("--crops", type=Path, default=CROPS)
    ap.add_argument("--models", type=Path, default=MODELS)
    ap.add_argument("--betas", default="0,0.25,0.5,1.0")
    ap.add_argument("--gate", type=float, default=0.5)
    ap.add_argument("--merge", choices=("nms", "soft"), default="soft")
    ap.add_argument("--soft-sigma", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with np.load(args.candidates, allow_pickle=True) as zf:
        c = {k: zf[k] for k in zf.files}
    with np.load(args.crops, allow_pickle=True) as zf:
        r = {k: zf[k] for k in zf.files}
    meta = json.loads(args.candidates.with_suffix(".json").read_text())
    n = int(meta["samples"])
    records = load_manifest(SPLIT_MANIFEST, split="val")[:n]
    gts = [gt_global(rr, math.ceil(rr.N / (N_TIME - 1))) for rr in records]
    offsets = c["sample_offsets"]
    ci = r["candidate_index"].astype(np.int64)
    labels = c["labels"].astype(np.int64)
    assert np.array_equal(r["label"].astype(np.int64), labels[ci])

    models = {}
    for g in GROUPS:
        models[g] = load_model(args.models / g / "best.pt", args.device)
    print("models", list(models), flush=True)

    def metrics(scores):
        out = []
        for i in range(n):
            a, b = int(offsets[i]), int(offsets[i + 1])
            p = Prediction(torch.from_numpy(c["boxes"][a:b]).float(),
                           torch.from_numpy(scores[a:b]).float(),
                           torch.from_numpy(labels[a:b]).long())
            if args.merge == "soft":
                f = soft_classwise_nms(p, 0.5, args.soft_sigma, 1e-3, 100)
            else:
                f = global_classwise_nms(p, 0.5, 100)
            out.append(ImagePredictions(f.boxes.numpy(), f.scores.numpy(),
                                        f.labels.numpy()))
        m = _eval_set(gts, out)
        m["n_pred"] = int(sum(len(p.boxes) for p in out))
        return m

    original = c["scores"].astype(np.float32).copy()
    results = {"A": metrics(original)}
    print("A", results["A"]["map50"], results["A"]["map50_95"], flush=True)

    # Per-group residuals, one-sided, own-class only.
    updates = []
    gate_skipped = 0
    for g in GROUPS:
        model, ck = models[g]
        cls = np.asarray(ck["classes"], dtype=np.int64)
        sel = np.isin(r["label"].astype(np.int64), cls)
        if not sel.any():
            continue
        meta_in = np.concatenate([
            r["class_scores"][sel][:, cls].astype(np.float32),
            r["score"][sel][:, None].astype(np.float32),
            np.log1p(r["bandwidth_mhz"][sel])[:, None].astype(np.float32),
            np.log1p(r["duration_ms"][sel])[:, None].astype(np.float32),
            r["view_type"][sel][:, None].astype(np.float32),
        ], axis=1).astype(np.float32)
        probs = predict_group(model, ck, r["local_power"][sel], r["features"][sel],
                              meta_in, True, args.device)
        for row, prob in zip(np.flatnonzero(sel), probs):
            idx = int(ci[row]); lab = int(labels[idx])
            j = int(np.where(cls == lab)[0][0])
            kp = float(prob[-1])
            pg = prob[:-1]
            if kp > 0.5 or float(pg.max()) < args.gate:
                gate_skipped += 1
                continue
            p_self = float(pg[j]) / max(1.0 - kp, 1e-6)
            s = float(np.clip(c["class_scores"][idx, lab], 1e-4, 1 - 1e-4))
            updates.append((idx, math.log(s / (1 - s)),
                            math.log(p_self + 1e-6) + math.log(len(cls))))
    print("updates", len(updates), "gate_skipped", gate_skipped, flush=True)

    betas = [float(x) for x in args.betas.split(",") if x.strip()]
    results["betas"] = {}
    for beta in betas:
        scores = original.copy()
        if beta:
            for idx, zdet, dz in updates:
                z = zdet + beta * max(0.0, dz)
                scores[idx] = 1.0 / (1.0 + math.exp(-z))
        m = metrics(scores)
        m["changed_scores"] = len(updates)
        m["gate_skipped"] = gate_skipped
        results["betas"][str(beta)] = m
        print(f"beta={beta} {m['map50']:.6f} {m['map50_95']:.6f} "
              f"n_pred={m['n_pred']}", flush=True)

    payload = {"betas": betas, "gate": args.gate, "merge": args.merge,
               "soft_sigma": args.soft_sigma, "results": results,
               "candidates": str(args.candidates), "crops": str(args.crops),
               "models": str(args.models)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
