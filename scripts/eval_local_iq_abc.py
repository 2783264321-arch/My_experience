#!/usr/bin/env python3
"""Evaluate A/B/C on the full val fixed-candidate pool.

A = original candidate scores. B/C update only the candidate's own class
score with a group-classifier residual; boxes and candidate membership stay
fixed. B uses local power; C adds physical features.
"""
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
from src.signals.infer import Prediction, global_classwise_nms, gt_global
from src.signals.metrics import ImagePredictions
from train_local_iq_abc import LocalNet
from validate_dual import _eval_set


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    feat_dim = int(ck.get("feature_dim", 0))
    n_out = int(ck.get("num_classes", len(ck["classes"]) +
                       (1 if ck.get("keep_class") else 0)))
    meta_dim = int(ck.get("meta_dim", 0))
    m = LocalNet(n_out, feat_dim=feat_dim, meta_dim=meta_dim).to(device).eval()
    m.load_state_dict(ck["state"])
    return m, ck


def predict_group(model, ck, power, feats, meta, use_feats, device, batch=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(power), batch):
            p = torch.from_numpy(power[i:i + batch]).float().to(device)
            p = (p - p.mean(dim=(1, 2, 3), keepdim=True)) / \
                p.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            p = p.clamp(-5, 5)
            f = None
            if use_feats:
                f = (torch.from_numpy(feats[i:i + batch]).float().to(device)
                     - torch.from_numpy(ck["feat_mu"]).float().to(device))
                f = f / torch.from_numpy(ck["feat_sd"]).float().to(device).clamp_min(1e-6)
                f = f.clamp(-5, 5)
            mm = (torch.from_numpy(meta[i:i + batch]).float().to(device)
                  - torch.from_numpy(ck["meta_mu"]).float().to(device))
            mm = mm / torch.from_numpy(ck["meta_sd"]).float().to(device).clamp_min(1e-6)
            mm = mm.clamp(-5, 5)
            out.append(torch.softmax(model(p, f, mm), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(ck["classes"])), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True,
                    help="directory with 012/{power,features}, 345/..., 1011/...")
    ap.add_argument("--betas", default="0,0.25,0.5,1.0")
    ap.add_argument("--gate", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with np.load(args.candidates, allow_pickle=True) as zf:
        c = {k: zf[k] for k in zf.files}
    with np.load(args.crops, allow_pickle=True) as zf:
        r = {k: zf[k] for k in zf.files}
    meta = json.loads(args.candidates.with_suffix(".json").read_text())
    cmeta = json.loads(args.crops.with_suffix(".json").read_text())
    assert meta.get("format") == "dfine-fixed-candidates-v1"
    assert cmeta.get("format") == "dfine-local-crops-v1"
    assert meta.get("split") == "val" and cmeta.get("split") == "val"
    assert len(c["sample_offsets"]) == int(meta["samples"]) + 1
    assert int(c["sample_offsets"][-1]) == len(c["scores"])
    ci = r["candidate_index"].astype(np.int64)
    assert int(cmeta["kept"]) == len(ci)
    assert np.all((ci >= 0) & (ci < len(c["scores"])))
    assert np.allclose(c["class_scores"][np.arange(len(c["labels"])), c["labels"].astype(np.int64)],
                       c["scores"], atol=2e-5)
    assert np.array_equal(r["label"].astype(np.int64), c["labels"][ci].astype(np.int64))
    n = int(meta["samples"])
    records = load_manifest(SPLIT_MANIFEST, split="val")[:n]
    gts = [gt_global(rr, math.ceil(rr.N / (N_TIME - 1))) for rr in records]
    offsets = c["sample_offsets"]
    pred_label = c["labels"].astype(np.int64)

    models = {}
    for group in ("012", "345", "1011"):
        for rep in ("power", "features"):
            path = args.models / group / rep / "best.pt"
            if path.exists():
                models[(group, rep)] = load_model(path, args.device)
    print("models", list(models), flush=True)
    for rep in ("power", "features"):
        missing = [g for g in ("012", "345", "1011") if (g, rep) not in models]
        if missing:
            raise SystemExit(f"{rep} missing group models: {missing}")

    # A metrics from original candidates.
    def metrics_for(scores):
        preds = []
        for i in range(n):
            a, b = int(offsets[i]), int(offsets[i + 1])
            p = Prediction(torch.from_numpy(c["boxes"][a:b]).float(),
                           torch.from_numpy(scores[a:b]).float(),
                           torch.from_numpy(c["labels"][a:b]).long())
            f = global_classwise_nms(p, iou=0.5, max_det=100)
            preds.append(ImagePredictions(f.boxes.numpy(), f.scores.numpy(),
                                          f.labels.numpy()))
        res = _eval_set(gts, preds)
        res["n_pred"] = int(sum(len(p.boxes) for p in preds))
        return res

    original = c["scores"].astype(np.float32).copy()
    results = {"A": metrics_for(original)}
    print("A", results["A"]["map50"], results["A"]["map50_95"], results["A"]["n_pred"],
          flush=True)

    updates = {}
    gate_skipped = {}
    for rep in ("power", "features"):
        ups = []
        n_gate = 0
        for group in ("012", "345", "1011"):
            key = (group, rep)
            if key not in models:
                continue
            model, ck = models[key]
            cls = np.asarray(ck["classes"], dtype=np.int64)
            sel = np.isin(r["label"].astype(np.int64), cls)
            if not sel.any():
                continue
            meta = np.concatenate([
                r["class_scores"][sel][:, cls].astype(np.float32),
                r["score"][sel][:, None].astype(np.float32),
                np.log1p(r["bandwidth_mhz"][sel])[:, None].astype(np.float32),
                np.log1p(r["duration_ms"][sel])[:, None].astype(np.float32),
                r["view_type"][sel][:, None].astype(np.float32),
            ], axis=1).astype(np.float32)
            probs = predict_group(model, ck, r["local_power"][sel], r["features"][sel],
                                  meta, rep == "features", args.device)
            rows = np.flatnonzero(sel)
            for row, prob in zip(rows, probs):
                idx = ci[row]
                lab = int(c["labels"][idx])
                if lab not in cls:
                    continue
                j = int(np.where(cls == lab)[0][0])
                kp = float(prob[-1]) if len(prob) == len(cls) + 1 else 0.0
                pg = prob[:-1] if len(prob) == len(cls) + 1 else prob
                if kp > 0.5 or float(pg.max()) < args.gate:
                    n_gate += 1
                    continue
                p_self = float(pg[j]) / max(1.0 - kp, 1e-6)
                s = float(np.clip(c["class_scores"][idx, lab], 1e-4, 1 - 1e-4))
                ups.append((idx, math.log(s / (1 - s)),
                            math.log(p_self + 1e-6) + math.log(len(cls))))
        updates[rep] = ups
        gate_skipped[rep] = n_gate
        print(rep, "updates", len(ups), "gate_skipped", n_gate, flush=True)

    betas = [float(x) for x in args.betas.split(",") if x.strip()]
    results["betas"] = {}
    for beta in betas:
        results["betas"][str(beta)] = {}
        for rep in ("power", "features"):
            scores = original.copy()
            if beta != 0:
                for idx, zdet, dz in updates[rep]:
                    z = zdet + beta * dz
                    scores[idx] = 1.0 / (1.0 + math.exp(-z))
            m = metrics_for(scores)
            m["changed_scores"] = len(updates[rep])
            m["gate_skipped"] = gate_skipped[rep]
            results["betas"][str(beta)][rep.upper()] = m
            print(f"beta={beta} {rep.upper()} {m['map50']:.6f} "
                  f"{m['map50_95']:.6f} n_pred={m['n_pred']} "
                  f"changed={len(updates[rep])}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"betas": betas, "gate": args.gate, "results": results,
               "candidates": str(args.candidates), "crops": str(args.crops),
               "models": str(args.models)}
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
