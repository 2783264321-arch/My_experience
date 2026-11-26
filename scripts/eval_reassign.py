#!/usr/bin/env python3
"""Reassign candidate labels from a local classifier, then Soft-NMS + AP.

The prior QRE experiments only reweighted the candidate's existing class
score.  This script tests the missing operation: replace the class label
when the local model is confident, or add its class as an extra detection.
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
from src.signals.infer import Prediction, soft_classwise_nms, gt_global
from src.signals.metrics import ImagePredictions
from eval_local_iq_abc import load_model, predict_group
from validate_dual import _eval_set

GROUPS = {"012": [0, 1, 2], "345": [3, 4, 5], "1011": [10, 11]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path,
                    default=Path("/root/autodl-fs/runs_local/dfine_local_iq/val_candidates.npz"))
    ap.add_argument("--crops", type=Path,
                    default=Path("/root/autodl-fs/runs_local/dfine_local_iq/val_local_crops.npz"))
    ap.add_argument("--models", type=Path,
                    default=Path("/root/autodl-fs/runs_local/dfine_local_iq/models"))
    ap.add_argument("--mode", choices=("replace", "add"), default="replace")
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--keep-gate", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--selected-groups", default="012,345,1011")
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
    labels = c["labels"].astype(np.int64).copy()
    boxes = c["boxes"].astype(np.float32)
    scores = c["scores"].astype(np.float32)
    sample_index = r["sample_index"].astype(np.int64)
    row_label = r["label"].astype(np.int64)
    gt_iou = r["gt_iou"].astype(np.float32)
    keep_prior = r["keep"].astype(np.int64)

    # Local class probabilities (class-only head; keep is separate).
    local = {}
    for g, cls in GROUPS.items():
        if g not in args.selected_groups.split(","):
            continue
        cls = np.asarray(cls, dtype=np.int64)
        sel = np.flatnonzero(np.isin(row_label, cls))
        model, ck = load_model(args.models / g / "features" / "best.pt", args.device)
        meta_in = np.concatenate([
            r["class_scores"][sel][:, cls].astype(np.float32),
            r["score"][sel][:, None].astype(np.float32),
            np.log1p(r["bandwidth_mhz"][sel])[:, None].astype(np.float32),
            np.log1p(r["duration_ms"][sel])[:, None].astype(np.float32),
            r["view_type"][sel][:, None].astype(np.float32),
        ], axis=1).astype(np.float32)
        probs = predict_group(model, ck, r["local_power"][sel], r["features"][sel],
                              meta_in, True, args.device)
        local[g] = (sel, cls, probs)
        print(g, "rows", len(sel), "acc_pos",
              float((probs[np.isin(row_label[sel], cls) & (gt_iou[sel] >= .5), :-1].argmax(1) ==
                     np.searchsorted(cls, r["gt_label"][sel][np.isin(row_label[sel], cls) & (gt_iou[sel] >= .5)])).mean()),
              flush=True)

    # Build a flat update table: candidate index -> (new class, confidence, keep_prob).
    upd = {}
    for g, (sel, cls, probs) in local.items():
        p = probs[:, :-1].argmax(1)
        conf = probs[:, :-1].max(1)
        pk = probs[:, -1]
        ok = (conf > args.tau) & (pk < args.keep_gate)
        for row, j, pp, cc, kk in zip(sel[ok], ci[sel[ok]], p[ok], conf[ok], pk[ok]):
            new_label = int(cls[pp])
            if new_label != int(labels[j]):
                # Keep the highest confidence if duplicates map to one candidate.
                if j not in upd or cc > upd[j][1]:
                    upd[int(j)] = (new_label, float(cc), float(kk))
    print("updates", len(upd), "tau", args.tau, "mode", args.mode, flush=True)

    def metrics_for(extra_by_sample=None):
        out = []
        for i in range(n):
            a, b = int(offsets[i]), int(offsets[i + 1])
            lab = labels[a:b].copy(); sco = scores[a:b].copy(); box = boxes[a:b]
            if extra_by_sample and i in extra_by_sample:
                eb, es, el = extra_by_sample[i]
                if eb:
                    box = np.concatenate([box, eb], 0)
                    sco = np.concatenate([sco, es], 0)
                    lab = np.concatenate([lab, el], 0)
            p = Prediction(torch.from_numpy(box).float(),
                           torch.from_numpy(sco).float(),
                           torch.from_numpy(lab).long())
            f = soft_classwise_nms(p, 0.5, 0.5, 1e-3, 100)
            out.append(ImagePredictions(f.boxes.numpy(), f.scores.numpy(), f.labels.numpy()))
        m = _eval_set(gts, out)
        m["n_pred"] = int(sum(len(p.boxes) for p in out))
        return m

    result = {"tau": args.tau, "mode": args.mode, "alpha": args.alpha,
              "keep_gate": args.keep_gate, "selected_groups": args.selected_groups,
              "updates": len(upd), "baseline": metrics_for()}
    print("baseline", result["baseline"]["map50"], result["baseline"]["map50_95"], flush=True)

    if args.mode == "replace":
        for j, (nl, cc, kk) in upd.items():
            labels[j] = nl
        result["variant"] = metrics_for()
    else:
        extra = {}
        for j, (nl, cc, kk) in upd.items():
            # sample index of this candidate
            si = int(np.searchsorted(offsets, j, side="right") - 1)
            eb, es, el = extra.setdefault(si, ([], [], []))
            eb.append(boxes[j]); es.append(float(scores[j]) * args.alpha); el.append(nl)
        extra = {k: (np.asarray(v[0], np.float32), np.asarray(v[1], np.float32),
                     np.asarray(v[2], np.int64)) for k, v in extra.items()}
        result["variant"] = metrics_for(extra)
        result["extra_dets"] = int(sum(len(v[0]) for v in extra.values()))
    print("variant", result["variant"]["map50"], result["variant"]["map50_95"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
