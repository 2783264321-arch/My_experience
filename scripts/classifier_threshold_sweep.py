#!/usr/bin/env python3
"""Threshold sweep for the existing coarse-crop 0/1/2 and 3/4/5 classifiers.

Uses the same fit crops and the same sample-level holdout split as
train_proposal_classifiers.py. For each GT-family holdout crop, computes the
classifier's proposed class and confidence, then sweeps confidence thresholds:
  changed / corrected / wrong-change / change ratio / net / accuracy.
Also compares:
  - GT-family routing (as in training): classifier chosen by GT family;
  - predicted-family routing (deployment): classifier chosen by detector class;
  - same-family-only deployment (clean subset) and cross-family cases.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from scripts.train_proposal_classifiers import SmallCNN, split_by_sample

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def load_cls(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = SmallCNN(len(ck["classes"])).cuda()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


@torch.no_grad()
def infer(model, crops, classes, batch=128):
    out = []
    for i in range(0, len(crops), batch):
        x = torch.from_numpy(crops[i:i + batch]).unsqueeze(1).float().cuda()
        logits = model(x)
        probs = torch.softmax(logits, 1)
        top2 = probs.topk(2, dim=1).values
        out.append(torch.stack([
            probs.argmax(1).float(),
            top2[:, 0],
            top2[:, 0] - top2[:, 1],
        ], 1).cpu())
    out = torch.cat(out).numpy()
    new_local = out[:, 0].astype(np.int64)
    conf = out[:, 1]
    margin = out[:, 2]
    new_global = np.array([classes[i] for i in new_local])
    return new_global, conf, margin


def stats(y_gt, y_old, y_new, conf, margin, thresholds):
    rows = []
    for t in thresholds:
        changed = (y_new != y_old) & (conf >= t)
        corrected = changed & (y_new == y_gt)
        wrong = changed & (y_new != y_gt)
        n = len(y_gt)
        rows.append({
            "threshold": t,
            "n": int(n),
            "changed": int(changed.sum()),
            "corrected": int(corrected.sum()),
            "wrong_change": int(wrong.sum()),
            "net": int(corrected.sum() - wrong.sum()),
            "change_ratio": float(changed.mean()) if n else None,
            "change_precision": float(corrected.sum() / max(changed.sum(), 1)),
            "acc_before": float((y_old == y_gt).mean()) if n else None,
            "acc_after": float((y_new == y_gt).mean()) if n else None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="reports/proposal_crops_fit.npz")
    ap.add_argument("--cls-dir", default="runs/yolo11s_c1_directreg_2048x512/proposal_classifiers")
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--out", default="reports/classifier_threshold_sweep.json")
    args = ap.parse_args()

    z = np.load(ROOT / args.crops if not Path(args.crops).is_absolute() else args.crops)
    crops = z["crops"].astype(np.float32)
    labels = z["labels"].astype(np.int64)
    pred_classes = z["pred_classes"].astype(np.int64)
    sample_ids = z["sample_ids"]
    cls_dir = ROOT / args.cls_dir
    models = {}
    for name, classes in [("012", [0, 1, 2]), ("345", [3, 4, 5])]:
        m, ck = load_cls(cls_dir / f"proposal_cls_{name}.pt")
        models[name] = (m, classes)

    hold = {}
    for name, classes in [("012", [0, 1, 2]), ("345", [3, 4, 5])]:
        keep = np.isin(labels, classes)
        sid = sample_ids[keep]
        y = labels[keep]
        tr, va = split_by_sample(sid, y, classes, holdout=args.holdout, seed=args.seed)
        idx = np.where(keep)[0][va]
        hold[name] = idx
        new, conf, margin = infer(models[name][0], crops[idx], classes)
        hold[name] = (idx, new, conf, margin)

    # union of GT-family holdout crops (each crop belongs to one GT family)
    idx_all = np.concatenate([hold["012"][0], hold["345"][0]])
    new_gt_route = np.concatenate([hold["012"][1], hold["345"][1]])
    conf_gt_route = np.concatenate([hold["012"][2], hold["345"][2]])
    margin_gt_route = np.concatenate([hold["012"][3], hold["345"][3]])
    y_gt = labels[idx_all]
    y_old = pred_classes[idx_all]
    sid_all = sample_ids[idx_all]

    # deployment routing by predicted class family; if pred outside 0-5, no change.
    new_pred = y_old.copy()
    conf_pred = np.zeros(len(idx_all), dtype=np.float32)
    for name, classes in [("012", [0, 1, 2]), ("345", [3, 4, 5])]:
        sel = np.where(np.isin(y_old, classes))[0]
        if len(sel) == 0:
            continue
        # classifier output for these crops (use model of predicted family)
        new_sel, conf_sel, _ = infer(models[name][0], crops[idx_all[sel]], classes)
        new_pred[sel] = new_sel
        conf_pred[sel] = conf_sel
    # for pred outside 0-5, new=old and conf=0 -> never changed.

    gt_family = np.where(np.isin(y_gt, [0, 1, 2]), "012", "345")
    pred_family = np.where(np.isin(y_old, [0, 1, 2]), "012",
                           np.where(np.isin(y_old, [3, 4, 5]), "345", "other"))
    same = gt_family == pred_family
    cross = ~same
    out = {
        "holdout_crops": int(len(idx_all)),
        "gt_family_counts": {k: int((gt_family == k).sum()) for k in ["012", "345"]},
        "pred_family_counts": {k: int((pred_family == k).sum()) for k in ["012", "345", "other"]},
        "routing": {
            "same_family": int(same.sum()),
            "cross_family": int(cross.sum()),
            "cross_family_ratio": float(cross.mean()),
            "cross_family_examples": [
                {"sample_id": str(sid_all[i]), "gt": int(y_gt[i]), "pred": int(y_old[i]),
                 "gt_family": str(gt_family[i]), "pred_family": str(pred_family[i])}
                for i in np.where(cross)[0][:20]
            ],
        },
        "gt_routed": stats(y_gt, y_old, new_gt_route, conf_gt_route, margin_gt_route, THRESHOLDS),
        "pred_routed_all": stats(y_gt, y_old, new_pred, conf_pred, np.zeros_like(conf_pred), THRESHOLDS),
        "pred_routed_same_family": stats(
            y_gt[same], y_old[same], new_pred[same], conf_pred[same], np.zeros_like(conf_pred[same]), THRESHOLDS
        ),
        "cross_family_only": stats(
            y_gt[cross], y_old[cross], new_pred[cross], conf_pred[cross], np.zeros_like(conf_pred[cross]), THRESHOLDS
        ),
        "thresholds": THRESHOLDS,
    }
    # margin sweep at threshold 0 for the clean GT-routed and same-family sets
    out["margin_sweep"] = {}
    for label, mask in [("gt_routed", np.ones(len(y_gt), dtype=bool)),
                        ("same_family", same)]:
        rows = []
        for mt in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
            changed = (new_gt_route[mask] != y_old[mask]) & (margin_gt_route[mask] >= mt)
            corrected = changed & (new_gt_route[mask] == y_gt[mask])
            wrong = changed & (new_gt_route[mask] != y_gt[mask])
            rows.append({
                "margin": mt, "n": int(mask.sum()),
                "changed": int(changed.sum()), "corrected": int(corrected.sum()),
                "wrong_change": int(wrong.sum()),
                "net": int(corrected.sum() - wrong.sum()),
                "change_precision": float(corrected.sum() / max(changed.sum(), 1)),
            })
        out["margin_sweep"][label] = rows

    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "holdout_crops": out["holdout_crops"],
        "gt_family_counts": out["gt_family_counts"],
        "pred_family_counts": out["pred_family_counts"],
        "routing": {k: out["routing"][k] for k in ["same_family", "cross_family", "cross_family_ratio"]},
        "gt_routed": out["gt_routed"],
        "pred_routed_same_family": out["pred_routed_same_family"],
        "cross_family_only": out["cross_family_only"],
        "margin_sweep": out["margin_sweep"],
    }, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
