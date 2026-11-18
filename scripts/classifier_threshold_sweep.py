#!/usr/bin/env python3
"""Threshold sweep for the existing coarse-crop 0/1/2 and 3/4/5 classifiers.

Uses the same fit crops and the same sample-level holdout split as
train_proposal_classifiers.py. For each gate (confidence or margin) and every
threshold, the adopted label is the classifier label only when the gate passes;
otherwise the detector label is kept. Accuracy is computed from these adopted
labels.

Transition counts:
  wrong_to_correct      : old != GT, adopted == GT
  correct_to_wrong      : old == GT, adopted != GT
  wrong_to_other_wrong  : old != GT, adopted != GT, adopted != old
  net_correct           = wrong_to_correct - correct_to_wrong
                        == correct_after - correct_before  (asserted)

Routing rows:
  - GT-routed: classifier chosen by GT family (clean training routing)
  - pred-routed all: classifier chosen by detector predicted family
  - pred-routed same: clean same-family subset
  - cross-family only: GT/pred family mismatch
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_proposal_classifiers import SmallCNN, split_by_sample

ROOT = Path(__file__).resolve().parents[1]
CONF_THRESHOLDS = [0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
MARGIN_THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


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
    new_global = np.array([classes[i] for i in out[:, 0].astype(np.int64)])
    return new_global, out[:, 1], out[:, 2]


def stats(y_gt, y_old, y_new, gate, thresholds, sample_ids, gate_name):
    rows = []
    n = len(y_gt)
    for t in thresholds:
        adopted = np.where(gate >= t, y_new, y_old)
        changed = adopted != y_old
        w2c = (y_old != y_gt) & (adopted == y_gt)
        c2w = (y_old == y_gt) & (adopted != y_gt)
        w2w = (y_old != y_gt) & (adopted != y_gt) & (adopted != y_old)
        unchanged_correct = (adopted == y_old) & (y_old == y_gt)
        unchanged_wrong = (adopted == y_old) & (y_old != y_gt)
        cb = int((y_old == y_gt).sum())
        ca = int((adopted == y_gt).sum())
        net = int(w2c.sum() - c2w.sum())
        net_acc = ca - cb
        assert net == net_acc, (gate_name, t, net, net_acc)
        assert int(changed.sum()) == int(w2c.sum() + c2w.sum() + w2w.sum())
        ids = sorted({str(x) for x in sample_ids[changed]})
        row = {
            "gate": gate_name,
            "threshold": float(t),
            "n": int(n),
            "changed": int(changed.sum()),
            "wrong_to_correct": int(w2c.sum()),
            "correct_to_wrong": int(c2w.sum()),
            "wrong_to_other_wrong": int(w2w.sum()),
            "unchanged_correct": int(unchanged_correct.sum()),
            "unchanged_wrong": int(unchanged_wrong.sum()),
            "correct_before": cb,
            "correct_after": ca,
            "net_correct": net,
            "net_acc_check": net_acc,
            "change_ratio": float(changed.mean()) if n else None,
            "change_precision": float(w2c.sum() / max(changed.sum(), 1)),
            "acc_before": float(cb / n) if n else None,
            "acc_after": float(ca / n) if n else None,
            "n_changed_samples": len(ids),
        }
        if abs(float(t) - 0.7) < 1e-9:
            row["changed_sample_ids"] = ids
        rows.append(row)
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
        new, conf, margin = infer(models[name][0], crops[idx], classes)
        hold[name] = (idx, new, conf, margin)

    idx_all = np.concatenate([hold["012"][0], hold["345"][0]])
    new_gt_route = np.concatenate([hold["012"][1], hold["345"][1]])
    conf_gt_route = np.concatenate([hold["012"][2], hold["345"][2]])
    margin_gt_route = np.concatenate([hold["012"][3], hold["345"][3]])
    y_gt = labels[idx_all]
    y_old = pred_classes[idx_all]
    sid_all = sample_ids[idx_all]

    # deployment routing by detector predicted family
    new_pred = y_old.copy()
    conf_pred = np.zeros(len(idx_all), dtype=np.float32)
    margin_pred = np.zeros(len(idx_all), dtype=np.float32)
    for name, classes in [("012", [0, 1, 2]), ("345", [3, 4, 5])]:
        sel = np.where(np.isin(y_old, classes))[0]
        if len(sel) == 0:
            continue
        new_sel, conf_sel, margin_sel = infer(models[name][0], crops[idx_all[sel]], classes)
        new_pred[sel] = new_sel
        conf_pred[sel] = conf_sel
        margin_pred[sel] = margin_sel

    gt_family = np.where(np.isin(y_gt, [0, 1, 2]), "012", "345")
    pred_family = np.where(np.isin(y_old, [0, 1, 2]), "012",
                           np.where(np.isin(y_old, [3, 4, 5]), "345", "other"))
    same = gt_family == pred_family
    cross = ~same
    masks = {
        "gt_routed": np.ones(len(y_gt), dtype=bool),
        "pred_routed_all": np.ones(len(y_gt), dtype=bool),
        "pred_routed_same_family": same,
        "cross_family_only": cross,
    }
    new_by_route = {
        "gt_routed": new_gt_route,
        "pred_routed_all": new_pred,
        "pred_routed_same_family": new_pred,
        "cross_family_only": new_pred,
    }
    conf_by_route = {
        "gt_routed": conf_gt_route,
        "pred_routed_all": conf_pred,
        "pred_routed_same_family": conf_pred,
        "cross_family_only": conf_pred,
    }
    margin_by_route = {
        "gt_routed": margin_gt_route,
        "pred_routed_all": margin_pred,
        "pred_routed_same_family": margin_pred,
        "cross_family_only": margin_pred,
    }

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
        "confidence": {},
        "margin": {},
    }
    for route, mask in masks.items():
        out["confidence"][route] = stats(
            y_gt[mask], y_old[mask], new_by_route[route][mask],
            conf_by_route[route][mask], CONF_THRESHOLDS, sid_all[mask], "confidence",
        )
        out["margin"][route] = stats(
            y_gt[mask], y_old[mask], new_by_route[route][mask],
            margin_by_route[route][mask], MARGIN_THRESHOLDS, sid_all[mask], "margin",
        )

    # explicit checks requested by the protocol
    t0 = {
        route: next(r for r in out["confidence"][route] if abs(r["threshold"]) < 1e-9)
        for route in out["confidence"]
    }
    t07 = {
        route: next(r for r in out["confidence"][route] if abs(r["threshold"] - 0.7) < 1e-9)
        for route in out["confidence"]
    }
    out["checks"] = {
        "expected_pred_t0_net": -48,
        "pred_routed_all_t0_net": t0["pred_routed_all"]["net_correct"],
        "pred_routed_same_t0_net": t0["pred_routed_same_family"]["net_correct"],
        "pred_t0_match_expected": (
            t0["pred_routed_all"]["net_correct"] == -48
            and t0["pred_routed_same_family"]["net_correct"] == -48
        ),
        "t0": t0,
        "t0_70": t07,
        "all_rows_net_check": all(
            r["net_correct"] == r["net_acc_check"]
            for sweep in [out["confidence"], out["margin"]]
            for rows in sweep.values() for r in rows
        ),
    }

    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    def slim(rows):
        return [{k: r[k] for k in [
            "threshold", "n", "changed", "wrong_to_correct", "correct_to_wrong",
            "wrong_to_other_wrong", "net_correct", "change_ratio",
            "change_precision", "acc_before", "acc_after", "n_changed_samples",
        ]} for r in rows]

    print(json.dumps({
        "holdout_crops": out["holdout_crops"],
        "routing": {k: out["routing"][k] for k in ["same_family", "cross_family", "cross_family_ratio"]},
        "checks": {
            "expected_pred_t0_net": out["checks"]["expected_pred_t0_net"],
            "pred_routed_all_t0_net": out["checks"]["pred_routed_all_t0_net"],
            "pred_routed_same_t0_net": out["checks"]["pred_routed_same_t0_net"],
            "pred_t0_match_expected": out["checks"]["pred_t0_match_expected"],
            "all_rows_net_check": out["checks"]["all_rows_net_check"],
        },
        "confidence_pred_routed_all_t0_t070": {
            "t0": t0["pred_routed_all"],
            "t0.70": t07["pred_routed_all"],
        },
        "confidence_pred_routed_same_t0_t070": {
            "t0": t0["pred_routed_same_family"],
            "t0.70": t07["pred_routed_same_family"],
        },
        "confidence_gt_routed": slim(out["confidence"]["gt_routed"]),
        "confidence_pred_routed_same": slim(out["confidence"]["pred_routed_same_family"]),
        "confidence_pred_routed_all": slim(out["confidence"]["pred_routed_all"]),
        "margin_gt_routed": slim(out["margin"]["gt_routed"]),
        "margin_pred_routed_same": slim(out["margin"]["pred_routed_same_family"]),
    }, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
