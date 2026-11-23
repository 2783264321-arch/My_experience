#!/usr/bin/env python3
"""Source-group calibration of the magnitude CNN and B0 on the same cache.

Uses only the original fit/train side:
  - inner train / calibration split by group, fixed seed;
  - magnitude CNN trained on inner train (same 5-epoch recipe);
  - B0 refit on the same inner train;
  - one global threshold per model/seed chosen on calibration;
  - evaluation on the untouched original 366-instance dev with the same
    weights, before vs after threshold calibration.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn

from scripts.baseline3srrc_b0_same_windows import instance_features
from scripts.baseline3srrc_gt_pilot import TinyResNet, batches, metrics, predict, read_pilot_cache

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/baseline3_srrc_gt_pilot"
BASE = Path("/root/autodl-fs/runs_local/srrc-gt-pilot")
SEEDS = (1729, 1730, 1731)
SPLIT_SEED = 20261123


def threshold_search(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    candidates = {0.0}
    order = np.sort(np.unique(scores))
    for a, b in zip(order[:-1], order[1:]):
        candidates.add(float(0.5 * (a + b)))
    best = None
    for t in sorted(candidates):
        pred = (scores >= t).astype(int)
        rec = []
        for c in (0, 1):
            m = labels == c
            rec.append(float((pred[m] == c).mean()) if m.any() else 0.0)
        bal = float(np.mean(rec))
        key = (bal, -abs(t), -t)
        if best is None or key > best[0]:
            best = (key, float(t), bal, int(pred.sum()))
    return {"threshold": best[1], "balanced_accuracy": best[2],
            "pred11": best[3], "n_candidates": len(candidates)}


def eval_scores(scores, labels, strata, threshold):
    scores = np.asarray(scores, float)
    pred = (scores >= threshold).astype(int)
    out = metrics(np.asarray(labels, int), pred, np.asarray(strata))
    out["pred11"] = int(pred.sum())
    out["pred11_frac"] = float(pred.mean())
    out["auc"] = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else None
    return out


def pair_vs_default(scores, labels, strata):
    scores = np.asarray(scores, float)
    y = np.asarray(labels, int)
    default = (scores >= 0).astype(int)
    # caller replaces this with calibrated preds; helper is below
    return default


def net_change(y, ref, new, strata):
    y = np.asarray(y, int); ref = np.asarray(ref, int); new = np.asarray(new, int)
    strata = np.asarray(strata)
    corr = int(((ref != y) & (new == y)).sum())
    broken = int(((ref == y) & (new != y)).sum())
    p = float(binomtest(corr, corr + broken, 0.5).pvalue) if corr + broken else 1.0
    out = {"corrected": corr, "broken": broken, "net": corr - broken,
           "mcnemar_exact_p": p, "strata": {}}
    for s in ("clean", "weak", "strong"):
        m = strata == s
        c = int(((ref[m] != y[m]) & (new[m] == y[m])).sum())
        b = int(((ref[m] == y[m]) & (new[m] != y[m])).sum())
        out["strata"][s] = {"n": int(m.sum()), "corrected": c,
                            "broken": b, "net": c - b}
    return out


def write_csv(path, cache, ids, scores, threshold, default_threshold=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "signal_idx", "group_id", "class", "overlap",
                    "score", "pred_default", "pred_calibrated", "threshold"])
        for j, i in enumerate(ids):
            s = float(scores[j])
            w.writerow([cache["sample_ids"][i], cache["signal_indices"][i],
                        cache["groups"][i], int(cache["labels"][i]) + 10,
                        cache["strata"][i], s,
                        int(s >= default_threshold), int(s >= threshold),
                        float(threshold)])


def train_magnitude(windows, labels, inner_ids, seed, device, epochs=5, batch=8, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TinyResNet(in_channels=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    y = torch.from_numpy(labels).long().to(device)
    t0 = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(epochs):
        model.train()
        for ids in batches(rng.permutation(inner_ids), batch):
            opt.zero_grad(set_to_none=True)
            logits = predict(model, windows[ids].to(device), "magnitude")
            loss = lossf(logits, y[ids])
            loss.backward()
            opt.step()
    model.eval()
    peak = torch.cuda.max_memory_reserved() if device == "cuda" else None
    return model, {"seconds": time.perf_counter() - t0,
                   "peak_cuda_reserved_bytes": peak}


@torch.inference_mode()
def score_model(model, windows, ids, device):
    out = []
    for chunk in batches(ids, 32):
        logits = predict(model, windows[chunk].to(device), "magnitude").cpu().numpy()
        out.append(logits[:, 1] - logits[:, 0])
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=BASE / "fit_8win.npz")
    ap.add_argument("--out", type=Path, default=OUT / "threshold_calibration.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache, metadata = read_pilot_cache(args.cache)
    windows_np = cache.pop("windows")
    windows = torch.from_numpy(windows_np)
    labels = np.asarray(cache["labels"], int)
    is_dev = np.asarray(cache["is_dev"], bool)
    groups = np.asarray(cache["groups"])
    strata = np.asarray(cache["strata"])

    train_ids = np.flatnonzero(~is_dev)
    dev_ids = np.flatnonzero(is_dev)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    inner_rel, cal_rel = next(sgkf.split(np.zeros(len(train_ids)),
                                         labels[train_ids], groups[train_ids]))
    inner_ids = train_ids[inner_rel]
    cal_ids = train_ids[cal_rel]
    if set(groups[inner_ids]) & set(groups[cal_ids]):
        raise RuntimeError("group leak between inner train and calibration")
    split = {
        "seed": SPLIT_SEED, "n_inner": int(len(inner_ids)),
        "n_cal": int(len(cal_ids)), "n_dev": int(len(dev_ids)),
        "groups_inner": int(len(set(groups[inner_ids]))),
        "groups_cal": int(len(set(groups[cal_ids]))),
        "class_inner": {str(c): int((labels[inner_ids] == c).sum()) for c in (0, 1)},
        "class_cal": {str(c): int((labels[cal_ids] == c).sum()) for c in (0, 1)},
        "class_dev": {str(c): int((labels[dev_ids] == c).sum()) for c in (0, 1)},
        "strata_cal": {s: int((strata[cal_ids] == s).sum()) for s in ("clean", "weak", "strong")},
        "strata_dev": {s: int((strata[dev_ids] == s).sum()) for s in ("clean", "weak", "strong")},
        "group_leak": 0,
    }

    result = {"split": split, "b0": {}, "magnitude": {}}

    # B0: same 53 features, refit on inner train.
    t0 = time.perf_counter()
    feats = np.stack([instance_features(w, metadata["fs_hz"]) for w in windows_np])
    b0 = make_pipeline(StandardScaler(),
                       LogisticRegression(C=1.0, class_weight="balanced",
                                          max_iter=5000, solver="lbfgs"))
    with threadpool_limits(limits=1):
        b0.fit(feats[inner_ids], labels[inner_ids])
        b0_cal = b0.decision_function(feats[cal_ids])
        b0_dev = b0.decision_function(feats[dev_ids])
    thr = threshold_search(b0_cal, labels[cal_ids])
    b0_default = eval_scores(b0_dev, labels[dev_ids], strata[dev_ids], 0.0)
    b0_calibrated = eval_scores(b0_dev, labels[dev_ids], strata[dev_ids], thr["threshold"])
    b0_net = net_change(labels[dev_ids], (b0_dev >= 0).astype(int),
                        (b0_dev >= thr["threshold"]).astype(int), strata[dev_ids])
    result["b0"] = {
        "seconds": time.perf_counter() - t0,
        "calibration": thr,
        "default": b0_default,
        "calibrated": b0_calibrated,
        "net_vs_default": b0_net,
    }
    write_csv(OUT / "raw/calibration_b0_dev.csv", cache, dev_ids, b0_dev, thr["threshold"])

    for seed in SEEDS:
        model, tr = train_magnitude(windows, labels, inner_ids, seed, args.device)
        cal_scores = score_model(model, windows, cal_ids, args.device)
        dev_scores = score_model(model, windows, dev_ids, args.device)
        thr = threshold_search(cal_scores, labels[cal_ids])
        default = eval_scores(dev_scores, labels[dev_ids], strata[dev_ids], 0.0)
        calibrated = eval_scores(dev_scores, labels[dev_ids], strata[dev_ids], thr["threshold"])
        net = net_change(labels[dev_ids], (dev_scores >= 0).astype(int),
                         (dev_scores >= thr["threshold"]).astype(int), strata[dev_ids])
        result["magnitude"][str(seed)] = {
            "train": tr, "calibration": thr,
            "default": default, "calibrated": calibrated,
            "net_vs_default": net,
        }
        write_csv(OUT / f"raw/calibration_magnitude_seed{seed}_dev.csv",
                  cache, dev_ids, dev_scores, thr["threshold"])

    # Compare calibrated magnitude vs calibrated B0 on dev.
    b0_pred = (b0_dev >= result["b0"]["calibration"]["threshold"]).astype(int)
    for seed in SEEDS:
        path = OUT / f"raw/calibration_magnitude_seed{seed}_dev.csv"
        rows = list(csv.DictReader(path.open()))
        mag_pred = np.array([int(r["pred_calibrated"]) for r in rows])
        result["magnitude"][str(seed)]["vs_b0_calibrated"] = net_change(
            labels[dev_ids], b0_pred, mag_pred, strata[dev_ids])

    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "split": split,
        "b0": {"threshold": result["b0"]["calibration"]["threshold"],
               "default_bal": b0_default["all"]["balanced_accuracy"],
               "calibrated_bal": b0_calibrated["all"]["balanced_accuracy"],
               "net": b0_net["net"]},
        "magnitude": {
            s: {"threshold": v["calibration"]["threshold"],
                "cal_default": v["default"]["all"]["balanced_accuracy"],
                "calibrated": v["calibrated"]["all"]["balanced_accuracy"],
                "rec_default": v["default"]["all"]["recall_10_11"],
                "rec_calibrated": v["calibrated"]["all"]["recall_10_11"],
                "net": v["net_vs_default"]["net"]}
            for s, v in result["magnitude"].items()
        },
    }, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
