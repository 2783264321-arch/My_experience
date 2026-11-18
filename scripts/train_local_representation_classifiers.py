#!/usr/bin/env python3
"""Train small 3-class heads on aligned coarse / local-power / complex crops.

The same shared sample-level fit holdout is used for:
  - both bandwidth groups (0/1/2 and 3/4/5);
  - all representations;
  - all model seeds.

Representations are intentionally aligned by `collect_local_proposal_crops.py`:
coarse power, local log power, and local complex STFT real/imag all use the
same proposals, the same physical context, and the same [128, 256] grid
(local power and complex also share the same complex coefficients).

The training budget is identical for every representation: same architecture
depth/width, optimizer, epochs, batch size, class weights, and seed list.
Only the first conv input channel count changes for complex (1 -> 2).

Reported metrics:
  - accuracy and 3x3 confusion per bandwidth group;
  - macro-F1 and per-class precision/recall/F1;
  - accuracy on WiFi-clean / WiFi-overlap / cross-class-overlap / strong-overlap
    subsets, plus any-signal overlap and clean subsets;
  - per-seed values and mean/std.

Usage:
  PYTHONPATH=. python scripts/train_local_representation_classifiers.py \
    --data reports/local_rep/proposal_local_win0100.npz \
    --tag win0100 --reps coarse,local_power,local_complex --seeds 1729,1730,1731
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_proposal_classifiers import SmallCNN, split_by_sample

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"012": [0, 1, 2], "345": [3, 4, 5]}
ALL_CLASSES = [0, 1, 2, 3, 4, 5]
REPS = ("coarse", "local_power", "local_complex")


def standardize_coarse(x):
    """x [N,H,W] -> per-crop zero mean/unit std."""
    m = x.mean(axis=(1, 2), keepdims=True)
    s = x.std(axis=(1, 2), keepdims=True)
    return (x - m) / np.maximum(s, 1e-6)


def standardize_complex(x):
    """x [N,2,H,W] -> joint per-crop zero mean/unit std (preserve I/Q scale)."""
    m = x.mean(axis=(1, 2, 3), keepdims=True)
    s = x.std(axis=(1, 2, 3), keepdims=True)
    return (x - m) / np.maximum(s, 1e-6)


def load_rep(z, name):
    if name == "coarse":
        x = z["coarse"].astype(np.float32)
        x = standardize_coarse(x)[:, None]
    elif name == "local_power":
        x = z["local_power"].astype(np.float32)
        x = standardize_coarse(x)[:, None]
    elif name == "local_complex":
        x = z["local_complex"].astype(np.float32)
        x = standardize_complex(x)
    else:
        raise ValueError(name)
    # A handful of near-empty crops have extreme standardized values. Clip
    # all representations identically; the network has BatchNorm afterwards.
    return np.clip(x, -10.0, 10.0)


def confusion_metrics(conf):
    conf = np.asarray(conf, dtype=np.int64)
    n = int(conf.sum())
    acc = float(np.trace(conf) / n) if n else None
    per = {}
    f1s = []
    for i in range(conf.shape[0]):
        tp = int(conf[i, i])
        fp = int(conf[:, i].sum() - tp)
        fn = int(conf[i, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = (2 * prec * rec / (prec + rec)) if prec is not None and rec is not None and (prec + rec) > 0 else None
        per[str(i)] = {
            "support": int(conf[i, :].sum()),
            "precision": prec, "recall": rec, "f1": f1,
        }
        if f1 is not None:
            f1s.append(f1)
    return {
        "n": n, "accuracy": acc,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "confusion": conf.tolist(),
        "per_class": per,
    }


def subset_metrics(pred, y, mask):
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "accuracy": None, "macro_f1": None, "confusion": None}
    conf = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y[mask].tolist(), pred[mask].tolist()):
        conf[int(t), int(p)] += 1
    out = confusion_metrics(conf)
    return out


def evaluate(model, x, y, device, batch=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch]).to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds) if preds else np.zeros((0,), dtype=np.int64)


def train_one(x_tr, y_tr, x_va, y_va, n_classes, seed, epochs, batch, device,
              lr=1e-3, weight_decay=1e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    model = SmallCNN(n_classes, in_channels=int(x_tr.shape[1])).to(device)
    counts = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))
    weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
    ds_tr = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr))
    ds_va = TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va))
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=2,
                       pin_memory=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=2,
                       pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        seen = 0
        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=weight_t)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(yb)
            seen += len(yb)
        sched.step()
        # training-set accuracy is cheap enough and helps diagnose collapse
        tr_pred = evaluate(model, x_tr, y_tr, device)
        tr_acc = float((tr_pred == y_tr).mean()) if len(y_tr) else None
        history.append({"epoch": epoch + 1, "loss": total / max(seen, 1),
                        "train_acc": tr_acc})
    pred = evaluate(model, x_va, y_va, device)
    return model, pred, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--reps", default="coarse,local_power,local_complex")
    ap.add_argument("--groups", default="012,345")
    ap.add_argument("--seeds", default="1729,1730,1731")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=1729)
    ap.add_argument("--out", default="reports/local_rep/representation_comparison.json")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    z = np.load(data_path)
    labels = z["labels"].astype(np.int64)
    sid = z["sample_ids"].astype(str)
    n = len(labels)
    if not np.all(np.isin(labels, ALL_CLASSES)):
        raise SystemExit("this trainer expects only 0-5 crops")

    reps = [r.strip() for r in args.reps.split(",") if r.strip()]
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for r in reps:
        if r not in REPS:
            raise SystemExit(f"unknown rep {r}")
    for g in groups:
        if g not in GROUPS:
            raise SystemExit(f"unknown group {g}")

    # One shared sample-level split for every representation/group/seed.
    tr, va = split_by_sample(sid, labels, ALL_CLASSES, holdout=args.holdout,
                             seed=args.split_seed)
    print(f"data {data_path} crops={n} samples={len(set(sid))} "
          f"train_crops={int(tr.sum())} holdout_crops={int(va.sum())} "
          f"train_samples={len(set(sid[tr]))} holdout_samples={len(set(sid[va]))}",
          flush=True)

    overlap_sets = {
        "wifi_clean": z["overlap_wifi"] == 0,
        "wifi_overlap": z["overlap_wifi"] == 1,
        "wifi_cross_overlap": z["overlap_wifi_cross"] == 1,
        "wifi_same_overlap": z["overlap_wifi_same"] == 1,
        "wifi_strong_overlap": z["overlap_wifi_frac_max"] >= 0.25,
        "any_clean": z["overlap_any"] == 0,
        "any_overlap": z["overlap_any"] == 1,
    }

    result = {
        "data": str(data_path),
        "tag": args.tag or data_path.stem,
        "crops": int(n),
        "samples": int(len(set(sid))),
        "split": {
            "holdout_fraction": args.holdout,
            "split_seed": args.split_seed,
            "train_crops": int(tr.sum()),
            "holdout_crops": int(va.sum()),
            "train_samples": int(len(set(sid[tr]))),
            "holdout_samples": int(len(set(sid[va]))),
            "class_counts_train": np.bincount(labels[tr], minlength=6).tolist(),
            "class_counts_holdout": np.bincount(labels[va], minlength=6).tolist(),
        },
        "reps": {},
    }

    device = "cuda"
    for rep in reps:
        x = load_rep(z, rep)
        result["reps"][rep] = {}
        for gname in groups:
            classes = GROUPS[gname]
            gmask = np.isin(labels, classes)
            trm = tr & gmask
            vam = va & gmask
            x_tr = x[trm].astype(np.float32)
            x_va = x[vam].astype(np.float32)
            local = {c: i for i, c in enumerate(classes)}
            y_tr = np.array([local[int(v)] for v in labels[trm]], dtype=np.int64)
            y_va = np.array([local[int(v)] for v in labels[vam]], dtype=np.int64)
            print(f"{rep} {gname}: train={len(y_tr)} holdout={len(y_va)} "
                  f"in_channels={x_tr.shape[1]}", flush=True)
            seed_rows = []
            for seed in seeds:
                t0 = time.perf_counter()
                model, pred, history = train_one(
                    x_tr, y_tr, x_va, y_va, len(classes), seed,
                    args.epochs, args.batch, device,
                    lr=args.lr, weight_decay=args.weight_decay,
                )
                conf = np.zeros((3, 3), dtype=np.int64)
                for t, p in zip(y_va.tolist(), pred.tolist()):
                    conf[int(t), int(p)] += 1
                main_metrics = confusion_metrics(conf)
                subsets = {}
                for name, mask_all in overlap_sets.items():
                    m = mask_all[vam]
                    subsets[name] = subset_metrics(pred, y_va, m)
                seed_rows.append({
                    "seed": seed,
                    "metrics": main_metrics,
                    "overlap_subsets": subsets,
                    "history": history,
                    "seconds": time.perf_counter() - t0,
                })
                print(f"  seed {seed}: acc={main_metrics['accuracy']:.4f} "
                      f"macroF1={main_metrics['macro_f1']} "
                      f"conf={main_metrics['confusion']} "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)

            accs = [r["metrics"]["accuracy"] for r in seed_rows
                    if r["metrics"]["accuracy"] is not None]
            f1s = [r["metrics"]["macro_f1"] for r in seed_rows
                   if r["metrics"]["macro_f1"] is not None]
            conf_sum = np.sum([r["metrics"]["confusion"] for r in seed_rows], axis=0)
            # Aggregate overlap accuracy across seeds (mean of per-seed values).
            sub_agg = {}
            for name in overlap_sets:
                vals = [r["overlap_subsets"][name]["accuracy"] for r in seed_rows
                        if r["overlap_subsets"][name]["accuracy"] is not None]
                ns = [r["overlap_subsets"][name]["n"] for r in seed_rows]
                sub_agg[name] = {
                    "n": int(ns[0]) if ns else 0,
                    "accuracy_mean": float(np.mean(vals)) if vals else None,
                    "accuracy_std": float(np.std(vals)) if vals else None,
                }
            result["reps"][rep][gname] = {
                "classes": classes,
                "train_crops": int(trm.sum()),
                "holdout_crops": int(vam.sum()),
                "train_samples": int(len(set(sid[trm]))),
                "holdout_samples": int(len(set(sid[vam]))),
                "in_channels": int(x_tr.shape[1]),
                "accuracy_mean": float(np.mean(accs)) if accs else None,
                "accuracy_std": float(np.std(accs)) if accs else None,
                "macro_f1_mean": float(np.mean(f1s)) if f1s else None,
                "macro_f1_std": float(np.std(f1s)) if f1s else None,
                "confusion_sum": conf_sum.tolist(),
                "confusion_mean_normalized": (conf_sum / max(1, conf_sum.sum())).tolist(),
                "overlap_subsets": sub_agg,
                "seeds": seed_rows,
            }

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("wrote", out)

    # Compact console table.
    for rep in reps:
        for gname in groups:
            r = result["reps"][rep][gname]
            print(f"{rep:14s} {gname}: acc {r['accuracy_mean']:.4f} "
                  f"± {r['accuracy_std']:.4f}  macroF1 {r['macro_f1_mean']} "
                  f"conf {r['confusion_sum']}")
    print(json.dumps({
        "tag": result["tag"],
        "split": result["split"],
        "summary": {
            rep: {g: {
                "acc": result["reps"][rep][g]["accuracy_mean"],
                "acc_std": result["reps"][rep][g]["accuracy_std"],
                "macro_f1": result["reps"][rep][g]["macro_f1_mean"],
                "conf": result["reps"][rep][g]["confusion_sum"],
                "overlap": result["reps"][rep][g]["overlap_subsets"],
            } for g in groups}
            for rep in reps
        },
    }, indent=2))


if __name__ == "__main__":
    main()
