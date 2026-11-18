#!/usr/bin/env python3
"""Train two small coarse-crop proposal classifiers: 0/1/2 and 3/4/5.

Inputs: reports/proposal_crops_fit.npz from collect_proposal_crops.py.
Split is by source sample (not by crop) to avoid leakage. The fixed val is not
used for training or early stopping; a 10% fit holdout is used for monitoring.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int, in_h: int = 128, in_w: int = 256,
                 in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))


def split_by_sample(sample_ids, labels, classes, holdout=0.1, seed=1729):
    rng = random.Random(seed)
    samples = sorted(set(str(s) for s in sample_ids))
    rng.shuffle(samples)
    n_hold = max(1, int(round(len(samples) * holdout)))
    hold = set(samples[:n_hold])
    tr = np.array([str(s) not in hold for s in sample_ids])
    va = ~tr
    # make sure each class exists in train; if a class is missing, move one
    # sample of that class from holdout to train
    for c in classes:
        if not np.any(tr & (labels == c)):
            idx = np.where((labels == c) & va)[0]
            if len(idx):
                s = str(sample_ids[idx[0]])
                tr |= np.array([str(x) == s for x in sample_ids])
                va = ~tr
    return tr, va


def evaluate(model, loader, device, n_classes):
    model.eval()
    correct = total = 0
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            y = y.to(device)
            p = model(x).argmax(1)
            correct += int((p == y).sum())
            total += len(y)
            for t, q in zip(y.cpu().tolist(), p.cpu().tolist()):
                conf[t, q] += 1
    return correct / max(total, 1), conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="reports/proposal_crops_fit.npz")
    ap.add_argument("--out-dir", default="runs/yolo11s_c1_directreg_2048x512/proposal_classifiers")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1729)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda"

    z = np.load(ROOT / args.crops if not Path(args.crops).is_absolute() else args.crops)
    crops = z["crops"].astype(np.float32)  # [N,H,W]
    labels = z["labels"].astype(np.int64)
    sample_ids = z["sample_ids"]
    pred_classes = z["pred_classes"].astype(np.int64)
    meta = json.loads(str(z["meta"][0])) if "meta" in z else {}
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"crops {crops.shape}, labels {np.bincount(labels, minlength=6).tolist()}, "
          f"wrong-original {int((labels != pred_classes).sum())}", flush=True)

    families = {"012": [0, 1, 2], "345": [3, 4, 5]}
    report = {"crops": str(args.crops), "n_crops": int(len(labels)),
              "meta": meta, "families": {}}
    for name, classes in families.items():
        keep = np.isin(labels, classes)
        c = crops[keep]
        y = labels[keep]
        sid = sample_ids[keep]
        if len(c) == 0:
            print(f"skip {name}: no crops", flush=True)
            continue
        tr, va = split_by_sample(sid, y, classes, holdout=args.holdout, seed=args.seed)
        x_tr = torch.from_numpy(c[tr]).unsqueeze(1)
        y_tr_global = y[tr]
        x_va = torch.from_numpy(c[va]).unsqueeze(1)
        y_va_global = y[va]
        local = {c: i for i, c in enumerate(classes)}
        y_tr = torch.tensor([local[int(v)] for v in y_tr_global.tolist()], dtype=torch.long)
        y_va = torch.tensor([local[int(v)] for v in y_va_global.tolist()], dtype=torch.long)
        counts = np.bincount(y_tr.numpy(), minlength=len(classes)).astype(np.float64)
        weights = counts.sum() / (len(classes) * np.maximum(counts, 1))
        weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
        ds_tr = TensorDataset(x_tr, y_tr)
        ds_va = TensorDataset(x_va, y_va)
        dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=2,
                           pin_memory=True, drop_last=False)
        dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=2,
                           pin_memory=True)
        model = SmallCNN(len(classes)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best_acc = -1.0
        best_state = None
        best_conf = None
        for epoch in range(args.epochs):
            model.train()
            t0 = time.perf_counter()
            total_loss = 0.0
            n = 0
            for x, yy in dl_tr:
                x = x.to(device, non_blocking=True)
                yy = yy.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, yy, weight=weight_t)
                loss.backward()
                opt.step()
                total_loss += float(loss.detach()) * len(yy)
                n += len(yy)
            sched.step()
            acc, conf = evaluate(model, dl_va, device, len(classes))
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_conf = conf
            print(f"{name} epoch {epoch+1}/{args.epochs} loss {total_loss/max(n,1):.4f} "
                  f"holdout_acc {acc:.4f} best {best_acc:.4f} "
                  f"{time.perf_counter()-t0:.1f}s", flush=True)
        payload = {
            "state_dict": best_state,
            "classes": classes,
            "input_hw": [int(crops.shape[1]), int(crops.shape[2])],
            "margin": meta.get("margin", 0.1),
            "source": str(args.crops),
            "holdout_acc": best_acc,
            "holdout_confusion": best_conf.tolist() if best_conf is not None else None,
            "train_samples": len(set(str(s) for s in sid[tr])),
            "holdout_samples": len(set(str(s) for s in sid[va])),
            "train_crops": int(tr.sum()), "holdout_crops": int(va.sum()),
            "class_counts_train": {str(c): int((y_tr_global == c).sum()) for c in classes},
        }
        path = out_dir / f"proposal_cls_{name}.pt"
        torch.save(payload, path)
        report["families"][name] = {k: v for k, v in payload.items() if k != "state_dict"}
        print(f"saved {path} best_holdout_acc {best_acc:.4f} "
              f"confusion {best_conf.tolist() if best_conf is not None else None}", flush=True)

    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2))
    print("wrote", out_dir / "training_report.json")


if __name__ == "__main__":
    main()
