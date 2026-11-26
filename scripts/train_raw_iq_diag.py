#!/usr/bin/env python3
"""Train a 1D raw-IQ classifier on canonical local crops.

Diagnostic only: compares candidate-level positive accuracy against the
local-power/physical-feature arm on the same fixed queries.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"012": [0, 1, 2], "345": [3, 4, 5], "1011": [10, 11]}


class IQNet(nn.Module):
    def __init__(self, n_classes: int, width: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(2, width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(width), nn.SiLU(inplace=True),
            nn.Conv1d(width, width * 2, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(width * 2), nn.SiLU(inplace=True),
            nn.Conv1d(width * 2, width * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(width * 4), nn.SiLU(inplace=True),
            nn.Conv1d(width * 4, width * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(width * 4), nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(nn.Linear(width * 4, 128), nn.SiLU(inplace=True),
                                nn.Linear(128, n_classes))
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):
        h = self.body(x)
        return self.fc(self.pool(h).flatten(1))


class IQDataset(Dataset):
    def __init__(self, iq, lengths, y, augment=False):
        self.iq = iq
        self.lengths = lengths
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.iq[i]).float()
        if self.augment:
            # Random global phase rotation; magnitude/phase structure preserved.
            th = torch.rand(()) * 6.283185307179586
            c, s = torch.cos(th), torch.sin(th)
            i0, q0 = x[0].clone(), x[1].clone()
            x[0] = c * i0 - s * q0
            x[1] = s * i0 + c * q0
        return x, int(self.y[i])


def split_by_sample(sids, y, n_classes, holdout, seed):
    rng = random.Random(seed)
    samples = sorted(set(str(s) for s in sids))
    rng.shuffle(samples)
    hold = set(samples[:max(1, int(round(len(samples) * holdout)))])
    va = np.array([str(s) in hold for s in sids])
    tr = ~va
    for c in range(n_classes):
        if not np.any(tr & (y == c)):
            idx = np.where((y == c) & va)[0]
            if len(idx):
                tr |= np.array([str(x) == str(sids[idx[0]]) for x in sids])
                va = ~tr
    return tr, va


def evaluate(model, loader, device):
    model.eval(); preds, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).argmax(1).cpu().numpy()); ys.append(y.numpy())
    p = np.concatenate(preds); y = np.concatenate(ys)
    conf = np.zeros((model.fc[-1].out_features, model.fc[-1].out_features), np.int64)
    for t, q in zip(y, p):
        conf[int(t), int(q)] += 1
    acc = float(np.trace(conf) / max(conf.sum(), 1))
    f1s = []
    for c in range(len(conf)):
        tp = conf[c, c]; fp = conf[:, c].sum() - tp; fn = conf[c, :].sum() - tp
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return acc, float(np.mean(f1s)), conf.tolist(), p, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--group", choices=tuple(GROUPS), required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=1729)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    z = np.load(args.data, allow_pickle=True)
    labels = z["label"].astype(np.int64); pred = z["pred_label"].astype(np.int64)
    keep = z["keep"].astype(np.int64)
    src = np.array(GROUPS[args.group])
    pos = (keep == 0) & np.isin(labels, src)
    neg = (keep == 1) & np.isin(pred, src)
    sel = pos | neg
    idx = np.flatnonzero(sel)
    y = np.where(keep[idx] == 1, len(src),
                 np.searchsorted(src, labels[idx])).astype(np.int64)
    n_out = len(src) + 1
    sids = z["sample_ids"][idx].astype(str)
    tr, va = split_by_sample(sids, y, n_out, args.holdout, args.split_seed)
    print(f"{args.group}: queries={len(idx)} train={int(tr.sum())} holdout={int(va.sum())} "
          f"classes={np.bincount(y, minlength=n_out).tolist()} max_len={z['iq'].shape[-1]}",
          flush=True)
    ds_tr = IQDataset(z["iq"][idx][tr], z["lengths"][idx][tr], y[tr], augment=True)
    ds_va = IQDataset(z["iq"][idx][va], z["lengths"][idx][va], y[va], augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True)
    model = IQNet(n_out, width=args.width).to(args.device)
    counts = np.bincount(y[tr], minlength=n_out).astype(np.float32)
    weight = torch.tensor(counts.sum() / (n_out * np.maximum(counts, 1)),
                          device=args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = None; hist = []; t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train(); total = seen = 0
        for x, yy in dl_tr:
            x = x.to(args.device, non_blocking=True); yy = yy.to(args.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, yy, weight=weight)
            loss.backward(); opt.step()
            total += float(loss.detach()) * len(yy); seen += len(yy)
        sched.step()
        acc, f1, conf, p, t = evaluate(model, dl_va, args.device)
        hist.append({"epoch": epoch + 1, "loss": total / max(seen, 1),
                     "holdout_acc": acc, "holdout_f1": f1})
        if best is None or (f1, acc) > (best["f1"], best["acc"]):
            best = {"epoch": epoch + 1, "acc": acc, "f1": f1, "conf": conf,
                    "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
        print(f"{args.group} ep {epoch+1}/{args.epochs} loss {total/max(seen,1):.4f} "
              f"acc {acc:.4f} f1 {f1:.4f}", flush=True)
    model.load_state_dict(best["state"])
    acc, f1, conf, p, t = evaluate(model, dl_va, args.device)
    pos_m = t < (n_out - 1); keep_m = t == (n_out - 1)
    pos_acc = float((p[pos_m] == t[pos_m]).mean()) if pos_m.any() else None
    keep_rec = float((p[keep_m] == (n_out - 1)).mean()) if keep_m.any() else None
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best["state"], "group": args.group, "classes": src.tolist(),
                "num_classes": n_out, "epoch": best["epoch"]}, args.out / "best.pt")
    meta = {"group": args.group, "queries": len(idx), "train": int(tr.sum()),
            "holdout": int(va.sum()), "best_epoch": best["epoch"],
            "holdout_acc": best["acc"], "holdout_f1": best["f1"],
            "holdout_pos_acc": pos_acc, "keep_recall": keep_rec,
            "confusion": best["conf"], "max_len": int(z["iq"].shape[-1]),
            "seconds": time.perf_counter() - t0,
            "params": sum(p.numel() for p in model.parameters()),
            "width": args.width}
    (args.out / "summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    (args.out / "history.json").write_text(json.dumps(hist, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
