#!/usr/bin/env python3
"""Train B (local power) / C (power + physical features) group classifiers."""
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
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"012": [0, 1, 2], "345": [3, 4, 5], "1011": [10, 11]}


class LocalNet(nn.Module):
    def __init__(self, n_classes: int, feat_dim: int = 0, meta_dim: int = 0):
        super().__init__()
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2 = m.layer1, m.layer2
        self.layer3, self.layer4 = m.layer3, m.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 256)
        self.feat_dim = feat_dim
        self.meta_dim = meta_dim
        if feat_dim:
            self.feat = nn.Sequential(nn.Linear(feat_dim, 128), nn.ReLU(inplace=True),
                                      nn.Linear(128, 64), nn.ReLU(inplace=True))
        if meta_dim:
            self.meta = nn.Sequential(nn.Linear(meta_dim, 32), nn.ReLU(inplace=True),
                                      nn.Linear(32, 32), nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(256 + (64 if feat_dim else 0) + (32 if meta_dim else 0),
                             n_classes)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, f=None, m=None):
        h = self.stem(x)
        h = self.layer1(h); h = self.layer2(h)
        h = self.layer3(h); h = self.layer4(h)
        h = self.pool(h).flatten(1)
        h = F.relu(self.fc1(h))
        if self.feat_dim:
            h = torch.cat([h, self.feat(f)], dim=1)
        if self.meta_dim:
            h = torch.cat([h, self.meta(m)], dim=1)
        return self.fc2(h)


class CropDataset(Dataset):
    def __init__(self, power, feats, meta, y, use_feats):
        self.power = power
        self.feats = feats
        self.meta = meta
        self.y = y
        self.use_feats = use_feats

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        p = torch.from_numpy(self.power[i]).float()
        p = (p - p.mean()) / p.std().clamp_min(1e-6)
        p = p.clamp(-5, 5)
        f = torch.from_numpy(self.feats[i]) if self.use_feats else torch.zeros(0)
        return p, f, torch.from_numpy(self.meta[i]).float(), int(self.y[i])


def split_by_sample(sample_ids, y, n_classes, holdout, seed):
    rng = random.Random(seed)
    samples = sorted(set(str(s) for s in sample_ids))
    rng.shuffle(samples)
    hold = set(samples[:max(1, int(round(len(samples) * holdout)))])
    va = np.array([str(s) in hold for s in sample_ids])
    tr = ~va
    for c in range(n_classes):
        if not np.any(tr & (y == c)):
            idx = np.where((y == c) & va)[0]
            if len(idx):
                tr |= np.array([str(x) == str(sample_ids[idx[0]]) for x in sample_ids])
                va = ~tr
    return tr, va


def evaluate(model, loader, device, n_classes, use_feats):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for p, f, m, y in loader:
            p = p.to(device, non_blocking=True)
            f = f.to(device, non_blocking=True) if use_feats else None
            m = m.to(device, non_blocking=True)
            preds.append(model(p, f, m).argmax(1).cpu().numpy())
            ys.append(y.numpy())
    pred = np.concatenate(preds) if preds else np.zeros((0,), np.int64)
    y = np.concatenate(ys) if ys else np.zeros((0,), np.int64)
    conf = np.zeros((n_classes, n_classes), np.int64)
    for t, q in zip(y, pred):
        conf[int(t), int(q)] += 1
    acc = float(np.trace(conf) / max(conf.sum(), 1))
    f1s = []
    for c in range(n_classes):
        tp = conf[c, c]; fp = conf[:, c].sum() - tp; fn = conf[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return acc, float(np.mean(f1s)), conf.tolist(), pred, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--group", choices=tuple(GROUPS), required=True)
    ap.add_argument("--rep", choices=("power", "features"), required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=1729)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-crops", type=int, default=0)
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--pairwise-weight", type=float, default=0.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    meta_in = json.loads(args.data.with_suffix(".json").read_text())
    assert meta_in.get("format") == "dfine-local-crops-v1", meta_in.get("format")
    assert meta_in.get("split") == "fit", meta_in.get("split")
    z = np.load(args.data, allow_pickle=True)
    labels = z["label"].astype(np.int64)
    pred_label = z["pred_label"].astype(np.int64)
    keep = z["keep"].astype(np.int64)
    src = np.array(GROUPS[args.group])
    pos = (keep == 0) & np.isin(labels, src)
    neg = (keep == 1) & np.isin(pred_label, src)
    sel = pos | neg
    idx = np.flatnonzero(sel)
    n_before = len(idx)
    if not args.no_dedupe:
        key = np.stack([z["sample_index"][idx], z["view_type"][idx],
                        z["view_start"][idx], z["qid"][idx], keep[idx]], axis=1)
        _, first = np.unique(key, axis=0, return_index=True)
        idx = idx[np.sort(first)]
    if args.max_crops:
        idx = idx[:args.max_crops]
    y = np.where(keep[idx] == 1, len(src),
                 np.searchsorted(src, labels[idx])).astype(np.int64)
    sid = z["sample_ids"][idx].astype(str)
    power = z["local_power"][idx]
    feats = z["features"][idx].astype(np.float32)
    use_feats = args.rep == "features"

    n_out = len(src) + 1
    tr, va = split_by_sample(sid, y, n_out, args.holdout, args.split_seed)
    mu = feats[tr].mean(0); sd = feats[tr].std(0) + 1e-6
    feats = np.clip((feats - mu) / sd, -5, 5).astype(np.float32)
    # Metadata available to both B and C: detector group scores and candidate
    # geometry. This lets the residual calibrate the detector instead of
    # ignoring its prior.
    meta = np.concatenate([
        z["class_scores"][idx][:, src].astype(np.float32),
        z["score"][idx][:, None].astype(np.float32),
        np.log1p(z["bandwidth_mhz"][idx])[:, None].astype(np.float32),
        np.log1p(z["duration_ms"][idx])[:, None].astype(np.float32),
        z["view_type"][idx][:, None].astype(np.float32),
    ], axis=1).astype(np.float32)
    mmu = meta[tr].mean(0); msd = meta[tr].std(0) + 1e-6
    meta = np.clip((meta - mmu) / msd, -5, 5).astype(np.float32)
    print(f"group={args.group} rep={args.rep} crops={len(y)} train={int(tr.sum())} "
          f"holdout={int(va.sum())} classes={np.bincount(y, minlength=n_out).tolist()}",
          flush=True)

    ds_tr = CropDataset(power[tr], feats[tr], meta[tr], y[tr], use_feats)
    ds_va = CropDataset(power[va], feats[va], meta[va], y[va], use_feats)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True)
    model = LocalNet(n_out, feat_dim=feats.shape[1] if use_feats else 0,
                     meta_dim=meta.shape[1]).to(args.device)
    counts = np.bincount(y[tr], minlength=n_out).astype(np.float32)
    weight = torch.tensor(counts.sum() / (n_out * np.maximum(counts, 1)),
                          device=args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = None
    hist = []
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total = seen = 0
        for p, f, m, yy in dl_tr:
            p = p.to(args.device, non_blocking=True)
            f = f.to(args.device, non_blocking=True) if use_feats else None
            m = m.to(args.device, non_blocking=True)
            yy = yy.to(args.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(p, f, m)
            loss = F.cross_entropy(logits, yy, weight=weight)
            if args.pairwise_weight > 0:
                pos_m = yy < (n_out - 1)
                neg_m = yy == (n_out - 1)
                if bool(pos_m.any()) and bool(neg_m.any()):
                    s_pos = logits[pos_m, yy[pos_m]]
                    s_neg = logits[neg_m][:, :-1].max(dim=1).values
                    pair = F.softplus(-(s_pos[:, None] - s_neg[None, :])).mean()
                    loss = loss + args.pairwise_weight * pair
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(yy); seen += len(yy)
        sched.step()
        acc, f1, conf, pred, true = evaluate(model, dl_va, args.device, n_out, use_feats)
        hist.append({"epoch": epoch + 1, "loss": total / max(seen, 1),
                     "holdout_acc": acc, "holdout_f1": f1})
        if best is None or (f1, acc) > (best["f1"], best["acc"]):
            best = {"epoch": epoch + 1, "acc": acc, "f1": f1, "conf": conf,
                    "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
        print(f"{args.group}/{args.rep} ep {epoch+1}/{args.epochs} "
              f"loss {total/max(seen,1):.4f} holdout acc {acc:.4f} f1 {f1:.4f}",
              flush=True)
    model.load_state_dict(best["state"])
    _, _, _, pred, true = evaluate(model, dl_va, args.device, n_out, use_feats)
    args.out.mkdir(parents=True, exist_ok=True)
    prior = counts / max(counts.sum(), 1)
    torch.save({"state": best["state"], "group": args.group, "rep": args.rep,
                "classes": src.tolist(), "num_classes": n_out, "keep_class": True,
                "feature_dim": int(feats.shape[1]) if use_feats else 0,
                "feat_mu": mu, "feat_sd": sd,
                "meta_mu": mmu, "meta_sd": msd, "meta_dim": int(meta.shape[1]),
                "prior": prior,
                "class_weighted": True, "data": str(args.data),
                "split_seed": args.split_seed, "seed": args.seed,
                "holdout": args.holdout, "epoch": best["epoch"]}, args.out / "best.pt")
    keep_mask = keep[idx][va]
    pos_mask = ~keep_mask.astype(bool)
    keep_true = keep_mask.astype(bool)
    holdout_pos_acc = float((pred[pos_mask] == true[pos_mask]).mean()) if pos_mask.any() else None
    keep_recall = float((pred[keep_true] == len(src)).mean()) if keep_true.any() else None
    meta = {"group": args.group, "rep": args.rep, "classes": src.tolist(),
            "keep_class": True, "num_classes": n_out,
            "crops": len(y), "train": int(tr.sum()), "holdout": int(va.sum()),
            "best_epoch": best["epoch"], "holdout_acc": best["acc"],
            "holdout_f1": best["f1"], "holdout_pos_acc": holdout_pos_acc,
            "keep_recall": keep_recall, "confusion": best["conf"],
            "pretrained": False, "epochs": args.epochs, "batch": args.batch,
            "lr": args.lr, "seconds": time.perf_counter() - t0,
            "params": sum(p.numel() for p in model.parameters()),
            "feature_dim": int(feats.shape[1]) if use_feats else 0,
            "meta_dim": int(meta.shape[1]),
            "class_weighted": True, "data": str(args.data),
            "split_seed": args.split_seed, "seed": args.seed,
            "holdout": args.holdout, "dedupe": not args.no_dedupe,
            "pairwise_weight": args.pairwise_weight,
            "candidates_before_dedupe": int(n_before),
            "prior": (counts / max(counts.sum(), 1)).tolist()}
    (args.out / "summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    (args.out / "history.json").write_text(json.dumps(hist, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
