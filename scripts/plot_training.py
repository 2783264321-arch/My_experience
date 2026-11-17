#!/usr/bin/env python3
"""Plot training loss and validation mAP from a run directory."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="yolo11s_c1_directreg_2048x512")
    args = ap.parse_args()
    run = ROOT / "runs" / args.run
    rows = list(csv.DictReader((run / "train_log.csv").open()))
    by_epoch = defaultdict(list)
    for r in rows:
        e = int(float(r["epoch"]))
        by_epoch[e].append(r)
    epochs, losses, boxes, clss, l1s, lrs = [], [], [], [], [], []
    for e in sorted(by_epoch):
        rs = by_epoch[e]
        epochs.append(e)
        losses.append(np.mean([float(x["loss"]) for x in rs]))
        boxes.append(np.mean([float(x["box"]) for x in rs]))
        clss.append(np.mean([float(x["cls"]) for x in rs]))
        l1s.append(np.mean([float(x["l1"]) for x in rs]))
        lrs.append(np.mean([float(x["lr"]) for x in rs]))

    hist = []
    hp = run / "val_history.json"
    if hp.exists():
        hist = json.loads(hp.read_text())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    axes[0].plot(epochs, losses, label="total")
    axes[0].plot(epochs, boxes, label="box/CIoU")
    axes[0].plot(epochs, clss, label="cls")
    axes[0].plot(epochs, l1s, label="L1")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("mean loss per view")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].set_title("training loss")

    axes[1].plot(epochs, lrs)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("lr")
    axes[1].set_title("learning rate")

    if hist:
        ve = [h["epoch"] for h in hist]
        vmap = [h["map50"] for h in hist]
        vagn = [h["map50_agnostic"] for h in hist]
        vrec = [h["localization_recall"] or 0 for h in hist]
        axes[2].plot(ve, vmap, "o-", label="mAP50 class-aware")
        axes[2].plot(ve, vagn, "s-", label="AP50 class-agnostic")
        axes[2].plot(ve, vrec, "^-", label="localization recall")
        axes[2].set_xlabel("epoch")
        axes[2].set_ylim(0, 1)
        axes[2].legend()
        axes[2].grid(alpha=0.3)
    axes[2].set_title("full-sample validation")
    fig.tight_layout()
    out = run / "training_curve.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
