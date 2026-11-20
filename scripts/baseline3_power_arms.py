#!/usr/bin/env python3
"""Standalone SmallCNN power-baseline arms for Baseline-3 round 1.

Trains the same-recipe SmallCNN on the GT-geometry coarse / local_power
crops for groups 012 / 345 / 1011 with the pre-registered per-family split
(seed 1729, holdout 0.2 -- identical to the separability study on the WiFi
part). Results are saved in the same JSON shape that baseline3_compare.py
produces for its power arms so the final comparison can merge them instead
of retraining.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.train_proposal_classifiers import SmallCNN, split_by_sample
from scripts.train_local_representation_classifiers import (
    train_one, confusion_metrics, standardize_coarse,
)

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"012": [0, 1, 2], "345": [3, 4, 5], "1011": [10, 11]}
REPS = ("coarse", "local_power")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", default="reports/local_iq_round1/power_reps.npz")
    ap.add_argument("--out", default="reports/local_iq_round1/power_arms.json")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", default="1729,1730,1731")
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=1729)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    z = np.load(ROOT / args.power)
    labels = z["label"].astype(np.int64)
    sid = z["sample_id"].astype(str)
    n = len(labels)
    print(f"frames={n}", flush=True)

    out = {"frames": int(n), "epochs": args.epochs, "seeds": seeds, "groups": {}}
    for gname, classes in GROUPS.items():
        gm = np.isin(labels, classes)
        idx = np.where(gm)[0]
        y = np.array([classes.index(int(v)) for v in labels[idx]], dtype=np.int64)
        tr_m, va_m = split_by_sample(sid[idx], labels[idx], classes,
                                     holdout=args.holdout, seed=args.split_seed)
        tr = idx[tr_m]; va = idx[va_m]
        g = {
            "classes": classes,
            "n": int(len(idx)),
            "train": int(len(tr)), "holdout": int(len(va)),
            "train_samples": int(len(set(sid[tr]))),
            "holdout_samples": int(len(set(sid[va]))),
            "class_counts_holdout": {str(c): int((labels[va] == c).sum()) for c in classes},
            "arms": {},
        }
        print(json.dumps({"group": gname, "n": g["n"], "train": g["train"],
                          "holdout": g["holdout"]}), flush=True)
        for rep in REPS:
            x = z[rep].astype(np.float32)
            x = np.clip(standardize_coarse(x)[:, None], -10.0, 10.0)
            x_tr, x_va = x[tr], x[va]
            y_tr, y_va = y[tr_m], y[va_m]
            seed_rows = []
            for seed in seeds:
                t0 = time.perf_counter()
                model, pred, _ = train_one(
                    x_tr, y_tr, x_va, y_va, len(classes), seed,
                    args.epochs, 64, device,
                )
                conf = np.zeros((len(classes), len(classes)), dtype=np.int64)
                for t, p in zip(y_va.tolist(), pred.tolist()):
                    conf[t, p] += 1
                row = {
                    "seed": seed,
                    "confusion": confusion_metrics(conf),
                    "seconds": round(time.perf_counter() - t0, 1),
                }
                if seed == seeds[0]:
                    row["pred_holdout"] = [int(v) for v in pred.tolist()]
                seed_rows.append(row)
                print(f"{gname} {rep} seed={seed} "
                      f"acc={row['confusion']['accuracy']:.4f} "
                      f"({row['seconds']}s)", flush=True)
            g["arms"][rep] = {
                "seeds": seed_rows,
                "holdout_acc_mean": float(np.mean(
                    [r["confusion"]["accuracy"] for r in seed_rows])),
            }
        out["groups"][gname] = g

    out_path = ROOT / args.out
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
