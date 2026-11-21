#!/usr/bin/env python3
"""Phase B' power baselines at the paired-comparison geometry.

Same instance set, same ONE-split protocol as dataset_manifest.json
(split_by_sample over ALL WiFi instances, seed 1729, holdout 0.2), same
GT-box geometry (margin-0 crops from power_reps_m0.npz), 3 seeds:

  pw_012   classes 0/1/2 (W20 family power baseline)
  pw_345   classes 3/4/5 (W40 family)
  pw_shared classes 0-5 pooled to QPSK/16QAM/64QAM (W-Shared control)

2 representations (coarse, local_power) x 3 seeds each. SmallCNN, the
published training recipe. Outputs power_baselines_bprime.json.
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
SRC = ROOT / "reports/local_iq_round1/power_reps_m0.npz"
OUT = ROOT / "reports/baseline3_wifi_expert/power_baselines_bprime.json"
REPS = ("coarse", "local_power")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", default="1729,1730,1731")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    z = np.load(SRC)
    labels_full = z["label"].astype(np.int64)
    sid_full = z["sample_id"].astype(str)
    wifi = np.isin(labels_full, [0, 1, 2, 3, 4, 5])
    idx = np.where(wifi)[0]                      # full-space -> compact
    labels = labels_full[idx]                    # compact, length n
    sid = sid_full[idx]
    n = len(idx)
    print(f"instances={n}", flush=True)

    # ONE split over all WiFi (matches dataset_manifest.json exactly)
    tr, va = split_by_sample(sid, labels, [0, 1, 2, 3, 4, 5],
                             holdout=0.2, seed=1729)
    tr = np.where(tr)[0]
    va = np.where(va)[0]
    print(f"train={len(tr)} holdout={len(va)} "
          f"train_samples={len(set(sid[tr]))} holdout_samples={len(set(sid[va]))}",
          flush=True)

    def masked_labels(classes, mask):
        m = {c: i for i, c in enumerate(classes)}
        y = np.full(n, -1, dtype=np.int64)
        y[mask] = np.array([m[int(v)] for v in labels[mask]], dtype=np.int64)
        return y

    arms = {
        "pw_012": (np.isin(labels, [0, 1, 2]),
                   masked_labels([0, 1, 2], np.isin(labels, [0, 1, 2]))),
        "pw_345": (np.isin(labels, [3, 4, 5]),
                   masked_labels([3, 4, 5], np.isin(labels, [3, 4, 5]))),
        "pw_shared": (np.ones(n, dtype=bool),
                      np.array([{0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2}[int(v)]
                                for v in labels], dtype=np.int64)),
    }

    out = {"source": str(SRC), "epochs": args.epochs, "seeds": seeds,
           "split": {"train_instances": int(len(tr)),
                     "holdout_instances": int(len(va))},
           "arms": {}}
    for gname, (gmask, y_all) in arms.items():
        gtr = tr[gmask[tr]]
        gva = va[gmask[va]]
        y_tr, y_va = y_all[gtr], y_all[gva]
        g = {"train": int(len(gtr)), "holdout": int(len(gva)),
             "class_counts_holdout": {str(c): int((y_va == c).sum())
                                      for c in range(3)},
             "arms": {}}
        print(json.dumps({"arm": gname, "train": g["train"],
                          "holdout": g["holdout"]}), flush=True)
        for rep in REPS:
            x = z[rep].astype(np.float32)[idx]
            x = np.clip(standardize_coarse(x)[:, None], -10.0, 10.0)
            seed_rows = []
            for seed in seeds:
                t0 = time.perf_counter()
                model, pred, _ = train_one(
                    x[gtr], y_tr, x[gva], y_va, 3, seed,
                    args.epochs, 64, device,
                )
                conf = np.zeros((3, 3), dtype=np.int64)
                for t, p in zip(y_va.tolist(), pred.tolist()):
                    conf[t, p] += 1
                row = {"seed": seed, "confusion": confusion_metrics(conf),
                       "seconds": round(time.perf_counter() - t0, 1)}
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
        out["arms"][gname] = g

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
