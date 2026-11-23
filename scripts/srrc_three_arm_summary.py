#!/usr/bin/env python3
"""Summarize the same-window B0, magnitude-CNN and I/Q-CNN pilot."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/baseline3_srrc_gt_pilot"
BASE = Path("/root/autodl-fs/runs_local/srrc-gt-pilot")
SEEDS = (1729, 1730, 1731)


def load_dev(path):
    rows = list(csv.DictReader(path.open()))
    return {(r["sample_id"], r["signal_idx"]): r for r in rows}


def pred(row):
    return int(np.argmax([float(row["logit_10"]), float(row["logit_11"])]))


def summary(history, epoch):
    return next(h for h in history if h["epoch"] == epoch)["dev"]


def pair_stats(ref, arm, keys):
    y = np.array([int(ref[k]["class"]) - 10 for k in keys])
    pr = np.array([pred(ref[k]) for k in keys])
    pa = np.array([pred(arm[k]) for k in keys])
    corr = int(((pr != y) & (pa == y)).sum())
    broken = int(((pr == y) & (pa != y)).sum())
    p = float(binomtest(corr, corr + broken, 0.5).pvalue) if corr + broken else 1.0
    out = {"n": len(keys), "agreement": float((pr == pa).mean()),
           "corrected": corr, "broken": broken, "net": corr - broken,
           "mcnemar_exact_p": p, "strata": {}}
    for s in ("clean", "weak", "strong"):
        idx = np.array([ref[k]["overlap"] == s for k in keys])
        c = int(((pr[idx] != y[idx]) & (pa[idx] == y[idx])).sum())
        b = int(((pr[idx] == y[idx]) & (pa[idx] != y[idx])).sum())
        out["strata"][s] = {"n": int(idx.sum()), "corrected": c,
                            "broken": b, "net": c - b}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--out", type=Path, default=OUT / "three_arm_results.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    b0_summary = json.loads((args.base / "b0_same_windows/summary.json").read_text())
    b0_dev = load_dev(args.base / "b0_same_windows/b0_same_windows_dev.csv")
    result = {
        "b0": {
            "dev": b0_summary["dev"],
            "seconds": b0_summary["seconds"],
            "n_features": b0_summary["n_features"],
            "aggregation": b0_summary["aggregation"],
        },
        "seeds": {},
        "aggregate": {},
    }
    for seed in SEEDS:
        sdir = args.base / f"seed{seed}"
        summary = json.loads((sdir / "summary.json").read_text())
        entry = {}
        for arm in ("magnitude", "iq"):
            hist = summary["results"][arm]["history"]
            final = hist[-1]["dev"]
            best_h = max(hist, key=lambda h: h["dev"]["all"]["balanced_accuracy"])
            dev = load_dev(sdir / f"{arm}_dev.csv")
            keys = sorted(set(b0_dev) & set(dev))
            entry[arm] = {
                "final": final,
                "best_epoch": best_h["epoch"],
                "best": best_h["dev"],
                "seconds": summary["results"][arm]["seconds"],
                "peak_cuda_reserved_bytes": summary["results"][arm]["peak_cuda_reserved_bytes"],
                "vs_b0": pair_stats(b0_dev, dev, keys),
            }
        result["seeds"][str(seed)] = entry
    for arm in ("magnitude", "iq"):
        for metric in ("balanced_accuracy", "accuracy"):
            vals = [result["seeds"][str(s)][arm]["final"]["all"][metric]
                    for s in SEEDS]
            result["aggregate"].setdefault(arm, {})[metric] = {
                "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
                "values": [float(x) for x in vals],
            }
        for s in ("clean", "weak", "strong"):
            vals = [result["seeds"][str(seed)][arm]["final"][s]["balanced_accuracy"]
                    for seed in SEEDS]
            result["aggregate"][arm].setdefault(s, {})["balanced_accuracy"] = {
                "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
                "values": [float(x) for x in vals],
            }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "b0": result["b0"]["dev"]["all"],
        "magnitude_final": result["aggregate"]["magnitude"]["balanced_accuracy"],
        "magnitude_clean": result["aggregate"]["magnitude"]["clean"]["balanced_accuracy"],
        "magnitude_weak": result["aggregate"]["magnitude"]["weak"]["balanced_accuracy"],
        "magnitude_strong": result["aggregate"]["magnitude"]["strong"]["balanced_accuracy"],
        "iq_final": result["aggregate"]["iq"]["balanced_accuracy"],
        "iq_clean": result["aggregate"]["iq"]["clean"]["balanced_accuracy"],
        "iq_weak": result["aggregate"]["iq"]["weak"]["balanced_accuracy"],
        "iq_strong": result["aggregate"]["iq"]["strong"]["balanced_accuracy"],
    }, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
