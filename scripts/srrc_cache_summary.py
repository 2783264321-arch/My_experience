#!/usr/bin/env python3
"""Summarize the SRRC full-fit cache and the paired D-FINE step-time run."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--monitor", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = {"cache": str(args.cache), "cache_exists": args.cache.exists()}
    if args.cache.exists():
        with np.load(args.cache, allow_pickle=False) as z:
            result["cache_bytes"] = args.cache.stat().st_size
            result["keys"] = list(z.files)
            w = z["windows"]
            result["windows_shape"] = list(w.shape)
            result["windows_dtype"] = str(w.dtype)
            result["finite"] = bool(np.isfinite(w).all())
            result["zero_windows"] = int(np.all(w == 0, axis=(1, 2, 3)).sum())
            rms = np.sqrt((w.astype(np.float64) ** 2).sum(axis=2).mean(axis=2))
            result["rms_min"] = float(rms.min())
            result["rms_max"] = float(rms.max())
            result["n_instances"] = int(len(z["labels"]))
            result["n_groups"] = int(len(set(z["groups"].tolist())))
            result["labels"] = dict(Counter(int(x) for x in z["labels"]))
            result["split"] = dict(Counter("dev" if x else "train" for x in z["is_dev"]))
            result["strata"] = dict(Counter(str(x) for x in z["strata"]))
            result["metadata"] = json.loads(str(z["metadata"]))
    mon = json.loads(args.monitor.read_text())
    result["monitor"] = mon
    # Per-epoch medians for the rows covered by the full run.
    rows = list(csv.DictReader(args.log.open()))
    if mon.get("during"):
        a = int(mon["during"]["first_step"]); b = int(mon["during"]["last_step"])
        sub = [r for r in rows if a <= int(r["step"]) <= b]
        per = {}
        for r in sub:
            per.setdefault(r["epoch"], []).append(float(r["step_s"]))
        result["during_per_epoch"] = {
            k: {"n": len(v), "median_s": float(np.median(v)),
                "p95_s": float(np.percentile(v, 95))}
            for k, v in sorted(per.items(), key=lambda kv: int(kv[0]))
        }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in
                      ("cache_exists", "cache_bytes", "windows_shape",
                       "n_instances", "n_groups", "labels", "split", "strata",
                       "finite", "zero_windows", "rms_min", "rms_max")},
                     indent=2))
    print("monitor stop", mon.get("stop_reason"),
          "baseline", mon.get("baseline", {}).get("median_s"),
          "during", (mon.get("during") or {}).get("median_s"),
          "delta_pp", (mon.get("during") or {}).get("delta_median_pp"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
