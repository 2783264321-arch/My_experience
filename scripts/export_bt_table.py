#!/usr/bin/env python3
"""Flatten stage-5 per-B/T metrics into a CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="yolo11s_c1_directreg_2048x512")
    ap.add_argument("--val-dir", default="")
    args = ap.parse_args()
    vdir = Path(args.val_dir) if args.val_dir else ROOT / "runs" / args.run / "stage5_val"
    s = json.loads((vdir / "summary.json").read_text())
    rows = []
    for key, v in s["per_bandwidth_duration"].items():
        B, T = key.replace("ms", "").split("MHz_")
        rows.append({
            "bandwidth_mhz": float(B), "duration_ms": float(T),
            "n_samples": v["n_samples"], "n_gt": v.get("n_gt"),
            "n_pred": v.get("n_pred"), "map50": v.get("map50"),
            "ap50_agnostic": v.get("ap50_agnostic"),
            "localization_recall": v.get("localization_recall"),
            "classification_accuracy": v.get("classification_accuracy"),
        })
    rows.sort(key=lambda r: (r["bandwidth_mhz"], r["duration_ms"]))
    out = vdir / "per_bt.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, len(rows), "rows")


if __name__ == "__main__":
    main()
