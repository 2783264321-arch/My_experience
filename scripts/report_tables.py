#!/usr/bin/env python3
"""Print Markdown tables from a run's val history / stage-5 summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="yolo11s_c1_directreg_2048x512")
    args = ap.parse_args()
    run = ROOT / "runs" / args.run
    hist = json.loads((run / "val_history.json").read_text())
    print("| epoch | mAP50 | AP50 agnostic | loc recall | cls acc | pred | val s |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for e in hist:
        print(f"| {e['epoch']} | {e['map50']:.4f} | {e['map50_agnostic']:.4f} | "
              f"{e['localization_recall']:.3f} | {e['classification_accuracy']:.3f} | "
              f"{e['n_pred']} | {e['val_elapsed_s']:.0f} |")
    best = max(hist, key=lambda e: e["map50"])
    print(f"\nbest epoch {best['epoch']} mAP50 {best['map50']:.4f}")
    print("\n| class | AP50 | GT | pred |")
    print("| ---: | ---: | ---: | ---: |")
    for c, v in best["per_class"].items():
        ap = v["ap"]
        print(f"| {c} | {ap:.3f} | {v['n_gt']} | {v['n_pred']} |")

    s5 = run / "stage5_val" / "summary.json"
    if s5.exists():
        s = json.loads(s5.read_text())
        print(f"\nstage5 mAP50 {s['map50']:.4f}, agnostic {s['map50_agnostic']:.4f}, "
              f"recall {s['localization_recall']:.3f}, acc {s['classification_accuracy']:.3f}")
        print("\n| bandwidth | samples | GT | mAP50 | AP50 agnostic |")
        print("| --- | ---: | ---: | ---: | ---: |")
        for k, v in s["per_bandwidth"].items():
            print(f"| {k} | {v['n_samples']} | {v['n_gt']} | {v['map50']:.4f} | "
                  f"{v['ap50_agnostic']:.4f} |")


if __name__ == "__main__":
    main()
