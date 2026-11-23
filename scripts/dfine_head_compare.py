#!/usr/bin/env python3
"""Compare round25, control and Wi-Fi20-group head fine-tunes on full val."""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path("/root/autodl-tmp/worktrees/formal-dfine/runs/dfine_l_c1_2048x512/val_full_best/summary.json")
ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "round25": BASE,
    "control": ROOT / "runs/dfine_head_finetune/control_val/summary.json",
    "group": ROOT / "runs/dfine_head_finetune/group_val/summary.json",
}
NAMES = {
    0: "0 W20 QPSK", 1: "1 W20 16QAM", 2: "2 W20 64QAM",
    3: "3 W40 QPSK", 4: "4 W40 16QAM", 5: "5 W40 64QAM",
    6: "6 BLE LE1M", 7: "7 BLE LE2M", 8: "8 Zigbee", 9: "9 LoRa",
    10: "10 SRRC QPSK", 11: "11 SRRC 16QAM", 12: "12 AM", 13: "13 FM",
}


def load(path):
    d = json.loads(Path(path).read_text())
    return d, d["settings"]["filtered_0_5"]


def main():
    data = {k: load(v) for k, v in RUNS.items()}
    base = data["round25"][1]
    print("| metric | round25 | control | group | Δcontrol | Δgroup |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    rows = [
        ("mAP50", "map50"), ("mAP50:95", "map50_95"),
        ("AP50 agnostic", "ap50_agnostic"), ("AP75 agnostic", "ap75_agnostic"),
        ("loc recall", "localization_recall"),
        ("cls accuracy", "classification_accuracy"),
    ]
    for label, key in rows:
        vals = [data[k][1][key] for k in RUNS]
        print(f"| {label} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} | "
              f"{vals[1]-vals[0]:+.6f} | {vals[2]-vals[0]:+.6f} |")
    print(f"| n_pred | {base['n_pred']} | {data['control'][1]['n_pred']} | "
          f"{data['group'][1]['n_pred']} | "
          f"{data['control'][1]['n_pred']-base['n_pred']:+d} | "
          f"{data['group'][1]['n_pred']-base['n_pred']:+d} |")
    print("\n| class | round25 | control | group | Δcontrol | Δgroup |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for c in range(14):
        vals = [data[k][1]["per_class"][str(c)]["ap"] for k in RUNS]
        print(f"| {NAMES[c]} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | "
              f"{vals[1]-vals[0]:+.4f} | {vals[2]-vals[0]:+.4f} |")
    print("\n| group | round25 | control | group | Δcontrol | Δgroup |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for g in base["groups"]:
        vals = [data[k][1]["groups"][g]["map50"] for k in RUNS]
        print(f"| {g} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | "
              f"{vals[1]-vals[0]:+.4f} | {vals[2]-vals[0]:+.4f} |")
    print("\n## group - control")
    for label, key in rows:
        print(f"- {label}: {data['group'][1][key]-data['control'][1][key]:+.6f}")
    for c in range(14):
        d = (data["group"][1]["per_class"][str(c)]["ap"]
             - data["control"][1]["per_class"][str(c)]["ap"])
        print(f"- class {NAMES[c]}: {d:+.4f}")
    print("\n## metadata")
    for k in RUNS:
        d, _ = data[k]
        print(k, "epoch", d.get("checkpoint_epoch"), "elapsed", round(d["elapsed_s"], 1),
              "n_views", d["n_views"], "checkpoint", d["checkpoint"])


if __name__ == "__main__":
    main()
