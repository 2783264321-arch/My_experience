#!/usr/bin/env python3
"""Compare corrected group-loss normalization against control and old group."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAL = Path("/root/autodl-tmp/worktrees/formal-dfine/runs/dfine_l_c1_2048x512")
RUNS = {
    "round25": FORMAL / "val_full_best/summary.json",
    "control": ROOT / "runs/dfine_head_finetune/control_val/summary.json",
    "group_old": ROOT / "runs/dfine_head_finetune/group_val/summary.json",
    "group_normfix": ROOT / "runs/dfine_head_finetune/group_normfix_val/summary.json",
}
TRAIN = {
    "control": ROOT / "runs/dfine_head_finetune/control",
    "group_old": ROOT / "runs/dfine_head_finetune/group",
    "group_normfix": ROOT / "runs/dfine_head_finetune/group_normfix",
}
NAMES = {
    0: "0 W20 QPSK", 1: "1 W20 16QAM", 2: "2 W20 64QAM",
    3: "3 W40 QPSK", 4: "4 W40 16QAM", 5: "5 W40 64QAM",
    6: "6 BLE LE1M", 7: "7 BLE LE2M", 8: "8 Zigbee", 9: "9 LoRa",
    10: "10 SRRC QPSK", 11: "11 SRRC 16QAM", 12: "12 AM", 13: "13 FM",
}


def load_val(path):
    d = json.loads(Path(path).read_text())
    return d, d["settings"]["filtered_0_5"]


def train_stats(path):
    rows = list(csv.DictReader(open(Path(path) / "train_log.csv")))
    g = [int(r["group_n"]) for r in rows]
    gl = [float(r["group_loss"]) for r in rows]
    s = json.loads((Path(path) / "summary.json").read_text())
    s["group_n_total"] = sum(g)
    s["group_n_mean_step"] = statistics.mean(g)
    s["group_n_median_step"] = statistics.median(g)
    s["group_n_zero_steps"] = sum(x == 0 for x in g)
    s["group_loss_mean"] = statistics.mean(gl)
    return s


def main():
    data = {k: load_val(v) for k, v in RUNS.items()}
    t = {k: v[1] for k, v in data.items()}
    cols = list(RUNS)
    print("| 模型 | mAP50 | mAP50:95 | 0 | 1 | 2 | Wi‑Fi20 组 | SRRC 组 | FM | 定位召回 | 最终候选 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for k in cols:
        x = t[k]
        print(f"| {k} | {x['map50']:.6f} | {x['map50_95']:.6f} | "
              f"{x['per_class']['0']['ap']:.4f} | {x['per_class']['1']['ap']:.4f} | "
              f"{x['per_class']['2']['ap']:.4f} | {x['groups']['WiFi_20MHz_0-2']['map50']:.4f} | "
              f"{x['groups']['SRRC_10-11']['map50']:.4f} | {x['per_class']['13']['ap']:.4f} | "
              f"{x['localization_recall']:.4f} | {x['n_pred']} |")

    print("\n| 对比 | Δ mAP50 | Δ mAP50:95 | Δ 0 | Δ 1 | Δ 2 | Δ Wi‑Fi20 组 | Δ SRRC 组 | Δ FM |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    base = t["round25"]
    comps = [("control", "round25"), ("group_old", "round25"),
             ("group_normfix", "round25"), ("group_normfix", "control"),
             ("group_normfix", "group_old")]
    for a, b in comps:
        def d(key, sub=None):
            av = t[a]["per_class"][str(key)]["ap"] if sub == "cls" else (
                t[a]["groups"][key]["map50"] if sub == "grp" else t[a][key])
            bv = t[b]["per_class"][str(key)]["ap"] if sub == "cls" else (
                t[b]["groups"][key]["map50"] if sub == "grp" else t[b][key])
            return (av - bv) * 100
        print(f"| {a} − {b} | {d('map50'):+.3f} | {d('map50_95'):+.3f} | "
              f"{d(0,'cls'):+.2f} | {d(1,'cls'):+.2f} | {d(2,'cls'):+.2f} | "
              f"{d('WiFi_20MHz_0-2','grp'):+.2f} | {d('SRRC_10-11','grp'):+.2f} | "
              f"{d(13,'cls'):+.2f} |")

    print("\n| 类 | round25 | control | group_old | group_normfix |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for c in range(14):
        vals = [t[k]["per_class"][str(c)]["ap"] for k in cols]
        print(f"| {NAMES[c]} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {vals[3]:.4f} |")

    out = {
        "normalization": {
            "old": "sum_v(mean_ce_v) / sum_v(n_v); default reduction=mean per view",
            "new": "sum_v(sum_i ce_vi) / sum_v(n_v); reduction=sum then one division",
            "smoke_first_batch": {"n": 14, "old_group_loss": 0.6159,
                                  "new_group_loss": 1.1252},
            "smoke_second_batch": {"n": 7, "old_group_loss": 0.8726,
                                   "new_group_loss": 0.8731},
        },
        "train": {k: train_stats(v) for k, v in TRAIN.items()},
        "val": {k: {x: t[k][x] for x in (
            "map50", "map50_95", "localization_recall", "classification_accuracy",
            "n_gt", "n_pred", "per_class", "groups")} for k in cols},
        "meta": {k: {"checkpoint": data[k][0]["checkpoint"],
                     "elapsed_s": data[k][0]["elapsed_s"],
                     "n_samples": data[k][0]["n_samples"],
                     "split": data[k][0]["split"]} for k in cols},
    }
    path = ROOT / "reports/detector_upgrade/dfine_group_normfix.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\nwrote", path, path.stat().st_size)


if __name__ == "__main__":
    main()
