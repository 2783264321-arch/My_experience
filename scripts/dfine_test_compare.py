#!/usr/bin/env python3
"""Four-checkpoint official-test comparison with matching val references."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAL = Path("/root/autodl-tmp/worktrees/formal-dfine/runs/dfine_l_c1_2048x512")
TEST = {
    "round25": ROOT / "runs/dfine_test/round25/summary.json",
    "round50": ROOT / "runs/dfine_test/round50/summary.json",
    "control": ROOT / "runs/dfine_test/control/summary.json",
    "group": ROOT / "runs/dfine_test/group/summary.json",
}
VAL = {
    "round25": FORMAL / "val_full_best/summary.json",
    "round50": FORMAL / "val_full_last/summary.json",
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
    tdata = {k: load(v) for k, v in TEST.items()}
    vdata = {k: load(v) for k, v in VAL.items()}
    t = {k: v[1] for k, v in tdata.items()}
    v = {k: v[1] for k, v in vdata.items()}
    cols = list(TEST)

    print("| 模型 | test mAP50 | test mAP50:95 | val mAP50 | val mAP50:95 | Δtest−val 50 | Δtest−val 95 | 定位召回 | 分类准确率 | 最终候选 | 耗时 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for k in cols:
        print(f"| {k} | {t[k]['map50']:.6f} | {t[k]['map50_95']:.6f} | "
              f"{v[k]['map50']:.6f} | {v[k]['map50_95']:.6f} | "
              f"{(t[k]['map50']-v[k]['map50'])*100:+.2f} | "
              f"{(t[k]['map50_95']-v[k]['map50_95'])*100:+.2f} | "
              f"{t[k]['localization_recall']:.4f} | {t[k]['classification_accuracy']:.4f} | "
              f"{t[k]['n_pred']} | {tdata[k][0]['elapsed_s']:.0f} s |")

    print("\n| 类 | test GT | round25 | round50 | control | group | Δ50 | Δcontrol | Δgroup |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    base = t["round25"]
    for c in range(14):
        vals = [t[k]["per_class"][str(c)]["ap"] for k in cols]
        n = t["round25"]["per_class"][str(c)]["n_gt"]
        print(f"| {NAMES[c]} | {n} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | "
              f"{vals[3]:.4f} | {(vals[1]-vals[0])*100:+.2f} | "
              f"{(vals[2]-vals[0])*100:+.2f} | {(vals[3]-vals[0])*100:+.2f} |")

    print("\n| 组 | test round25 | round50 | control | group | Δ50 | Δcontrol | Δgroup |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for g in base["groups"]:
        vals = [t[k]["groups"][g]["map50"] for k in cols]
        print(f"| {g} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {vals[3]:.4f} | "
              f"{(vals[1]-vals[0])*100:+.2f} | {(vals[2]-vals[0])*100:+.2f} | "
              f"{(vals[3]-vals[0])*100:+.2f} |")

    print("\n## aggregate JSON")
    out = {"test": {}, "val": {}, "test_minus_val_pp": {}}
    for k in cols:
        out["test"][k] = {x: t[k][x] for x in (
            "map50", "map50_95", "ap50_agnostic", "ap75_agnostic",
            "localization_recall", "classification_accuracy", "n_gt", "n_pred",
            "per_class", "groups", "confusion")}
        out["test"][k]["elapsed_s"] = tdata[k][0]["elapsed_s"]
        out["test"][k]["n_samples"] = tdata[k][0]["n_samples"]
        out["test"][k]["n_views"] = tdata[k][0]["n_views"]
        out["test"][k]["checkpoint"] = tdata[k][0]["checkpoint"]
        out["test"][k]["split"] = tdata[k][0]["split"]
        out["val"][k] = {x: v[k][x] for x in (
            "map50", "map50_95", "localization_recall", "classification_accuracy",
            "n_gt", "n_pred", "per_class", "groups")}
        out["test_minus_val_pp"][k] = {
            "map50": (t[k]["map50"] - v[k]["map50"]) * 100,
            "map50_95": (t[k]["map50_95"] - v[k]["map50_95"]) * 100,
        }
    path = ROOT / "reports/detector_upgrade/dfine_official_test.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(path, path.stat().st_size)


if __name__ == "__main__":
    main()
