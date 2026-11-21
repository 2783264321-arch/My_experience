#!/usr/bin/env python3
"""Post-hoc metric audit for final-dev and relabel accounting.

1. all-instance classification: correct-family candidates are the primary
   group; cross-family routing errors are reported separately and are
   never mapped to a shared 3-class label as if correct.
2. expert invocation / sync-ok / actual label changes for full val and
   final-dev, per arm and seed.

This does not change any model or mAP.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.baseline3cb_e0_eval import conf_matrix, metrics_from_conf, predict
from scripts.baseline3ce_evaluate import build_pair_rows, load_all, seed_model

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports/baseline3_wifi_cascade"
SEEDS = (1729, 1730, 1731)
ARMS = ("E0", "E1", "E2", "E3")
SHARED_MAP = {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2}


def fam(label: int) -> str:
    return "W20" if label <= 2 else "W40" if label <= 5 else "nonwifi"


def build_label_map(model, hz, yz, device):
    h_pred_pos = {int(pi): k for k, pi in enumerate(hz["pred_idx"])}
    pred_status = hz["pred_status"]; pred_off = hz["pred_offsets"]
    c0_labels = yz["c0_labels"]
    label_map = {}
    for pi, pos in h_pred_pos.items():
        if pred_status[pos] != "ok":
            continue
        b = hz["pred_blocks"][pred_off[pos]:pred_off[pos + 1]]
        p = int(np.argmax(predict(model, b, device)))
        orig = int(c0_labels[pi])
        label_map[pi] = p if orig <= 2 else p + 3
    return label_map


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hz, yz, pairs, split = load_all()
    assign = split["assignment"]
    sample_ids = [str(s) for s in yz["sample_ids"]]
    c0_labels = yz["c0_labels"]; c0_counts = yz["c0_counts"]
    c0_off = np.concatenate([[0], np.cumsum(c0_counts)])
    h_pred_pos = {int(pi): k for k, pi in enumerate(hz["pred_idx"])}
    pred_status = hz["pred_status"]

    out = {"final_dev_group": {}, "relabel": {}}

    # ---------- final-dev all-instance audit ----------
    rows = build_pair_rows(hz, pairs, assign, "final-dev")
    cross = [r for r in rows if fam(r["gt_label"]) != fam(r["pred_label"])]
    correct_fam = [r for r in rows if fam(r["gt_label"]) == fam(r["pred_label"])]
    print(f"final rows={len(rows)} correct-family={len(correct_fam)} "
          f"cross-family={len(cross)}")
    for arm in ARMS:
        out["final_dev_group"][arm] = {}
        for seed in SEEDS:
            model = seed_model(arm, seed, device)
            correct = 0
            for r in correct_fam:
                if len(r["pred_blocks"]) > 0:
                    p = int(np.argmax(predict(model, r["pred_blocks"], device)))
                else:
                    p = SHARED_MAP[int(r["pred_label"])]
                correct += int(p == r["y"])
            group_acc = correct / max(len(correct_fam), 1)
            strict_acc = correct / max(len(rows), 1)
            out["final_dev_group"][arm][str(seed)] = {
                "n_correct_family": len(correct_fam),
                "n_cross_family": len(cross),
                "n_all_matched": len(rows),
                "correct_family_correct": correct,
                "group_accuracy": group_acc,
                "strict_all_matched_accuracy": strict_acc,
                "cross_family_items": [
                    {"sample_id": r["sample_id"], "gt_label": r["gt_label"],
                     "pred_label": r["pred_label"], "y": r["y"],
                     "has_pred_hist": len(r["pred_blocks"]) > 0}
                    for r in cross],
            }
            print(f"  {arm} seed{seed}: group={group_acc:.4f} "
                  f"strict={strict_acc:.4f}")

    # ---------- relabel accounting ----------
    for scope in ("full-val", "final-dev"):
        out["relabel"][scope] = {}
        if scope == "full-val":
            sids = set(sample_ids)
        else:
            sids = {s for s, v in assign.items() if v == "final-dev"}
        wifi = [pi for pi in h_pred_pos if int(c0_labels[pi]) <= 5]
        wifi = [pi for pi in wifi
                if sample_ids[int(np.searchsorted(c0_off, pi, side="right") - 1)]
                in sids]
        sync = [pi for pi in wifi
                if pred_status[h_pred_pos[pi]] == "ok"]
        matched = {int(p["pred_idx"]) for p in pairs}
        out["relabel"][scope]["candidates_wifi"] = len(wifi)
        out["relabel"][scope]["sync_ok"] = len(sync)
        out["relabel"][scope]["matched_sync_ok"] = sum(
            1 for pi in sync if pi in matched)
        out["relabel"][scope]["unmatched_sync_ok"] = sum(
            1 for pi in sync if pi not in matched)
        out["relabel"][scope]["arms"] = {}
        for arm in ARMS:
            out["relabel"][scope]["arms"][arm] = {}
            for seed in SEEDS:
                model = seed_model(arm, seed, device)
                lm = build_label_map(model, hz, yz, device)
                lm = {pi: v for pi, v in lm.items() if pi in set(sync)}
                changed = [pi for pi, v in lm.items()
                           if int(v) != int(c0_labels[pi])]
                fam_changed = [pi for pi, v in lm.items()
                               if fam(int(c0_labels[pi])) != fam(int(v))]
                out["relabel"][scope]["arms"][arm][str(seed)] = {
                    "expert_classified": len(lm),
                    "label_changed": len(changed),
                    "label_changed_matched": sum(1 for pi in changed
                                                 if pi in matched),
                    "label_changed_unmatched": sum(1 for pi in changed
                                                   if pi not in matched),
                    "family_changed": len(fam_changed),
                    "labels": {
                        "yolo": dict(Counter(int(c0_labels[pi])
                                             for pi in sync)),
                        "expert": dict(Counter(int(lm[pi]) for pi in sync)),
                    },
                }
                print(f"  {scope} {arm} seed{seed}: sync={len(sync)} "
                      f"changed={len(changed)}")

    (EXP / "finaldev_metric_audit.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("wrote finaldev_metric_audit.json")


if __name__ == "__main__":
    main()
