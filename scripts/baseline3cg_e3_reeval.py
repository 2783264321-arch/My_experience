#!/usr/bin/env python3
"""Re-evaluate E3 after the class-stratified jitter subset was selected.

Updates only the E3 blocks of cascade_adaptation_eval.json (E0/E1/E2 and
C0 are unchanged). No bootstrap here; the main E2 CI is unaffected.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.baseline3cb_e0_eval import bootstrap_ci, conf_matrix, \
    metrics_from_conf, predict
from scripts.baseline3ce_evaluate import (
    build_pair_rows, load_all, seed_model,
)
from src.signals.metrics import ImagePredictions, ImageTargets, evaluate

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports/baseline3_wifi_cascade"
SEEDS = (1729, 1730, 1731)
SHARED_MAP = {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = EXP / "cascade_adaptation_eval.json"
    d = json.loads(out_path.read_text())
    hz, yz, pairs, split = load_all()
    assign = split["assignment"]
    final_sids = sorted({s for s, v in assign.items() if v == "final-dev"})
    final_set = set(final_sids)
    rows = build_pair_rows(hz, pairs, assign, "final-dev")
    dual = [r for r in rows if r["dual"]]

    sample_ids = [str(s) for s in yz["sample_ids"]]
    c0_boxes = yz["c0_boxes"]; c0_scores = yz["c0_scores"]
    c0_labels = yz["c0_labels"]; c0_counts = yz["c0_counts"]
    gt_boxes = yz["gt_boxes"]; gt_labels = yz["gt_labels"]
    gt_counts = yz["gt_counts"]
    c0_off = np.concatenate([[0], np.cumsum(c0_counts)])
    gt_off = np.concatenate([[0], np.cumsum(gt_counts)])
    final_si = [i for i, s in enumerate(sample_ids) if s in final_set]
    h_pred_pos = {int(pi): k for k, pi in enumerate(hz["pred_idx"])}
    pred_status = hz["pred_status"]; pred_off = hz["pred_offsets"]

    per_seed = {}
    maps = []
    cond = {}
    allinst = {}
    for seed in SEEDS:
        model = seed_model("E3", seed, device)
        ys, ps = [], []
        for r in dual:
            p = int(np.argmax(predict(model, r["pred_blocks"], device)))
            ys.append(r["y"]); ps.append(p)
        C = conf_matrix(ys, ps)
        cond[str(seed)] = {
            **metrics_from_conf(C),
            "ci": bootstrap_ci([(r["sample_id"], int(pp == r["y"]))
                                for r, pp in zip(dual, ps)], seed=seed),
        }
        ys2, ps2 = [], []
        for r in rows:
            if len(r["pred_blocks"]) > 0:
                p = int(np.argmax(predict(model, r["pred_blocks"], device)))
            else:
                p = SHARED_MAP.get(int(r["pred_label"]), -1)
            ys2.append(r["y"]); ps2.append(p)
        C2 = conf_matrix(ys2, ps2)
        allinst[str(seed)] = {
            **metrics_from_conf(C2),
            "ci": bootstrap_ci([(r["sample_id"], int(pp == r["y"]))
                                for r, pp in zip(rows, ps2)], seed=seed),
        }

        label_map = {}
        for pi, pos in h_pred_pos.items():
            if pred_status[pos] != "ok":
                continue
            b = hz["pred_blocks"][pred_off[pos]:pred_off[pos + 1]]
            p = int(np.argmax(predict(model, b, device)))
            orig = int(c0_labels[pi])
            label_map[pi] = p if orig <= 2 else p + 3
        preds, targets = [], []
        for si in final_si:
            lo, hi = int(c0_off[si]), int(c0_off[si + 1])
            l = c0_labels[lo:hi].copy()
            for pi in range(lo, hi):
                if pi in label_map:
                    l[pi - lo] = label_map[pi]
            preds.append(ImagePredictions(c0_boxes[lo:hi],
                                          c0_scores[lo:hi], l))
            glo, ghi = int(gt_off[si]), int(gt_off[si + 1])
            targets.append(ImageTargets(gt_boxes[glo:ghi],
                                        gt_labels[glo:ghi]))
        ev = evaluate(targets, preds, iou_thresholds=(0.5,), max_dets=100)
        m = ev["per_iou"]["0.5"]["map_class_aware"]
        maps.append(m)
        per_seed[str(seed)] = {
            "map50": m,
            "delta_pp": round(100 * (m - d["cascade"]["C0"]["map50"]), 3),
            "n_relabeled": len(label_map),
            "per_class": {k: v["ap"]
                          for k, v in ev["per_iou"]["0.5"]["per_class"].items()},
        }

    d["classification"]["conditional"]["E3"] = cond
    d["classification"]["all_instance"]["E3"] = allinst
    d["cascade"]["E3"] = {
        "seeds": per_seed,
        "map50_mean": float(np.mean(maps)),
        "map50_std": float(np.std(maps)),
        "delta_pp_mean": round(100 * (np.mean(maps)
                                      - d["cascade"]["C0"]["map50"]), 3),
    }
    out_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(f"E3 cond={np.mean([v['accuracy'] for v in cond.values()]):.4f} "
          f"all={np.mean([v['accuracy'] for v in allinst.values()]):.4f} "
          f"mAP50={np.mean(maps):.6f} "
          f"delta={d['cascade']['E3']['delta_pp_mean']:+.3f}pp")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
