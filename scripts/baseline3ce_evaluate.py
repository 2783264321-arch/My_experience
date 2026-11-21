#!/usr/bin/env python3
"""Phase C'-Adaptation final evaluation on final-dev.

1. Classification on final-dev matched pairs (conditional dual-sync +
   all-instance with YOLO fallback) for E0/E1/E2/E3 (3 seeds each).
2. Full 14-class cascade mAP50 on the 140 final-dev IQ files, with
   per-seed and seed-mean-softmax variants, per-class AP, sample-clustered
   bootstrap CI for the mAP delta vs C0.
3. Unmatched sync-success tracking (WiFi preds with ok sync that have no
   IoU>=0.5 match to a WiFi GT): categorized vs non-WiFi GT or unmatched.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.baseline3b_train_experts import HistCNN
from scripts.baseline3ca_build_hist_cache import sync_box  # noqa: F401
from scripts.baseline3cb_e0_eval import conf_matrix, metrics_from_conf, \
    bootstrap_ci, predict
from src.signals.metrics import ImagePredictions, ImageTargets, evaluate

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs_local/baseline3c"
EXP = ROOT / "reports/baseline3_wifi_cascade"
SHARED_MAP = {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2}
SEEDS = (1729, 1730, 1731)


def load_all():
    hz = np.load(CACHE / "hist_cache.npz", allow_pickle=True)
    yz = np.load(CACHE / "yolo_val_cache.npz", allow_pickle=True)
    pairs = json.loads((EXP / "adaptation_pairs.json").read_text())
    split = json.loads((EXP / "adaptation_split_v2.json").read_text())
    return hz, yz, pairs, split


def build_pair_rows(hz, pairs, assign, split_name):
    pred_pos = {int(pi): k for k, pi in enumerate(hz["pred_idx"])}
    gt_off = hz["gt_offsets"]; pred_off = hz["pred_offsets"]
    rows = []
    for i, p in enumerate(pairs):
        sid = p["sample_id"]
        if assign.get(sid) != split_name:
            continue
        pp = pred_pos.get(int(p["pred_idx"]))
        if pp is None:
            continue
        g_off = int(gt_off[i]); g_n = int(hz["gt_n_blocks"][i])
        p_off = int(pred_off[pp]); p_n = int(hz["pred_n_blocks"][pp])
        rows.append({
            "pair_index": i, "sample_id": sid,
            "gt_label": p["gt_label"], "pred_label": p["pred_label"],
            "y": SHARED_MAP[p["gt_label"]],
            "gt_blocks": hz["gt_blocks"][g_off:g_off + g_n],
            "pred_blocks": hz["pred_blocks"][p_off:p_off + p_n],
            "dual": g_n > 0 and p_n > 0,
        })
    return rows


def seed_model(arm, seed, device):
    m = HistCNN(3).to(device)
    if arm == "E0":
        p = ROOT / f"runs_local/baseline3b/weights/SHARED_seed{seed}.pt"
    else:
        p = ROOT / f"runs_local/baseline3ca/weights/{arm}_seed{seed}.pt"
    m.load_state_dict(torch.load(p, map_location=device))
    m.eval()
    return m


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hz, yz, pairs, split = load_all()
    assign = split["assignment"]
    final_sids = sorted({s for s, v in assign.items() if v == "final-dev"})
    print(f"final-dev samples: {len(final_sids)}")

    rows = build_pair_rows(hz, pairs, assign, "final-dev")
    dual = [r for r in rows if r["dual"]]
    print(f"final-dev matched pairs {len(rows)}, dual-sync {len(dual)}")

    # ---------------- classification ----------------
    class_out = {"conditional": {}, "all_instance": {}}
    for arm in ("E0", "E1", "E2", "E3"):
        class_out["conditional"][arm] = {}
        class_out["all_instance"][arm] = {}
        for seed in SEEDS:
            model = seed_model(arm, seed, device)
            # conditional
            ys, ps = [], []
            for r in dual:
                p = int(np.argmax(predict(model, r["pred_blocks"], device)))
                ys.append(r["y"]); ps.append(p)
            C = conf_matrix(ys, ps)
            class_out["conditional"][arm][str(seed)] = {
                **metrics_from_conf(C),
                "ci": bootstrap_ci([(r["sample_id"], int(pp == r["y"]))
                                    for r, pp in zip(dual, ps)], seed=seed),
            }
            # all-instance: every matched pair; sync-fail keeps YOLO class
            ys2, ps2 = [], []
            for r in rows:
                if len(r["pred_blocks"]) > 0:
                    p = int(np.argmax(predict(model, r["pred_blocks"],
                                              device)))
                else:
                    p = SHARED_MAP[r["pred_label"]]
                ys2.append(r["y"]); ps2.append(p)
            C2 = conf_matrix(ys2, ps2)
            class_out["all_instance"][arm][str(seed)] = {
                **metrics_from_conf(C2),
                "ci": bootstrap_ci([(r["sample_id"], int(pp == r["y"]))
                                    for r, pp in zip(rows, ps2)], seed=seed),
            }
        print(f"{arm}: cond={np.mean([v['accuracy'] for v in class_out['conditional'][arm].values()]):.4f} "
              f"all={np.mean([v['accuracy'] for v in class_out['all_instance'][arm].values()]):.4f}",
              flush=True)

    # ---------------- cascade mAP on final-dev ----------------
    sample_ids = [str(s) for s in yz["sample_ids"]]
    si_of = {s: i for i, s in enumerate(sample_ids)}
    c0_boxes = yz["c0_boxes"]; c0_scores = yz["c0_scores"]
    c0_labels = yz["c0_labels"]; c0_counts = yz["c0_counts"]
    gt_boxes = yz["gt_boxes"]; gt_labels = yz["gt_labels"]
    gt_counts = yz["gt_counts"]
    c0_off = np.concatenate([[0], np.cumsum(c0_counts)])
    gt_off = np.concatenate([[0], np.cumsum(gt_counts)])
    final_set = set(final_sids)
    final_si = [si_of[s] for s in final_sids if s in si_of]
    print(f"final-dev samples present in yolo cache: {len(final_si)}")

    # pred cache: global c0 index -> hist cache position
    h_pred_pos = {int(pi): k for k, pi in enumerate(hz["pred_idx"])}
    pred_status = hz["pred_status"]; pred_off = hz["pred_offsets"]

    def expert_labels(arm, seed):
        """Return dict global c0 idx -> mapped expert class (sync-ok only)."""
        model = seed_model(arm, seed, device)
        mapping = {}
        for pi, pos in h_pred_pos.items():
            if pred_status[pos] != "ok":
                continue
            b = hz["pred_blocks"][pred_off[pos]:pred_off[pos + 1]]
            p = int(np.argmax(predict(model, b, device)))
            orig = int(c0_labels[pi])
            if orig <= 2:
                mapping[pi] = p
            else:
                mapping[pi] = p + 3
        return mapping

    def build_eval(label_map):
        preds, targets = [], []
        for si in final_si:
            lo, hi = int(c0_off[si]), int(c0_off[si + 1])
            b = c0_boxes[lo:hi]; s = c0_scores[lo:hi]
            l = c0_labels[lo:hi].copy()
            for pi in range(lo, hi):
                if pi in label_map:
                    l[pi - lo] = label_map[pi]
            preds.append(ImagePredictions(b, s, l))
            glo, ghi = int(gt_off[si]), int(gt_off[si + 1])
            targets.append(ImageTargets(gt_boxes[glo:ghi],
                                        gt_labels[glo:ghi]))
        return preds, targets

    preds0, targets = build_eval({})
    ev0 = evaluate(targets, preds0, iou_thresholds=(0.5,), max_dets=100)
    m0 = ev0["per_iou"]["0.5"]["map_class_aware"]
    pc0 = ev0["per_iou"]["0.5"]["per_class"]
    print(f"C0 final-dev subset mAP50={m0:.6f} "
          f"n_pred={sum(len(p.boxes) for p in preds0)}")

    cascade = {"C0": {"map50": m0,
                      "n_pred": int(sum(len(p.boxes) for p in preds0)),
                      "per_class": {k: v["ap"] for k, v in pc0.items()}}}
    for arm in ("E0", "E1", "E2", "E3"):
        per_seed = {}
        maps = []
        for seed in SEEDS:
            lm = expert_labels(arm, seed)
            preds, _ = build_eval(lm)
            ev = evaluate(targets, preds, iou_thresholds=(0.5,), max_dets=100)
            m = ev["per_iou"]["0.5"]["map_class_aware"]
            per_seed[str(seed)] = {
                "map50": m, "delta_pp": round(100 * (m - m0), 3),
                "n_relabeled": len(lm),
                "per_class": {k: v["ap"]
                              for k, v in ev["per_iou"]["0.5"]["per_class"].items()},
            }
            maps.append(m)
        cascade[arm] = {"seeds": per_seed,
                        "map50_mean": float(np.mean(maps)),
                        "map50_std": float(np.std(maps)),
                        "delta_pp_mean": round(100 * (np.mean(maps) - m0), 3)}
        print(f"{arm}: mAP50 mean={np.mean(maps):.6f} "
              f"delta={100*(np.mean(maps)-m0):+.3f}pp", flush=True)

    # bootstrap CI for the mAP delta (E2 seed-mean vs C0), by sample
    rng = np.random.default_rng(20261121)
    lm2 = {}
    # seed-mean softmax for E2: average probabilities then argmax
    models = [seed_model("E2", s, device) for s in SEEDS]
    for pi, pos in h_pred_pos.items():
        if pred_status[pos] != "ok":
            continue
        b = hz["pred_blocks"][pred_off[pos]:pred_off[pos + 1]]
        probs = np.mean([predict(m, b, device) for m in models], axis=0)
        p = int(np.argmax(probs))
        orig = int(c0_labels[pi])
        lm2[pi] = p if orig <= 2 else p + 3
    preds2, _ = build_eval(lm2)
    ev2 = evaluate(targets, preds2, iou_thresholds=(0.5,), max_dets=100)
    m2 = ev2["per_iou"]["0.5"]["map_class_aware"]
    deltas = []
    for _ in range(120):
        pick = rng.integers(0, len(final_si), size=len(final_si))
        preds_c, preds_e, targ_b = [], [], []
        for j in pick:
            si = final_si[j]
            lo, hi = int(c0_off[si]), int(c0_off[si + 1])
            b = c0_boxes[lo:hi]; s = c0_scores[lo:hi]
            l0 = c0_labels[lo:hi].copy()
            le = l0.copy()
            for pi in range(lo, hi):
                if pi in lm2:
                    le[pi - lo] = lm2[pi]
            preds_c.append(ImagePredictions(b, s, l0))
            preds_e.append(ImagePredictions(b, s, le))
            glo, ghi = int(gt_off[si]), int(gt_off[si + 1])
            targ_b.append(ImageTargets(gt_boxes[glo:ghi],
                                       gt_labels[glo:ghi]))
        mc = evaluate(targ_b, preds_c, iou_thresholds=(0.5,),
                      max_dets=100)["per_iou"]["0.5"]["map_class_aware"]
        me = evaluate(targ_b, preds_e, iou_thresholds=(0.5,),
                      max_dets=100)["per_iou"]["0.5"]["map_class_aware"]
        deltas.append(100 * (me - mc))
    cascade["E2_seedmean"] = {
        "map50": m2, "delta_pp": round(100 * (m2 - m0), 3),
        "bootstrap_delta_ci": {
            "mean": float(np.mean(deltas)),
            "lo": float(np.quantile(deltas, 0.025)),
            "hi": float(np.quantile(deltas, 0.975))},
    }
    print(f"E2 seed-mean: mAP50={m2:.6f} "
          f"delta CI={cascade['E2_seedmean']['bootstrap_delta_ci']}")

    # ---------------- unmatched sync-success tracking ----------------
    matched_pred_idx = {int(p["pred_idx"]) for p in pairs}
    unmatched = []
    for pi, pos in h_pred_pos.items():
        if pred_status[pos] != "ok" or pi in matched_pred_idx:
            continue
        si = int(np.searchsorted(c0_off, pi, side="right") - 1)
        sid = sample_ids[si]
        # best IoU to any GT in the sample (all classes)
        glo, ghi = int(gt_off[si]), int(gt_off[si + 1])
        gb = gt_boxes[glo:ghi]; gl = gt_labels[glo:ghi]
        best = 0.0; best_lab = None
        b = c0_boxes[pi]
        for k in range(len(gb)):
            ix1 = max(b[0], gb[k][0]); iy1 = max(b[1], gb[k][1])
            ix2 = min(b[2], gb[k][2]); iy2 = min(b[3], gb[k][3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            a1 = (b[2] - b[0]) * (b[3] - b[1])
            a2 = (gb[k][2] - gb[k][0]) * (gb[k][3] - gb[k][1])
            iou = inter / max(a1 + a2 - inter, 1e-12)
            if iou > best:
                best, best_lab = float(iou), int(gl[k])
        unmatched.append({
            "pred_idx": int(pi),
            "sample_id": sid, "pred_label": int(c0_labels[pi]),
            "pred_score": float(c0_scores[pi]),
            "best_gt_iou": round(best, 3),
            "best_gt_label": best_lab,
            "gt_is_wifi": (best_lab is not None and best_lab <= 5),
            "unmatched_category": (
                "wifi_gt_secondary_duplicate" if best_lab is not None
                and best_lab <= 5 and best >= 0.5 else
                "wifi_gt_low_iou" if best_lab is not None
                and best_lab <= 5 else
                "nonwifi_gt" if best_lab is not None
                and best >= 0.5 else "no_match"),
        })
    unmatched_counts = Counter(u["unmatched_category"] for u in unmatched)
    print(f"unmatched sync-ok WiFi preds: {len(unmatched)} "
          f"{dict(unmatched_counts)}")

    out = {"final_dev_samples": len(final_sids),
           "matched_pairs": len(rows), "dual_sync_pairs": len(dual),
           "classification": class_out,
           "cascade": cascade,
           "unmatched_sync_ok": {"count": len(unmatched),
                                 "categories": dict(unmatched_counts),
                                 "items": unmatched}}
    (EXP / "cascade_adaptation_eval.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("wrote cascade_adaptation_eval.json")


if __name__ == "__main__":
    main()
