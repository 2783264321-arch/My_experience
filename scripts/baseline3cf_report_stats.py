#!/usr/bin/env python3
"""Phase C'-Adaptation: final report statistics.

Reads the frozen evaluation artefacts and adds:
  - conditional/all-instance flip counts vs E0 (same seed)
  - per-class precision/recall for the 3-class W-Shared output
  - clean/overlap, IoU and class strata on final-dev
  - IQ-file paired bootstrap CI for the classification delta vs E0
  - corrected unmatched-sync-ok categories (duplicate / low-IoU /
    non-WiFi overlap / no match) and their AP cost (oracle diagnostic)
  - final-dev sync coverage

Writes reports/baseline3_wifi_cascade/adaptation_report_stats.json.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.baseline3b_train_experts import HistCNN
from scripts.baseline3cb_e0_eval import bootstrap_ci, predict
from scripts.baseline3ce_evaluate import (
    build_pair_rows, load_all, seed_model, SHARED_MAP,
)
from src.signals.metrics import ImagePredictions, ImageTargets, evaluate

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports/baseline3_wifi_cascade"
CACHE = ROOT / "runs_local/baseline3c"
SEEDS = (1729, 1730, 1731)
ARMS = ("E0", "E1", "E2", "E3")


def conf_pr(C):
    C = np.asarray(C, float)
    out = {}
    for c in range(C.shape[0]):
        tp = C[c, c]; fp = C[:, c].sum() - tp; fn = C[c, :].sum() - tp
        out[str(c)] = {
            "precision": float(tp / (tp + fp)) if tp + fp else None,
            "recall": float(tp / (tp + fn)) if tp + fn else None,
            "support": int(C[c, :].sum()),
        }
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hz, yz, pairs, split = load_all()
    assign = split["assignment"]
    final_sids = sorted({s for s, v in assign.items() if v == "final-dev"})
    rows = build_pair_rows(hz, pairs, assign, "final-dev")
    for r in rows:
        p = pairs[r["pair_index"]]
        r["iou"] = p["iou"]
        r["f_center_delta"] = p["f_center_delta"]
        r["overlap_any"] = p.get("overlap_any")
    dual = [r for r in rows if r["dual"]]
    print(f"final rows={len(rows)} dual={len(dual)} "
          f"samples={len(final_sids)}")

    # ---------- predictions for every arm/seed ----------
    cond = {}   # arm -> seed -> list of (row, pred, correct)
    allinst = {}
    for arm in ARMS:
        cond[arm] = {}
        allinst[arm] = {}
        for seed in SEEDS:
            model = seed_model(arm, seed, device)
            preds_c = []
            for r in dual:
                p = int(np.argmax(predict(model, r["pred_blocks"], device)))
                preds_c.append((r, p, int(p == r["y"])))
            preds_a = []
            for r in rows:
                if len(r["pred_blocks"]) > 0:
                    p = int(np.argmax(predict(model, r["pred_blocks"], device)))
                else:
                    pl = int(r["pred_label"])
                    p = SHARED_MAP.get(pl, -1)
                preds_a.append((r, p, int(p == r["y"])))
            cond[arm][seed] = preds_c
            allinst[arm][seed] = preds_a

    # ---------- flips vs E0 ----------
    flips = {}
    for arm in ARMS[1:]:
        for kind, table in (("conditional", cond), ("all_instance", allinst)):
            per_seed = {}
            for seed in SEEDS:
                e0 = table["E0"][seed]
                a = table[arm][seed]
                corrected = sum(1 for x, y in zip(e0, a)
                                if x[2] == 0 and y[2] == 1)
                broken = sum(1 for x, y in zip(e0, a)
                             if x[2] == 1 and y[2] == 0)
                per_seed[str(seed)] = {
                    "corrected": corrected, "broken": broken,
                    "net": corrected - broken,
                    "n": len(e0),
                }
            agg = {k: float(np.mean([v[k] for v in per_seed.values()]))
                   for k in ("corrected", "broken", "net")}
            flips.setdefault(arm, {})[kind] = {
                "per_seed": per_seed, "seed_mean": agg}

    # ---------- per-class P/R ----------
    pr = {}
    for arm in ARMS:
        pr[arm] = {}
        for seed in SEEDS:
            C = np.zeros((3, 3), np.int64)
            for r, p, _ in cond[arm][seed]:
                if 0 <= p < 3:
                    C[r["y"], p] += 1
            pr[arm][str(seed)] = {"confusion": C.tolist(),
                                  "per_class": conf_pr(C)}

    # ---------- strata ----------
    def stratum_acc(arm, sel_fn):
        vals = []
        for seed in SEEDS:
            sub = [(r, p, c) for r, p, c in cond[arm][seed] if sel_fn(r)]
            if sub:
                vals.append(np.mean([c for _, _, c in sub]))
        return (float(np.mean(vals)) if vals else None,
                int(len([r for r in dual if sel_fn(r)])))

    defs = {
        "clean": lambda r: r.get("overlap_any") == 0,
        "overlap": lambda r: r.get("overlap_any") == 1,
        "iou_hi>=0.75": lambda r: r["iou"] >= 0.75,
        "iou_lo<0.75": lambda r: r["iou"] < 0.75,
        "offset<0.5MHz": lambda r: abs(r.get("f_center_delta", 0)) < 0.5,
        "offset>=0.5MHz": lambda r: abs(r.get("f_center_delta", 0)) >= 0.5,
    }
    for c in range(6):
        defs[f"class{c}"] = (lambda cc: (lambda r: r["gt_label"] == cc))(c)
    strata = {}
    for arm in ARMS:
        strata[arm] = {}
        for name, fn in defs.items():
            acc, n = stratum_acc(arm, fn)
            strata[arm][name] = {"n": n, "acc": acc}

    # ---------- paired bootstrap CI for delta vs E0 (by IQ file) ----------
    ci = {}
    for arm in ARMS[1:]:
        for kind, table in (("conditional", cond), ("all_instance", allinst)):
            per_seed = {}
            for seed in SEEDS:
                e0 = table["E0"][seed]; a = table[arm][seed]
                by = defaultdict(list)
                for x, y in zip(e0, a):
                    by[x[0]["sample_id"]].append(y[2] - x[2])
                keys = sorted(by)
                base = np.array([np.mean(by[k]) for k in keys])
                rng = np.random.default_rng(20261121 + seed)
                vals = np.zeros(2000)
                for t in range(2000):
                    pick = rng.integers(0, len(keys), size=len(keys))
                    vals[t] = base[pick].mean()
                per_seed[str(seed)] = {
                    "delta": float(base.mean()),
                    "lo": float(np.quantile(vals, .025)),
                    "hi": float(np.quantile(vals, .975))}
            ci.setdefault(arm, {})[kind] = per_seed

    # ---------- unmatched sync-ok, corrected categories ----------
    sample_ids = [str(s) for s in yz["sample_ids"]]
    c0_boxes = yz["c0_boxes"]; c0_scores = yz["c0_scores"]
    c0_labels = yz["c0_labels"]; c0_counts = yz["c0_counts"]
    gt_boxes = yz["gt_boxes"]; gt_labels = yz["gt_labels"]
    gt_counts = yz["gt_counts"]
    c0_off = np.concatenate([[0], np.cumsum(c0_counts)])
    gt_off = np.concatenate([[0], np.cumsum(gt_counts)])
    final_set = set(final_sids)
    h_pred_pos = {int(pi): k for k, pi in enumerate(hz["pred_idx"])}
    pred_status = hz["pred_status"]; pred_off = hz["pred_offsets"]
    matched_pred_idx = {int(p["pred_idx"]) for p in pairs}

    unmatched = []
    for pi, pos in h_pred_pos.items():
        if pred_status[pos] != "ok" or pi in matched_pred_idx:
            continue
        si = int(np.searchsorted(c0_off, pi, side="right") - 1)
        sid = sample_ids[si]
        glo, ghi = int(gt_off[si]), int(gt_off[si + 1])
        gb = gt_boxes[glo:ghi]; gl = gt_labels[glo:ghi]
        b = c0_boxes[pi]
        best = 0.0; best_lab = None
        for k in range(len(gb)):
            ix1 = max(b[0], gb[k][0]); iy1 = max(b[1], gb[k][1])
            ix2 = min(b[2], gb[k][2]); iy2 = min(b[3], gb[k][3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            a1 = (b[2] - b[0]) * (b[3] - b[1])
            a2 = (gb[k][2] - gb[k][0]) * (gb[k][3] - gb[k][1])
            iou = inter / max(a1 + a2 - inter, 1e-12)
            if iou > best:
                best, best_lab = float(iou), int(gl[k])
        if best_lab is not None and best_lab <= 5 and best >= 0.5:
            cat = "wifi_gt_secondary_duplicate"
        elif best_lab is not None and best_lab <= 5:
            cat = "wifi_gt_low_iou"
        elif best_lab is not None and best >= 0.5:
            cat = "nonwifi_gt"
        else:
            cat = "no_match"
        unmatched.append({
            "pred_idx": int(pi), "sample_id": sid,
            "split": assign.get(sid, "not-in-split"),
            "pred_label": int(c0_labels[pi]),
            "pred_score": float(c0_scores[pi]),
            "best_gt_iou": round(best, 3),
            "best_gt_label": best_lab, "category": cat,
        })
    cat_counts = Counter(u["category"] for u in unmatched)
    cat_final = Counter(u["category"] for u in unmatched
                        if u["split"] == "final-dev")

    # sync coverage on final-dev
    final_si = [i for i, s in enumerate(sample_ids) if s in final_set]
    n_wifi = 0; n_ok = 0
    for si in final_si:
        lo, hi = int(c0_off[si]), int(c0_off[si + 1])
        for pi in range(lo, hi):
            if c0_labels[pi] in (0, 1, 2, 3, 4, 5):
                n_wifi += 1
                pos = h_pred_pos.get(pi)
                if pos is not None and pred_status[pos] == "ok":
                    n_ok += 1
    n_ok_unmatched = sum(1 for u in unmatched if u["split"] == "final-dev")
    n_ok_matched = n_ok - n_ok_unmatched

    # ---------- AP diagnostic: drop unmatched sync-ok preds ----------
    def build_preds(label_map, drop=set()):
        preds, targets = [], []
        for si in final_si:
            lo, hi = int(c0_off[si]), int(c0_off[si + 1])
            keep = [pi for pi in range(lo, hi) if pi not in drop]
            b = c0_boxes[keep]; s = c0_scores[keep]
            l = c0_labels[keep].copy()
            for j, pi in enumerate(keep):
                if pi in label_map:
                    l[j] = label_map[pi]
            preds.append(ImagePredictions(b, s, l))
            glo, ghi = int(gt_off[si]), int(gt_off[si + 1])
            targets.append(ImageTargets(gt_boxes[glo:ghi],
                                        gt_labels[glo:ghi]))
        return preds, targets

    def map50(preds, targets):
        ev = evaluate(targets, preds, iou_thresholds=(0.5,), max_dets=100)
        return float(ev["per_iou"]["0.5"]["map_class_aware"])

    # E2 seed-mean softmax expert map (same as the evaluation script)
    models = [seed_model("E2", s, device) for s in SEEDS]
    e2_map = {}
    for pi, pos in h_pred_pos.items():
        if pred_status[pos] != "ok":
            continue
        b = hz["pred_blocks"][pred_off[pos]:pred_off[pos + 1]]
        probs = np.mean([predict(m, b, device) for m in models], axis=0)
        p = int(np.argmax(probs))
        orig = int(c0_labels[pi])
        e2_map[pi] = p if orig <= 2 else p + 3
    e0_map = {}
    m0 = seed_model("E0", 1729, device)
    for pi, pos in h_pred_pos.items():
        if pred_status[pos] != "ok":
            continue
        b = hz["pred_blocks"][pred_off[pos]:pred_off[pos + 1]]
        p = int(np.argmax(predict(m0, b, device)))
        orig = int(c0_labels[pi])
        e0_map[pi] = p if orig <= 2 else p + 3

    drop_final = {u["pred_idx"] for u in unmatched
                  if u["split"] == "final-dev"}
    p_c0, t = build_preds({})
    p_e0, _ = build_preds(e0_map)
    p_e2, _ = build_preds(e2_map)
    p_c0d, _ = build_preds({}, drop_final)
    p_e0d, _ = build_preds(e0_map, drop_final)
    p_e2d, _ = build_preds(e2_map, drop_final)
    ap_diag = {
        "n_dropped": len(drop_final),
        "C0": {"map50": map50(p_c0, t), "map50_no_unmatched": map50(p_c0d, t)},
        "E0": {"map50": map50(p_e0, t), "map50_no_unmatched": map50(p_e0d, t)},
        "E2_seedmean": {"map50": map50(p_e2, t),
                        "map50_no_unmatched": map50(p_e2d, t)},
    }
    for v in ap_diag.values():
        if isinstance(v, dict):
            v["delta_pp"] = round(100 * (v["map50_no_unmatched"] - v["map50"]), 3)

    out = {
        "final_dev": {"samples": len(final_sids), "matched_pairs": len(rows),
                      "dual_sync_pairs": len(dual),
                      "wifi_preds": n_wifi, "sync_ok": n_ok,
                      "sync_coverage": round(n_ok / max(n_wifi, 1), 4),
                      "sync_ok_matched_pairs": n_ok_matched,
                      "sync_ok_unmatched": n_ok_unmatched},
        "flips_vs_E0": flips,
        "per_class_pr": pr,
        "strata": strata,
        "bootstrap_delta_vs_E0": ci,
        "unmatched_categories_all": dict(cat_counts),
        "unmatched_categories_final": dict(cat_final),
        "unmatched_items": unmatched,
        "ap_diagnostic_drop_unmatched": ap_diag,
    }
    (EXP / "adaptation_report_stats.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("unmatched_items",)}, indent=2)[:4000])
    print("wrote adaptation_report_stats.json")


if __name__ == "__main__":
    main()
