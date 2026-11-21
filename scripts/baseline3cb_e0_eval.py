#!/usr/bin/env python3
"""Phase C'-Adaptation E0: frozen W-Shared GT-vs-Pred paired diagnosis.

For dual-sync matched pairs (GT and Pred both sync ok) in adaptation-train
(primary) and calibration-dev (auxiliary), evaluate the three frozen
W-Shared seeds on GT-block inputs and Pred-block inputs. Reports
conditional accuracy, per-instance agreement, confusion matrices,
per-class paired changes, IoU/geometry strata, and sample-clustered
bootstrap CIs.

No training. final-dev not used here (model selection stays on cal-dev).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.baseline3b_train_experts import HistCNN

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs_local/baseline3c/hist_cache.npz"
WEIGHTS = ROOT / "runs_local/baseline3b/weights"
EXP = ROOT / "reports/baseline3_wifi_cascade"
SHARED_MAP = {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2}
SEEDS = (1729, 1730, 1731)


def bootstrap_ci(pairs, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for k, v in pairs:
        by[k].append(v)
    keys = sorted(by)
    base = np.array([np.mean(by[k]) for k in keys])
    wts = np.array([len(by[k]) for k in keys], float)
    accs = np.zeros(n_boot)
    for t in range(n_boot):
        pick = rng.integers(0, len(keys), size=len(keys))
        accs[t] = np.sum(base[pick] * wts[pick]) / np.sum(wts[pick])
    return {"mean": float(np.mean(accs)), "lo": float(np.quantile(accs, .025)),
            "hi": float(np.quantile(accs, .975))}


def conf_matrix(ys, ps, n=3):
    C = np.zeros((n, n), np.int64)
    for y, p in zip(ys, ps):
        C[int(y), int(p)] += 1
    return C


def metrics_from_conf(C):
    n = int(C.sum())
    if n == 0:
        return {"n": 0, "accuracy": None, "macro_f1": None,
                "confusion": C.tolist()}
    acc = float(np.trace(C) / n)
    f1s = []
    for c in range(3):
        tp = int(C[c, c]); fp = int(C[:, c].sum() - tp)
        fn = int(C[c, :].sum() - tp)
        if tp and tp + fp and tp + fn:
            prec = tp / (tp + fp); rec = tp / (tp + fn)
            f1s.append(2 * prec * rec / (prec + rec))
    return {"n": n, "accuracy": acc,
            "macro_f1": float(np.mean(f1s)) if f1s else None,
            "confusion": C.tolist()}


def predict(model, blocks, device):
    if len(blocks) == 0:
        return np.zeros(3, np.float32)
    x = torch.from_numpy(blocks).to(device)
    with torch.no_grad():
        p = F.softmax(model(x), dim=1).mean(dim=0)
    return p.cpu().numpy()


def main():
    z = np.load(CACHE, allow_pickle=True)
    pairs = json.loads((EXP / "adaptation_pairs.json").read_text())
    split = json.loads((EXP / "adaptation_split_v2.json").read_text())
    assign = split["assignment"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pred_pos = {int(pi): k for k, pi in enumerate(z["pred_idx"])}
    gt_offsets = z["gt_offsets"]; pred_offsets = z["pred_offsets"]

    rows = []
    for i, p in enumerate(pairs):
        sid = p["sample_id"]
        if sid not in assign:
            continue
        g_off = int(gt_offsets[i]); g_n = int(z["gt_n_blocks"][i])
        pp = pred_pos.get(int(p["pred_idx"]))
        if pp is None:
            continue
        p_off = int(pred_offsets[pp]); p_n = int(z["pred_n_blocks"][pp])
        if g_n == 0 or p_n == 0:
            continue   # dual-sync only
        # training/eval sets use the correct-family routed pairs only
        # (C set; the task's 169 dual-sync instances exclude the 2
        # cross-family pred-sync pairs)
        if not (0 <= p["pred_label"] <= 5):
            continue
        fam_pred = "W20" if p["pred_label"] <= 2 else "W40"
        fam_gt = "W20" if p["gt_label"] <= 2 else "W40"
        if fam_pred != fam_gt:
            continue
        rows.append({
            "pair_index": i, "sample_id": sid,
            "split": assign[sid], "gt_label": p["gt_label"],
            "pred_label": p["pred_label"],
            "y": SHARED_MAP[p["gt_label"]],
            "iou": p["iou"], "f_center_delta": p["f_center_delta"],
            "overlap_any": p.get("overlap_any"),
            "strong_overlap": p.get("strong_overlap"),
            "gt_blocks": z["gt_blocks"][g_off:g_off + g_n],
            "pred_blocks": z["pred_blocks"][p_off:p_off + p_n],
        })
    print(f"dual-sync pairs: {len(rows)}")
    for s in ("adaptation-train", "calibration-dev", "final-dev"):
        sel = [r for r in rows if r["split"] == s]
        print(f"  {s}: {len(sel)} pairs, "
              f"{len({r['sample_id'] for r in sel})} IQ files, "
              f"class={dict(sorted(Counter(r['gt_label'] for r in sel).items()))}")

    # ---- evaluate frozen seeds ----
    results = {}
    for seed in SEEDS:
        model = HistCNN(3).to(device)
        model.load_state_dict(torch.load(WEIGHTS / f"SHARED_seed{seed}.pt",
                                         map_location=device))
        model.eval()
        for sname in ("adaptation-train", "calibration-dev"):
            sel = [r for r in rows if r["split"] == sname]
            ys, gt_pred, pr_pred = [], [], []
            agree = 0
            for r in sel:
                pg = predict(model, r["gt_blocks"], device)
                pp = predict(model, r["pred_blocks"], device)
                lg = int(np.argmax(pg)); lp = int(np.argmax(pp))
                ys.append(r["y"]); gt_pred.append(lg); pr_pred.append(lp)
                agree += int(lg == lp)
            m_gt = metrics_from_conf(conf_matrix(ys, gt_pred))
            m_pr = metrics_from_conf(conf_matrix(ys, pr_pred))
            pairs_gt = [(r["sample_id"], int(g == r["y"]))
                        for r, g in zip(sel, gt_pred)]
            pairs_pr = [(r["sample_id"], int(p == r["y"]))
                        for r, p in zip(sel, pr_pred)]
            # per-class paired changes
            changes = {}
            for c in range(6):
                sub = [k for k, r in enumerate(sel) if r["gt_label"] == c]
                if not sub:
                    continue
                changes[f"c{c}"] = {
                    "n": len(sub),
                    "gt_acc": float(np.mean([gt_pred[k] == ys[k] for k in sub])),
                    "pred_acc": float(np.mean([pr_pred[k] == ys[k] for k in sub])),
                    "agree": float(np.mean([gt_pred[k] == pr_pred[k] for k in sub])),
                }
            results.setdefault(sname, {})[str(seed)] = {
                "gt": m_gt, "pred": m_pr,
                "agreement": agree / max(len(sel), 1),
                "gt_ci": bootstrap_ci(pairs_gt, seed=seed),
                "pred_ci": bootstrap_ci(pairs_pr, seed=seed),
                "per_class": changes,
            }

    # strata on adaptation-train (pooled across seeds via seed means)
    def seed_mean(metric_key):
        out = []
        for seed in SEEDS:
            out.append(results["adaptation-train"][str(seed)][metric_key]["accuracy"])
        return float(np.mean(out))

    strata = {}
    train_rows = [r for r in rows if r["split"] == "adaptation-train"]
    for name, sel_fn in (
            ("iou_hi>=0.75", lambda r: r["iou"] >= 0.75),
            ("iou_lo<0.75", lambda r: r["iou"] < 0.75),
            ("offset<0.5MHz", lambda r: abs(r["f_center_delta"]) < 0.5),
            ("offset>=0.5MHz", lambda r: abs(r["f_center_delta"]) >= 0.5),
            ("clean", lambda r: r["overlap_any"] == 0),
            ("overlap", lambda r: r["overlap_any"] == 1)):
        sub = [r for r in train_rows if sel_fn(r)]
        if not sub:
            continue
        accs_gt, accs_pr = [], []
        for seed in SEEDS:
            model = HistCNN(3).to(device)
            model.load_state_dict(torch.load(
                WEIGHTS / f"SHARED_seed{seed}.pt", map_location=device))
            model.eval()
            cg = cp = 0
            for r in sub:
                pg = int(np.argmax(predict(model, r["gt_blocks"], device)))
                pp = int(np.argmax(predict(model, r["pred_blocks"], device)))
                cg += int(pg == r["y"]); cp += int(pp == r["y"])
            accs_gt.append(cg / len(sub)); accs_pr.append(cp / len(sub))
        strata[name] = {"n": len(sub),
                        "gt_acc_mean": float(np.mean(accs_gt)),
                        "pred_acc_mean": float(np.mean(accs_pr))}

    out = {
        "n_dual_sync_pairs": len(rows),
        "seeds": list(SEEDS),
        "results": results,
        "strata_adaptation_train": strata,
    }
    (EXP / "frozen_gt_pred_comparison.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(strata, indent=2))
    print(f"wrote frozen_gt_pred_comparison.json")


if __name__ == "__main__":
    main()
