#!/usr/bin/env python3
"""Paired 4-window vs 8-window inference with existing magnitude weights.

Splits the 8 cached windows into two interleaved halves:
  even = [0,2,4,6], odd = [1,3,5,7]
and compares their dev scores/AUC with the full 8-window score.
No training; uses the saved magnitude_last.pt weights.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest

from scripts.baseline3srrc_gt_pilot import TinyResNet, metrics, predict, read_pilot_cache

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/baseline3_srrc_gt_pilot"
BASE = Path("/root/autodl-fs/runs_local/srrc-gt-pilot")
SEEDS = (1729, 1730, 1731)
EVEN = (0, 2, 4, 6)
ODD = (1, 3, 5, 7)


def auc(y, s):
    y = np.asarray(y, int)
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def auc_by_stratum(y, s, strata):
    out = {}
    for name in ("all", "clean", "weak", "strong"):
        m = np.ones(len(y), bool) if name == "all" else (strata == name)
        out[name] = {"n": int(m.sum()), "auc": auc(y[m], s[m])}
    return out


def bootstrap_auc_diff(y, a, b, groups, n_boot=5000, seed=1729):
    """Paired source-group bootstrap for AUC(a)-AUC(b)."""
    y = np.asarray(y, int); a = np.asarray(a); b = np.asarray(b)
    groups = np.asarray(groups)
    keys = sorted(set(groups.tolist()))
    idx = {k: np.flatnonzero(groups == k) for k in keys}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        rows = np.concatenate([idx[keys[j]] for j in pick])
        if len(set(y[rows])) < 2:
            continue
        vals.append(auc(y[rows], a[rows]) - auc(y[rows], b[rows]))
    v = np.asarray(vals, float)
    return {"mean": float(v.mean()), "lo": float(np.quantile(v, .025)),
            "hi": float(np.quantile(v, .975)), "n_boot": len(v)}


@torch.inference_mode()
def score_subset(model, windows, ids, cols, device, batch=32):
    out = []
    w = windows[:, list(cols)]
    for start in range(0, len(ids), batch):
        chunk = ids[start:start + batch]
        logits = predict(model, w[chunk].to(device), "magnitude").cpu().numpy()
        out.append(logits[:, 1] - logits[:, 0])
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=BASE / "fit_8win.npz")
    ap.add_argument("--out", type=Path, default=OUT / "window_count_probe.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache, metadata = read_pilot_cache(args.cache)
    windows = torch.from_numpy(cache.pop("windows"))
    labels = np.asarray(cache["labels"], int)
    strata = np.asarray(cache["strata"])
    groups = np.asarray(cache["groups"])
    dev_ids = np.flatnonzero(np.asarray(cache["is_dev"], bool))
    y = labels[dev_ids]
    st = strata[dev_ids]
    grp = groups[dev_ids]

    result = {"protocol": {
        "split": metadata, "n_dev": int(len(dev_ids)),
        "even_windows": list(EVEN), "odd_windows": list(ODD),
        "score": "mean logit difference over selected windows",
    }, "seeds": {}}
    for seed in SEEDS:
        ck = BASE / f"seed{seed}/magnitude_last.pt"
        model = TinyResNet(in_channels=1).to(args.device)
        model.load_state_dict(torch.load(ck, map_location=args.device))
        model.eval()
        full = score_subset(model, windows, dev_ids, range(8), args.device)
        even = score_subset(model, windows, dev_ids, EVEN, args.device)
        odd = score_subset(model, windows, dev_ids, ODD, args.device)
        per_window = np.stack([
            score_subset(model, windows, dev_ids, [i], args.device)
            for i in range(8)
        ], axis=1)
        # Verify against the saved full-8 score.
        saved = list(csv.DictReader((BASE / f"seed{seed}/magnitude_dev.csv").open()))
        saved_score = np.array([float(r["logit_11"]) - float(r["logit_10"])
                                for r in saved])
        result["seeds"][str(seed)] = {
            "checkpoint": str(ck),
            "full8_vs_saved_max_abs": float(np.max(np.abs(full - saved_score))),
            "auc": {
                "full8": auc_by_stratum(y, full, st),
                "even4": auc_by_stratum(y, even, st),
                "odd4": auc_by_stratum(y, odd, st),
            },
            "delta_full8_minus_even4": (auc(y[st == "strong"], full[st == "strong"]) -
                                        auc(y[st == "strong"], even[st == "strong"])),
            "delta_full8_minus_odd4": (auc(y[st == "strong"], full[st == "strong"]) -
                                       auc(y[st == "strong"], odd[st == "strong"])),
            "bootstrap_full8_minus_even4_strong": bootstrap_auc_diff(
                y[st == "strong"], full[st == "strong"], even[st == "strong"],
                grp[st == "strong"], seed=seed),
            "bootstrap_full8_minus_odd4_strong": bootstrap_auc_diff(
                y[st == "strong"], full[st == "strong"], odd[st == "strong"],
                grp[st == "strong"], seed=seed + 100),
            "agreement": {
                "full8_even4": float(np.mean((full >= 0) == (even >= 0))),
                "full8_odd4": float(np.mean((full >= 0) == (odd >= 0))),
                "even4_odd4": float(np.mean((even >= 0) == (odd >= 0))),
            },
            "corr_spearman_strong": {
                "full8_even4": float(np.corrcoef(
                    np.argsort(np.argsort(full[st == "strong"])),
                    np.argsort(np.argsort(even[st == "strong"])))[0, 1]),
                "full8_odd4": float(np.corrcoef(
                    np.argsort(np.argsort(full[st == "strong"])),
                    np.argsort(np.argsort(odd[st == "strong"])))[0, 1]),
            },
            "within_instance_sd_strong": float(np.mean(
                np.std(per_window[st == "strong"], axis=1, ddof=1))),
            "default_bal": {
                "full8": metrics(y, (full >= 0).astype(int), st)["all"]["balanced_accuracy"],
                "even4": metrics(y, (even >= 0).astype(int), st)["all"]["balanced_accuracy"],
                "odd4": metrics(y, (odd >= 0).astype(int), st)["all"]["balanced_accuracy"],
            },
        }
        # save scores
        path = OUT / f"raw/window_count_scores_seed{seed}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "signal_idx", "group_id", "class", "overlap",
                        "full8", "even4", "odd4"] +
                       [f"win{i}" for i in range(8)])
            for j, i in enumerate(dev_ids):
                w.writerow([cache["sample_ids"][i], cache["signal_indices"][i],
                            cache["groups"][i], int(labels[i]) + 10, strata[i],
                            float(full[j]), float(even[j]), float(odd[j]),
                            *[float(per_window[j, k]) for k in range(8)]])
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        s: {
            "strong_auc": {k: v["strong"]["auc"]
                           for k, v in r["auc"].items()},
            "full8_minus_even4": r["delta_full8_minus_even4"],
            "full8_minus_odd4": r["delta_full8_minus_odd4"],
            "boot_even": r["bootstrap_full8_minus_even4_strong"],
            "boot_odd": r["bootstrap_full8_minus_odd4_strong"],
            "agreement": r["agreement"],
            "within_sd": r["within_instance_sd_strong"],
        }
        for s, r in result["seeds"].items()
    }, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
