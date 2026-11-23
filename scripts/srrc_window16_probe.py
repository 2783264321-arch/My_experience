#!/usr/bin/env python3
"""Nested 16-window probe for the fixed dev set.

Keeps the original 8 cache windows and adds 8 fixed positions:
  - 7 midpoints between consecutive original starts;
  - 1 quarter-point of the first gap.
All new windows stay inside the GT interval. Reuses the existing
magnitude CNN weights; no training, no detector, no test.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from scripts.baseline3srrc_gt_pilot import TinyResNet, metrics, predict, read_pilot_cache
from scripts.baseline3srrc_phase0_verify import (
    crop_and_resample, windows_from_crop, WIN, N_WIN,
)
from src.signals.data import SPLIT_MANIFEST, load_manifest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/baseline3_srrc_gt_pilot"
BASE = Path("/root/autodl-fs/runs_local/srrc-gt-pilot")
SEEDS = (1729, 1730, 1731)


def auc(y, s):
    y = np.asarray(y, int)
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def auc_strata(y, s, strata):
    out = {}
    for name in ("all", "clean", "weak", "strong"):
        m = np.ones(len(y), bool) if name == "all" else (strata == name)
        out[name] = {"n": int(m.sum()), "auc": auc(y[m], s[m])}
    return out


def bootstrap_diff(y, a, b, groups, n_boot=5000, seed=1729):
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


def new_starts(winfo):
    s = np.array([w["start"] for w in winfo["windows"]], dtype=int)
    gt_start = int(winfo["gt_start"]); gt_end = int(winfo["gt_end"])
    if len(s) != N_WIN:
        raise ValueError("expected 8 original windows")
    mids = [int(round((s[i] + s[i + 1]) / 2.0)) for i in range(7)]
    extra = int(round(s[0] + (s[1] - s[0]) / 4.0))
    new = mids + [extra]
    if len(set(new)) != 8 or set(new) & set(s.tolist()):
        raise ValueError(f"new starts not unique: {new} vs {s.tolist()}")
    if min(new) < gt_start or max(new) > gt_end - WIN:
        raise ValueError(f"new start outside GT: {min(new)}..{max(new)}")
    return s, np.array(new, dtype=int)


@torch.inference_mode()
def score(model, windows, device, batch=32):
    out = []
    for start in range(0, len(windows), batch):
        x = windows[start:start + batch].to(device)
        logits = predict(model, x, "magnitude").cpu().numpy()
        out.append(logits[:, 1] - logits[:, 0])
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=BASE / "fit_8win.npz")
    ap.add_argument("--out", type=Path, default=OUT / "window16_probe.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache, meta = read_pilot_cache(args.cache)
    old8 = cache["windows"]
    labels = np.asarray(cache["labels"], int)
    strata = np.asarray(cache["strata"])
    groups = np.asarray(cache["groups"])
    dev_ids = np.flatnonzero(np.asarray(cache["is_dev"], bool))
    recs = {r.sample_id: r for r in load_manifest(SPLIT_MANIFEST, split="fit")}

    t0 = time.perf_counter()
    w16 = np.zeros((len(dev_ids), 16, 2, WIN), dtype=np.float32)
    max_old_diff = 0.0
    unique_new = 0
    examples = []
    for row, i in enumerate(dev_ids):
        sid = str(cache["sample_ids"][i]); si = int(cache["signal_indices"][i])
        rec = recs[sid]; sig = rec.signals[si]
        crop = crop_and_resample(rec, sig)
        winfo = windows_from_crop(crop, sig, n_win=8)
        s, new = new_starts(winfo)
        old = np.zeros((8, 2, WIN), dtype=np.float32)
        for k, w in enumerate(winfo["windows"]):
            samples = crop["x"][w["start"]:w["stop"]]
            old[k, 0, :len(samples)] = samples.real
            old[k, 1, :len(samples)] = samples.imag
        max_old_diff = max(max_old_diff, float(np.max(np.abs(old - old8[i]))))
        w16[row, :8] = old
        for k, start in enumerate(new):
            stop = min(start + WIN, winfo["gt_end"])
            samples = crop["x"][start:stop]
            if len(samples) < WIN:
                samples = np.pad(samples, (0, WIN - len(samples)))
            w16[row, 8 + k, 0] = samples.real
            w16[row, 8 + k, 1] = samples.imag
        unique_new += 1
        if len(examples) < 3:
            examples.append({"sample_id": sid, "B_obs_mhz": rec.B_mhz,
                             "old_starts": s.tolist(), "new_starts": new.tolist(),
                             "gt_start": int(winfo["gt_start"]),
                             "gt_end": int(winfo["gt_end"])})
    extraction_s = time.perf_counter() - t0
    np.savez_compressed(BASE / "dev_16win.npz",
                        windows=w16, labels=labels[dev_ids],
                        is_dev=np.ones(len(dev_ids), bool),
                        groups=groups[dev_ids], strata=strata[dev_ids],
                        sample_ids=cache["sample_ids"][dev_ids],
                        signal_indices=cache["signal_indices"][dev_ids],
                        metadata=json.dumps({**meta, "n_windows": 16,
                                             "new_window_rule":
                                             "7 midpoints + first-gap quarter"}))

    y = labels[dev_ids]; st = strata[dev_ids]; grp = groups[dev_ids]
    w8_t = torch.from_numpy(old8[dev_ids])
    w16_t = torch.from_numpy(w16)
    result = {
        "protocol": {"n_dev": int(len(dev_ids)), "old8": list(range(8)),
                     "new8_rule": "7 midpoints between original starts + "
                                  "first-gap quarter point",
                     "extraction_seconds": extraction_s,
                     "old8_recompute_max_abs": max_old_diff,
                     "unique_new_per_instance": unique_new,
                     "examples": examples,
                     "score": "mean logit difference over selected windows"},
        "seeds": {},
    }
    for seed in SEEDS:
        model = TinyResNet(in_channels=1).to(args.device)
        model.load_state_dict(torch.load(BASE / f"seed{seed}/magnitude_last.pt",
                                         map_location=args.device))
        model.eval()
        # warmup
        _ = score(model, w8_t[:2], args.device)
        if args.device == "cuda":
            torch.cuda.synchronize()
        t8 = time.perf_counter(); s8 = score(model, w8_t, args.device)
        if args.device == "cuda":
            torch.cuda.synchronize()
        t8 = time.perf_counter() - t8
        t16 = time.perf_counter(); s16 = score(model, w16_t, args.device)
        if args.device == "cuda":
            torch.cuda.synchronize()
        t16 = time.perf_counter() - t16
        saved = list(csv.DictReader((BASE / f"seed{seed}/magnitude_dev.csv").open()))
        saved_score = np.array([float(r["logit_11"]) - float(r["logit_10"])
                                for r in saved])
        result["seeds"][str(seed)] = {
            "checkpoint": str(BASE / f"seed{seed}/magnitude_last.pt"),
            "saved_full8_max_abs": float(np.max(np.abs(s8 - saved_score))),
            "auc": {"8": auc_strata(y, s8, st), "16": auc_strata(y, s16, st)},
            "strong_delta": float(auc(y[st == "strong"], s16[st == "strong"]) -
                                  auc(y[st == "strong"], s8[st == "strong"])),
            "bootstrap_strong": bootstrap_diff(y[st == "strong"], s16[st == "strong"],
                                               s8[st == "strong"], grp[st == "strong"],
                                               seed=seed),
            "default": {
                "8": metrics(y, (s8 >= 0).astype(int), st),
                "16": metrics(y, (s16 >= 0).astype(int), st),
            },
            "inference_s": {"8": t8, "16": t16, "extra": t16 - t8},
        }
        path = OUT / f"raw/window16_scores_seed{seed}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "signal_idx", "group_id", "class", "overlap",
                        "score8", "score16"])
            for j, i in enumerate(dev_ids):
                w.writerow([cache["sample_ids"][i], cache["signal_indices"][i],
                            cache["groups"][i], int(labels[i]) + 10,
                            strata[i], float(s8[j]), float(s16[j])])
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "extraction_s": extraction_s,
        "old8_max_diff": max_old_diff,
        "seeds": {s: {"strong_auc_8": r["auc"]["8"]["strong"]["auc"],
                      "strong_auc_16": r["auc"]["16"]["strong"]["auc"],
                      "delta": r["strong_delta"],
                      "ci": [r["bootstrap_strong"]["lo"], r["bootstrap_strong"]["hi"]],
                      "default_strong_8": r["default"]["8"]["strong"],
                      "default_strong_16": r["default"]["16"]["strong"],
                      "infer_s": r["inference_s"]}
                  for s, r in result["seeds"].items()},
    }, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
