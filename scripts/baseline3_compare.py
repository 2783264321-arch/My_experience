#!/usr/bin/env python3
"""Baseline-3 round-1 paired comparison: IQ physical features vs same-frame
power baselines on the shared GT-box frames.

Arms (per bandwidth group 0/1/2, 3/4/5 and SRRC 10/11):

  coarse       SmallCNN on the 0.7 ms coarse crop (same recipe as the
               separability study, rebuilt at GT-box geometry)
  local_power  SmallCNN on the 0.1 ms local power crop, same geometry
  iq_full      multinomial logistic on the per-frame physical census features
  iq_band      ablation: 16-bin in-box energy layout + fill/edge only
  iq_line     ablation: envelope/x^2 lines, autocorr, kurtosis, symmetry

Split: by IQ sample, holdout 0.2, seed 1729 -- computed per family exactly as
split_by_sample does, so the WiFi split matches the published separability
study. Holdout is evaluated once, after all arms are frozen. Class labels are
used only to fit classifiers on the train side and to score holdout.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from scripts.train_proposal_classifiers import SmallCNN, split_by_sample
from scripts.train_local_representation_classifiers import (
    train_one, confusion_metrics, standardize_coarse,
)

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"012": [0, 1, 2], "345": [3, 4, 5], "1011": [10, 11]}
FAMILY = {"012": "wifi", "345": "wifi", "1011": "srrc"}

BAND_FEATURES = ["fill10", "fill20", "edge_db"] + [f"band{q:02d}" for q in range(16)]
LINE_FEATURES = ["sym_db", "r32", "r64", "r128",
                 "env_line_str", "env_line2_str", "x2_line_str",
                 "env_at_a20", "env_at_a35", "env_at_a50",
                 "kurt_time", "n_peaks", "tone_width_mhz"]
POS_FEATURES = ["peak_off_mhz", "env_line_mhz", "x2_line_mhz", "tone_center_mhz",
                "box_to_cap_db", "floor_ratio_db"]
FULL_FEATURES = BAND_FEATURES + LINE_FEATURES + POS_FEATURES


def standardize_fit(xtr, xva):
    m = xtr.mean(axis=0)
    s = np.maximum(xtr.std(axis=0), 1e-8)
    return (xtr - m) / s, (xva - m) / s


def logistic_fit(xtr, ytr, n_classes, seed=0, wd=1e-3, iters=500, lr=0.1):
    """Full-batch Adam multinomial logistic regression with class weights."""
    torch.manual_seed(seed)
    xtr = torch.from_numpy(xtr.astype(np.float32))
    ytr = torch.from_numpy(ytr.astype(np.int64))
    d = xtr.shape[1]
    w = torch.zeros(d + 1, n_classes, requires_grad=True)
    counts = np.bincount(ytr.numpy(), minlength=n_classes).astype(np.float64)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))
    wt = torch.tensor(weights, dtype=torch.float32)
    opt = torch.optim.Adam([w], lr=lr)
    xc = torch.cat([xtr, torch.ones(len(xtr), 1)], dim=1)
    for _ in range(iters):
        opt.zero_grad()
        logits = xc @ w
        loss = torch.nn.functional.cross_entropy(logits, ytr, weight=wt) + \
            wd * (w[:-1] ** 2).mean()
        loss.backward()
        opt.step()
    return w.detach().numpy()


def logistic_predict(w, x):
    xc = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    return np.argmax(xc @ w, axis=1)


def one_vs_rest_auc(x, y, c):
    """Rank-based AUC of feature x for class c vs rest (train-side aid)."""
    m = np.isfinite(x)
    s = np.argsort(x[m])
    ranks = np.empty(int(m.sum()))
    ranks[s] = np.arange(1, len(ranks) + 1)
    xv = x[m]
    yv = (y[m] == c).astype(int)
    n1 = int(yv.sum()); n0 = len(yv) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r1 = ranks[yv == 1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n1 * n0))


def binom_p_two_sided(k, n):
    if n == 0:
        return 1.0
    from math import comb
    p = 0.0
    for i in range(0, n + 1):
        if comb(n, i) <= comb(n, min(k, n - k)) * (1 + 1e-9):
            p += comb(n, i) * 0.5 ** n
    return min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", default="reports/local_iq_round1/power_reps.npz")
    ap.add_argument("--census", default="reports/local_iq_round1/census_all.npz")
    ap.add_argument("--out", default="reports/local_iq_round1/comparison.json")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", default="1729,1730,1731")
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=1729)
    ap.add_argument("--skip-cnn", action="store_true")
    ap.add_argument("--power-arms", default="",
                    help="merge pre-trained power arms from baseline3_power_arms.py JSON")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    z = np.load(ROOT / args.power)
    labels = z["label"].astype(np.int64)
    sid = z["sample_id"].astype(str)
    n = len(labels)

    cz = np.load(ROOT / args.census)
    key = {}
    csid = cz["sample_id"].astype(str)
    csig = cz["signal_id"].astype(np.int64)
    for i in range(len(csid)):
        key[(str(csid[i]), int(csig[i]))] = i
    feats = np.full((n, len(FULL_FEATURES)), np.nan, dtype=np.float64)
    fidx = {k: j for j, k in enumerate(FULL_FEATURES)}
    missing = 0
    for i in range(n):
        j = key.get((str(sid[i]), int(z["signal_id"][i])))
        if j is None:
            missing += 1
            continue
        for k, col in cz.files:
            if k in fidx:
                feats[i, fidx[k]] = float(cz[col][j])
    ok = np.all(np.isfinite(feats), axis=1)
    print(f"frames={n} census_missing={missing} nan_rows={int((~ok).sum())}", flush=True)

    result = {"frames": int(n), "census_missing": missing,
              "nan_rows": int((~ok).sum()), "groups": {}}
    power_arms = None
    if args.power_arms:
        power_arms = json.loads((ROOT / args.power_arms).read_text())

    for gname, classes in GROUPS.items():
        fam = FAMILY[gname]
        gm = np.isin(labels, classes) & ok
        idx = np.where(gm)[0]
        y = np.array([classes.index(int(v)) for v in labels[idx]], dtype=np.int64)
        s_g = sid[idx]
        tr_m, va_m = split_by_sample(s_g, labels[idx], classes,
                                     holdout=args.holdout, seed=args.split_seed)
        tr = np.where(gm)[0][tr_m]
        va = np.where(gm)[0][va_m]
        g = {
            "classes": classes,
            "n": int(len(idx)),
            "train": int(len(tr)), "holdout": int(len(va)),
            "train_samples": int(len(set(sid[tr]))),
            "holdout_samples": int(len(set(sid[va]))),
            "class_counts_holdout": {str(c): int((labels[va] == c).sum()) for c in classes},
            "arms": {},
        }

        # ---- power baselines (SmallCNN, same recipe) ----
        for rep in ("coarse", "local_power"):
            if args.skip_cnn and not power_arms:
                continue
            if power_arms is not None and gname in power_arms.get("groups", {}):
                g["arms"][rep] = power_arms["groups"][gname]["arms"][rep]
                continue
            x = z[rep].astype(np.float32)
            x = np.clip(standardize_coarse(x)[:, None], -10.0, 10.0)
            x_tr = x[tr]; x_va = x[va]
            y_tr = y[tr_m]; y_va = y[va_m]
            seed_rows = []
            for seed in seeds:
                t0 = time.perf_counter()
                model, pred, _ = train_one(
                    x_tr, y_tr, x_va, y_va, len(classes), seed,
                    args.epochs, 64, device,
                )
                conf = np.zeros((len(classes), len(classes)), dtype=np.int64)
                for t, p in zip(y_va.tolist(), pred.tolist()):
                    conf[t, p] += 1
                seed_rows.append({
                    "seed": seed,
                    "confusion": confusion_metrics(conf),
                    "seconds": round(time.perf_counter() - t0, 1),
                })
            g["arms"][rep] = {
                "seeds": seed_rows,
                "holdout_acc_mean": float(np.mean([r["confusion"]["accuracy"]
                                                   for r in seed_rows])),
            }

        # ---- IQ feature arms ----
        for arm, cols in [
            ("iq_full", FULL_FEATURES),
            ("iq_band", BAND_FEATURES),
            ("iq_line", LINE_FEATURES),
        ]:
            sel = [FULL_FEATURES.index(c) for c in cols]
            X = feats[:, sel]
            x_tr, x_va = standardize_fit(X[tr], X[va])
            y_tr = y[tr_m]; y_va = y[va_m]
            seed_rows = []
            for seed in seeds:
                w = logistic_fit(x_tr, y_tr, len(classes), seed=seed)
                pred = logistic_predict(w, x_va)
                conf = np.zeros((len(classes), len(classes)), dtype=np.int64)
                for t, p in zip(y_va.tolist(), pred.tolist()):
                    conf[t, p] += 1
                seed_rows.append({
                    "seed": seed,
                    "confusion": confusion_metrics(conf),
                })
            g["arms"][arm] = {
                "features": cols,
                "seeds": seed_rows,
                "holdout_acc_mean": float(np.mean([r["confusion"]["accuracy"]
                                                   for r in seed_rows])),
            }
            if arm == "iq_full" and seed == seeds[-1]:
                # train-side AUC diagnostics for the report
                auc = {}
                for j, cname in enumerate(cols):
                    vals = [one_vs_rest_auc(X[tr][:, j], y_tr, c) for c in range(len(classes))]
                    auc[cname] = [None if (v != v) else round(v, 3) for v in vals]
                g["arm_train_auc_iq_full"] = auc

        # ---- paired McNemar: iq_full vs best power arm, per instance ----
        best_power = None
        if power_arms is not None and gname in power_arms.get("groups", {}):
            reps_avail = [r for r in ("coarse", "local_power")
                          if r in g["arms"]]
            best_power = max(reps_avail,
                             key=lambda r: g["arms"][r]["holdout_acc_mean"])
        elif not args.skip_cnn:
            reps_avail = [r for r in ("coarse", "local_power") if r in g["arms"]]
            best_power = max(reps_avail,
                             key=lambda r: g["arms"][r]["holdout_acc_mean"])
        g["best_power_arm"] = best_power
        if best_power:
            y_va_arr = y[va_m]
            sel = list(range(len(FULL_FEATURES)))
            X = feats[:, sel]
            x_tr, x_va_f = standardize_fit(X[tr], X[va])
            w = logistic_fit(x_tr, y[tr_m], len(classes), seed=seeds[0])
            iq_pred = logistic_predict(w, x_va_f)
            iq_correct = (iq_pred == y_va_arr)
            arm0 = g["arms"][best_power]["seeds"][0]
            if "pred_holdout" in arm0:
                pred_pw = np.asarray(arm0["pred_holdout"], dtype=np.int64)
            else:
                xrep = z[best_power].astype(np.float32)
                xrep = np.clip(standardize_coarse(xrep)[:, None], -10.0, 10.0)
                _, pred_pw, _ = train_one(
                    xrep[tr], y[tr_m], xrep[va], y_va_arr, len(classes), seeds[0],
                    args.epochs, 64, device,
                )
            pw_correct = (pred_pw == y_va_arr)
            n01 = int((iq_correct & ~pw_correct).sum())
            n10 = int((~iq_correct & pw_correct).sum())
            g["paired_iq_vs_power"] = {
                "power_arm": best_power,
                "iq_correct_power_wrong": n01,
                "power_correct_iq_wrong": n10,
                "binom_p": binom_p_two_sided(min(n01, n10), n01 + n10),
                "acc_iq": float(iq_correct.mean()),
                "acc_power": float(pw_correct.mean()),
            }
            ov = z["overlap_wifi"] if "overlap_wifi" in z.files else np.zeros(n, dtype=np.int8)
            anyv = z["overlap_any"]
            ovf = z["overlap_wifi_frac_max"] if "overlap_wifi_frac_max" in z.files else np.zeros(n)
            masks = {
                "wifi_clean": (ov == 0) if fam == "wifi" else (anyv == 0),
                "wifi_overlap": (ov == 1) if fam == "wifi" else (anyv == 1),
                "strong_overlap_25pct": ovf >= 0.25,
                "any_overlap": anyv == 1,
            }
            g["strata"] = {}
            for name, mk in masks.items():
                mva = mk[va]
                if mva.sum():
                    g["strata"][name] = {
                        "n": int(mva.sum()),
                        "acc_iq": float(iq_correct[mva].mean()),
                        "acc_power": float(pw_correct[mva].mean()),
                    }
        result["groups"][gname] = g
        print(json.dumps({
            "group": gname,
            **{a: round(g["arms"][a]["holdout_acc_mean"], 4)
               for a in g["arms"]},
        }, indent=2), flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
