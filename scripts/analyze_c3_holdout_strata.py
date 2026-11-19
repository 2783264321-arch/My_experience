#!/usr/bin/env python3
"""Stratify the frozen 1,159-sample holdout predictions by observation B/T.

Reuses existing eval_holdout NPZ predictions. No inference, training, weight
selection or threshold tuning. For each 3/4/5 GT, matches each model
class-agnostically at IoU>=0.5; common boxes are those matched by both.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.signals.data import SPLIT_MANIFEST, load_manifest
from src.signals.metrics import ImagePredictions, ImageTargets, _match_single

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [
    ("s1729_best", "c3_pilot_dense", "best", "c3_pilot_zero3_s1729", "best"),
    ("s1729_last", "c3_pilot_dense", "last", "c3_pilot_zero3_s1729", "last"),
    ("s1730_best", "c3_pilot_dense_s1730", "best", "c3_pilot_zero3_s1730", "best"),
    ("s1730_last", "c3_pilot_dense_s1730", "last", "c3_pilot_zero3_s1730", "last"),
]
CLASSES = [3, 4, 5]


def holdout_records():
    subset = json.loads((ROOT / "cache/c3_subset.json").read_text())
    sub = set(subset["val_ids"])
    val = load_manifest(SPLIT_MANIFEST, "val")
    hold = [r for r in val if r.sample_id not in sub]
    if len(hold) != 1159:
        raise SystemExit(f"holdout {len(hold)}")
    return hold


def arrays(path):
    z = np.load(path)
    out = {}
    for name in ("gt", "pred"):
        boxes, scores, labels, idx = (z[f"{name}_boxes"], z[f"{name}_scores"],
                                      z[f"{name}_labels"], z[f"{name}_sample_index"])
        per = []
        for i in range(int(idx.max()) + 1 if len(idx) else 0):
            m = idx == i
            per.append((boxes[m], scores[m], labels[m]))
        out[name] = per
    return out


def empty():
    return {"n_samples": 0, "n_gt": 0, "matched_dense": 0, "matched_zero": 0,
            "common": 0, "dense_correct": 0, "zero_correct": 0,
            "dense_only": 0, "zero_only": 0, "both_correct": 0, "both_wrong": 0}


def add(dst, src):
    for k in dst:
        dst[k] += src[k]


def analyze_pair(hold, dense_path, zero_path, tag):
    d = arrays(dense_path)
    z = arrays(zero_path)
    by_b = defaultdict(empty); by_t = defaultdict(empty); by_bt = defaultdict(empty)
    for i, rec in enumerate(hold):
        gb, gs, gl = d["gt"][i] if i < len(d["gt"]) else (np.zeros((0,4)), np.zeros((0,)), np.zeros((0,),dtype=int))
        gt = ImageTargets(gb, gl)
        pa = ImagePredictions(*d["pred"][i]) if i < len(d["pred"]) else ImagePredictions(np.zeros((0,4)),np.zeros((0,)),np.zeros((0,),dtype=int))
        pz = ImagePredictions(*z["pred"][i]) if i < len(z["pred"]) else ImagePredictions(np.zeros((0,4)),np.zeros((0,)),np.zeros((0,),dtype=int))
        ma, _ = _match_single(gt, pa, 0.5, class_aware=False, max_dets=100)
        mz, _ = _match_single(gt, pz, 0.5, class_aware=False, max_dets=100)
        amap = {gi: pi for pi, gi in ma}; zmap = {gi: pi for pi, gi in mz}
        B = float(round(rec.B_mhz, 3)); T = int(round(rec.T_ms))
        groups = [by_b[B], by_t[T], by_bt[(B, T)]]
        for g in groups:
            g["n_samples"] += 1
        for gi, c in enumerate(gl):
            if int(c) not in CLASSES:
                continue
            ha, hz = gi in amap, gi in zmap
            for g in groups:
                g["n_gt"] += 1
                g["matched_dense"] += int(ha)
                g["matched_zero"] += int(hz)
                if ha and hz:
                    g["common"] += 1
                    ca = int(pa.labels[amap[gi]]) == int(c)
                    cz = int(pz.labels[zmap[gi]]) == int(c)
                    g["dense_correct"] += int(ca)
                    g["zero_correct"] += int(cz)
                    if ca and cz: g["both_correct"] += 1
                    elif ca: g["dense_only"] += 1
                    elif cz: g["zero_only"] += 1
                    else: g["both_wrong"] += 1
    def finish(g):
        g["dense_recall"] = g["matched_dense"]/g["n_gt"] if g["n_gt"] else None
        g["zero_recall"] = g["matched_zero"]/g["n_gt"] if g["n_gt"] else None
        g["dense_common_acc"] = g["dense_correct"]/g["common"] if g["common"] else None
        g["zero_common_acc"] = g["zero_correct"]/g["common"] if g["common"] else None
        g["delta_dense_minus_zero"] = g["dense_only"] - g["zero_only"]
        return g
    return {
        "tag": tag,
        "by_B": {str(k): finish(v) for k, v in sorted(by_b.items())},
        "by_T": {str(k): finish(v) for k, v in sorted(by_t.items())},
        "by_BT": {f"{k[0]:g}MHz_{k[1]}ms": finish(v) for k, v in sorted(by_bt.items())},
    }


def consistency(rows, key):
    """rows: list of pair results; key: 'by_B', etc. Condition -> signed deltas."""
    conditions = sorted(set().union(*[set(p[key]) for p in rows]))
    out = {}
    for cond in conditions:
        ds = [p[key][cond]["delta_dense_minus_zero"] for p in rows]
        commons = [p[key][cond]["common"] for p in rows]
        signs = ["+" if d > 0 else "-" if d < 0 else "0" for d in ds]
        out[cond] = {
            "deltas": ds, "signs": signs, "common": commons,
            "n_samples": rows[0][key][cond]["n_samples"],
            "n_gt": rows[0][key][cond]["n_gt"],
            "consistent_dense": all(d > 0 for d in ds) and min(commons) >= 10,
            "consistent_zero3": all(d < 0 for d in ds) and min(commons) >= 10,
            "mixed": not (all(d > 0 for d in ds) or all(d < 0 for d in ds)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="reports/c3_holdout_strata.json")
    ap.add_argument("--md", default="reports/c3_holdout_strata.md")
    args = ap.parse_args()
    hold = holdout_records()
    rows = []
    for tag, dr, dc, zr, zc in PAIRS:
        rows.append(analyze_pair(
            hold,
            ROOT / f"runs/{dr}/eval_holdout_{dc}.npz",
            ROOT / f"runs/{zr}/eval_holdout_{zc}.npz",
            tag,
        ))
    out = {
        "holdout_n": len(hold),
        "pairs": [r["tag"] for r in rows],
        "by_B": {b: {r["tag"]: r["by_B"][b] for r in rows} for b in rows[0]["by_B"]},
        "by_T": {t: {r["tag"]: r["by_T"][t] for r in rows} for t in rows[0]["by_T"]},
        "by_BT": {bt: {r["tag"]: r["by_BT"][bt] for r in rows} for bt in rows[0]["by_BT"]},
        "consistency_B": consistency(rows, "by_B"),
        "consistency_T": consistency(rows, "by_T"),
        "consistency_BT": consistency(rows, "by_BT"),
        "pair_results": rows,
    }
    p = ROOT / args.output
    p.write_text(json.dumps(out, indent=2))

    def fmt(x, nd=3):
        return "n/a" if x is None else f"{x:.{nd}f}"
    lines = ["# C3 holdout stratified by B/T (1,159 samples)", "",
             "Frozen existing predictions; no training/inference. Common = both models match the same 3/4/5 GT at IoU>=0.5.",
             "Deltas in the consistency tables are dense-only-correct minus zero3-only-correct.",
             "B x T table only shows groups with n_samples>=10 and n_gt_345>=10.", ""]
    # B table
    for dim, title in [("by_B", "B"), ("by_T", "T")]:
        lines += [f"## By observation {title}", "",
                  "| cond | pair | IQ | GT345 | common | R dense | R zero3 | dense-only | zero3-only | acc dense | acc zero3 |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for cond in sorted(out[dim], key=lambda x: float(x)):
            for tag in out["pairs"]:
                r = out[dim][cond][tag]
                lines.append(f"| {cond} | {tag} | {r['n_samples']} | {r['n_gt']} | {r['common']} | "
                             f"{fmt(r['dense_recall'])} | {fmt(r['zero_recall'])} | {r['dense_only']} | {r['zero_only']} | "
                             f"{fmt(r['dense_common_acc'])} | {fmt(r['zero_common_acc'])} |")
        lines.append("")
    # BxT sufficient
    lines += ["## B x T groups (n_samples>=10, n_gt_345>=10)", "",
              "| B x T | pair | IQ | GT345 | common | R dense | R zero3 | dense-only | zero3-only | acc dense | acc zero3 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for cond, pairs in out["by_BT"].items():
        if any(v["n_samples"] < 10 or v["n_gt"] < 10 for v in pairs.values()):
            continue
        for tag in out["pairs"]:
            r = pairs[tag]
            lines.append(f"| {cond} | {tag} | {r['n_samples']} | {r['n_gt']} | {r['common']} | "
                         f"{fmt(r['dense_recall'])} | {fmt(r['zero_recall'])} | {r['dense_only']} | {r['zero_only']} | "
                         f"{fmt(r['dense_common_acc'])} | {fmt(r['zero_common_acc'])} |")
    lines.append("")
    # consistency
    for dim, title in [("consistency_B", "B"), ("consistency_T", "T")]:
        lines += [f"## Consistency by {title}", "",
                  "| cond | signs (4 pairs) | dense-only total | zero3-only total | common | flag |",
                  "| --- | --- | ---: | ---: | --- | --- |"]
        for cond, c in out[dim].items():
            flag = "dense" if c["consistent_dense"] else "zero3" if c["consistent_zero3"] else "mixed"
            lines.append(f"| {cond} | {' '.join(c['signs'])} | {sum(max(d,0) for d in c['deltas'])} | "
                         f"{sum(max(-d,0) for d in c['deltas'])} | {c['common']} | {flag} |")
    lines += ["", "## Consistency by B x T (n_samples>=10, n_gt_345>=10)", "",
              "| B x T | signs (4 pairs) | deltas | common | flag |",
              "| --- | --- | --- | --- | --- |"]
    for cond, c in out["consistency_BT"].items():
        if c["n_samples"] < 10 or c["n_gt"] < 10:
            continue
        flag = "dense" if c["consistent_dense"] else "zero3" if c["consistent_zero3"] else "mixed"
        lines.append(f"| {cond} | {' '.join(c['signs'])} | {c['deltas']} | {c['common']} | {flag} |")
    # 40x80 special
    key = "40MHz_80ms"
    if key in out["by_BT"]:
        lines += ["", "## 40 MHz x 80 ms (single-group check; not a target-selection rule)", "",
                  "| pair | IQ | GT345 | common | R dense | R zero3 | dense-only | zero3-only | acc dense | acc zero3 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for tag in out["pairs"]:
            r = out["by_BT"][key][tag]
            lines.append(f"| {tag} | {r['n_samples']} | {r['n_gt']} | {r['common']} | {fmt(r['dense_recall'])} | "
                         f"{fmt(r['zero_recall'])} | {r['dense_only']} | {r['zero_only']} | "
                         f"{fmt(r['dense_common_acc'])} | {fmt(r['zero_common_acc'])} |")
    md = ROOT / args.md
    md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", p, md)


if __name__ == "__main__":
    main()
