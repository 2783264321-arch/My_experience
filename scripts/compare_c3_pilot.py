#!/usr/bin/env python3
"""Assemble the C3 front-end pilot comparison (C1 mean + complex arms).

All arms use the same 1536/384 subset, 20,000 optimizer steps, effective batch
16 and paired initialization. Absolute mAP is not comparable to the full
0.6497 baseline; only paired deltas are meaningful.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("C1 s1729", "c3_pilot_c1"),
    ("C1 s1730", "c3_pilot_c1_s1730"),
    ("overlap s1729", "c3_pilot_overlap"),
    ("overlap s1730", "c3_pilot_overlap_s1730"),
    ("sparse s1729", "c3_pilot_sparse"),
    ("sparse s1730", "c3_pilot_sparse_s1730"),
    ("dense s1729", "c3_pilot_dense"),
]
GROUPS = ["0/1/2", "3/4/5", "10/11", "9/12/13"]


def load(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return json.loads(p.read_text()) if p.exists() else None


def best_entry(summary):
    if not summary:
        return None
    hist = summary.get("val_history", [])
    if not hist:
        return None
    target = summary.get("best_step", summary.get("best_epoch"))
    for e in hist:
        if target is not None and int(e.get("global_step", -1)) == int(target):
            return e
    return max(hist, key=lambda x: x.get("mAP50", -1))


def wifi_counts(confusion):
    if not confusion:
        return {}
    c = np.asarray(confusion)
    out = {}
    for name, classes in [("0/1/2", [0, 1, 2]), ("3/4/5", [3, 4, 5])]:
        correct = wrong = 0
        for g in classes:
            row = c[g]
            correct += int(row[g])
            wrong += int(row.sum() - row[g])
        out[name] = {"correct": correct, "wrong": wrong}
    return out


def mean_entry(entries):
    out = {"groups": {}, "confusion": None, "n_pred": None}
    for key in ("mAP50", "map50_agnostic", "localization_recall",
                "classification_accuracy"):
        vals = [e.get(key) for e in entries if e.get(key) is not None]
        out[key] = float(np.mean(vals)) if vals else None
    for g in GROUPS:
        vals = [e.get("groups", {}).get(g, {}).get("map50") for e in entries]
        vals = [v for v in vals if v is not None]
        out["groups"][g] = {"map50": float(np.mean(vals)) if vals else None}
    vals = [e.get("n_pred") for e in entries if e.get("n_pred") is not None]
    out["n_pred"] = int(round(float(np.mean(vals)))) if vals else None
    return out


def mean_wifi(counts):
    out = {}
    for g in ("0/1/2", "3/4/5"):
        vals = [c.get(g, {}) for c in counts if c]
        out[g] = {
            "correct": float(np.mean([v.get("correct", 0) for v in vals])) if vals else None,
            "wrong": float(np.mean([v.get("wrong", 0) for v in vals])) if vals else None,
        }
    return out


def fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    return f"{float(x):.{nd}f}"


def build_seconds(meta):
    if not meta:
        return None
    v = meta.get("build_seconds")
    if isinstance(v, dict):
        return sum(float(x) for x in v.values() if isinstance(x, (int, float)))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-md", default="reports/c3_pilot_results.md")
    ap.add_argument("--output-json", default="reports/c3_pilot_results.json")
    args = ap.parse_args()
    data = {}
    for label, run in RUNS:
        s = load(ROOT / "runs" / run / "summary.json")
        e = best_entry(s)
        data[label] = {"run": run, "summary": s, "best": e,
                       "wifi": wifi_counts((e or {}).get("confusion"))}
    c1_entries = [data["C1 s1729"]["best"], data["C1 s1730"]["best"]]
    c1_entries = [e for e in c1_entries if e]
    c1_mean = mean_entry(c1_entries)
    c1_wifi = mean_wifi([data["C1 s1729"]["wifi"], data["C1 s1730"]["wifi"]])

    cache = {}
    for name in ("overlap", "sparse", "dense"):
        m = load(ROOT / "reports" / f"c3_cache_meta_{name}.json")
        if m:
            cache[name] = {"build_seconds": build_seconds(m),
                           "shape": m.get("shape"), "scale": m.get("scale")}
    front = load(ROOT / "reports" / "c3_frontend_validation.json")

    lines = [
        "# C3 front-end pilot (matched 20k steps)", "",
        "Same power image, labels, 1536/384 subset, 20,000 optimizer steps and",
        "effective batch 16. C1 is the power-only control; C1 mean is the average",
        "of seeds 1729/1730. Absolute mAP is a subset result, not comparable to",
        "the full 0.6497 baseline. Best EMA per arm is used.", "",
        "## Best EMA", "",
        "| arm | mAP | best step | AP_agn | R_agn | 0/1/2 | 3/4/5 | 10/11 | 9/12/13 | n_pred | train h | peak GiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    # C1 mean row
    lines.append(
        f"| **C1 mean** | **{fmt(c1_mean.get('mAP50'))}** | - | "
        f"{fmt(c1_mean.get('map50_agnostic'))} | {fmt(c1_mean.get('localization_recall'))} | "
        f"{fmt(c1_mean['groups']['0/1/2']['map50'])} | {fmt(c1_mean['groups']['3/4/5']['map50'])} | "
        f"{fmt(c1_mean['groups']['10/11']['map50'])} | {fmt(c1_mean['groups']['9/12/13']['map50'])} | "
        f"{c1_mean.get('n_pred')} | 0.79 | 8.6 |"
    )
    for label, _ in RUNS:
        d = data[label]
        s, e = d["summary"], d["best"]
        if not s or not e:
            lines.append(f"| {label} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        g = e.get("groups", {})
        lines.append(
            f"| {label} | {fmt(e.get('mAP50'))} | {s.get('best_step', 'n/a')} | "
            f"{fmt(e.get('map50_agnostic'))} | {fmt(e.get('localization_recall'))} | "
            f"{fmt(g.get('0/1/2', {}).get('map50'))} | {fmt(g.get('3/4/5', {}).get('map50'))} | "
            f"{fmt(g.get('10/11', {}).get('map50'))} | {fmt(g.get('9/12/13', {}).get('map50'))} | "
            f"{e.get('n_pred', 'n/a')} | {fmt(s.get('total_h'), 2)} | {fmt(s.get('peak_gpu_gib'), 1)} |"
        )

    lines += ["", "## Wi-Fi matched counts (best EMA)", "",
              "| arm | 0/1/2 correct | 0/1/2 wrong | 3/4/5 correct | 3/4/5 wrong |",
              "| --- | ---: | ---: | ---: | ---: |",
              f"| **C1 mean** | **{fmt(c1_wifi['0/1/2']['correct'],1)}** | **{fmt(c1_wifi['0/1/2']['wrong'],1)}** | "
              f"**{fmt(c1_wifi['3/4/5']['correct'],1)}** | **{fmt(c1_wifi['3/4/5']['wrong'],1)}** |"]
    for label, _ in RUNS:
        w = data[label]["wifi"] or {}
        a, b = w.get("0/1/2", {}), w.get("3/4/5", {})
        lines.append(f"| {label} | {a.get('correct','n/a')} | {a.get('wrong','n/a')} | "
                     f"{b.get('correct','n/a')} | {b.get('wrong','n/a')} |")

    lines += ["", "## Paired deltas vs C1 mean (best EMA; pp)", "",
              "| arm | ΔmAP | ΔAP_agn | ΔR_agn | Δ0/1/2 | Δ3/4/5 | Δ10/11 | Δ9/12/13 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, _ in RUNS[2:]:
        e = data[label]["best"]
        if not e:
            lines.append(f"| {label} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        g = e.get("groups", {})
        def d(key):
            a, b = e.get(key), c1_mean.get(key)
            return (a - b) * 100 if a is not None and b is not None else None
        def dg(group):
            a = g.get(group, {}).get("map50")
            b = c1_mean["groups"][group]["map50"]
            return (a - b) * 100 if a is not None and b is not None else None
        lines.append(
            f"| {label} | {fmt(d('mAP50'),2)} | {fmt(d('map50_agnostic'),2)} | "
            f"{fmt(d('localization_recall'),2)} | {fmt(dg('0/1/2'),2)} | "
            f"{fmt(dg('3/4/5'),2)} | {fmt(dg('10/11'),2)} | {fmt(dg('9/12/13'),2)} |"
        )

    lines += ["", "## Cache / front-end cost", "",
              "| frontend | build s | shape | scale |", "| --- | ---: | --- | ---: |"]
    for name in ("overlap", "sparse", "dense"):
        c = cache.get(name)
        if not c:
            lines.append(f"| {name} | n/a | n/a | n/a |")
        else:
            lines.append(f"| {name} | {fmt(c.get('build_seconds'), 0)} | "
                         f"{c.get('shape')} | {fmt(c.get('scale'), 6)} |")
    if front:
        t = front.get("timing", {})
        lines += ["", f"Front-end per-sample: {fmt(t.get('per_sample_seconds'),2)} s; "
                  f"1536+384 estimate {fmt((t.get('estimated_1536_fit_plus_384_val_seconds') or 0)/60,1)} min."]

    lines += ["", "## Judgment", "",
              "Pending final synthesis. The key reading is 0/1/2 vs 3/4/5; 10/11",
              "has a large C1 seed spread and should not be used as a stable ranking.",
              "All complex arms are single-seed except overlap, which has two seeds.", ""]

    out_json = ROOT / args.output_json
    out_json.write_text(json.dumps({
        "runs": {k: {"run": v["run"], "best": v["best"], "summary": v["summary"]}
                 for k, v in data.items()},
        "c1_mean": c1_mean, "c1_wifi_mean": c1_wifi,
        "cache": cache, "frontend": front,
    }, indent=2))
    out_md = ROOT / args.output_md
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
