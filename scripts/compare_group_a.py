#!/usr/bin/env python3
"""Assemble the Group A vs B vs C comparison report.

Inputs are the artifacts produced by validate_run.py,
analyze_signal_characteristics.py, class_group_metrics.py, and
bench_view_modes.py. Missing optional inputs are rendered as "n/a" rather
than failing, so the report can be built incrementally.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/compare_group_a.py \
    --a-dir runs/A/stage5_val --b-dir runs/regression/c_coarse \
    --c-dir runs/regression/c_all \
    --a-groups reports/group_a_class_groups_stage5_val.json \
    --b-groups reports/group_b_class_groups.json \
    --c-groups reports/group_c_class_groups.json \
    --latency reports/group_a_latency.json \
    --a-run runs/A --c-run runs/C \
    --output-md reports/group_a_results.md \
    --output-json reports/group_a_results.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASS_GROUPS = ["0/1/2", "3/4/5", "10/11", "9/12/13"]
SIGNAL_GROUPS = [
    ("narrow_<=0.25MHz", "narrow ≤0.25 MHz"),
    ("medium_0.3-2MHz", "medium 0.3–2 MHz"),
    ("wide_2.1-10MHz", "wide 2.1–10 MHz"),
    ("verywide_10.1-20MHz", "very wide 10.1–20 MHz"),
    ("ultra_>20MHz", "ultra >20 MHz"),
]


def load(path: str | Path | None):
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(x, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    return f"{float(x):.{nd}f}"


def delta(a, b):
    if a is None or b is None:
        return None
    return float(a) - float(b)


def overall(summary):
    if not summary:
        return {}
    return {
        "map50": summary.get("map50"),
        "ap50_agnostic": summary.get("map50_agnostic"),
        "recall": summary.get("localization_recall"),
        "cls_acc": summary.get("classification_accuracy"),
        "n_gt": summary.get("n_gt"),
        "n_pred": summary.get("n_pred"),
        "n_views": (summary.get("stats") or {}).get("n_views"),
    }


def group_ap(groups, name):
    if not groups:
        return None
    return (groups.get("class_groups", {}).get(name) or {}).get("mean_per_class_ap")


def signal_metric(signal, name, key):
    if not signal:
        return None
    return ((signal.get("dims", {}).get("abs_bw", {}).get(name) or {}).get(key))


def peak_reserved(train_log: Path | None):
    if not train_log or not train_log.exists():
        return None
    vals = []
    with train_log.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                vals.append(float(row["gpu_reserved_gib"]))
            except (KeyError, ValueError):
                pass
    return max(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", default="runs/yolo11s_c1_directreg_coarseonly_2048x512/stage5_val")
    ap.add_argument("--b-dir", default="runs/regression/c_coarse")
    ap.add_argument("--c-dir", default="runs/regression/c_all")
    ap.add_argument("--a-groups", default="reports/group_a_class_groups_stage5_val.json")
    ap.add_argument("--b-groups", default="reports/group_b_class_groups.json")
    ap.add_argument("--c-groups", default="reports/group_c_class_groups.json")
    ap.add_argument("--latency", default="reports/group_a_latency.json")
    ap.add_argument("--a-run", default="runs/yolo11s_c1_directreg_coarseonly_2048x512")
    ap.add_argument("--c-run", default="runs/yolo11s_c1_directreg_2048x512")
    ap.add_argument("--c-run-root", default="/root/autodl-tmp/workspace/baseline/runs/yolo11s_c1_directreg_2048x512")
    ap.add_argument("--cache-meta", default="cache/coarse_fit_bf16.json")
    ap.add_argument("--cache-verify", default="reports/group_a_cache_verification.json")
    ap.add_argument("--output-md", default="reports/group_a_results.md")
    ap.add_argument("--output-json", default="reports/group_a_results.json")
    args = ap.parse_args()

    a_sum = load(Path(args.a_dir) / "summary.json")
    b_sum = load(Path(args.b_dir) / "summary.json")
    c_sum = load(Path(args.c_dir) / "summary.json")
    a_groups = load(args.a_groups)
    b_groups = load(args.b_groups)
    c_groups = load(args.c_groups)
    a_signal = load(Path(args.a_dir) / "signal_characteristics.json")
    b_signal = load(Path(args.b_dir) / "signal_characteristics.json")
    c_signal = load(Path(args.c_dir) / "signal_characteristics.json")
    latency = load(args.latency)
    cache_meta = load(args.cache_meta)
    cache_verify = load(args.cache_verify)

    a_over, b_over, c_over = overall(a_sum), overall(b_sum), overall(c_sum)
    a_cfg = load(Path(args.a_run) / "config.json")
    c_cfg = load(Path(args.c_run_root) / "config.json")
    a_train_sum = load(Path(args.a_run) / "summary.json")
    c_train_sum = load(Path(args.c_run_root) / "summary.json")
    a_reserved = peak_reserved(Path(args.a_run) / "train_log.csv")
    c_reserved = peak_reserved(Path(args.c_run_root) / "train_log.csv")

    out = {
        "overall": {"A": a_over, "B": b_over, "C": c_over},
        "class_groups": {
            g: {"A": group_ap(a_groups, g), "B": group_ap(b_groups, g), "C": group_ap(c_groups, g)}
            for g in CLASS_GROUPS
        },
        "signal_groups": {
            g: {
                "A": {k: signal_metric(a_signal, g, k) for k in ("map50", "ap50_agnostic", "recall_agnostic")},
                "B": {k: signal_metric(b_signal, g, k) for k in ("map50", "ap50_agnostic", "recall_agnostic")},
                "C": {k: signal_metric(c_signal, g, k) for k in ("map50", "ap50_agnostic", "recall_agnostic")},
            }
            for g, _ in SIGNAL_GROUPS
        },
        "latency": (latency or {}).get("summary"),
        "latency_ratios": (latency or {}).get("ratios"),
        "training": {
            "A": {"config": a_cfg, "summary": a_train_sum, "peak_reserved_gib": a_reserved},
            "C": {"config": c_cfg, "summary": c_train_sum, "peak_reserved_gib": c_reserved},
            "cache": cache_meta,
            "cache_verification": cache_verify,
        },
    }
    out_path = Path(args.output_json)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# Group A / B / C comparison",
        "",
        "A = full-band single-image, independent training, coarse-only inference.",
        "B = joint coarse+fine weights, coarse-only inference (same weights as C).",
        "C = joint coarse+fine weights, all-view inference.",
        "The primary A/C checkpoint is the last-epoch EMA. A best is reported separately.",
        "",
        "## Overall",
        "",
        "| metric | A | B | C | B−A | C−B | C−A |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in [
        ("map50", "mAP@0.5"), ("ap50_agnostic", "AP50 agnostic"),
        ("recall", "localization recall"), ("cls_acc", "classification accuracy"),
        ("n_gt", "GT"), ("n_pred", "predictions"), ("n_views", "val views"),
    ]:
        av, bv, cv = a_over.get(key), b_over.get(key), c_over.get(key)
        lines.append(
            f"| {label} | {fmt(av)} | {fmt(bv)} | {fmt(cv)} | "
            f"{fmt(delta(bv, av))} | {fmt(delta(cv, bv))} | {fmt(delta(cv, av))} |"
        )

    lines += [
        "",
        "## Class groups (mean per-class AP@0.5)",
        "",
        "| group | A | B | C | B−A | C−B | C−A |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g in CLASS_GROUPS:
        av = out["class_groups"][g]["A"]
        bv = out["class_groups"][g]["B"]
        cv = out["class_groups"][g]["C"]
        lines.append(
            f"| {g} | {fmt(av)} | {fmt(bv)} | {fmt(cv)} | "
            f"{fmt(delta(bv, av))} | {fmt(delta(cv, bv))} | {fmt(delta(cv, av))} |"
        )

    lines += [
        "",
        "## Signal-level bandwidth groups (AP@0.5)",
        "",
        "| group | A | B | C | B−A | C−B | C−A |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g, label in SIGNAL_GROUPS:
        av = out["signal_groups"][g]["A"]["map50"]
        bv = out["signal_groups"][g]["B"]["map50"]
        cv = out["signal_groups"][g]["C"]["map50"]
        lines.append(
            f"| {label} | {fmt(av)} | {fmt(bv)} | {fmt(cv)} | "
            f"{fmt(delta(bv, av))} | {fmt(delta(cv, bv))} | {fmt(delta(cv, av))} |"
        )

    lat = out["latency"] or {}
    ratios = out["latency_ratios"] or {}
    lines += [
        "",
        "## Inference latency",
        "",
        "| mode | views/sample | build p50 s | model p50 s | end-to-end p50 s | end-to-end p95 s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, key in [("A coarse", "A_coarse"), ("C all", "C_all")]:
        d = lat.get(key) or {}
        v = (d.get("views_per_sample") or {}).get("mean")
        b = (d.get("build_s") or {}).get("p50")
        m = (d.get("model_s") or {}).get("p50")
        t = (d.get("total_s") or {}).get("p50")
        p95 = (d.get("total_s") or {}).get("p95")
        lines.append(f"| {mode} | {fmt(v, 2)} | {fmt(b)} | {fmt(m)} | {fmt(t)} | {fmt(p95)} |")
    if ratios:
        lines += [
            "",
            f"Views ratio C/A: {fmt(ratios.get('views_c_over_a'), 2)}; "
            f"end-to-end p50 ratio C/A: {fmt(ratios.get('total_p50_c_over_a'), 3)}; "
            f"model-only p50 ratio C/A: {fmt(ratios.get('model_p50_c_over_a'), 3)}.",
        ]

    lines += [
        "",
        "## Training budget and cost",
        "",
        "| item | C (B uses same weights) | A |",
        "| --- | ---: | ---: |",
    ]
    def c_budget(key):
        val = (c_cfg or {}).get(key)
        if val is not None:
            return val
        if not c_cfg:
            return None
        epochs = int(c_cfg.get("epochs", 50))
        bs = int(c_cfg.get("batch_sources", 8))
        micro = math.ceil(int(c_cfg.get("fit_samples", 5957)) / max(bs, 1))
        if key == "views_per_source":
            return 2
        if key == "total_optimizer_steps":
            return micro * epochs
        if key == "warmup_steps":
            return int(round(float(c_cfg.get("warmup_epochs", 3)) * micro))
        if key == "total_image_views":
            return int(c_cfg.get("fit_samples", 5957)) * 2 * epochs
        return None

    for label, key in [
        ("epochs", "epochs"), ("views/source/epoch", "views_per_source"),
        ("effective views", "effective_views"), ("optimizer steps", "total_optimizer_steps"),
        ("warmup steps", "warmup_steps"), ("image presentations", "total_image_views"),
    ]:
        cv = c_budget(key)
        av = (a_cfg or {}).get(key) if key != "views_per_source" else (a_cfg or {}).get(key, 1)
        lines.append(f"| {label} | {fmt(cv)} | {fmt(av)} |")
    c_h = (c_train_sum or {}).get("total_h")
    a_h = (a_train_sum or {}).get("total_h")
    lines.append(f"| training wall time, h | {fmt(c_h, 2)} | {fmt(a_h, 2)} |")
    lines.append(f"| peak reserved, GiB | {fmt(c_reserved, 1)} | {fmt(a_reserved, 1)} |")
    if cache_meta:
        lines.append(f"| cache build, s | n/a | {fmt(cache_meta.get('build_seconds'), 0)} |")
        lines.append(f"| cache size, GiB | n/a | {fmt((cache_meta.get('shape') or [0])[0] * 2048 * 512 * 2 / 1024 ** 3, 2)} |")
    if cache_verify:
        lines.append(f"| cache training-path check | n/a | {fmt(cache_verify.get('pass'))} |")

    lines += [
        "",
        "Notes:",
        "- B−A isolates the effect of training with fine views when inference is coarse-only.",
        "- C−B isolates inference-time fine slices with the same joint weights.",
        "- C−A is the full slicing scheme relative to the full-band single-image baseline.",
        "- Test split remains sealed.",
        "",
    ]
    md_path = Path(args.output_md)
    if not md_path.is_absolute():
        md_path = ROOT / md_path
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path} and {md_path}")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
