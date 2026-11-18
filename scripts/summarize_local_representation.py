#!/usr/bin/env python3
"""Render the local-representation comparison JSON as compact markdown tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default="reports/local_rep/representation_summary.md")
    args = ap.parse_args()

    lines = ["# Local representation comparison: generated tables", ""]
    for path in args.results:
        d = json.loads(Path(path).read_text())
        lines += [f"## {d.get('tag', Path(path).stem)}", ""]
        s = d["split"]
        lines += [
            f"Shared sample-level holdout: {s['holdout_crops']} crops / "
            f"{s['holdout_samples']} samples; train {s['train_crops']} crops / "
            f"{s['train_samples']} samples; class counts train "
            f"{s['class_counts_train']}, holdout {s['class_counts_holdout']}.",
            "",
            "| rep | group | acc mean | acc std | macro-F1 | confusion (sum) |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
        for rep, groups in d["reps"].items():
            for g, r in groups.items():
                lines.append(
                    f"| {rep} | {g} | {fmt(r['accuracy_mean'])} | "
                    f"{fmt(r['accuracy_std'])} | {fmt(r['macro_f1_mean'])} | "
                    f"`{r['confusion_sum']}` |"
                )
        lines.append("")
        for rep, groups in d["reps"].items():
            for g, r in groups.items():
                conf = r["confusion_sum"]
                tot = max(1, sum(sum(row) for row in conf))
                names = [str(c) for c in r["classes"]]
                lines += [f"### {rep} / {g} confusion (rows GT, cols pred)", ""]
                lines.append("| GT \\ pred | " + " | ".join(names) + " | recall |")
                lines.append("| --- | " + " | ".join("---:" for _ in names) + " | ---: |")
                for i, row in enumerate(conf):
                    rec = row[i] / max(1, sum(row))
                    lines.append(
                        f"| {r['classes'][i]} | " + " | ".join(str(x) for x in row) +
                        f" | {rec:.3f} |"
                    )
                lines.append(f"Total: {tot}")
                lines.append("")
        lines += [
            "### Overlap-condition accuracy (mean over seeds)",
            "",
            "| rep | group | clean WiFi | WiFi overlap | cross-class WiFi | strong WiFi >=25% |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for rep, groups in d["reps"].items():
            for g, r in groups.items():
                ov = r["overlap_subsets"]
                def cell(name):
                    x = ov[name]
                    return f"{fmt(x['accuracy_mean'])} (n={x['n']})"
                lines.append(
                    f"| {rep} | {g} | {cell('wifi_clean')} | "
                    f"{cell('wifi_overlap')} | {cell('wifi_cross_overlap')} | "
                    f"{cell('wifi_strong_overlap')} |"
                )
        lines.append("")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
