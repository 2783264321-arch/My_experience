#!/usr/bin/env python3
"""After training: full-val best.pt and write formal result artifacts."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/root/autodl-tmp/workspace/baseline/.venv/bin/python"
OUT = ROOT / "reports/detector_upgrade"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="yolo26l_c1_2048x512")
    args = ap.parse_args()
    run_dir = ROOT / "runs" / args.run
    summary_p = run_dir / "summary.json"
    if not summary_p.exists():
        msg = {"run": args.run, "status": "failed_or_incomplete",
               "note": "no summary.json; check train log"}
        (OUT / "formal_yolo26_results.json").write_text(json.dumps(msg, indent=2))
        print(json.dumps(msg))
        return
    training = json.loads(summary_p.read_text())
    ck = run_dir / "best.pt"
    if not ck.exists():
        ck = run_dir / "last.pt"
    val_dir = run_dir / ("val_full_best" if ck.name == "best.pt" else "val_full_last")
    cmd = [PYTHON, str(ROOT / "scripts/validate_yolo26.py"), "--run", args.run,
           "--checkpoint", ck.name, "--output", str(val_dir)]
    print("running final full val:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    val = json.loads((val_dir / "summary.json").read_text())
    (OUT / "formal_yolo26_val_full.json").write_text(json.dumps(val, indent=2))
    ref = 0.649725
    f = val["settings"]["filtered_0_5"]
    u = val["settings"]["unfiltered"]
    res = {"run": args.run, "status": "completed",
           "reference_filtered_map50": ref,
           "filtered_map50": f["map50"],
           "filtered_minus_reference_pp": (f["map50"] - ref) * 100,
           "filtered_map50_95": f["map50_95"],
           "unfiltered_map50": u["map50"],
           "unfiltered_map50_95": u["map50_95"],
           "filtered_ap50_agnostic": f["ap50_agnostic"],
           "filtered_localization_recall": f["localization_recall"],
           "groups": f["groups"],
           "best_epoch": training.get("best_epoch"),
           "best_filtered_map50_training": training.get("best_filtered_map50"),
           "total_h": training.get("total_h"),
           "val_history": training.get("val_history"),
           "config": training.get("config"),
           "val_summary_path": str(val_dir / "summary.json")}
    (OUT / "formal_yolo26_results.json").write_text(json.dumps(res, indent=2))
    md = [f"# YOLO26l 正式训练最终结果", "",
          f"- run: `{args.run}`；best epoch **{res['best_epoch']}**；总耗时 **{res['total_h']:.2f} h**",
          "", "| 指标 | 值 | 参照 0.649725 | 差值 |", "| --- | ---: | ---: | ---: |",
          f"| 过滤 mAP@0.5（主） | **{f['map50']:.6f}** | 0.649725 | {res['filtered_minus_reference_pp']:+.2f} pp |",
          f"| 无过滤 mAP@0.5 | {u['map50']:.6f} | — | — |",
          f"| 过滤 mAP@0.5:0.95 | {f['map50_95']:.6f} | — | — |",
          f"| 无过滤 mAP@0.5:0.95 | {u['map50_95']:.6f} | — | — |",
          f"| 不计类 AP@0.5 | {f['ap50_agnostic']:.6f} | 0.9060 | {f['ap50_agnostic']-0.9060:+.4f} |",
          f"| 定位召回 | {f['localization_recall']:.6f} | 0.9578 | {f['localization_recall']-0.9578:+.4f} |",
          "", "## 分组（过滤）", "", "| 组 | mAP@0.5 | mAP@0.5:0.95 | recall |", "| --- | ---: | ---: | ---: |"]
    for g, v in f["groups"].items():
        md.append(f"| {g} | {v['map50']:.4f} | {v['map50_95']:.4f} | {v['recall']:.4f} |")
    (OUT / "formal_yolo26_results.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"filtered_map50": f["map50"], "delta_pp": res["filtered_minus_reference_pp"],
                      "unfiltered_map50": u["map50"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
