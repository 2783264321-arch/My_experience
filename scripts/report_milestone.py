#!/usr/bin/env python3
"""Print the 10-epoch milestone summary from val_history.json."""
from __future__ import annotations
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/yolo26l_c1_2048x512/val_history.json"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epoch", type=int, required=True)
    a = ap.parse_args()
    if not RUN.exists():
        print("no val_history yet"); return
    h = json.loads(RUN.read_text())
    e = next((x for x in h if x["epoch"] == a.epoch), None)
    if e is None:
        print(f"milestone epoch {a.epoch} not found yet"); return
    prev = [x for x in h if x["epoch"] < a.epoch]
    f = e["settings"]["filtered_0_5"]; u = e["settings"]["unfiltered"]
    print(f"MILESTONE epoch {a.epoch} (round {a.epoch+1}/50)  val {e['wall_val_s']:.0f}s")
    print(f"filtered mAP50 {f['map50']:.4f}  AP50:95 {f['map50_95']:.4f}  "
          f"AP_agn {f['ap50_agnostic']:.4f}  recall {f['localization_recall']:.4f}  "
          f"cls {f['classification_accuracy']:.4f}")
    print(f"unfiltered mAP50 {u['map50']:.4f}  AP50:95 {u['map50_95']:.4f}")
    if prev:
        p = prev[-1]["settings"]["filtered_0_5"]
        print(f"delta vs round {prev[-1]['epoch']+1}: {(f['map50']-p['map50'])*100:+.2f} pp")
    print("groups(filtered):", {g: round(v["map50"], 3) for g, v in f["groups"].items()})
    pc = {int(c): v["ap"] for c, v in f["per_class"].items()}
    print("strongest:", sorted(pc.items(), key=lambda x: -x[1])[:5])
    print("weakest:", sorted(pc.items(), key=lambda x: x[1])[:5])


if __name__ == "__main__":
    main()
