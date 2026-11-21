#!/usr/bin/env python3
"""Wi-Fi sync-failure diagnostic: whole-window vs 2 ms short segments.

Frozen YOLO/expert/rules. Uses adaptation-train + calibration-dev only
(final-dev and test untouched). For matched Wi-Fi pairs whose Pred box
failed the current whole-window CP check, scan 2 ms segments inside the
same central 20 ms window with the identical thresholds; run the frozen
W-Shared 3-seed expert on the best passing segment; score against GT
only after selection. Also scans GT boxes whose whole-window GT check
failed, and a few unmatched class-4/5 candidates (no truth).

No training, no rule changes, no inference changes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.signals.data import SPLIT_MANIFEST, load_manifest
from scripts.baseline3ca_build_hist_cache import sync_box
from scripts.baseline3b_train_experts import HistCNN

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports/baseline3_wifi_cascade"
OUT = ROOT / "reports/baseline3_wifi_sync_diag"
CACHE = ROOT / "runs_local/baseline3c"
SEEDS = (1729, 1730, 1731)
DEV_SPLITS = {"adaptation-train", "calibration-dev"}
SEG_MS = 2.0
HOP_MS = 1.0
MAX_SEG = 20
GATE_METRIC = 0.34
GATE_RATIO = 15.0
SHARED_MAP = {3: 0, 4: 1, 5: 2}


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def seg_windows(t1, t2):
    """2 ms windows inside the same central 20 ms used by the whole check."""
    dur = t2 - t1
    win = min(dur, 20.0)
    wc = 0.5 * (t1 + t2)
    ws = wc - 0.5 * win
    we = wc + 0.5 * win
    if win <= SEG_MS:
        return [(ws, we)]
    starts = np.arange(ws, max(ws, we - SEG_MS) + 1e-9, HOP_MS)
    if len(starts) > MAX_SEG:
        starts = np.linspace(ws, we - SEG_MS, MAX_SEG)
    return [(float(s), float(s + SEG_MS)) for s in starts]


def seg_worker(args):
    try:
        r = sync_box(args["job"])
        return {"case": args["case"], "seg": args["seg"], **r,
                "t1": args["t1"], "t2": args["t2"]}
    except Exception as e:
        return {"case": args["case"], "seg": args["seg"],
                "status": "error", "error": str(e)[:200],
                "t1": args["t1"], "t2": args["t2"]}


def load_experts(device):
    models = []
    for s in SEEDS:
        m = HistCNN(3).to(device)
        m.load_state_dict(torch.load(
            ROOT / f"runs_local/baseline3b/weights/SHARED_seed{s}.pt",
            map_location=device))
        m.eval()
        models.append(m)
    return models


def expert_predict(models, blocks, device):
    if blocks is None or len(blocks) == 0:
        return None
    x = torch.from_numpy(np.asarray(blocks, np.float32)).to(device)
    probs = []
    with torch.no_grad():
        for m in models:
            probs.append(F.softmax(m(x), dim=1).mean(dim=0).cpu().numpy())
    return int(np.argmax(np.mean(probs, axis=0)))


def rect_intersection(a, b):
    iw = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    return iw * ih


def other_signal_risk(sigs, target_gt_idx, f_center, bw, t1, t2):
    """Max overlap of the segment rectangle with other annotated signals."""
    a = (f_center - bw / 2, f_center + bw / 2, t1, t2)
    area = max((a[1] - a[0]) * (a[3] - a[2]), 1e-9)
    best = 0.0; best_lab = None
    for j, s in enumerate(sigs):
        if j == target_gt_idx:
            continue
        b = (float(s["start_frequency"]), float(s["end_frequency"]),
             float(s["start_time"]), float(s["end_time"]))
        frac = rect_intersection(a, b) / area
        if frac > best:
            best, best_lab = float(frac), int(s["class"])
    return best, best_lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--unmatched", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()

    split = json.loads((EXP / "adaptation_split_v2.json").read_text())
    assign = split["assignment"]
    pairs = json.loads((EXP / "adaptation_pairs.json").read_text())
    paired = json.loads((EXP / "paired_gt_pred_sync.json").read_text())["pairs"]
    recs = {r.sample_id: r for r in load_manifest(SPLIT_MANIFEST, split="val")}
    sigs_by = {r.sample_id: r.signals for r in recs.values()}

    # ---- matched cases ----
    cases = []
    for i, (p, q) in enumerate(zip(pairs, paired)):
        sid = p["sample_id"]
        if sid not in assign or assign[sid] not in DEV_SPLITS:
            continue
        gt = int(p["gt_label"]); pred = int(p["pred_label"])
        if gt not in (3, 4, 5) or pred not in (3, 4, 5):
            continue
        if q["pred_sync"] == "ok":
            continue
        if gt == 3 and pred != 3:
            continue  # class-3 control uses same-class prediction
        # Pred-box short check
        cases.append({
            "kind": "matched", "box_type": "pred", "pair_index": i,
            "sample_id": sid, "split": assign[sid],
            "gt_label": gt, "yolo_label": pred,
            "f_center": float(p["f_center"]), "t1": float(p["t1_ms"]),
            "t2": float(p["t2_ms"]), "bw": int(p["nominal_bw"]),
            "gt_idx": int(p["gt_idx"]),
            "whole_status": q["pred_sync"],
            "whole_metric": q["pred_metric"], "whole_ratio": q["pred_ratio"],
            "gt_sync": q["gt_sync"],
        })
        # GT-box short check, only when GT whole check also failed
        if q["gt_sync"] != "ok":
            cases.append({
                "kind": "matched", "box_type": "gt", "pair_index": i,
                "sample_id": sid, "split": assign[sid],
                "gt_label": gt, "yolo_label": pred,
                "f_center": float(p["gt_f_center"]),
                "t1": float(p["gt_t1"]), "t2": float(p["gt_t2"]),
                "bw": int(p["nominal_bw"]), "gt_idx": int(p["gt_idx"]),
                "whole_status": q["gt_sync"],
                "whole_metric": q["gt_metric"], "whole_ratio": q["gt_ratio"],
                "gt_sync": q["gt_sync"],
            })

    # ---- unmatched failures (dev only), 12 deterministic ----
    z = np.load(CACHE / "yolo_val_cache.npz", allow_pickle=True)
    z = {k: z[k] for k in z.files}
    hz = np.load(CACHE / "hist_cache.npz", allow_pickle=True)
    hz = {k: hz[k] for k in hz.files}
    h_pos = {int(pi): k for k, pi in enumerate(hz["pred_idx"])}
    matched_pred = {int(p["pred_idx"]) for p in pairs}
    off = np.concatenate([[0], np.cumsum(z["c0_counts"])])
    unmatched = []
    for pi, pos in h_pos.items():
        if int(z["c0_labels"][pi]) not in (4, 5):
            continue
        if hz["pred_status"][pos] == "ok" or pi in matched_pred:
            continue
        si = int(np.searchsorted(off, pi, side="right") - 1)
        sid = str(z["sample_ids"][si])
        if sid not in assign or assign[sid] not in DEV_SPLITS:
            continue
        x1, y1, x2, y2 = z["c0_boxes"][pi]
        m = json.loads(str(z["meta"][si]))
        h = (int(m["N"]) + 510) // 511
        fs = float(m["B_mhz"]) * 1e6
        t1 = (float(x1) * 512 - 0.5) * h / fs * 1000.0
        t2 = (float(x2) * 512 - 0.5) * h / fs * 1000.0
        f1 = float(m["f_lo"]) + float(y1) * float(m["B_mhz"])
        f2 = float(m["f_lo"]) + float(y2) * float(m["B_mhz"])
        unmatched.append({
            "kind": "unmatched", "box_type": "pred", "pred_idx": pi,
            "sample_id": sid, "split": assign[sid],
            "gt_label": None, "yolo_label": int(z["c0_labels"][pi]),
            "f_center": 0.5 * (f1 + f2),
            "t1": max(0.0, min(float(m["T_ms"]), t1)),
            "t2": max(0.0, min(float(m["T_ms"]), t2)),
            "bw": 40, "gt_idx": None, "whole_status": "sync_fail",
            "whole_metric": float(hz["pred_metric"][pos]),
            "whole_ratio": float(hz["pred_ratio"][pos]), "gt_sync": None,
        })
    unmatched.sort(key=lambda x: sha256(f"20261121:{x['sample_id']}:{x['pred_idx']}"))
    cases.extend(unmatched[:args.unmatched])
    print(f"matched cases: {sum(c['kind']=='matched' for c in cases)} "
          f"(pred {sum(c['box_type']=='pred' and c['kind']=='matched' for c in cases)}, "
          f"gt {sum(c['box_type']=='gt' for c in cases)}); "
          f"unmatched {sum(c['kind']=='unmatched' for c in cases)}")

    # ---- segment jobs ----
    jobs = []
    case_id = 0
    for c in cases:
        rec = recs[c["sample_id"]]
        windows = seg_windows(c["t1"], c["t2"])
        for k, (a, b) in enumerate(windows):
            jobs.append(({
                "case": case_id, "seg": k,
                "job": (str(rec.bin_path), rec.f_lo, rec.B_mhz, rec.N,
                        c["f_center"], a, b, c["bw"]),
                "t1": a, "t2": b,
            }))
        c["case_id"] = case_id
        c["n_segments"] = len(windows)
        c["segment_jobs"] = list(range(len(jobs) - len(windows), len(jobs)))
        case_id += 1
    # convert tuple jobs to dict expected by worker
    worker_args = []
    for j in jobs:
        worker_args.append({"case": j["case"], "seg": j["seg"],
                            "job": j["job"], "t1": j["t1"], "t2": j["t2"]})
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, r in enumerate(ex.map(seg_worker, worker_args, chunksize=16)):
            results.append(r)
            if (k + 1) % 500 == 0:
                print(f"  segments {k+1}/{len(worker_args)} "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)
    by_case = defaultdict(list)
    for r in results:
        by_case[r["case"]].append(r)

    models = load_experts(device)
    out_rows = []
    for c in cases:
        rs = by_case[c["case_id"]]
        passed = [r for r in rs if r.get("status") == "ok"]
        total_sym = sum(int(r.get("n_kept", 0)) for r in passed)
        total_blocks = sum(len(r.get("blocks", [])) for r in passed)
        best = max(passed, key=lambda r: (float(r.get("metric", 0)),
                                          int(r.get("n_kept", 0)))) if passed else None
        expert = None
        if best is not None:
            expert3 = expert_predict(models, best.get("blocks"), device)
            if expert3 is not None:
                family_offset = 3 if c["bw"] == 40 else 0
                # all selected cases are W40 classes 3/4/5
                expert = expert3 + 3
        other_frac = other_lab = None
        emo = False
        if best is not None:
            other_frac, other_lab = other_signal_risk(
                sigs_by[c["sample_id"]], c["gt_idx"], c["f_center"],
                c["bw"], best["t1"], best["t2"])
            emo = bool(expert is not None and other_lab is not None and
                       other_lab in (3, 4, 5) and expert == other_lab)
        yolo = c["yolo_label"]
        gt = c["gt_label"]
        row = {**{k: v for k, v in c.items()
                  if k not in ("segment_jobs",)},
               "n_pass_segments": len(passed),
               "n_kept_total": total_sym,
               "n_blocks_total": total_blocks,
               "best_metric": None if best is None else best.get("metric"),
               "best_ratio": None if best is None else best.get("ratio"),
               "best_t1": None if best is None else best["t1"],
               "best_t2": None if best is None else best["t2"],
               "expert_label": expert,
               "expert_correct": None if gt is None or expert is None else int(expert == gt),
               "yolo_correct": None if gt is None else int(yolo == gt),
               "corrected": None if gt is None or expert is None else int(expert == gt and yolo != gt),
               "broken": None if gt is None or expert is None else int(expert != gt and yolo == gt),
               "other_frac": other_frac, "other_label": other_lab,
               "high_other_overlap": None if other_frac is None else int(other_frac >= 0.5),
               "expert_matches_other": int(emo),
               "changed": None if expert is None else int(expert != yolo)}
        out_rows.append(row)

    # CSV
    fields = ["kind", "box_type", "split", "sample_id", "pair_index",
              "pred_idx", "gt_label", "yolo_label", "whole_status",
              "whole_metric", "whole_ratio", "gt_sync", "n_segments",
              "n_pass_segments", "n_kept_total", "n_blocks_total",
              "best_metric", "best_ratio", "best_t1", "best_t2",
              "expert_label", "expert_correct", "yolo_correct",
              "corrected", "broken", "changed", "other_frac", "other_label",
              "high_other_overlap", "expert_matches_other"]
    with (OUT / "sync_segment_cases.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k) for k in fields})

    # summaries
    def summarize(rows):
        n = len(rows)
        rec = sum(1 for r in rows if r["n_pass_segments"] > 0)
        ent = [r for r in rows if r["expert_label"] is not None]
        return {
            "n": n, "n_recovered": rec,
            "recovery_rate": rec / max(n, 1),
            "n_with_expert": len(ent),
            "n_kept_total": int(sum(r["n_kept_total"] for r in rows)),
            "n_blocks_total": int(sum(r["n_blocks_total"] for r in rows)),
            "n_pass_segments": int(sum(r["n_pass_segments"] for r in rows)),
            "n_high_other_overlap": int(sum(1 for r in rows if r["high_other_overlap"])),
            "n_expert_matches_other": int(sum(1 for r in rows if r["expert_matches_other"])),
            "expert_acc": (float(np.mean([r["expert_correct"] for r in ent]))
                           if ent else None),
            "yolo_acc": (float(np.mean([r["yolo_correct"] for r in ent]))
                         if ent else None),
            "corrected": int(sum(r["corrected"] or 0 for r in ent)),
            "broken": int(sum(r["broken"] or 0 for r in ent)),
            "net": int(sum((r["corrected"] or 0) - (r["broken"] or 0) for r in ent)),
            "both_correct": int(sum(1 for r in ent if r["expert_correct"] and r["yolo_correct"])),
            "both_wrong": int(sum(1 for r in ent if not r["expert_correct"] and not r["yolo_correct"])),
            "confusion": np.array([
                [sum(1 for r in ent if r["gt_label"] == a and r["expert_label"] == b)
                 for b in (3, 4, 5)] for a in (3, 4, 5)]).tolist(),
        }

    summary = {"n_cases": len(out_rows),
               "matched": {}, "unmatched": {}}
    matched = [r for r in out_rows if r["kind"] == "matched"]
    unmatched_rows = [r for r in out_rows if r["kind"] == "unmatched"]
    for box in ("pred", "gt"):
        for cls in (3, 4, 5):
            sub = [r for r in matched if r["box_type"] == box and
                   r["gt_label"] == cls]
            if sub:
                summary["matched"][f"{box}_class{cls}"] = summarize(sub)
        sub = [r for r in matched if r["box_type"] == box]
        if sub:
            summary["matched"][f"{box}_all"] = summarize(sub)
    for split_name in ("adaptation-train", "calibration-dev"):
        sub = [r for r in matched if r["split"] == split_name]
        if sub:
            summary["matched"][f"pred_{split_name}"] = summarize(
                [r for r in sub if r["box_type"] == "pred"])
    summary["unmatched"] = {
        "n": len(unmatched_rows),
        "n_recovered": sum(1 for r in unmatched_rows if r["n_pass_segments"] > 0),
        "n_changed": sum(1 for r in unmatched_rows if r["changed"]),
        "n_high_other_overlap": sum(1 for r in unmatched_rows
                                    if r["high_other_overlap"]),
        "n_expert_matches_other": sum(1 for r in unmatched_rows
                                      if r["expert_matches_other"]),
        "expert_labels": dict(Counter(r["expert_label"] for r in unmatched_rows
                                      if r["expert_label"] is not None)),
        "n_kept_total": int(sum(r["n_kept_total"] for r in unmatched_rows)),
    }
    (OUT / "sync_segment_results.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2)[:12000])
    print("wrote", OUT / "sync_segment_cases.csv")


if __name__ == "__main__":
    main()
