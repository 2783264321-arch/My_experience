#!/usr/bin/env python3
"""Evaluate the frozen-detector proposal classifiers on the filtered val set.

Base: stage5_val_filtered_shared (fine 0-5 removed) with mAP50 0.649725.
For every final 0-5 box, crop the coarse view and reclassify with the
0/1/2 or 3/4/5 classifier. Only labels change; boxes and scores stay fixed.
Re-run class-wise NMS + 100 cap. Report group AP, total mAP, high-score
misclass, localization recall and extra time.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.signals.data import SPLIT_MANIFEST, load_manifest
from src.signals.infer import (
    Prediction, global_classwise_nms, gt_global,
)
from src.signals.metrics import ImagePredictions, ImageTargets, evaluate
from scripts.collect_proposal_crops import CoarseOnly, crop_coarse
from scripts.train_proposal_classifiers import SmallCNN
from scripts.view_source_attribution import (
    CLASS_GROUPS, GROUPS, evaluate_group,
)

ROOT = Path(__file__).resolve().parents[1]


def strip_examples(d):
    d = dict(d)
    d.pop("fp_examples", None)
    d.pop("fn_examples", None)
    return d


def load_cls(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = SmallCNN(len(ck["classes"])).cuda()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="yolo11s_c1_directreg_2048x512")
    ap.add_argument("--val-dir", default="stage5_val_filtered_shared")
    ap.add_argument("--cls-dir", default="runs/yolo11s_c1_directreg_2048x512/proposal_classifiers")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run = ROOT / "runs" / args.run
    vdir = run / args.val_dir
    z = np.load(vdir / "val_predictions.npz")
    ids = json.loads((vdir / "val_sample_ids.json").read_text())
    records = load_manifest(SPLIT_MANIFEST, split="val")
    by_id = {r.sample_id: r for r in records}
    recs = [by_id[s] for s in ids]
    if args.max_samples:
        recs = recs[: args.max_samples]
        ids = ids[: args.max_samples]
    sample_ids = [r.sample_id for r in recs]

    # before predictions from the filtered full val
    pred_idx = z["pred_sample_index"]
    gt_idx = z["gt_sample_index"]
    before = []
    gts = []
    for i in range(len(recs)):
        pm = pred_idx == i
        gm = gt_idx == i
        before.append(ImagePredictions(
            z["pred_boxes"][pm], z["pred_scores"][pm], z["pred_labels"][pm],
        ))
        gts.append(ImageTargets(z["gt_boxes"][gm], z["gt_labels"][gm]))
    ev_before = evaluate(gts, before, iou_thresholds=(0.5,), max_dets=100)
    pi_b = ev_before["per_iou"]["0.5"]
    print(f"filtered before: mAP={pi_b['map_class_aware']:.6f} "
          f"agn={pi_b['ap_class_agnostic']:.6f} "
          f"n_pred={sum(len(p.boxes) for p in before)}", flush=True)

    cls012, ck012 = load_cls(Path(args.cls_dir) / "proposal_cls_012.pt")
    cls345, ck345 = load_cls(Path(args.cls_dir) / "proposal_cls_345.pt")
    norm_cfg = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm_cfg["lo"]), float(norm_cfg["hi"]))
    ds = CoarseOnly(recs, norm)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=args.workers,
        collate_fn=lambda b: b[0], pin_memory=False,
        prefetch_factor=4 if args.workers else None,
        persistent_workers=args.workers > 0,
    )
    after = []
    changed = Counter()
    n_05 = 0
    n_changed = 0
    t_coarse = 0.0
    t_extra = 0.0
    started = time.perf_counter()
    for ni, item in enumerate(loader):
        t0 = time.perf_counter()
        idx = item["index"]
        rec = recs[idx]
        tensor = item["tensor"]
        t_coarse += time.perf_counter() - t0
        pb = before[idx].boxes
        ps = before[idx].scores
        pl = before[idx].labels.copy()
        old = pl.copy()
        # gather crops by family
        for fam, model, ck in [
            ("012", cls012, ck012),
            ("345", cls345, ck345),
        ]:
            classes = ck["classes"]
            sel = np.where(np.isin(pl, classes))[0]
            if len(sel) == 0:
                continue
            t1 = time.perf_counter()
            pairs = []
            for j in sel:
                c = crop_coarse(
                    tensor, pb[j], margin=ck.get("margin", 0.1),
                    size=tuple(ck.get("input_hw", [128, 256])),
                )
                if c is not None:
                    pairs.append((int(j), c))
            if not pairs:
                continue
            sel = [p[0] for p in pairs]
            crops = [p[1] for p in pairs]
            x = torch.stack(crops, 0).unsqueeze(1).float().cuda()
            with torch.no_grad():
                logits = model(x)
                pred_local = logits.argmax(1).cpu().numpy()
            for j, li in zip(sel, pred_local.tolist()):
                new_c = int(classes[li])
                changed[(int(pl[j]), new_c)] += 1
                if new_c != int(pl[j]):
                    n_changed += 1
                pl[j] = new_c
            t_extra += time.perf_counter() - t1
        n_05 += int(np.isin(old, [0, 1, 2, 3, 4, 5]).sum())
        t2 = time.perf_counter()
        merged = global_classwise_nms(
            Prediction(torch.from_numpy(pb), torch.from_numpy(ps), torch.from_numpy(pl)),
            iou=0.5, max_det=100,
        )
        t_extra += time.perf_counter() - t2
        after.append(ImagePredictions(
            merged.boxes.numpy(), merged.scores.numpy(), merged.labels.numpy(),
        ))
        if (ni + 1) % 100 == 0 or ni + 1 == len(recs):
            print(f"{ni+1}/{len(recs)} samples, changed {n_changed}/{n_05}, "
                  f"{time.perf_counter()-started:.0f}s", flush=True)

    ev_after = evaluate(gts, after, iou_thresholds=(0.5,), max_dets=100)
    pi_a = ev_after["per_iou"]["0.5"]
    print(f"after: mAP={pi_a['map_class_aware']:.6f} "
          f"agn={pi_a['ap_class_agnostic']:.6f} "
          f"n_pred={sum(len(p.boxes) for p in after)}", flush=True)

    out = {
        "protocol": {
            "run": args.run, "val_dir": str(vdir), "cls_dir": str(args.cls_dir),
            "filter": "fine 0-5 removed; shared validation",
            "classifier": "frozen detector; coarse-crop 012/345; labels changed only; boxes/scores fixed",
            "n_samples": len(recs), "n_05_boxes": int(n_05),
            "n_changed": int(n_changed),
            "changed_pairs": {f"{a}->{b}": int(v) for (a, b), v in sorted(changed.items()) if a != b},
            "extra_crop_cls_nms_s": t_extra,
            "coarse_view_s": t_coarse,
            "total_s": time.perf_counter() - started,
        },
        "before": {},
        "after": {},
    }
    for name, ev, preds in [("before", ev_before, before), ("after", ev_after, after)]:
        pi = ev["per_iou"]["0.5"]
        entry = {
            "mAP50": pi["map_class_aware"],
            "ap50_agnostic": pi["ap_class_agnostic"],
            "localization_recall": ev["classification"]["localization_recall"],
            "classification_accuracy": ev["classification"]["classification_accuracy"],
            "n_pred": sum(len(p.boxes) for p in preds),
            "per_class": pi["per_class"],
            "groups": {},
        }
        for g in GROUPS:
            gm = evaluate_group(
                name, gts, preds, CLASS_GROUPS[g], recs, sample_ids,
            )
            aps = [pi["per_class"][str(c)]["ap"] for c in CLASS_GROUPS[g]
                   if str(c) in pi["per_class"] and pi["per_class"][str(c)]["ap"] is not None]
            gm["group_class_ap_mean"] = float(np.mean(aps)) if aps else None
            gm["class_ap"] = {str(c): (pi["per_class"].get(str(c), {}) or {}).get("ap")
                               for c in CLASS_GROUPS[g]}
            entry["groups"][g] = strip_examples(gm)
        allg = evaluate_group(name, gts, preds, list(range(14)), recs, sample_ids)
        entry["all"] = strip_examples(allg)
        out[name] = entry

    out_path = Path(args.out) if args.out else (vdir.parent / "proposal_classifier_experiment.json")
    out_path.write_text(json.dumps(out, indent=2))
    print("wrote", out_path)

    def brief(e):
        return {
            "mAP50": e["mAP50"], "ap50_agnostic": e["ap50_agnostic"],
            "r_agn": e["localization_recall"], "n_pred": e["n_pred"],
            "tp/fp/fn": [e["all"]["tp"], e["all"]["fp"], e["all"]["fn"]],
            "fp_type": [e["all"]["fp_misclass"], e["all"]["fp_duplicate"],
                        e["all"]["fp_background"]],
            "fp_high": [e["all"]["fp_high_misclass"], e["all"]["fp_high_duplicate"],
                        e["all"]["fp_high_background"]],
            "groups": {g: {
                "ap": e["groups"][g]["group_class_ap_mean"],
                "r_agn": e["groups"][g]["r_agn"],
                "tp/fp/fn": [e["groups"][g]["tp"], e["groups"][g]["fp"], e["groups"][g]["fn"]],
                "fp_high": [e["groups"][g]["fp_high_misclass"],
                            e["groups"][g]["fp_high_duplicate"],
                            e["groups"][g]["fp_high_background"]],
            } for g in GROUPS},
        }
    print(json.dumps({
        "protocol": out["protocol"],
        "before": brief(out["before"]),
        "after": brief(out["after"]),
    }, indent=2))


if __name__ == "__main__":
    main()
