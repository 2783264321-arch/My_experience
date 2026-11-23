#!/usr/bin/env python3
"""Score a frozen YOLO26l validation set after SRRC label-only correction."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.baseline3srrc_gt_pilot import INPUT_CHANNELS, TinyResNet, instance_windows, predict
from scripts.baseline3srrc_phase0_verify import FS_OUT, N_WIN
from src.signals.data import N_TIME, SPLIT_MANIFEST, h_for_n, load_manifest
from src.signals.infer import gt_global
from src.signals.metrics import ImagePredictions, ImageTargets, _match_single, evaluate


DEFAULT_PRED_DIR = Path(
    "/root/autodl-tmp/worktrees/formal-yolo26/runs/yolo26l_c1_2048x512/val_full_best"
)
SRRC_CLASSES = (10, 11)
IOU_THRESHOLDS = tuple(i / 100 for i in range(50, 96, 5))


def load_frozen_val(manifest, pred_dir):
    sample_ids = json.loads((pred_dir / "val_sample_ids.json").read_text(encoding="utf-8"))
    records = {r.sample_id: r for r in load_manifest(manifest, split="val")}
    if len(sample_ids) != len(records) or set(sample_ids) != records.keys():
        raise ValueError("Prediction sample IDs must cover the full validation manifest once.")
    with np.load(pred_dir / "val_predictions_filtered_0_5.npz", allow_pickle=False) as archive:
        boxes = archive["pred_boxes"]
        scores = archive["pred_scores"]
        labels = archive["pred_labels"]
        sample_index = archive["pred_sample_index"]
    if (
        boxes.shape != (len(scores), 4) or labels.shape != scores.shape
        or sample_index.shape != scores.shape
        or np.any((sample_index < 0) | (sample_index >= len(sample_ids)))
    ):
        raise ValueError("Invalid frozen prediction archive.")
    predictions = []
    for i in range(len(sample_ids)):
        mask = sample_index == i
        predictions.append(ImagePredictions(boxes[mask], scores[mask], labels[mask]))
    return [records[sid] for sid in sample_ids], predictions


def local_box(box, record):
    """Invert the detector's global canvas convention to milliseconds and MHz."""
    x1, y1, x2, y2 = np.clip(box.astype(np.float64), 0, 1)
    hop_ms = h_for_n(record.N) / (record.B_mhz * 1e3)
    t1, t2 = np.clip((np.array([x1, x2]) * N_TIME - 0.5) * hop_ms, 0, record.T_ms)
    return {
        "start_time": float(t1),
        "end_time": float(t2),
        "start_frequency": record.f_lo + float(y1) * record.B_mhz,
        "end_frequency": record.f_lo + float(y2) * record.B_mhz,
    }


def read_candidate_windows(records, predictions, min_score):
    keys, windows = [], []
    n_routed = 0
    for image, (record, pred) in enumerate(zip(records, predictions)):
        widths = (pred.boxes[:, 3] - pred.boxes[:, 1]) * record.B_mhz
        routed = (
            np.isin(pred.labels, SRRC_CLASSES) & (pred.scores >= min_score)
            & (widths >= 9.5) & (widths <= 10.5)
        )
        for index in np.flatnonzero(routed):
            n_routed += 1
            box = local_box(pred.boxes[index], record)
            if box["end_time"] - box["start_time"] < 1000 / FS_OUT:
                continue
            windows.append(instance_windows(record, box, N_WIN, allow_short=True))
            keys.append((image, int(index)))
    return n_routed, keys, windows


@torch.inference_mode()
def classify(windows, arm, checkpoint, device):
    model = TinyResNet(in_channels=INPUT_CHANNELS[arm]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    result = []
    for start in range(0, len(windows), 32):
        batch = torch.from_numpy(np.stack(windows[start:start + 32])).to(device)
        logits = predict(model, batch, arm)
        confidence, label = logits.softmax(dim=1).max(dim=1)
        result.extend(zip((label + 10).cpu().tolist(), confidence.cpu().tolist()))
    return result


def report_ap(targets, predictions):
    stats = evaluate(targets, predictions, iou_thresholds=IOU_THRESHOLDS, max_dets=100)
    by_iou = stats["per_iou"]
    return {
        "map50": by_iou["0.5"]["map_class_aware"],
        "map50_95": float(np.mean([entry["map_class_aware"] for entry in by_iou.values()])),
        "ap50": {name: entry["ap"] for name, entry in by_iou["0.5"]["per_class"].items()},
    }


def score_rule(targets, predictions, keys, decisions, known_labels, threshold, baseline):
    updated = [ImagePredictions(p.boxes, p.scores, p.labels.copy()) for p in predictions]
    corrected, broken, unmatched, other = 0, 0, 0, 0
    for (image, index), (label, confidence) in zip(keys, decisions):
        previous = predictions[image].labels[index]
        if label == previous or confidence < threshold:
            continue
        updated[image].labels[index] = label
        truth = known_labels[image].get(index)
        if truth is None:
            unmatched += 1
        elif truth not in SRRC_CLASSES:
            other += 1
        else:
            corrected += int(label == truth)
            broken += int(previous == truth)
    scores = report_ap(targets, updated)
    return {
        "expert_confidence_floor": threshold,
        "n_changed": corrected + broken + unmatched + other,
        "matched_corrected": corrected,
        "matched_broken": broken,
        "unmatched_changed": unmatched,
        "other_matched_class_changed": other,
        "delta_map50_pp": (scores["map50"] - baseline["map50"]) * 100,
        **scores,
    }


def write_candidates(path, records, predictions, keys, decisions, known_labels):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("sample_id", "pred_idx", "yolo_class", "yolo_score",
                         "expert_class", "expert_confidence", "matched_gt_class"))
        for (image, index), (label, confidence) in zip(keys, decisions):
            pred = predictions[image]
            writer.writerow((records[image].sample_id, index, int(pred.labels[index]),
                             float(pred.scores[index]), label, confidence,
                             known_labels[image].get(index, "")))


def run(args):
    if args.device == "cpu":
        torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    records, predictions = load_frozen_val(args.manifest, args.pred_dir)
    targets = []
    for record in records:
        gt = gt_global(record, h_for_n(record.N))
        targets.append(ImageTargets(gt.boxes.numpy().reshape(-1, 4), gt.labels.numpy()))
    baseline = report_ap(targets, predictions)
    print(json.dumps({"val_samples": len(records), "baseline": baseline}), flush=True)

    start = time.perf_counter()
    n_routed, keys, windows = read_candidate_windows(records, predictions, args.min_yolo_score)
    result = {
        "predictions": str(args.pred_dir),
        "val_samples": len(records),
        "min_yolo_score": args.min_yolo_score,
        "pred_width_mhz": [9.5, 10.5],
        "routed_candidates": n_routed,
        "scored_candidates": len(keys),
        "crop_seconds": time.perf_counter() - start,
        "baseline": baseline,
        "experts": {},
    }
    # Matches are for reporting only. Routing and relabeling use prediction data.
    known_labels = []
    for gt, pred in zip(targets, predictions):
        matches, _ = _match_single(gt, pred, 0.5, class_aware=False)
        known_labels.append({index: int(gt.labels[match]) for index, match in matches})

    for arm, checkpoint in args.expert:
        decisions = classify(windows, arm, checkpoint, args.device)
        write_candidates(args.out / f"{arm}_candidates.csv", records, predictions,
                         keys, decisions, known_labels)
        result["experts"][arm] = {
            "checkpoint": str(checkpoint),
            "routed_argmax": score_rule(targets, predictions, keys, decisions,
                                       known_labels, 0.5, baseline),
            "confidence_gated": score_rule(targets, predictions, keys, decisions,
                                           known_labels, args.min_expert_confidence, baseline),
        }
        print(json.dumps({arm: result["experts"][arm]}), flush=True)
        (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n",
                                               encoding="utf-8")
    return result


def expert_argument(value):
    arm, _, path = value.partition("=")
    if arm not in INPUT_CHANNELS or not path:
        raise argparse.ArgumentTypeError("Use magnitude=PATH, iq=PATH or amplitude_phase_change=PATH.")
    return arm, Path(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expert", type=expert_argument, action="append", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min-yolo-score", type=float, default=0.01)
    parser.add_argument("--min-expert-confidence", type=float, default=0.70)
    args = parser.parse_args()
    if len({arm for arm, _ in args.expert}) != len(args.expert):
        parser.error("Specify one checkpoint per representation.")
    run(args)


if __name__ == "__main__":
    main()
