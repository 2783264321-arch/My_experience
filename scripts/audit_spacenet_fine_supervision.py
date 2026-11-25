"""Audit the existing fine-tile training sampler using only train export labels."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


TILE = 2048
STEP = 1639
MARGIN = 16


def train_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "train":
                yield row


def starts_for(rows):
    starts = list(range(0, rows - TILE + 1, STEP))
    if starts[-1] != rows - TILE:
        starts.append(rows - TILE)
    return starts


def audit(analysis_dir, seed):
    signals_by_sample = defaultdict(list)
    for signal in train_rows(analysis_dir / "instances.csv"):
        signals_by_sample[signal["sample_id"]].append(signal)

    totals = Counter()
    by_class = defaultdict(Counter)
    by_bandwidth = defaultdict(Counter)
    near_duplicates = 0
    max_coarse, max_fine = 0, 0

    for sample in train_rows(analysis_dir / "samples.csv"):
        sample_id = sample["sample_id"]
        signals = signals_by_sample[sample_id]
        bandwidth = float(sample["bandwidth_mhz"])
        rows = round(bandwidth * 700)
        starts = starts_for(rows)
        tiled_counts = [0] * len(starts)
        epochs_by_tile = [0] * len(starts)
        order = list(range(len(starts)))
        random.Random(f"{seed}:{sample_id}").shuffle(order)
        for epoch in range(25):
            epochs_by_tile[order[epoch % len(order)]] += 1

        totals["samples"] += 1
        totals["instances"] += len(signals)
        totals["full_fine_views"] += len(starts)
        stat = by_bandwidth[bandwidth]
        stat["samples"] += 1
        stat["tiles_per_sample"] = len(starts)
        stat["distinct_tiles_through_epoch25"] = min(len(starts), 25)
        max_coarse = max(max_coarse, len(signals))

        exact_boxes = Counter()
        for signal in signals:
            cls = int(signal["class"])
            left = (float(signal["f1_mhz"]) - float(sample["f_lo_mhz"])) / bandwidth * rows
            right = (float(signal["f2_mhz"]) - float(sample["f_lo_mhz"])) / bandwidth * rows
            exact_boxes[(cls, left, right, signal["t1_ms"], signal["t2_ms"])] += 1
            class_stat = by_class[cls]
            class_stat["instances"] += 1
            full_views, full_training_views = 0, 0
            for tile_index, start in enumerate(starts):
                end = start + TILE
                if min(right, end) <= max(left, start):
                    continue
                repeats = epochs_by_tile[tile_index]
                tiled_counts[tile_index] += 1
                class_stat["all_intersecting_targets"] += 1
                class_stat["epoch25_intersecting_targets"] += repeats
                if left < start or right > end:
                    class_stat["all_truncated_targets"] += 1
                    class_stat["epoch25_truncated_targets"] += repeats
                    continue
                if (start > 0 and left <= start + MARGIN) or (
                    end < rows and right >= end - MARGIN
                ):
                    continue
                full_views += 1
                full_training_views += repeats
            class_stat["instances_with_full_fine_view_margin16"] += int(full_views > 0)
            class_stat["instances_with_full_fine_training_view_margin16_epoch25"] += int(full_training_views > 0)

        near_duplicates += sum(n * (n - 1) // 2 for n in exact_boxes.values())
        max_fine = max(max_fine, max(tiled_counts))

    totals["max_coarse_instances"] = max_coarse
    totals["max_fine_intersecting_targets"] = max_fine
    totals["same_class_exact_box_pairs"] = near_duplicates
    totals["epoch25_fine_views"] = totals["samples"] * 25
    for name in ("all_intersecting_targets", "all_truncated_targets",
                 "epoch25_intersecting_targets", "epoch25_truncated_targets"):
        totals[name] = sum(stat[name] for stat in by_class.values())
        totals[f"wifi_{name}"] = sum(by_class[cls][name] for cls in range(6))

    return {
        "scope": "original train split only, full original 7500, not the subset used for fit",
        "training_epoch_count_emulated": 25,
        "seed": seed,
        "tile_frequency_bins": TILE,
        "tile_step": STEP,
        "inference_boundary_margin": MARGIN,
        "totals": totals,
        "by_class": dict(sorted(by_class.items())),
        "by_observation_bandwidth_mhz": dict(sorted(by_bandwidth.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-outputs", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = json.dumps(audit(args.analysis_outputs, args.seed), indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(result + "\n", encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
