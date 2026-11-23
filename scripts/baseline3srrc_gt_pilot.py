#!/usr/bin/env python3
"""Build a 10 MHz SRRC cache and train waveform classifiers on GT boxes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from scripts.baseline3srrc_phase0_verify import (
    EDGE_MS,
    FS_OUT,
    KEEP_BW,
    N_WIN,
    WIN,
    crop_and_resample,
    overlap_frac,
    windows_from_crop,
)
from scripts.baseline3srrc_pipeline_smoke import TinyResNet
from src.signals.data import SPLIT_MANIFEST, load_manifest

INPUT_CHANNELS = {"magnitude": 1, "iq": 2, "amplitude_phase_change": 3}


def gt_split(manifest, seed, max_groups):
    with manifest.open(newline="", encoding="utf-8") as f:
        groups_by_file = {
            row["sample_id"]: row["group_id"] or row["sample_id"]
            for row in csv.DictReader(f)
            if row["split_new"] == "fit"
        }
    boxes = [
        (rec, index, sig)
        for rec in sorted(load_manifest(manifest, split="fit"), key=lambda r: r.sample_id)
        for index, sig in enumerate(rec.signals)
        if int(sig["class"]) in (10, 11)
        and 9.5 <= float(sig["end_frequency"]) - float(sig["start_frequency"]) <= 10.5
    ]
    if max_groups is not None:
        pool = sorted({groups_by_file[rec.sample_id] for rec, _, _ in boxes})
        selected = set(np.random.default_rng(seed).permutation(pool)[:max_groups])
        boxes = [box for box in boxes if groups_by_file[box[0].sample_id] in selected]
    groups = np.array([groups_by_file[rec.sample_id] for rec, _, _ in boxes])
    labels = np.array([int(sig["class"]) - 10 for _, _, sig in boxes])
    if len(set(groups)) < 5 or set(labels) != {0, 1}:
        raise ValueError("Need at least five source groups and both SRRC classes")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train, dev = next(splitter.split(np.zeros(len(boxes)), labels, groups))
    if any(set(labels[idx]) != {0, 1} for idx in (train, dev)):
        raise ValueError("Both classes must be present in train and dev. Increase --max-groups.")
    is_dev = np.zeros(len(boxes), dtype=bool)
    is_dev[dev] = True
    return boxes, groups, labels, is_dev


def instance_windows(rec, sig, n_windows, *, allow_short=False):
    crop = crop_and_resample(rec, sig)
    windows = windows_from_crop(crop, sig, n_win=n_windows)["windows"]
    if allow_short and len(windows) == 1:
        # Repeating the padded short segment keeps its weight under mean pooling.
        windows = windows * n_windows
    if len(windows) != n_windows:
        raise ValueError(f"Unexpected short GT interval: {rec.sample_id}")
    arrays = np.zeros((n_windows, 2, WIN), dtype=np.float32)
    for out, w in zip(arrays, windows):
        if not w["finite"] or not np.isfinite(w["rms"]):
            raise ValueError(f"Non-finite GT window: {rec.sample_id}")
        samples = crop["x"][w["start"]:w["stop"]]
        out[0, : len(samples)] = samples.real
        out[1, : len(samples)] = samples.imag
    return arrays


def prepare(args):
    started = time.perf_counter()
    boxes, groups, labels, is_dev = gt_split(args.manifest, args.seed, args.max_groups)
    overlaps = [overlap_frac(rec, sig) for rec, _, sig in boxes]
    strata = np.array(["strong" if o["strong"] else "weak" if o["any"] else "clean" for o in overlaps])
    sample_ids = np.array([r.sample_id for r, _, _ in boxes])
    signal_indices = np.array([i for _, i, _ in boxes])
    metadata = {
        "protocol": "fit-only SRRC 10 MHz, grouped dev fold 0 of 5",
        "seed": args.seed,
        "fs_hz": FS_OUT,
        "band_limit_hz": KEEP_BW,
        "edge_ms_per_side": EDGE_MS,
        "window_points": WIN,
        "n_windows": args.n_windows,
        "max_groups": args.max_groups,
        "n_train": int((~is_dev).sum()),
        "n_dev": int(is_dev.sum()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.metadata_only:
        records = [
            {"sample_id": sid, "signal_idx": int(idx), "group_id": group,
             "class": int(label) + 10, "split": "dev" if dev else "train", "overlap": stratum}
            for sid, idx, group, label, dev, stratum in
            zip(sample_ids, signal_indices, groups, labels, is_dev, strata)
        ]
        args.out.write_text(json.dumps({"metadata": metadata, "records": records}, indent=2) + "\n",
                            encoding="utf-8")
    else:
        windows = np.empty((len(boxes), args.n_windows, 2, WIN), dtype=np.float32)
        for row, (rec, _, sig) in enumerate(boxes):
            windows[row] = instance_windows(rec, sig, args.n_windows)
        # Publish the cache only after all records and metadata are written.
        with tempfile.NamedTemporaryFile(dir=args.out.parent, prefix=args.out.name + ".",
                                         delete=False) as f:
            temp_path = Path(f.name)
        try:
            with temp_path.open("wb") as f:
                np.savez(f, windows=windows, labels=labels, is_dev=is_dev, groups=groups, strata=strata,
                         sample_ids=sample_ids, signal_indices=signal_indices,
                         metadata=json.dumps(metadata))
            temp_path.replace(args.out)
        finally:
            temp_path.unlink(missing_ok=True)
    print(json.dumps({**metadata, "seconds": round(time.perf_counter() - started, 2)}, indent=2))


def predict(model, windows, arm):
    rms = windows.square().sum(dim=2).mean(dim=-1).sqrt().clamp_min(1e-8)
    windows = windows / rms[..., None, None]
    if arm == "magnitude":
        windows = torch.linalg.vector_norm(windows, dim=2, keepdim=True)
    elif arm == "amplitude_phase_change":
        magnitude = torch.linalg.vector_norm(windows, dim=2, keepdim=True)
        i, q = windows.unbind(dim=2)
        real = i[..., 1:] * i[..., :-1] + q[..., 1:] * q[..., :-1]
        imag = q[..., 1:] * i[..., :-1] - i[..., 1:] * q[..., :-1]
        phase_change = nn.functional.pad(torch.stack((real, imag), dim=2), (1, 0))
        windows = torch.cat((magnitude, phase_change), dim=2)
    logits = model(windows.flatten(0, 1))
    return logits.unflatten(0, windows.shape[:2]).mean(dim=1)


def metrics(labels, preds, strata):
    result = {}
    for key in ("all", "clean", "weak", "strong"):
        use = np.ones(len(labels), dtype=bool) if key == "all" else strata == key
        if not use.any():
            continue
        cm = np.bincount(2 * labels[use] + preds[use], minlength=4).reshape(2, 2)
        totals = cm.sum(axis=1)
        recall = [float(cm[i, i] / totals[i]) if totals[i] else None for i in range(2)]
        result[key] = {
            "n": int(use.sum()),
            "accuracy": float(cm.trace() / cm.sum()),
            "balanced_accuracy": float(np.mean(recall)) if all(totals) else None,
            "recall_10_11": recall,
            "confusion_10_11": cm.tolist(),
        }
    return result


def batches(indices, batch_size):
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def read_pilot_cache(path):
    with np.load(path, allow_pickle=False) as f:
        cache = dict(f)
    metadata = json.loads(str(cache["metadata"]))
    if metadata["edge_ms_per_side"] != EDGE_MS or metadata["fs_hz"] != FS_OUT:
        raise ValueError("Cache settings differ from the current front end. Rebuild it.")
    return cache, metadata


def write_dev_logits(path, cache, dev_ids, logits):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "signal_idx", "group_id", "class", "overlap", "logit_10", "logit_11"])
        for i, pair in zip(dev_ids, logits):
            writer.writerow([cache["sample_ids"][i], cache["signal_indices"][i], cache["groups"][i],
                             int(cache["labels"][i]) + 10, cache["strata"][i], *map(float, pair)])


def run_arm(args, cache, windows, labels, arm):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = TinyResNet(in_channels=INPUT_CHANNELS[arm]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    train_ids = np.flatnonzero(~cache["is_dev"])
    dev_ids = np.flatnonzero(cache["is_dev"])
    history = []
    started = time.perf_counter()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for ids in batches(rng.permutation(train_ids), args.batch):
            optimizer.zero_grad(set_to_none=True)
            logits = predict(model, windows[ids].to(args.device), arm)
            loss = loss_fn(logits, labels[ids].to(args.device))
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise ValueError(f"Non-finite {arm} loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            total_loss += loss_value * len(ids)
        model.eval()
        with torch.inference_mode():
            dev_logits = torch.cat([
                predict(model, windows[ids].to(args.device), arm).cpu()
                for ids in batches(dev_ids, args.batch)
            ]).numpy()
        scores = metrics(cache["labels"][dev_ids], dev_logits.argmax(axis=1), cache["strata"][dev_ids])
        history.append({"epoch": epoch + 1, "loss": total_loss / len(train_ids), "dev": scores})
        print(f"{arm} epoch {epoch + 1}: loss={history[-1]['loss']:.4f}, "
              f"dev balanced accuracy={scores['all']['balanced_accuracy']:.4f}", flush=True)
    peak = torch.cuda.max_memory_reserved() if args.device == "cuda" else None
    write_dev_logits(args.out_dir / f"{arm}_dev.csv", cache, dev_ids, dev_logits)
    torch.save(model.state_dict(), args.out_dir / f"{arm}_last.pt")
    return {"seconds": round(time.perf_counter() - started, 2), "peak_cuda_reserved_bytes": peak,
            "history": history}


def train(args):
    if args.device == "cpu":
        torch.set_num_threads(1)
    elif not torch.cuda.is_available():
        raise ValueError("CUDA was explicitly requested but is unavailable")
    cache, metadata = read_pilot_cache(args.cache)
    windows = torch.from_numpy(cache.pop("windows"))
    labels = torch.from_numpy(cache["labels"]).long()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arms = ("magnitude", "iq") if args.arm == "both" else (args.arm,)
    results = {arm: run_arm(args, cache, windows, labels, arm) for arm in arms}
    summary = {"cache": metadata, "device": args.device, "seed": args.seed, "batch_instances": args.batch,
               "lr": args.lr, "results": results}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=1729)
    prep.add_argument("--max-groups", type=positive_int)
    prep.add_argument("--n-windows", type=int, choices=(8, 16), default=N_WIN)
    prep.add_argument("--metadata-only", action="store_true")
    prep.set_defaults(run=prepare)

    pilot = commands.add_parser("train")
    pilot.add_argument("--cache", type=Path, required=True)
    pilot.add_argument("--out-dir", type=Path, required=True)
    pilot.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    pilot.add_argument("--arm", choices=(*INPUT_CHANNELS, "both"), default="both")
    pilot.add_argument("--seed", type=int, default=1729)
    pilot.add_argument("--batch", type=positive_int, default=8, help="instances per batch")
    pilot.add_argument("--epochs", type=positive_int, default=5)
    pilot.add_argument("--lr", type=float, default=1e-3)
    pilot.set_defaults(run=train)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
