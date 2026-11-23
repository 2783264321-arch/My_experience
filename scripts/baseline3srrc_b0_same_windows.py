#!/usr/bin/env python3
"""Fit the unsynchronized B0 baseline on the same windows and dev split as the CNNs."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.baseline3srrc_gt_pilot import metrics, read_pilot_cache, write_dev_logits
from scripts.baseline3srrc_phase0_verify import SIG_BW


def _histogram(v, bins, lo, hi):
    counts, _ = np.histogram(v, bins=np.linspace(lo, hi, bins + 1))
    return counts.astype(np.float64) / len(v)


def window_features(x, fs):
    """B0_unsync features from 2c5e815:scripts/srrc_frontend.py."""
    magnitude = np.abs(x)
    power = magnitude ** 2
    p = float(power.mean())
    rms = math.sqrt(max(p, 1e-30))
    spec = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(len(x), d=1.0 / fs))
    edges = np.linspace(-SIG_BW / 2, SIG_BW / 2, 17)
    bands = np.zeros(16)
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        selected = spec[(freqs >= lo) & (freqs < hi)]
        if selected.size:
            bands[index] = selected.mean()
    bands /= max(bands.sum(), 1e-30)
    segments = np.array([part.mean() for part in np.array_split(power, 8)], dtype=float)
    segments /= max(segments.sum(), 1e-30)

    m4 = float((magnitude ** 4).mean() / max(p * p, 1e-30))
    papr = float(power.max() / max(p, 1e-30))
    positive = spec[spec > 0]
    entropy = flatness = 0.0
    if positive.size:
        positive /= max(positive.sum(), 1e-30)
        logp = np.log(positive)
        entropy = float(-np.sum(positive * logp) / math.log(max(len(positive), 2)))
        flatness = float(np.exp(logp.mean()) / max(positive.mean(), 1e-30))
    amp_hist = _histogram(magnitude / rms, 12, 0.0, 3.0)
    phase = np.angle(x[1:] * np.conj(x[:-1]))
    phase_hist = _histogram(phase, 12, -np.pi, np.pi)
    return np.concatenate(([math.log10(max(p, 1e-30))], bands, segments,
                           [m4, papr, entropy, flatness], amp_hist, phase_hist))


def instance_features(windows, fs):
    rows = [window_features(w[0] + 1j * w[1], fs) for w in windows]
    return np.mean(rows, axis=0)


def run(cache_path, out_dir):
    started = time.perf_counter()
    cache, metadata = read_pilot_cache(cache_path)
    windows = cache.pop("windows")
    if not np.isfinite(windows).all():
        raise ValueError("Expected finite IQ windows")
    features = np.stack([instance_features(w, metadata["fs_hz"]) for w in windows])
    labels = cache["labels"]
    dev_ids = np.flatnonzero(cache["is_dev"])
    train_mask = ~cache["is_dev"]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, solver="lbfgs"),
    )
    with threadpool_limits(limits=1):
        model.fit(features[train_mask], labels[train_mask])
        binary_logits = model.decision_function(features[dev_ids])
    logits = np.column_stack((np.zeros_like(binary_logits), binary_logits))
    scores = metrics(labels[dev_ids], logits.argmax(axis=1), cache["strata"][dev_ids])
    summary = {
        "arm": "b0_same_windows",
        "feature_source": "2c5e815:scripts/srrc_frontend.py",
        "aggregation": "mean of per-window features",
        "n_features": int(features.shape[1]),
        "cache": metadata,
        "seconds": round(time.perf_counter() - started, 2),
        "dev": scores,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dev_logits(out_dir / "b0_same_windows_dev.csv", cache, dev_ids, logits)
    text = json.dumps(summary, indent=2) + "\n"
    (out_dir / "summary.json").write_text(text, encoding="utf-8")
    print(text, end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.out_dir)


if __name__ == "__main__":
    main()
