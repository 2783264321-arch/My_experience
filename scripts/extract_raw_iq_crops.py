#!/usr/bin/env python3
"""Extract canonical local raw-IQ crops for the representation diagnostic.

Uses the same selected local-crop rows as the local power/physical-feature
experiment. Each crop is band-limited to its predicted box, resampled to
2x its bandwidth, and centered on a fixed 0.1 ms physical window. No image
resizing; variable lengths are padded and masked at training time.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from collect_local_proposal_crops import local_geometry
from src.signals.data import SPLIT_MANIFEST, load_iq, load_manifest

TARGET = list(range(6)) + [10, 11]


def canonical(x: np.ndarray, fs: float, f1_mhz: float, f2_mhz: float,
              fc_mhz: float, win_ms: float = 0.1, oversample: float = 2.0):
    # Band-limit to the predicted box.
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(X), d=1.0 / fs)
    mask = (freqs >= (f1_mhz - fc_mhz) * 1e6) & (freqs <= (f2_mhz - fc_mhz) * 1e6)
    if not mask.any():
        return None
    xb = np.fft.ifft(X * mask)
    bw = max((f2_mhz - f1_mhz) * 1e6, 1e3)
    fs_t = max(oversample * bw, 1e3)
    L = max(64, int(round(fs_t * win_ms / 1000.0)))
    # Resample the complex baseband by linear interpolation of I/Q.
    src = np.arange(len(xb), dtype=np.float64)
    dst = np.linspace(0, len(xb) - 1, L)
    re = np.interp(dst, src, xb.real)
    im = np.interp(dst, src, xb.imag)
    z = re + 1j * im
    # Center on the requested physical window; pad if shorter.
    if len(z) > L:
        st = (len(z) - L) // 2
        z = z[st:st + L]
    elif len(z) < L:
        z = np.pad(z, (0, L - len(z)))
    rms = np.sqrt(np.mean(np.abs(z) ** 2)) + 1e-9
    z = z / rms
    return np.stack([z.real, z.imag]).astype(np.float16)


_WORK = {}


def _init(split):
    _WORK["records"] = load_manifest(SPLIT_MANIFEST, split=split)


def worker(args):
    sidx, rows = args
    rec = _WORK["records"][sidx]
    iq = load_iq(rec.bin_path).numpy()
    fs = rec.B_mhz * 1e6
    fc = (rec.f_lo + rec.f_hi) / 2.0
    out = []
    for row in rows:
        i, box = row[0], row[1]
        geom = local_geometry(rec, box, 0.1)
        if geom is None:
            continue
        i0, i1, f1, f2 = geom
        seg = iq[max(0, i0):min(len(iq), i1)]
        if len(seg) < 32:
            continue
        # Only a short physical segment is needed for modulation; avoid FFT
        # over a 150 ms / 12M-point burst.
        cap = int(0.005 * fs)
        if len(seg) > cap:
            mid = len(seg) // 2
            seg = seg[mid - cap // 2:mid + cap // 2]
        z = canonical(seg, fs, f1, f2, fc)
        if z is None:
            continue
        out.append((i, z, z.shape[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("fit", "val"), required=True)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-per-class", type=int, default=3000)
    ap.add_argument("--max-neg-per-class", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    with np.load(args.crops, allow_pickle=True) as zf:
        z = {k: zf[k] for k in zf.files}
    keep = z["keep"].astype(np.int64)
    labels = z["label"].astype(np.int64)
    pred = z["pred_label"].astype(np.int64)
    rng = np.random.default_rng(1729)
    # Deduplicate by query identity; keep positives/negatives separately.
    key = np.stack([z["sample_index"], z["view_type"], z["view_start"],
                    z["qid"], keep], axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    idx_all = np.sort(first)
    idx = []
    for c in TARGET:
        pos = idx_all[(keep[idx_all] == 0) & (labels[idx_all] == c)]
        neg = idx_all[(keep[idx_all] == 1) & (pred[idx_all] == c)]
        if len(pos) > args.max_per_class:
            pos = rng.choice(pos, args.max_per_class, replace=False)
        if len(neg) > args.max_neg_per_class:
            neg = rng.choice(neg, args.max_neg_per_class, replace=False)
        idx.extend(pos.tolist()); idx.extend(neg.tolist())
    idx = np.sort(np.asarray(idx, dtype=np.int64))
    print(f"{args.split}: selected {len(idx)} unique queries", flush=True)

    by_sample = defaultdict(list)
    for i in idx:
        by_sample[int(z["sample_index"][i])].append(
            (int(i), z["boxes"][i].astype(np.float32)))
    tasks = [(s, rows) for s, rows in by_sample.items()]
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.split,)) as ex:
        for out in ex.map(worker, tasks):
            results.extend(out)
    L = max(x[2] for x in results)
    iq = np.zeros((len(results), 2, L), dtype=np.float16)
    lengths = np.zeros((len(results),), dtype=np.int32)
    rows = []
    for j, (i, arr, ln) in enumerate(results):
        iq[j, :, :ln] = arr
        lengths[j] = ln
        rows.append(i)
    rows = np.asarray(rows, dtype=np.int64)
    payload = {
        "iq": iq, "lengths": lengths,
        "source_index": rows,
        "label": labels[rows], "pred_label": pred[rows], "keep": keep[rows],
        "sample_ids": z["sample_ids"][rows],
        "sample_index": z["sample_index"][rows],
        "view_type": z["view_type"][rows], "view_start": z["view_start"][rows],
        "qid": z["qid"][rows], "boxes": z["boxes"][rows],
        "class_scores": z["class_scores"][rows],
        "score": z["score"][rows], "bandwidth_mhz": z["bandwidth_mhz"][rows],
        "duration_ms": z["duration_ms"][rows],
        "features": z["features"][rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **payload)
    meta = {"format": "dfine-raw-iq-crops-v1", "split": args.split,
            "source": str(args.crops), "queries": len(rows), "max_len": int(L),
            "win_ms": 0.1, "oversample": 2.0, "elapsed_s": time.perf_counter() - t0,
            "class_counts": np.bincount(labels[rows], minlength=14).tolist(),
            "keep_counts": np.bincount(keep[rows], minlength=2).tolist()}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
