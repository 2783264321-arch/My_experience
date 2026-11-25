#!/usr/bin/env python3
"""Collect aligned local power crops + physical features for fixed candidates.

Input is the fixed-candidate cache from build_dfine_candidate_cache.py. B uses
`local_power`; C uses the same crop plus `features`. Boxes and candidate
membership never change.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from collect_local_proposal_crops import (
    load_iq_segment, local_geometry, local_stft_crop,
)
from src.signals.data import SPLIT_MANIFEST, load_manifest

TARGET = list(range(6)) + [10, 11]
FEATURE_NAMES = [
    "env_mean", "env_std", "env_cv", "env_p10", "env_p50", "env_p90", "env_p99",
    "env_m2", "env_m4", "env_m6", "env_papr", "env_peak_ratio",
    "p_mean", "p_std", "p_cv", "p_flatness", "p_p90_p50", "p_p99_p50",
    "freq_mean_std", "freq_mean_cv", "freq_flatness",
    "time_mean_std", "time_mean_cv",
    "cp_04us", "cp_08us", "cp_16us", "cp_32us",
    "strong_frac", "empty_frac", "snr_db",
]


def physical_features(seg: torch.Tensor, p: np.ndarray, fs: float) -> np.ndarray:
    x = seg.detach().cpu().numpy().astype(np.complex64)
    env = np.abs(x).astype(np.float32)
    eps = 1e-8
    em = float(env.mean()) + eps
    e = env / em
    m2 = float((e ** 2).mean())
    m4 = float((e ** 4).mean())
    m6 = float((e ** 6).mean())
    q = np.percentile(e, [10, 50, 90, 99])
    papr = float((e.max() ** 2) / max(m2, eps))
    peak_ratio = float(e.max() / max(q[1], eps))

    p = np.asarray(p, dtype=np.float64)
    p = p - p.min()
    pmean = float(p.mean()) + eps
    pstd = float(p.std())
    flat = float(np.exp(np.mean(np.log(p + eps))) / pmean)
    pp = np.percentile(p, [50, 90, 99])
    freq_mean = p.mean(axis=1)
    freq_std = p.std(axis=1)
    time_mean = p.mean(axis=0)

    def corr(lag_s):
        lag = int(round(lag_s * fs))
        if lag < 1 or lag >= len(x):
            return 0.0
        num = np.mean(x[lag:] * np.conj(x[:-lag]))
        den = np.mean(np.abs(x) ** 2) + eps
        return float(abs(num) / den)

    total = float((np.abs(x) ** 2).mean()) + eps
    noise = float(np.percentile(np.abs(x) ** 2, 10)) + eps
    snr_db = 10.0 * math.log10(max(total / noise - 1.0, eps))

    return np.asarray([
        float(env.mean()), float(env.std()), float(env.std() / (env.mean() + eps)),
        float(q[0]), float(q[1]), float(q[2]), float(q[3]),
        m2, m4, m6, papr, peak_ratio,
        pmean, pstd, pstd / pmean, flat,
        float(pp[1] / max(pp[0], eps)), float(pp[2] / max(pp[0], eps)),
        float(freq_mean.std()), float(freq_mean.std() / (freq_mean.mean() + eps)),
        float(np.exp(np.mean(np.log(freq_mean + eps))) / (freq_mean.mean() + eps)),
        float(time_mean.std()), float(time_mean.std() / (time_mean.mean() + eps)),
        corr(0.4e-6), corr(0.8e-6), corr(1.6e-6), corr(3.2e-6),
        float((p > 0.5 * p.max()).mean()), float((p < 0.05 * p.max()).mean()), snr_db,
    ], dtype=np.float32)


_WORK = {}


def _worker_init(split, win_ms, height, width, margin):
    torch.set_num_threads(1)
    records = load_manifest(SPLIT_MANIFEST, split=split)
    _WORK["by_id"] = {r.sample_id: r for r in records}
    _WORK.update(win_ms=win_ms, height=height, width=width, margin=margin)


def _worker_one(task):
    pos, sid, box = task[0], task[1], task[2]
    rec = _WORK["by_id"].get(sid)
    if rec is None:
        return pos, None, None
    geom = local_geometry(rec, box, _WORK["margin"])
    if geom is None:
        return pos, None, None
    i0, i1, f1, f2 = geom
    seg = load_iq_segment(rec, i0, i1)
    c, p = local_stft_crop(seg, rec, f1, f2, _WORK["win_ms"],
                           _WORK["height"], _WORK["width"])
    if c is None:
        return pos, None, None
    return pos, p.astype(np.float16), physical_features(seg, p, rec.B_mhz * 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--split", choices=("fit", "val"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--win-ms", type=float, default=0.1)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--iou-min", type=float, default=0.5)
    ap.add_argument("--score-min", type=float, default=0.001)
    ap.add_argument("--max-crops", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    with np.load(args.cache, allow_pickle=True) as zf:
        z = {k: zf[k] for k in zf.files}
    meta = json.loads(args.cache.with_suffix(".json").read_text())
    n_samples = int(meta["samples"])
    records = load_manifest(SPLIT_MANIFEST, split=args.split)[:n_samples]
    by_id = {r.sample_id: r for r in records}
    sid = z["sample_ids"]
    sample_index = z["sample_index"]
    labels = z["labels"].astype(np.int64)
    gt_label = z["gt_label"].astype(np.int64)
    gt_iou = z["gt_iou"].astype(np.float32)
    scores = z["scores"].astype(np.float32)
    target = np.isin(labels, TARGET)
    if args.split == "fit":
        select = target & (gt_label >= 0) & (gt_iou >= args.iou_min)
        train_label = gt_label
    else:
        select = target & (scores >= args.score_min)
        train_label = labels
    idx = np.flatnonzero(select)
    if args.max_crops:
        idx = idx[:args.max_crops]
    print(f"{args.split}: candidates={len(labels)} selected={len(idx)} "
          f"target={int(target.sum())}", flush=True)

    n = len(idx)
    power = np.empty((n, 1, args.height, args.width), dtype=np.float16)
    feats = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    valid = np.zeros((n,), dtype=bool)
    out = {k: np.empty((n,), dtype=t) for k, t in (
        ("label", np.int64), ("pred_label", np.int64), ("gt_label", np.int64),
        ("gt_iou", np.float32), ("score", np.float32), ("sample_index", np.int32),
        ("candidate_index", np.int64),
        ("view_type", np.int8), ("view_start", np.int32), ("qid", np.int16),
        ("bandwidth_mhz", np.float32), ("duration_ms", np.float32))}
    boxes = np.empty((n, 4), dtype=np.float32)
    class_scores = np.empty((n, 14), dtype=np.float32)
    sample_ids = np.empty((n,), dtype=object)

    t0 = time.perf_counter()
    done = skipped = 0

    def make_task(j, i):
        rec = by_id[str(sid[sample_index[i]])]
        b = z["boxes"][i]
        return (j, str(sid[sample_index[i]]), b,
                float((b[3] - b[1]) * rec.B_mhz),
                float((b[2] - b[0]) * rec.T_ms))

    tasks = (make_task(j, i) for j, i in enumerate(idx, 1))
    if args.workers > 1:
        it = mp.Pool(args.workers, initializer=_worker_init,
                     initargs=(args.split, args.win_ms, args.height,
                               args.width, args.margin)).imap_unordered(
                                   _worker_one, tasks, chunksize=8)
    else:
        _worker_init(args.split, args.win_ms, args.height, args.width, args.margin)
        it = map(_worker_one, tasks)
    for pos, p, f in it:
        i = idx[pos - 1]
        rec = by_id[str(sid[sample_index[i]])]
        if p is None:
            skipped += 1
        else:
            power[pos - 1, 0] = p
            feats[pos - 1] = f
            out["label"][pos - 1] = int(train_label[i])
            out["pred_label"][pos - 1] = int(labels[i])
            out["gt_label"][pos - 1] = int(gt_label[i])
            out["gt_iou"][pos - 1] = float(gt_iou[i])
            out["score"][pos - 1] = float(scores[i])
            out["sample_index"][pos - 1] = int(sample_index[i])
            out["candidate_index"][pos - 1] = int(i)
            out["view_type"][pos - 1] = int(z["view_type"][i])
            out["view_start"][pos - 1] = int(z["view_start"][i])
            out["qid"][pos - 1] = int(z["qid"][i])
            boxes[pos - 1] = z["boxes"][i]
            class_scores[pos - 1] = z["class_scores"][i]
            sample_ids[pos - 1] = str(sid[sample_index[i]])
            b = z["boxes"][i]
            out["bandwidth_mhz"][pos - 1] = float((b[3] - b[1]) * rec.B_mhz)
            out["duration_ms"][pos - 1] = float((b[2] - b[0]) * rec.T_ms)
            valid[pos - 1] = True
            done += 1
        if pos % args.progress_every == 0 or pos == n:
            print(f"{pos}/{n} kept={done} skipped={skipped} "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    keep = np.flatnonzero(valid)
    payload = {k: v[keep] for k, v in out.items()}
    payload.update({
        "local_power": power[keep], "features": feats[keep],
        "boxes": boxes[keep], "class_scores": class_scores[keep],
        "sample_ids": sample_ids[keep],
        "feature_names": np.asarray(FEATURE_NAMES),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **payload)
    meta_out = {
        "format": "dfine-local-crops-v1", "split": args.split,
        "source_cache": str(args.cache), "candidates": int(len(idx)),
        "kept": int(len(keep)), "skipped": int(skipped),
        "win_ms": args.win_ms, "height": args.height, "width": args.width,
        "margin": args.margin, "iou_min": args.iou_min,
        "score_min": args.score_min, "feature_names": FEATURE_NAMES,
        "class_counts": np.bincount(payload["label"], minlength=14).tolist(),
        "elapsed_s": time.perf_counter() - t0,
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta_out, indent=2) + "\n",
                                             encoding="utf-8")
    print(json.dumps({k: meta_out[k] for k in
                      ("split", "candidates", "kept", "skipped", "class_counts",
                       "elapsed_s")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
