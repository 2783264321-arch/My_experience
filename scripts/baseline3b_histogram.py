#!/usr/bin/env python3
"""Phase B' stage 1: build the P1-faithful 50x50 amp-phase histograms.

Locked spec (reports/baseline3_wifi_expert/histogram_spec.md):
  H[amp_bin, phase_bin], 50x50, per 128-consecutive-symbol block:
    A  = |Y| over the block's symbols x data subcarriers (calibration pool)
    thr= P99(A); amax = max(A[A<=thr])
    pairs: within each 8-symbol group, consecutive (s, s+1):
           (|Y^s_k|, wrap(angle(Y^{s+1,k}) - angle(Y^{s,k})))
    discard pairs with |Y^s_k| > thr; amp = |Y^s_k| / amax in [0,1]
    v = (wrap(Dphi) + pi) / (2pi) in [0,1)   # Dphi=0 -> col 24/25
    bins: ceil(x*50)-1 clipped to [0,49]; counts accumulate
  training input: H / H.max() (per-instance max norm, P1 4.2)

Instance level: symbols from the cache (keep_idx) are split into maximal
consecutive runs; complete 128-symbol blocks only; up to MAX_BLOCKS
earliest blocks per instance; instance prediction = mean softmax.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports/baseline3_wifi_expert"
CACHE = EXP / "cache/symbols"

N_BINS = 50
GROUP = 8
N_GROUPS = 16
BLOCK = GROUP * N_GROUPS          # 128
MAX_BLOCKS = 4
AMP_PCT = 99.0


def consecutive_runs(keep_idx):
    """Maximal runs of originally-consecutive symbol rows."""
    runs = []
    start = 0
    for i in range(1, len(keep_idx)):
        if keep_idx[i] != keep_idx[i - 1] + 1:
            runs.append((start, i))
            start = i
    if len(keep_idx):
        runs.append((start, len(keep_idx)))
    return runs


def block_histogram(Xblk, data_bins):
    """One 128-symbol block -> count histogram [50,50] float64."""
    amps_pool = np.abs(Xblk[:, data_bins]).ravel()
    if amps_pool.size == 0 or not np.any(amps_pool > 0):
        return None
    thr = np.percentile(amps_pool, AMP_PCT)
    amax = float(amps_pool[amps_pool <= thr].max()) if np.any(amps_pool <= thr) else 0.0
    if amax <= 0:
        return None
    amps, vs = [], []
    for g in range(N_GROUPS):
        s0 = g * GROUP
        for s in range(s0, s0 + GROUP - 1):
            a = np.abs(Xblk[s, data_bins])
            d = np.angle(Xblk[s + 1, data_bins]) - np.angle(Xblk[s, data_bins])
            d = np.mod(d + np.pi, 2.0 * np.pi) - np.pi
            amps.append(a)
            vs.append((d + np.pi) / (2.0 * np.pi))
    amps = np.concatenate(amps)
    vs = np.concatenate(vs)
    keep = amps <= thr
    amp = amps[keep] / amax
    v = vs[keep]
    r = np.clip(np.ceil(amp * N_BINS) - 1, 0, N_BINS - 1).astype(np.int64)
    c = np.clip(np.ceil(v * N_BINS) - 1, 0, N_BINS - 1).astype(np.int64)
    H = np.zeros((N_BINS, N_BINS), dtype=np.float64)
    np.add.at(H, (r, c), 1.0)
    return H


def instance_histograms(inst_id):
    """Load the cache entry and build up to MAX_BLOCKS histograms.

    Returns dict(blocks=[H...], n_symbols, coverage) or None.
    """
    npz = CACHE / f"{inst_id:05d}.npz"
    if not npz.exists():
        return None
    with np.load(npz) as z:
        X = z["X"]
        keep_idx = z["keep_idx"]
        data_bins = z["data_bins"]
        meta = json.loads(str(z["meta"]))
    if X.shape[0] < BLOCK:
        return {"blocks": [], "n_symbols": int(X.shape[0]),
                "status": "insufficient_symbols", "meta": meta}
    blocks = []
    used = 0
    for a, b in consecutive_runs(keep_idx):
        run = X[a:b]
        n_blk = len(run) // BLOCK
        for k in range(min(n_blk, MAX_BLOCKS - len(blocks))):
            H = block_histogram(run[k * BLOCK:(k + 1) * BLOCK], data_bins)
            if H is not None:
                blocks.append(H)
        used += n_blk * BLOCK
        if len(blocks) >= MAX_BLOCKS:
            break
    return {"blocks": blocks, "n_symbols": int(X.shape[0]),
            "n_blocks_available": used // BLOCK,
            "status": "ok" if blocks else "insufficient_symbols",
            "meta": meta}


def histogram_for_training(H):
    """per-instance max normalization -> float32 [1,50,50]."""
    m = H.max()
    if m <= 0:
        return None
    return (H / m).astype(np.float32)[None]


def build_all(out_npz=EXP / "cache/histograms.npz"):
    """Build histograms for every cached instance; store one npz."""
    man = json.loads((EXP / "dataset_manifest.json").read_text())
    Hs, ids, metas = [], [], []
    stats = {"instances_with_histograms": 0, "total_blocks": 0,
             "insufficient": 0, "missing_cache": 0}
    for inst in man["instances"]:
        res = instance_histograms(inst["instance_id"])
        if res is None:
            stats["missing_cache"] += 1
            continue
        if not res["blocks"]:
            stats["insufficient"] += 1
            continue
        stats["instances_with_histograms"] += 1
        stats["total_blocks"] += len(res["blocks"])
        for k, H in enumerate(res["blocks"]):
            Ht = histogram_for_training(H)
            if Ht is None:
                continue
            Hs.append(Ht)
            ids.append(inst["instance_id"])
            metas.append(json.dumps({
                "block": k, "class": inst["class"], "family": inst["family"],
                "shared_label": inst["shared_label"], "split": inst["split"],
                "overlap_any": inst["overlap_any"], "B_mhz": inst["B_mhz"],
                "n_blocks_available": res.get("n_blocks_available", 0),
            }))
    out = np.savez_compressed(
        out_npz,
        H=np.stack(Hs).astype(np.float32) if Hs else np.zeros((0, 1, 50, 50), np.float32),
        instance_id=np.array(ids, dtype=np.int64),
        meta=np.array(metas),
    )
    print(json.dumps(stats))
    print(f"wrote {out_npz}")
    return stats


if __name__ == "__main__":
    build_all()
