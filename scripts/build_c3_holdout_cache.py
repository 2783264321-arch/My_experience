#!/usr/bin/env python3
"""Build power + dense complex caches for the 1,159-sample val holdout.

The holdout is the full val split minus the fixed 384-sample C3 subset. The
dense operator and scale are unchanged; only the sample set differs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.signals.c3_frontend import COARSE_NFFT, dense_correlation
from src.signals.data import COARSE_H, N_TIME, SPLIT_MANIFEST, load_manifest
from scripts.build_c3_subset_cache import build_val_power

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cache/c3_holdout")
    ap.add_argument("--dense-out", default="cache/c3_holdout_dense")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out = ROOT / args.out
    dense_out = ROOT / args.dense_out
    out.mkdir(parents=True, exist_ok=True)
    dense_out.mkdir(parents=True, exist_ok=True)

    subset = json.loads((ROOT / "cache/c3_subset.json").read_text())
    sub = set(subset["val_ids"])
    val = load_manifest(SPLIT_MANIFEST, "val")
    hold = [r for r in val if r.sample_id not in sub]
    if len(hold) != len(val) - len(subset["val_ids"]):
        raise SystemExit("holdout count mismatch")
    if len(hold) != 1159:
        raise SystemExit(f"unexpected holdout size {len(hold)}")
    ids = [r.sample_id for r in hold]
    (out / "holdout.json").write_text(json.dumps({"val_ids": ids, "n": len(ids)}, indent=2))
    print(f"holdout {len(hold)}", flush=True)

    norm_cfg = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm_cfg["lo"]), float(norm_cfg["hi"]))
    power = out / "val_power_bf16.npy"
    build_val_power(hold, norm, power)
    pmm = np.load(power, mmap_mode="r")
    for i in (0, len(hold) - 1):
        bits = np.asarray(pmm[i]).copy()
        p = torch.from_numpy(bits).view(torch.bfloat16).float()[0]
        ref, _, _ = __import__("src.signals.data", fromlist=["build_sample_views"]).build_sample_views(hold[i], norm, [])
        if not torch.equal(p, ref.tensor[0].bfloat16().float()):
            raise SystemExit(f"power mismatch at {hold[i].sample_id}")
    print("power exact on holdout endpoints", flush=True)

    cpath = dense_out / "val_complex.npy"
    if cpath.exists() and not args.overwrite:
        raise SystemExit(f"{cpath} exists; pass --overwrite")
    mm = np.lib.format.open_memmap(cpath, mode="w+", dtype=np.float16,
                                   shape=(len(hold), 2, COARSE_H, N_TIME))
    t0 = time.perf_counter()
    diags = {}
    for i, rec in enumerate(hold):
        from src.signals.data import load_iq
        x, diag = dense_correlation(rec, load_iq(rec.bin_path))
        mm[i] = x.numpy().astype(np.float16)
        diags[rec.sample_id] = diag
        if (i + 1) % 50 == 0:
            mm.flush()
            rate = (i + 1) / (time.perf_counter() - t0)
            print(f"dense {i+1}/{len(hold)} {rate:.1f}/s", flush=True)
    mm.flush(); del mm
    meta = {
        "format": "c3-holdout-dense-f16-v1", "frontend": "dense",
        "channels": 2, "dtype": "float16",
        "shape": [len(hold), 2, COARSE_H, N_TIME],
        "scale": 1.0, "dense_hop_us": 10.0,
        "val_ids": ids, "holdout_json": str(out / "holdout.json"),
        "power_cache": str(power),
        "build_seconds": time.perf_counter() - t0,
        "per_sample": diags,
        "note": "production per-pair normalized dense correlation; no operator change",
    }
    (dense_out / "meta.json").write_text(json.dumps(meta, indent=2))
    # spot verification
    check = []
    for i in (0, len(hold) // 2, len(hold) - 1):
        from src.signals.data import load_iq
        x, _ = dense_correlation(hold[i], load_iq(hold[i].bin_path))
        cached = torch.from_numpy(np.asarray(np.load(cpath, mmap_mode="r")[i]).copy()).float()
        check.append({"sample_id": hold[i].sample_id,
                      "max_abs_diff_f16": float((cached - x.to(torch.float16).float()).abs().max())})
    meta["verification"] = check
    (dense_out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("dense cache done", json.dumps(check), flush=True)


if __name__ == "__main__":
    main()
