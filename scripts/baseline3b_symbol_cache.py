#!/usr/bin/env python3
"""Phase B' stage-1 prerequisite: per-instance OFDM symbol cache.

For every WiFi instance in dataset_manifest.json:
  central window min(20 ms, GT duration) -> extract_band (GT box + 2%
  transition guard) -> cp_scan_full (full-cycle, pre-registered grid,
  label-blind) -> if the pre-registered pass rule (metric > 0.34 AND
  ratio > 15, phase-A' calibration) holds, strip symbols at the winning
  candidate, CFO-corrected, energy-filter rows, and store the complex
  symbol matrix + sync metadata. Failures keep their best-candidate
  metrics for the sync-availability statistics.

Output: reports/baseline3_wifi_expert/cache/symbols/<instance_id>.npz
(complex64 [n_sym, n_fft] + meta) plus a summary JSON. Local cache only,
never committed. Parallel over 12 workers.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from scripts.baseline3b_ofdm_frontend import (
    extract_band, cp_scan_full, strip_symbols, filter_symbol_rows, CP_TU_US,
    CP_CP_US,
)

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports/baseline3_wifi_expert"
CACHE = EXP / "cache" / "symbols"
MANIFEST = EXP / "dataset_manifest.json"

PASS_METRIC = 0.34
PASS_RATIO = 15.0
CAP_MS = 20.0
MAX_SYMBOLS = 4000  # bound the cache; ~10 ms at 4us period is ~2500


def process_one(inst):
    bin_path = f"/root/autodl-fs/SpaceNet_Dataset/advanced/{inst['sample_id']}.bin"
    try:
        xb, fs, bw = extract_band(
            bin_path, inst["f_lo_mhz"], inst["B_mhz"] * 1e6,
            inst["f1_mhz"], inst["f2_mhz"],
            inst["t1_ms"], inst["t2_ms"], cap_ms=CAP_MS)
        res = cp_scan_full(xb, fs)
        b = res["best"]
        if b is None:
            return {"instance_id": inst["instance_id"], "status": "no_candidate"}
        out = {
            "instance_id": inst["instance_id"],
            "sample_id": inst["sample_id"], "class": inst["class"],
            "family": inst["family"],
            "tu_us": b["tu_us"], "cp_us": b["cp_us"],
            "metric": round(b["metric"], 4), "ratio": round(b["ratio"], 2),
            "phi": b["phi"],
            "cfo_hz": round(b["cfo_cycles"] * fs, 2),
            "n_win": b["n_sym"], "status": "ok",
        }
        if b["metric"] < PASS_METRIC or b["ratio"] < PASS_RATIO:
            out["status"] = "sync_fail"
            return out
        X = strip_symbols(xb, fs, b["tu_us"], b["cp_us"], b["phi"],
                          cfo_cycles=-b["cfo_cycles"],
                          max_sym=MAX_SYMBOLS)
        # The occupied-bin mask is determined by the TARGET bandwidth
        # (SCS = 1/Tu_us is fixed), NOT by the capture-rate FFT length:
        # 20 MHz -> 64-FFT layout (52 occupied, +-26); 40 MHz -> 128-FFT
        # layout (104 occupied, +-52). Bin indices coincide at any capture
        # rate because the subcarrier spacing is unchanged.
        SCS_HZ = 312.5e3
        n_fft_sig = int(round(inst["bw_mhz"] * 1e6 / SCS_HZ))
        occ, pilots, data = legacy_masks_public(n_fft_sig)
        X, keep_idx, n_kept, n_drop = filter_symbol_rows(X, data)
        if n_kept < 16:
            out["status"] = "too_few_symbols"
            out["n_kept"] = n_kept
            return out
        out["n_kept"] = n_kept
        out["n_dropped"] = n_drop
        npz = CACHE / f"{inst['instance_id']:05d}.npz"
        np.savez_compressed(npz, X=X,
                            keep_idx=keep_idx.astype(np.int32),
                            data_bins=np.asarray(data, dtype=np.int64),
                            meta=json.dumps(out))
        out["stored"] = True
        return out
    except Exception as e:
        return {"instance_id": inst["instance_id"], "status": "error",
                "error": str(e)[:200]}


def legacy_masks_public(n_fft):
    # imported lazily to keep the worker import surface small
    from scripts.baseline3b_ofdm_frontend import legacy_masks
    return legacy_masks(n_fft)


def main():
    man = json.loads(MANIFEST.read_text())
    insts = man["instances"]
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for j, r in enumerate(ex.map(process_one, insts, chunksize=8), 1):
            results.append(r)
            if j % 250 == 0:
                print(f"{j}/{len(insts)} {time.perf_counter()-t0:.0f}s",
                      flush=True)
    from collections import Counter
    cnt = Counter(r["status"] for r in results)
    ok = [r for r in results if r.get("stored")]
    print(json.dumps({"status_counts": dict(cnt),
                      "stored": len(ok),
                      "seconds": round(time.perf_counter() - t0, 1)}))
    (EXP / "symbol_cache_summary.json").write_text(
        json.dumps({"status_counts": dict(cnt),
                    "pass_rule": {"metric": PASS_METRIC, "ratio": PASS_RATIO},
                    "cap_ms": CAP_MS,
                    "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
