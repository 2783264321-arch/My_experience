#!/usr/bin/env python3
"""Independent front-end audit for the existing SRRC 8-window cache.

Checks IQ byte order/time crop, centre shift sign, 20 MS/s resampling,
band-edge gain, synthetic SRRC power/m4 retention, and exact re-use of
the cached windows. CPU-only; no training.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.signal import resample_poly

from scripts.baseline3_dsp import band_limit_fft
from scripts.baseline3_pa_srrc import gen_srrc_clean
from scripts.baseline3srrc_phase0_verify import (
    crop_and_resample, windows_from_crop, FS_OUT, KEEP_BW, WIN, N_WIN,
)
from src.signals.data import SPLIT_MANIFEST, load_manifest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/baseline3_srrc_gt_pilot"
CACHE = Path("/root/autodl-fs/runs_local/srrc-gt-pilot/fit_8win.npz")


def write_tone_bin(path, fs, dur_ms, f_off):
    n = int(round(fs * dur_ms / 1000.0))
    t = np.arange(n) / fs
    x = np.exp(2j * np.pi * f_off * t).astype(np.complex64)
    inter = np.empty(2 * n, dtype="<f2")
    inter[0::2] = x.real.astype("<f2")
    inter[1::2] = x.imag.astype("<f2")
    inter.tofile(path)
    return n


def synthetic_tone_gain(fs, f_off, dur_ms=10.0):
    f_lo = 2400e6
    center = f_lo + fs / 2.0
    with tempfile.TemporaryDirectory(dir="/tmp/opencode") as d:
        p = Path(d) / "tone.bin"
        n = write_tone_bin(p, fs, dur_ms, f_off)
        rec = SimpleNamespace(sample_id="synth", B_mhz=fs / 1e6,
                              T_ms=dur_ms, N=n, f_lo=f_lo, bin_path=p)
        sig = {"signal_id": 0, "class": 10,
               "start_frequency": (center - 5e6) / 1e6,
               "end_frequency": (center + 5e6) / 1e6,
               "start_time": 0.0, "end_time": dur_ms}
        crop = crop_and_resample(rec, sig)
    x = crop["x"]
    rms = float(np.sqrt(np.mean(np.abs(x) ** 2)))
    return {"fs_in_hz": fs, "f_off_hz": f_off, "rms": rms,
            "gain_db": 20.0 * math.log10(max(rms, 1e-30)),
            "out_points": int(len(x)), "raw_points": int(crop["raw_points"])}


def synthetic_srrc(fs=80e6, order=4, alpha=0.2):
    x, _ = gen_srrc_clean(10e6, alpha, 5000, order, fs, 1729 + order)
    n = min(len(x), int(0.01 * fs))
    return x[:n].astype(np.complex64)


def srrc_retention(fs=80e6, order=4, alpha=0.2):
    x = synthetic_srrc(fs, order, alpha)
    trim = min(int(0.2e-3 * fs), len(x) // 10)
    if trim:
        x = x[trim:len(x) - trim]
    for keep, tag in ((KEEP_BW, "cache_10p2"), (12e6, "ref_12")):
        y = band_limit_fft(x, fs, keep, edge_frac=0.05)
        ratio = Fraction(FS_OUT / fs).limit_denominator(2000)
        y = resample_poly(y, ratio.numerator, ratio.denominator)
        rms = float(np.sqrt(np.mean(np.abs(y) ** 2)))
        p = float(np.mean(np.abs(y) ** 2))
        m4 = float(np.mean(np.abs(y) ** 4) / max(p * p, 1e-30))
        if tag == "cache_10p2":
            cache = {"rms": rms, "m4": m4, "points": len(y)}
        else:
            ref = {"rms": rms, "m4": m4, "points": len(y)}
    return {
        "fs_in_hz": fs, "order": order, "alpha": alpha,
        "cache": cache, "ref": ref,
        "power_ratio_cache_vs_ref": (cache["rms"] / ref["rms"]) ** 2,
        "m4_cache_minus_ref": cache["m4"] - ref["m4"],
    }


def cache_audit(cache_path, n_recompute=6):
    z = np.load(cache_path, allow_pickle=False)
    cache = {k: z[k] for k in z.files}
    metadata = json.loads(str(cache["metadata"]))
    w = cache["windows"]
    result = {
        "path": str(cache_path),
        "bytes": cache_path.stat().st_size,
        "shape": list(w.shape),
        "dtype": str(w.dtype),
        "metadata": metadata,
        "n_instances": int(len(cache["labels"])),
        "n_groups": int(len(set(cache["groups"].tolist()))),
        "labels": {str(k): int(v) for k, v in
                   zip(*np.unique(cache["labels"], return_counts=True))},
        "split": {"dev": int(cache["is_dev"].sum()),
                  "train": int((~cache["is_dev"]).sum())},
        "strata": {str(k): int(v) for k, v in
                   zip(*np.unique(cache["strata"], return_counts=True))},
        "finite": bool(np.isfinite(w).all()),
        "zero_windows": int(np.all(w == 0, axis=(1, 2, 3)).sum()),
    }
    recs = {r.sample_id: r for r in load_manifest(SPLIT_MANIFEST, split="fit")}
    ids = np.linspace(0, len(cache["labels"]) - 1, n_recompute).astype(int)
    checks = []
    for i in ids:
        sid = str(cache["sample_ids"][i])
        si = int(cache["signal_indices"][i])
        rec = recs[sid]
        sig = rec.signals[si]
        if int(sig["class"]) != int(cache["labels"][i]) + 10:
            raise ValueError(f"class mismatch {sid}/{si}")
        crop = crop_and_resample(rec, sig)
        winfo = windows_from_crop(crop, sig, n_win=N_WIN)
        wins = winfo["windows"]
        arr = np.zeros((N_WIN, 2, WIN), dtype=np.float32)
        for out, win in zip(arr, wins):
            samples = crop["x"][win["start"]:win["stop"]]
            out[0, :len(samples)] = samples.real
            out[1, :len(samples)] = samples.imag
        diff = float(np.max(np.abs(arr - w[i])))
        checks.append({
            "sample_id": sid, "signal_idx": si, "class": int(sig["class"]),
            "B_obs_mhz": float(rec.B_mhz),
            "duration_ms": float(sig["end_time"]) - float(sig["start_time"]),
            "raw_points": int(crop["raw_points"]),
            "out_points": int(crop["out_points"]),
            "gt_start": int(winfo["gt_start"]),
            "gt_end": int(winfo["gt_end"]),
            "window_stops_inside_gt": bool(all(x["stop"] <= winfo["gt_end"]
                                               for x in wins)),
            "max_abs_cache_diff": diff,
            "cache_reusable": bool(diff < 1e-6),
        })
    result["recompute_checks"] = checks
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, default=OUT / "frontend_audit.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    audit = {
        "tone_gain": [synthetic_tone_gain(fs, off)
                      for fs in (20e6, 40e6, 80e6)
                      for off in (0.0, 4.4e6, 4.59e6, 5.0e6, 5.1e6,
                                  5.5e6, 5.61e6, 6.0e6, 8.0e6)],
        "srrc_retention": [srrc_retention(80e6, order)
                           for order in (4, 16)],
        "cache": cache_audit(args.cache),
    }
    args.out.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("tone gain (dB):")
    for fs in (20e6, 40e6, 80e6):
        vals = [x for x in audit["tone_gain"] if x["fs_in_hz"] == fs]
        print(f"  Fs={fs/1e6:.0f} MHz: " +
              " ".join(f"{x['f_off_hz']/1e6:.2f}MHz={x['gain_db']:.2f}"
                       for x in vals))
    print("SRRC retention:")
    for x in audit["srrc_retention"]:
        print(f"  order={x['order']} power_ratio={x['power_ratio_cache_vs_ref']:.6f} "
              f"m4={x['cache']['m4']:.4f} ref={x['ref']['m4']:.4f}")
    print("cache:", audit["cache"]["shape"], audit["cache"]["bytes"],
          "recompute:", [(c["sample_id"], c["max_abs_cache_diff"])
                         for c in audit["cache"]["recompute_checks"]])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
