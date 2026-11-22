#!/usr/bin/env python3
"""Phase-0 data and front-end verification for the SRRC 10 MHz local-IQ plan.

CPU-only, read-only. Verifies:
  - manifest counts for 10 MHz SRRC 10/11 (fit/val) and full class10/11;
  - file-level split disjointness;
  - GT-box crop / centre down-conversion / 10.2 MHz band-limit / 20 MS/s
    resampling on real IQ;
  - synthetic tone centre and band-edge checks;
  - 8 x 1024-point window coverage at 20 MS/s.

No training, no detector inference, no test usage.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from src.signals.data import SPLIT_MANIFEST, load_manifest
from scripts.baseline3_dsp import read_segment, demod_to_center, band_limit_fft

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/baseline3_srrc_phase0"
FS_OUT = 20.0e6
SIG_BW = 10.0e6
KEEP_BW = 10.2e6
EDGE_MS = 0.2
WIN = 1024
N_WIN = 8


def fft_peak_hz(x: np.ndarray, fs: float) -> float:
    n = len(x)
    spec = np.abs(np.fft.fftshift(np.fft.fft(x)))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
    return float(freqs[int(np.argmax(spec))])


def spectral_centroid_hz(x: np.ndarray, fs: float) -> float:
    n = len(x)
    spec = np.abs(np.fft.fft(x)) ** 2
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    s = float(spec.sum())
    return float(np.sum(freqs * spec) / s) if s > 0 else 0.0


def band_power(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    n = len(x)
    spec = np.abs(np.fft.fft(x)) ** 2
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    m = (freqs >= f_lo) & (freqs <= f_hi)
    return float(np.mean(spec[m])) if m.any() else 0.0


def overlap_frac(rec, sig) -> dict:
    a = (float(sig["start_frequency"]), float(sig["end_frequency"]),
         float(sig["start_time"]), float(sig["end_time"]))
    area = max((a[1] - a[0]) * (a[3] - a[2]), 1e-9)
    anyf = wifi = srrc = 0.0
    for o in rec.signals:
        if o is sig:
            continue
        b = (float(o["start_frequency"]), float(o["end_frequency"]),
             float(o["start_time"]), float(o["end_time"]))
        iw = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
        ih = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
        f = (iw * ih) / area
        anyf = max(anyf, f)
        if int(o["class"]) in (10, 11):
            srrc = max(srrc, f)
        if 0 <= int(o["class"]) <= 5:
            wifi = max(wifi, f)
    return {"any": int(anyf > 0), "strong": int(anyf >= 0.25),
            "srrc": int(srrc > 0), "wifi": int(wifi > 0),
            "any_frac": float(anyf), "srrc_frac": float(srrc)}


def crop_and_resample(rec, sig, fs_out=FS_OUT, edge_ms=EDGE_MS):
    fs_in = float(rec.B_mhz) * 1e6
    t1 = float(sig["start_time"]); t2 = float(sig["end_time"])
    ta = max(0.0, t1 - edge_ms)
    tb = min(float(rec.T_ms), t2 + edge_ms)
    i0 = max(0, int(round(ta / 1000.0 * fs_in)))
    i1 = min(int(rec.N), int(round(tb / 1000.0 * fs_in)))
    if i1 - i0 < 256:
        raise RuntimeError("crop too short")
    x = read_segment(str(rec.bin_path), i0, i1)
    if len(x) != i1 - i0:
        raise ValueError(f"IQ file shorter than manifest: {rec.bin_path}")
    fc = 0.5 * (float(sig["start_frequency"]) + float(sig["end_frequency"])) * 1e6
    xd = demod_to_center(x, fs_in, fc, float(rec.f_lo) * 1e6)
    xb = band_limit_fft(xd, fs_in, KEEP_BW, edge_frac=0.05)
    ratio = Fraction(fs_out / fs_in).limit_denominator(2000)
    xr = resample_poly(xb, ratio.numerator, ratio.denominator)
    return {
        "x": xr.astype(np.complex64), "fs_out": fs_out,
        "fs_in": fs_in, "i0": i0, "i1": i1,
        "ta": i0 / fs_in * 1000.0, "tb": i1 / fs_in * 1000.0,
        "crop_ms": (i1 - i0) / fs_in * 1000.0,
        "raw_points": int(i1 - i0), "out_points": int(len(xr)),
        "ratio": [ratio.numerator, ratio.denominator],
    }


def windows_from_crop(crop, sig, n_win=N_WIN, win=WIN):
    fs = crop["fs_out"]
    ta = crop["ta"]
    t1 = float(sig["start_time"]); t2 = float(sig["end_time"])
    a = int(round((t1 - ta) / 1000.0 * fs))
    b = int(round((t2 - ta) / 1000.0 * fs))
    a = max(0, min(len(crop["x"]), a))
    b = max(0, min(len(crop["x"]), b))
    if b <= a:
        raise ValueError("Box interval has no samples inside IQ crop")
    span = b - a
    if span <= win:
        starts = [a]
    else:
        starts = np.linspace(a, b - win, n_win)
        starts = [int(round(s)) for s in starts]
    wins = []
    for start in starts:
        stop = min(start + win, b)
        w = crop["x"][start:stop]
        rms = float(np.sqrt(np.mean(np.abs(w) ** 2)))
        wins.append({"start": start, "stop": stop, "rms": rms,
                     "finite": bool(np.isfinite(w).all())})
    return {"gt_start": a, "gt_end": b, "span": span,
            "n_windows": len(wins), "windows": wins,
            "coverage": min(1.0, len(wins) * win / span)}


def read_segment_check() -> dict:
    """Synthetic interleaved float16 check for read_segment offsets."""
    tmp = Path("/tmp/opencode/phase0_iq_check.bin")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    n = 1000
    i = np.arange(n, dtype=np.float32)
    q = (2.0 * i + 0.5).astype(np.float32)
    inter = np.empty(2 * n, dtype="<f2")
    inter[0::2] = i.astype("<f2")
    inter[1::2] = q.astype("<f2")
    inter.tofile(tmp)
    got = read_segment(str(tmp), 100, 200)
    want = (i[100:200] + 1j * q[100:200]).astype(np.complex64)
    return {
        "n": int(len(got)),
        "max_abs_err": float(np.max(np.abs(got - want))),
        "ok": bool(len(got) == 100 and np.allclose(got, want, atol=1e-3)),
    }


def group_audit() -> dict:
    """Check label-file repeats / group_id across fit and val."""
    p = Path("/root/autodl-tmp/workspace/spacenet/analysis/outputs/"
             "split_manifest.csv")
    rows = [r for r in csv.DictReader(p.open()) if r["split_new"] in ("fit", "val")]
    by_sha = defaultdict(set)
    by_group = defaultdict(set)
    for r in rows:
        by_sha[r["json_sha1"]].add(r["split_new"])
        by_group[r["group_id"]].add(r["split_new"])
    group_rows = Counter(r["group_id"] for r in rows)
    dup_groups = {k: v for k, v in group_rows.items() if v > 1}
    srrc_groups = []
    for gid in dup_groups:
        hit = False
        for r in rows:
            if r["group_id"] != gid:
                continue
            jp = (Path("/root/autodl-fs/SpaceNet_Dataset/advanced") /
                  ("train" if r["split"] == "train" else "test") /
                  f"{r['stem']}.json")
            try:
                sigs = json.loads(jp.read_text())["signals"]
            except Exception:
                continue
            for s in sigs:
                bw = float(s["end_frequency"]) - float(s["start_frequency"])
                if int(s["class"]) in (10, 11) and 9.5 <= bw <= 10.5:
                    hit = True
        if hit:
            srrc_groups.append(gid)
    return {
        "n_rows": len(rows),
        "n_json_sha": len(by_sha),
        "n_group_id": len(by_group),
        "json_sha_cross_split": sum(1 for v in by_sha.values() if len(v) > 1),
        "group_id_cross_split": sum(1 for v in by_group.values() if len(v) > 1),
        "duplicate_group_ids": len(dup_groups),
        "duplicate_groups_with_10mhz_srrc": len(srrc_groups),
        "duplicate_group_example": next(iter(dup_groups), None),
    }


def select_examples(recs):
    """Pick three 10 MHz examples spanning 20/40/80 MHz observations."""
    out = []
    for cls in (10, 11):
        for B in (20, 40, 80):
            cand = []
            for r in recs:
                if round(float(r.B_mhz)) != B:
                    continue
                for s in r.signals:
                    bw = float(s["end_frequency"]) - float(s["start_frequency"])
                    if int(s["class"]) == cls and 9.5 <= bw <= 10.5:
                        cand.append((float(s["end_time"]) - float(s["start_time"]), r, s))
            if cand:
                cand.sort(key=lambda z: z[0])
                # prefer a 5-30 ms example for CPU speed
                pick = min(cand, key=lambda z: abs(z[0] - 20.0))
                out.append(pick)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "frontend_verification.json"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- manifest counts ----
    manifest = {}
    for split in ("fit", "val"):
        recs = load_manifest(SPLIT_MANIFEST, split=split)
        manifest[split] = recs
    fit_ids = {r.sample_id for r in manifest["fit"]}
    val_ids = {r.sample_id for r in manifest["val"]}
    overlap_ids = sorted(fit_ids & val_ids)
    if overlap_ids:
        raise RuntimeError(f"fit/val leak: {overlap_ids[:5]}")

    counts = {}
    for split, recs in manifest.items():
        c = Counter()
        files_10 = set(); files_all = set()
        durations = []
        overlap = Counter()
        for r in recs:
            for s in r.signals:
                if int(s["class"]) not in (10, 11):
                    continue
                bw = float(s["end_frequency"]) - float(s["start_frequency"])
                c[f"class{int(s['class'])}_all"] += 1
                files_all.add(r.sample_id)
                if 9.5 <= bw <= 10.5:
                    c[f"class{int(s['class'])}_10mhz"] += 1
                    files_10.add(r.sample_id)
                    durations.append(float(s["end_time"]) - float(s["start_time"]))
                    ov = overlap_frac(r, s)
                    if ov["any"] == 0:
                        overlap["clean"] += 1
                    else:
                        overlap["any"] += 1
                        if ov["strong"]:
                            overlap["strong"] += 1
        counts[split] = {
            "n_files": len(recs),
            "n_all": int(sum(v for k, v in c.items() if k.endswith("_all"))),
            "n_10mhz": int(sum(v for k, v in c.items() if k.endswith("_10mhz"))),
            "files_all": len(files_all),
            "files_10mhz": len(files_10),
            "class10_all": int(c["class10_all"]),
            "class11_all": int(c["class11_all"]),
            "class10_10mhz": int(c["class10_10mhz"]),
            "class11_10mhz": int(c["class11_10mhz"]),
            "narrow": int(sum(1 for r in recs for s in r.signals
                              if int(s["class"]) in (10, 11) and
                              (float(s["end_frequency"]) - float(s["start_frequency"])) < 9.5)),
            "duration_ms": {
                "min": float(np.min(durations)),
                "median": float(np.median(durations)),
                "max": float(np.max(durations)),
            },
            "points_20msps": {
                "min": int(round(np.min(durations) * 20_000)),
                "median": int(round(np.median(durations) * 20_000)),
                "max": int(round(np.max(durations) * 20_000)),
            },
            "overlap_10mhz": dict(overlap),
        }

    expected = {
        "fit": {"n_10mhz": 1824, "files_10mhz": 1572,
                "n_all": 2752, "files_all": 2179},
        "val": {"n_10mhz": 469, "files_10mhz": 404,
                "n_all": 716, "files_all": 576},
    }
    checks = {}
    for split in ("fit", "val"):
        checks[split] = {
            k: {"expected": v, "actual": counts[split][k],
                "ok": int(v == counts[split][k])}
            for k, v in expected[split].items()
        }

    # ---- synthetic tone checks ----
    # Tones are generated at baseband offsets, not at absolute RF
    # frequencies: with Fs=40 MHz, a 2405 MHz RF tone aliases.
    fs = 40e6; n = int(0.02 * fs); t = np.arange(n) / fs
    f_lo = 2400e6; f_center = 2420e6
    synth = {}
    for label, f_off in (("below_center_15mhz", -15e6),
                         ("above_center_15mhz", 15e6),
                         ("in_band_3mhz", 3e6)):
        x = np.exp(2j * np.pi * f_off * t).astype(np.complex64)
        y = demod_to_center(x, fs, f_center + f_off, f_lo)  # shift this tone to DC
        synth[label] = {"peak_hz": fft_peak_hz(y, fs)}
    # fixed centre shift: a tone at centre-15 MHz should land at -15 MHz
    x = np.exp(2j * np.pi * (-15e6) * t).astype(np.complex64)
    y = demod_to_center(x, fs, f_center, f_lo)
    synth["fixed_center_below_peak_hz"] = fft_peak_hz(y, fs)
    # band limit: +3 MHz passes, +15 MHz rejected
    x3 = np.exp(2j * np.pi * 3e6 * t).astype(np.complex64)
    x15 = np.exp(2j * np.pi * 15e6 * t).astype(np.complex64)
    b3 = band_limit_fft(x3, fs, KEEP_BW, edge_frac=0.05)
    b15 = band_limit_fft(x15, fs, KEEP_BW, edge_frac=0.05)
    p3 = band_power(b3, fs, -5.1e6, 5.1e6)
    p15 = band_power(b15, fs, -5.1e6, 5.1e6)
    synth["bandlimit"] = {
        "inband_3mhz_power": p3,
        "outband_15mhz_power": p15,
        "rejection_db": float(10 * np.log10(max(p3, 1e-30) / max(p15, 1e-30))),
    }

    # ---- real-IQ front-end examples ----
    examples = []
    for dur, rec, sig in select_examples(manifest["val"] or manifest["fit"]):
        crop = crop_and_resample(rec, sig)
        pre_peak = fft_peak_hz(crop["x"], crop["fs_out"])
        win = windows_from_crop(crop, sig)
        examples.append({
            "sample_id": rec.sample_id, "signal_id": int(sig["signal_id"]),
            "class": int(sig["class"]), "B_obs_mhz": float(rec.B_mhz),
            "duration_ms": float(sig["end_time"]) - float(sig["start_time"]),
            "raw_points": crop["raw_points"],
            "out_points_20msps": crop["out_points"],
            "resample_ratio": crop["ratio"],
            "post_resample_peak_hz": pre_peak,
            "post_resample_centroid_hz": spectral_centroid_hz(
                crop["x"], crop["fs_out"]),
            "windows": win,
            "finite": bool(np.isfinite(crop["x"]).all()),
        })

    # ---- window coverage over all 10 MHz val ----
    cov = []
    for r in manifest["val"]:
        for s in r.signals:
            bw = float(s["end_frequency"]) - float(s["start_frequency"])
            if int(s["class"]) in (10, 11) and 9.5 <= bw <= 10.5:
                dur = float(s["end_time"]) - float(s["start_time"])
                npts = int(round(dur * 20_000))
                cov.append(N_WIN * WIN / max(npts, 1))
    coverage = {
        "n": len(cov),
        "min": float(np.min(cov)), "median": float(np.median(cov)),
        "max": float(np.max(cov)),
    }

    result = {
        "status": "ok",
        "split_leak": overlap_ids,
        "read_segment_check": read_segment_check(),
        "group_audit": group_audit(),
        "counts": counts,
        "checks": checks,
        "synthetic": synth,
        "real_examples": examples,
        "val_10mhz_window_coverage": coverage,
        "front_end": {
            "fs_out_hz": FS_OUT, "signal_bw_hz": SIG_BW,
            "keep_bw_hz": KEEP_BW, "edge_ms": EDGE_MS,
            "window_points": WIN, "n_windows": N_WIN,
        },
        "notes": [
            "Down-conversion and filtering preserve the time-sample count; "
            "only time cropping and resampling/decimation change it.",
            "The 20 MS/s choice is 2x the 10 MHz SRRC nominal bandwidth.",
            "GT boxes are used for phase-0 front-end verification only; "
            "predictions are evaluated separately.",
        ],
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "counts": {k: {x: v[x] for x in ("n_10mhz", "files_10mhz", "n_all",
                                         "files_all", "narrow")}
                   for k, v in counts.items()},
        "checks_ok": all(c["ok"] for s in checks.values() for c in s.values()),
        "read_segment_check": result["read_segment_check"],
        "group_audit": result["group_audit"],
        "synthetic": synth,
        "examples": [{k: e[k] for k in ("sample_id", "class", "B_obs_mhz",
                                        "duration_ms", "raw_points",
                                        "out_points_20msps",
                                        "post_resample_peak_hz")}
                     for e in examples],
        "window_coverage": coverage,
    }, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
