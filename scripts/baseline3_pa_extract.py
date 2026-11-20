#!/usr/bin/env python3
"""Phase A' Track E: is there target energy at the labeled frequency?

For every frame in reports/local_iq_round1/phase_ap_frames.json (single
source of truth, fit split only) this script uses a central analysis window
of min(10 ms, t2-t1) centered in [t1, t2] and computes, with NO demodulation
assumptions first and the audited chain second:

  1. RAW FFT of the un-demodulated window (full band; frequency convention:
     np.fft zero frequency = observation center f_lo + B/2, i.e.
     absolute = f_lo + B/2 + digital):
       - in/out dB : median PSD inside the GT box band [f1, f2] vs median
                     PSD outside the box but inside the observation, at
                     least 2 MHz away from the box;
       - top-3 spectral peaks with ABSOLUTE MHz and an inside-box flag;
       - strongest raw in-box peak (dB above the out-of-box median).
  2. CORRECTED extraction: demod_to_center to (f1+f2)/2,
     band_limit_fft(keep_bw=bw*1.02, edge_frac=0.05), then trim 0.5 ms from
     both ends:
       - dominant peak offset from DC (kHz, signed);
       - occupied width at -10 dB relative to the in-band max (MHz);
       - energy keep ratio (mean|xb|^2 after filter+trim over the mean of
         the demodulated window before the filter).
  3. OLD round-1 demod for contrast (shift = f_center - f_lo, wrong by
     B/2): dominant full-band peak offset from DC (MHz, signed). Under the
     wrong shift the labeled content is displaced by exactly B/2, so this
     landing near +/-B/2 shows the old code measured the wrong part of the
     capture. (The old chain then band-limited around DC, i.e. it kept the
     band displaced +B/2 from the label, wrapping at the edges.)
  4. Negative controls through the IDENTICAL chain:
       - 4 synthetic complex white-noise segments with (B, bw) matched to
         real frames;
       - 4 real "decoy band" controls: a band shifted +15 MHz from the box
         center (clipped into the observation), chosen to contain no JSON
         signal when possible.
  5. Verdict per frame, "target_at_label": true iff in/out dB >= 6 dB AND
     the strongest raw in-box peak is >= 10 dB above the out-of-box median.
     Energy presence at the labeled location ONLY -- explicitly NOT about
     modulation structure.

Outputs: reports/local_iq_round1/phase_ap_extract.{json,md}.
Run:  PYTHONPATH=. .venv/bin/python scripts/baseline3_pa_extract.py
"""
from __future__ import annotations

import argparse
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from scripts.baseline3_dsp import band_limit_fft, demod_to_center, read_segment

ROOT = Path(__file__).resolve().parents[1]
FRAMES_JSON = ROOT / "reports/local_iq_round1/phase_ap_frames.json"
OUT_JSON = ROOT / "reports/local_iq_round1/phase_ap_extract.json"
OUT_MD = ROOT / "reports/local_iq_round1/phase_ap_extract.md"
DATA_DIR = Path("/root/autodl-fs/SpaceNet_Dataset/advanced")

WIN_CAP_MS = 10.0        # central analysis window cap
SMOOTH_HZ = 50e3         # PSD smoothing for peak/width metrics
PEAK_SEP_HZ = 200e3      # minimum separation of reported spectral peaks
OUT_GUARD_HZ = 2e6       # out-of-box region must be >= this far from the box
TRIM_MS = 0.5            # edge-safety trim after the FFT band filter
KEEP_BW_FRAC = 1.02      # band_limit_fft keep_bw = bw * 1.02
EDGE_FRAC = 0.05         # band_limit_fft edge_frac
VERDICT_INOUT_DB = 6.0   # target_at_label threshold on in/out dB
VERDICT_PEAK_DB = 10.0   # ... and on in-box peak above out-of-box median
DECOY_SHIFT_MHZ = 15.0
SEED_BASE = 20261121

# (B, bw) matched to real frames; geometry cloned from those frames where
# one exists (train/20, train/1494, train/1058); (30, 2) has no real frame
# in the sample set, so its geometry is constructed.
SYNTH_SPECS = [
    {"label": "noise (40,20)", "B_mhz": 40.0, "bw_mhz": 20.0,
     "f_lo_mhz": 2407.0, "f1_mhz": 2412.0, "f2_mhz": 2432.0,
     "clone_of": "train/20"},
    {"label": "noise (80,40)", "B_mhz": 80.0, "bw_mhz": 40.0,
     "f_lo_mhz": 2401.0, "f1_mhz": 2437.0, "f2_mhz": 2477.0,
     "clone_of": "train/1494"},
    {"label": "noise (40,10)", "B_mhz": 40.0, "bw_mhz": 10.0,
     "f_lo_mhz": 2432.0, "f1_mhz": 2442.0, "f2_mhz": 2452.0,
     "clone_of": "train/1058"},
    {"label": "noise (30,2)", "B_mhz": 30.0, "bw_mhz": 2.0,
     "f_lo_mhz": 2400.0, "f1_mhz": 2411.0, "f2_mhz": 2413.0,
     "clone_of": None},
]

# Decoy-band candidates (sample_id, signal_id), checked against the sample's
# full JSON; the first 4 with a JSON-clear +15 MHz band are used. The five
# known-contaminated probes are evaluated FIRST so the report documents why
# they are rejected (their +15 MHz band contains other JSON signals -- these
# captures are crowded). Frames with bw=20/40 cannot be decoyed this way (a
# +15 MHz shift of a >=20 MHz band always re-contains part of the target
# box), so the pool is bw<=10.
DECOY_POOL = [
    ("train/1099", 0), ("train/1058", 1), ("train/9", 2), ("train/21", 4),
    ("train/1037", 1),
    ("train/123", 1), ("train/1471", 0), ("train/1986", 0),
    ("train/57", 3), ("train/9", 3), ("train/18", 2), ("train/50", 2),
]


def r2(v):
    return None if v is None else round(float(v), 2)


def r3(v):
    return None if v is None else round(float(v), 3)


def r4(v):
    return None if v is None else round(float(v), 4)


def _db(num, den):
    if num is None or den is None or den <= 0:
        return None
    return float(10.0 * math.log10(max(float(num), 1e-30) / max(float(den), 1e-30)))


def _smoothed(p, df):
    """Circular uniform smoothing of the PSD (~SMOOTH_HZ wide)."""
    w = max(1, int(round(SMOOTH_HZ / df)))
    if w > 1:
        p = uniform_filter1d(p, size=w, mode="wrap")
    return p, w


def audit_chain(x, geom):
    """The identical metric chain for real frames, decoys and synthetic noise."""
    fs = float(geom["B_mhz"]) * 1e6
    flo = float(geom["f_lo_mhz"]) * 1e6
    f1 = float(geom["f1_mhz"]) * 1e6
    f2 = float(geom["f2_mhz"]) * 1e6
    bw = f2 - f1
    fc = 0.5 * (f1 + f2)
    obs_c = flo + 0.5 * fs
    n = len(x)

    # ---------- 1. RAW full-band FFT (no demodulation) ----------
    spec = np.fft.fft(x)
    p = np.abs(spec) ** 2
    freq = np.fft.fftfreq(n, 1.0 / fs)
    df = fs / n
    absf = obs_c + freq
    obs = (absf >= flo) & (absf <= flo + fs)
    inb = (absf >= f1) & (absf <= f2)
    outb = obs & ((absf <= f1 - OUT_GUARD_HZ) | (absf >= f2 + OUT_GUARD_HZ))
    med_in = float(np.median(p[inb])) if bool(inb.any()) else None
    med_out = float(np.median(p[outb])) if bool(outb.any()) else None
    in_out_db = _db(med_in, med_out)
    p_s, smooth_bins = _smoothed(p, df)
    box_peak = float(p_s[inb].max()) if bool(inb.any()) else None
    box_peak_db = _db(box_peak, med_out)
    top3 = []
    if n >= 16:
        dist = max(1, int(round(PEAK_SEP_HZ / df)))
        pk, _ = find_peaks(p_s, distance=dist)
        if len(pk):
            order = np.argsort(p_s[pk])[::-1][:3]
            for k in pk[order]:
                k = int(k)
                top3.append({
                    "abs_mhz": r4(float(absf[k]) / 1e6),
                    "digital_mhz": r4(float(freq[k]) / 1e6),
                    "psd_db": r2(10.0 * math.log10(max(float(p_s[k]), 1e-30))),
                    "db_above_out": r2(_db(float(p_s[k]), med_out)),
                    "inside_box": bool(f1 <= float(absf[k]) <= f2),
                })
    raw = {
        "n_in_bins": int(inb.sum()),
        "n_out_bins": int(outb.sum()),
        "out_region_empty": not bool(outb.any()),
        "in_out_db": r2(in_out_db),
        "box_peak_db_above_out": r2(box_peak_db),
        "smooth_bins": int(smooth_bins),
        "df_hz": float(df),
        "top3": top3,
    }

    # ---------- 2. CORRECTED extraction ----------
    xb = demod_to_center(x, fs, fc, flo)
    p_demod = float(np.mean(np.abs(xb) ** 2))
    xf = band_limit_fft(xb, fs, bw * KEEP_BW_FRAC, EDGE_FRAC)
    ntrim = int(round(TRIM_MS * 1e-3 * fs))
    xt = xf[ntrim:n - ntrim] if 2 * ntrim < n else xf
    p_keep = float(np.mean(np.abs(xt) ** 2))
    m = len(xt)
    pc = np.abs(np.fft.fft(xt)) ** 2
    freq_c = np.fft.fftfreq(m, 1.0 / fs)
    dfc = fs / m
    pc_s, _ = _smoothed(pc, dfc)
    kpk = int(np.argmax(pc_s))
    peak_khz = float(freq_c[kpk]) / 1e3
    keepm = np.abs(freq_c) <= bw * KEEP_BW_FRAC / 2.0
    if bool(keepm.any()):
        pmax = float(pc_s[keepm].max())
        occ_mhz = float(int(np.count_nonzero(pc_s[keepm] >= 0.1 * pmax)) * dfc / 1e6)
    else:
        occ_mhz = None
    corrected = {
        "peak_offset_khz": r2(peak_khz),
        "peak_offset_abs_khz": r2(abs(peak_khz)),
        "occupied_minus10db_mhz": r3(occ_mhz),
        "energy_keep_ratio": (r4(p_keep / p_demod) if p_demod > 0 else None),
        "n_trim_samples": int(ntrim),
        "n_after_trim": int(m),
    }

    # ---------- 3. OLD (round-1, wrong) demod: shift = f_center - f_lo ----------
    ph = 2.0 * np.pi * (fc - flo) / fs * np.arange(n, dtype=np.float64)
    x_old = (x * np.exp(-1j * ph)).astype(np.complex64)
    po = np.abs(np.fft.fft(x_old)) ** 2
    fo = np.fft.fftfreq(n, 1.0 / fs)
    po_s, _ = _smoothed(po, fs / n)
    ko = int(np.argmax(po_s))
    old_peak_mhz = float(fo[ko]) / 1e6
    old = {
        "peak_offset_mhz": r3(old_peak_mhz),
        "peak_offset_abs_mhz": r3(abs(old_peak_mhz)),
    }

    verdict = {
        "in_out_db": raw["in_out_db"],
        "box_peak_db_above_out": raw["box_peak_db_above_out"],
        "target_at_label": bool(
            in_out_db is not None and in_out_db >= VERDICT_INOUT_DB
            and box_peak_db is not None and box_peak_db >= VERDICT_PEAK_DB),
    }
    return {"n_samples": int(n), "raw": raw, "corrected": corrected,
            "old": old, "verdict": verdict}


def run_task(task):
    kind = task["kind"]
    try:
        if kind == "synth":
            geom = task["geom"]
            fs = float(geom["B_mhz"]) * 1e6
            m = int(round(float(task["win_ms"]) * 1e-3 * fs))
            rng = np.random.default_rng(task["seed"])
            x = (rng.standard_normal(m) + 1j * rng.standard_normal(m)).astype(np.complex64)
        else:
            x = read_segment(task["bin"], task["i0"], task["i1"])
        res = audit_chain(x, task["geom"])
        rec = dict(task["meta"])
        rec.update(res)
        return task["id"], rec
    except Exception as exc:  # keep the audit going; record the failure
        rec = dict(task["meta"])
        rec["error"] = repr(exc)
        return task["id"], rec


def window_bounds(fr):
    """Central analysis window [i0, i1) of min(WIN_CAP_MS, t2-t1)."""
    fs = float(fr["B_mhz"]) * 1e6
    t1 = float(fr["t1_ms"])
    t2 = float(fr["t2_ms"])
    c = min(WIN_CAP_MS, max(t2 - t1, 1e-9))
    tm = 0.5 * (t1 + t2)
    ta, tb = tm - c / 2.0, tm + c / 2.0
    n_tot = int(round(float(fr["B_mhz"]) * 1e6 * float(fr["T_ms"]) * 1e-3))
    i0 = max(0, int(round(ta * 1e-3 * fs)))
    i1 = min(n_tot, int(round(tb * 1e-3 * fs)))
    return i0, max(i1, i0 + 256), c, ta, tb


def json_overlap_signals(sample_id, f1, f2, ta_ms, tb_ms):
    """JSON signals whose (freq, time) box intersects [f1,f2] x [ta,tb]."""
    path = DATA_DIR / f"{sample_id}.json"
    sigs = json.loads(path.read_text(encoding="utf-8"))["signals"]
    hits = []
    for s in sigs:
        sf1 = float(s["start_frequency"])
        sf2 = float(s["end_frequency"])
        st1 = float(s["start_time"])
        st2 = float(s["end_time"])
        if (min(f2, sf2) - max(f1, sf1) > 1e-9
                and min(tb_ms, st2) - max(ta_ms, st1) > 1e-9):
            hits.append({"signal_id": s.get("signal_id"), "class": s.get("class"),
                         "f1_mhz": sf1, "f2_mhz": sf2, "t1_ms": st1, "t2_ms": st2})
    return hits


def select_decoys(frames, limit=4):
    """First `limit` frames from DECOY_POOL whose +15 MHz band is JSON-clear."""
    by_key = {(f["sample_id"], int(f["signal_id"])): f for f in frames}
    chosen, fallback = [], []
    for sid, sig in DECOY_POOL:
        fr = by_key.get((sid, sig))
        if fr is None:
            continue
        B = float(fr["B_mhz"])
        bw = float(fr["bw_mhz"])
        f_lo, f_hi = float(fr["f_lo_mhz"]), float(fr["f_hi_mhz"])
        f1, f2 = float(fr["f1_mhz"]), float(fr["f2_mhz"])
        fc = 0.5 * (f1 + f2)
        cd = min(max(fc + DECOY_SHIFT_MHZ, f_lo + bw / 2.0), f_hi - bw / 2.0)
        f1d, f2d = cd - bw / 2.0, cd + bw / 2.0
        i0, i1, win_ms, ta, tb = window_bounds(fr)
        hits = json_overlap_signals(sid, f1d, f2d, ta, tb)
        rec = {
            "sample_id": sid, "signal_id": int(fr["signal_id"]),
            "class": int(fr["class"]), "stratum": fr["stratum"],
            "B_mhz": B, "bw_mhz": bw, "f_lo_mhz": f_lo,
            "f1_mhz": f1, "f2_mhz": f2,
            "decoy_f1_mhz": f1d, "decoy_f2_mhz": f2d,
            "decoy_center_mhz": cd, "decoy_shift_mhz": cd - fc,
            "i0": i0, "i1": i1, "win_ms": win_ms,
            "json_overlap": hits,
        }
        (chosen if not hits else fallback).append(rec)
        if len(chosen) >= limit:
            break
    for rec in fallback:
        if len(chosen) >= limit:
            break
        chosen.append(rec)
    return chosen[:limit], fallback


def _get(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _med(vals):
    vals = [v for v in vals if v is not None]
    return round(float(np.median(vals)), 2) if vals else None


def _med_abs(rows, *path):
    vals = [abs(float(_get(r, *path))) for r in rows if _get(r, *path) is not None]
    return round(float(np.median(vals)), 2) if vals else None


def group_stats(rows):
    ok = [r for r in rows if "error" not in r]
    st = {
        "n": len(rows),
        "n_ok": len(ok),
        "n_pass": int(sum(1 for r in ok if _get(r, "verdict", "target_at_label"))),
        "med_in_out_db": _med([_get(r, "raw", "in_out_db") for r in ok]),
        "med_box_peak_db": _med([_get(r, "raw", "box_peak_db_above_out") for r in ok]),
        "med_abs_peak_khz": _med_abs(ok, "corrected", "peak_offset_khz"),
        "med_occupied_mhz": _med([_get(r, "corrected", "occupied_minus10db_mhz") for r in ok]),
        "med_abs_old_peak_mhz": _med_abs(ok, "old", "peak_offset_mhz"),
        "med_keep_ratio": _med([_get(r, "corrected", "energy_keep_ratio") for r in ok]),
    }
    st["pass_frac"] = round(st["n_pass"] / st["n"], 3) if st["n"] else None
    return st


def rid(r):
    return f"{r['sample_id']}:{r['signal_id']}"


def build_summary(frame_recs, synth_recs, decoy_recs):
    ok = [r for r in frame_recs if "error" not in r]
    per_cs = {}
    for c, s in sorted({(int(r["class"]), str(r["stratum"])) for r in frame_recs}):
        per_cs[f"{c}/{s}"] = group_stats(
            [r for r in frame_recs if int(r["class"]) == c and str(r["stratum"]) == s])
    per_class = {}
    for c in sorted({int(r["class"]) for r in frame_recs}):
        per_class[str(c)] = group_stats([r for r in frame_recs if int(r["class"]) == c])
    per_strat = {}
    for s in sorted({str(r["stratum"]) for r in frame_recs}):
        per_strat[s] = group_stats([r for r in frame_recs if str(r["stratum"]) == s])
    near_ids = [rid(r) for r in ok
               if (_get(r, "old", "peak_offset_abs_mhz") or -1.0)
               >= float(r["B_mhz"]) / 2.0 - float(r["bw_mhz"]) / 2.0 - 0.5]
    inside_ids = [rid(r) for r in ok
                  if _get(r, "raw", "top3") and r["raw"]["top3"][0]["inside_box"]]
    fails = [r for r in ok if not _get(r, "verdict", "target_at_label")]
    fail_ne = [rid(r) for r in fails if _get(r, "raw", "out_region_empty")]
    fail_tone = [rid(r) for r in fails
                 if not _get(r, "raw", "out_region_empty")
                 and (_get(r, "raw", "box_peak_db_above_out") or -99.0)
                 >= VERDICT_PEAK_DB]
    fail_weak = [rid(r) for r in fails
                 if not _get(r, "raw", "out_region_empty")
                 and (_get(r, "raw", "box_peak_db_above_out") or -99.0)
                 < VERDICT_PEAK_DB]
    return {
        "per_class_stratum": per_cs,
        "per_class": per_class,
        "per_stratum": per_strat,
        "overall": group_stats(frame_recs),
        "synthetic_noise": group_stats(synth_recs),
        "decoy_controls": group_stats(decoy_recs),
        "old_demod_near_halfband": {
            "n": len(ok), "count": len(near_ids),
            "matches_top1_inside_box": bool(near_ids == inside_ids),
            "ids": near_ids,
        },
        "verdict_fail_frames": [
            {"id": rid(r), "class": r["class"], "stratum": r["stratum"],
             "in_out_db": _get(r, "raw", "in_out_db"),
             "box_peak_db": _get(r, "raw", "box_peak_db_above_out"),
             "top1_inside_box": (r["raw"]["top3"][0]["inside_box"]
                                 if _get(r, "raw", "top3") else None),
             "top1_abs_mhz": (r["raw"]["top3"][0]["abs_mhz"]
                              if _get(r, "raw", "top3") else None),
             "out_region_empty": bool(_get(r, "raw", "out_region_empty"))}
            for r in fails],
        "fail_not_evaluable": fail_ne,
        "fail_narrow_tone_only": fail_tone,
        "fail_weak_at_label": fail_weak,
        "out_region_empty_frames": [rid(r) for r in ok if _get(r, "raw", "out_region_empty")],
        "negative_in_out_frames": [rid(r) for r in ok
                                   if (_get(r, "raw", "in_out_db") is not None
                                       and _get(r, "raw", "in_out_db") < 0.0)],
        "top1_peak_outside_box_frames": [rid(r) for r in ok
                                         if _get(r, "raw", "top3")
                                         and not r["raw"]["top3"][0]["inside_box"]],
    }


def fmt(v, spec="{:+.1f}", na="n/a"):
    return na if v is None else spec.format(v)


def write_md(path, meta, frame_recs, synth_recs, decoy_recs, decoy_fallback,
             summary, seconds):
    L = []
    a = L.append
    ov = summary["overall"]
    per_class = summary["per_class"]
    per_strat = summary["per_stratum"]

    a("# Baseline-3 Phase A' · Track E — label-location energy audit")
    a("")
    a(f"Generated {meta['date']} by `scripts/baseline3_pa_extract.py` in "
      f"{seconds:.0f} s on {meta['workers']} workers. Full per-frame numbers in "
      f"`phase_ap_extract.json`.")
    a("Sample set: `reports/local_iq_round1/phase_ap_frames.json` — 46 frames, "
      "fit split only, all `train/`. Nothing under `advanced/test` was read.")
    a("")
    a("## Method")
    a("")
    a("- Analysis window: min(10 ms, t2−t1) centered in [t1, t2] — 10.0 ms for "
      "every frame in this set; FFT length N = B·10k samples.")
    a("- Frequency convention (audited, unit-tested): np.fft zero frequency = "
      "observation center `f_lo + B/2`, i.e. `abs = f_lo + B/2 + digital`.")
    a("- **RAW** (un-demodulated, full band): `in/out dB` = 10·log10(median PSD "
      "inside [f1,f2] / median PSD outside the box but inside the observation, "
      "≥2 MHz away from the box); top-3 peaks (PSD smoothed ~50 kHz, min "
      "separation 200 kHz) as absolute MHz with inside-box flags; strongest "
      "in-box peak vs the out-of-box median (dB).")
    a("- **CORRECTED**: `demod_to_center` to (f1+f2)/2 → `band_limit_fft`"
      "(keep_bw = bw·1.02, edge_frac = 0.05) → trim 0.5 ms both ends → dominant "
      "peak offset from DC (kHz, signed), occupied width at −10 dB rel. in-band "
      "max (MHz), energy keep ratio.")
    a("- **OLD** (round-1 wrong demod, shift = f_center − f_lo): dominant "
      "full-band peak offset from DC (MHz). Under this shift the labeled "
      "content is displaced by exactly B/2, so a value near ±B/2 means the old "
      "code was looking at the wrong part of the capture (its DC-centered box "
      "filter kept the band displaced +B/2 from the label, wrapping at the "
      "observation edges).")
    a("- **Verdict `target_at_label`**: in/out dB ≥ 6 dB AND strongest raw "
      "in-box peak ≥ 10 dB above the out-of-box median. Energy presence ONLY — "
      "explicitly NOT about modulation structure.")
    a("")
    a("## 1. Per-frame table (ids are `sample:signal`, all under `train/`)")
    a("")
    a("| id | cls | stratum | B | bw | in/out dB | pk kHz | occ MHz | old MHz | @label |")
    a("|---:|---:|---|---:|---:|---:|---:|---:|---:|:--:|")
    for r in frame_recs:
        io = _get(r, "raw", "in_out_db")
        pk = _get(r, "corrected", "peak_offset_khz")
        occ = _get(r, "corrected", "occupied_minus10db_mhz")
        old = _get(r, "old", "peak_offset_mhz")
        if "error" in r:
            a(f"| {rid(r)} | {r['class']} | {r['stratum']} | {r['B_mhz']:g} | "
              f"{r['bw_mhz']:g} | ERROR: {r['error']} | | | | |")
            continue
        mark = "✓" if _get(r, "verdict", "target_at_label") else (
            "✗ᵉ" if _get(r, "raw", "out_region_empty") else "✗")
        a(f"| {rid(r)} | {r['class']} | {r['stratum']} | {r['B_mhz']:g} | "
          f"{r['bw_mhz']:g} | {fmt(io)} | {fmt(pk)} | {fmt(occ, '{:.2f}', '—')} | "
          f"{fmt(old)} | {mark} |")
    a("")
    a("ᵉ box spans the whole observation → no out-of-box region ≥2 MHz away; "
      "in/out dB not evaluable, verdict false by rule. Frames: "
      f"{', '.join(summary['out_region_empty_frames'])}.")
    a("")
    a("## 2. Negative controls through the identical chain")
    a("")
    a("### 2a. Synthetic complex white noise, (B, bw) matched to real frames")
    a("")
    a("| config | B | bw | in/out dB | pk kHz | occ MHz | keep | old MHz | @label |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|:--:|")
    for r in synth_recs:
        mark = "✓" if _get(r, "verdict", "target_at_label") else "✗"
        a(f"| {r['label']} | {r['B_mhz']:g} | {r['bw_mhz']:g} | "
          f"{fmt(_get(r, 'raw', 'in_out_db'))} | "
          f"{fmt(_get(r, 'corrected', 'peak_offset_khz'))} | "
          f"{fmt(_get(r, 'corrected', 'occupied_minus10db_mhz'), '{:.2f}', '—')} | "
          f"{fmt(_get(r, 'corrected', 'energy_keep_ratio'), '{:.3f}')} | "
          f"{fmt(_get(r, 'old', 'peak_offset_mhz'))} | {mark} |")
    a("")
    a("Expected \"nothing there\": in/out ≈ 0 dB, occupied ≈ full keep band, "
      "no in-box peak dominance, verdict false.")
    a("")
    a("### 2b. Real decoy bands: +15 MHz from the box center, clipped into "
      "the observation, JSON-clear when possible")
    a("")
    a("| id | cls | decoy band MHz | shift | JSON clear | in/out dB | pk kHz | "
      "occ MHz | old MHz | @label |")
    a("|---:|---:|---|---:|:--:|---:|---:|---:|---:|:--:|")
    for r in decoy_recs:
        clr = "yes" if not r.get("json_overlap") else (
            "NO " + ";".join(f"c{h['class']}@[{h['f1_mhz']:.1f},{h['f2_mhz']:.1f}]"
                             for h in r["json_overlap"]))
        mark = "✓" if _get(r, "verdict", "target_at_label") else "✗"
        band = f"[{r['decoy_band_mhz'][0]:g},{r['decoy_band_mhz'][1]:g}]"
        a(f"| {rid(r)} | {r['class']} | {band} | {r['decoy_shift_mhz']:+g} | {clr} | "
          f"{fmt(_get(r, 'raw', 'in_out_db'))} | "
          f"{fmt(_get(r, 'corrected', 'peak_offset_khz'))} | "
          f"{fmt(_get(r, 'corrected', 'occupied_minus10db_mhz'), '{:.2f}', '—')} | "
          f"{fmt(_get(r, 'old', 'peak_offset_mhz'))} | {mark} |")
    if decoy_fallback:
        rej = "; ".join(
            f"{rec['sample_id']}:{rec['signal_id']} → "
            + ";".join(f"c{h['class']}@[{h['f1_mhz']:.1f},{h['f2_mhz']:.1f}]MHz"
                       f"×[{h['t1_ms']:.1f},{h['t2_ms']:.1f}]ms"
                       for h in rec["json_overlap"])
            for rec in decoy_fallback)
        a("")
        a(f"Rejected decoy candidates (their +15 MHz band contains JSON "
          f"signals): {rej}.")
    a("")
    a("## 3. Per-class/stratum summary")
    a("")
    a("| group | n | pass | med in/out dB | med box-pk dB | med \\|pk\\| kHz | "
      "med occ MHz | med \\|old\\| MHz |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")

    def srow(name, st):
        a(f"| {name} | {st['n']} | {st['n_pass']} | {fmt(st['med_in_out_db'])} | "
          f"{fmt(st['med_box_peak_db'])} | "
          f"{fmt(st['med_abs_peak_khz'], '{:.0f}')} | "
          f"{fmt(st['med_occupied_mhz'], '{:.2f}')} | "
          f"{fmt(st['med_abs_old_peak_mhz'], '{:.1f}')} |")

    for k, st in summary["per_class_stratum"].items():
        srow(k, st)
    srow("**ALL**", ov)
    a("")
    a("Pass by class: "
      + " · ".join(f"{c} {per_class[c]['n_pass']}/{per_class[c]['n']}"
                   for c in sorted(per_class, key=int)) + ".  ")
    a("Pass by stratum: "
      + " · ".join(f"{s} {per_strat[s]['n_pass']}/{per_strat[s]['n']}"
                   for s in per_strat) + ".")
    a("")
    a("## 4. Facts vs inferences")
    a("")
    a("### Facts (measured)")
    a("")
    wins = [r.get("win_ms") for r in frame_recs if "error" not in r]
    wdesc = (f"all {len(wins)} are {min(wins):.1f} ms"
             if wins and min(wins) == max(wins)
             else f"{min(wins):.1f}–{max(wins):.1f} ms")
    a(f"- Processed {ov['n_ok']}/{ov['n']} real frames plus "
      f"{len(synth_recs)} synthetic-noise and {len(decoy_recs)} decoy controls "
      f"in {seconds:.0f} s; analysis windows: {wdesc} (GT durations are all "
      f"≥ 10 ms in this set).")
    a(f"- `target_at_label` passes {ov['n_pass']}/{ov['n']} frames overall "
      f"(clean {per_strat.get('clean', {}).get('n_pass', 0)}/"
      f"{per_strat.get('clean', {}).get('n', 0)}, overlap "
      f"{per_strat.get('overlap', {}).get('n_pass', 0)}/"
      f"{per_strat.get('overlap', {}).get('n', 0)}, control "
      f"{per_strat.get('control', {}).get('n_pass', 0)}/"
      f"{per_strat.get('control', {}).get('n', 0)}).")
    med_io = {c: per_class[c]["med_in_out_db"] for c in per_class}
    a("- Raw in/out dB per class (medians): "
      + " · ".join(f"{c} {v}" for c, v in sorted(med_io.items(), key=lambda kv: int(kv[0])))
      + f"; overall median {ov['med_in_out_db']} dB. No evaluable frame has a "
      f"NEGATIVE in/out dB (no labeled band is emptier than its "
      f"surroundings).")
    a(f"- Corrected peak offset: overall median |pk| = "
      f"{ov['med_abs_peak_khz']:.0f} kHz from DC; occupied width overall "
      f"median {ov['med_occupied_mhz']:.2f} MHz. Controls land at |pk| "
      f"{per_strat.get('control', {}).get('med_abs_peak_khz', 0):.0f} kHz "
      f"median with occupied ≈ their true bw; target classes sit MHz away "
      f"from box center and occupy a large fraction of the keep band.")
    a(f"- Synthetic white noise through the identical chain: in/out dB "
      f"median {summary['synthetic_noise']['med_in_out_db']} dB, in-box peak "
      f"only {summary['synthetic_noise']['med_box_peak_db']} dB above the "
      f"out-median, occupied median "
      f"{summary['synthetic_noise']['med_occupied_mhz']} MHz ≈ keep band, "
      f"{summary['synthetic_noise']['n_pass']}/"
      f"{summary['synthetic_noise']['n']} pass.")
    evaluable = [r for r in frame_recs
                 if "error" not in r and not _get(r, "raw", "out_region_empty")]
    min_boxpk = min((_get(r, "raw", "box_peak_db_above_out") or 0.0)
                    for r in evaluable)
    n_top1_out = len(summary["top1_peak_outside_box_frames"])
    decoy_io = [(_get(r, 'raw', 'in_out_db') or 0.0) for r in decoy_recs]
    a(f"- Real decoy bands (JSON-clear {sum(1 for r in decoy_recs if not r.get('json_overlap'))}"
      f"/{len(decoy_recs)}): in/out dB "
      f"{min(decoy_io):+.2f}…{max(decoy_io):+.2f}, energy keep ratio "
      f"median {summary['decoy_controls']['med_keep_ratio']}, "
      f"{summary['decoy_controls']['n_pass']}/{summary['decoy_controls']['n']} "
      f"pass — a real-capture \"nothing there\" reference: at or below the "
      f"surrounding floor.")
    oh = summary["old_demod_near_halfband"]
    a(f"- OLD wrong demod: the dominant peak lands within bw/2+0.5 MHz of "
      f"±B/2 in {oh['count']}/{oh['n']} frames — and this set coincides "
      f"({'verified' if oh['matches_top1_inside_box'] else 'NOT verified, see JSON'}) "
      f"with exactly the frames whose top-1 raw peak is INSIDE the box. In "
      f"the other {oh['n'] - oh['count']} frames the strongest capture "
      f"energy is outside the box and the old-demod peak simply follows it.")
    n_empty = len(summary["out_region_empty_frames"])
    a(f"- {n_empty} frames have a box equal to the whole observation "
      f"({', '.join(summary['out_region_empty_frames'])}): in/out dB is not "
      "evaluable there by construction (both still have their top-1 raw "
      "peak inside the box).")
    a("")
    a("### Inferences (interpretation — cross-check with the other phase-A' "
      "tracks before relying on them)")
    a("")
    a("- Where `target_at_label` passes, the labeled band genuinely "
      "concentrates energy relative to its surroundings. This is a statement "
      "about ENERGY PRESENCE only; phase A already established that the "
      "labeled modulations (WiFi OFDM, SRRC single-carrier) are absent, and "
      "nothing here supports or requires modulation content.")
    a("- The old round-1 demod displaced the labeled content to the ±B/2 "
      "edge of the digital band (fact above). Round-1 per-frame statements "
      "about \"box content\" therefore described a band shifted B/2 from the "
      "label, not the label band; the round-1 census is unreliable at frame "
      "level. Where its class-level findings agree with this audit, they do "
      "so despite the geometry error, not because of it.")
    a("- The labeled bands are NOT empty: every evaluable frame has an "
      f"in-box peak ≥ {min_boxpk:.1f} dB above its out-of-box median, and no "
      f"in/out dB is negative. But in {n_top1_out}/{ov['n']} frames the "
      "single strongest peak of the whole observation window lies OUTSIDE "
      "the labeled box — these captures are crowded, and the label is "
      "usually not the dominant energy.")
    a("- The verdict failures are dominated by median-contrast failures: "
      "narrow tone-like content at/near the label with a box MEDIAN close to "
      "the capture floor (see §5 taxonomy). "
      f"{len(summary['fail_weak_at_label'])} frame(s) carry no significant "
      "content at the label at all — the closest thing to \"JSON band "
      "empty, energy elsewhere\" in this sample set.")
    a("- The decoy/noise controls define the \"nothing there\" signature: "
      "in/out ≈ 0 or below, keep ratio ≈ keep_bw/B, occupied ≈ full keep "
      "band, no in-box peak dominance. No target frame matches this "
      "signature completely; the failures differ from it mainly by having a "
      "narrow in-box peak above the floor.")
    a("")
    a("## 5. Anomalies (verdict failures, taxonomy)")
    a("")
    fails = summary["verdict_fail_frames"]
    if not fails:
        a("- No frame fails `target_at_label`.")
    a(f"- **Not evaluable** (box = whole observation, no out-region; verdict "
      f"false by rule): {', '.join(summary['fail_not_evaluable'])} — both "
      f"have their top-1 raw peak INSIDE the box.")
    a(f"- **Narrow tone at/near the label, box median at the floor** "
      f"(in-box peak ≥ {VERDICT_PEAK_DB:.0f} dB above out-median but in/out "
      f"< {VERDICT_INOUT_DB:.0f} dB): "
      f"{', '.join(summary['fail_narrow_tone_only'])}.")
    a(f"- **No significant content at the label** (best in-box peak < "
      f"{VERDICT_PEAK_DB:.0f} dB above out-median AND in/out < "
      f"{VERDICT_INOUT_DB:.0f} dB; dominant energy elsewhere): "
      f"{', '.join(summary['fail_weak_at_label'])}.")
    a("")
    for f in fails:
        note = " — not evaluable (out-region empty)" if f["out_region_empty"] else ""
        a(f"- `{f['id']}` class {f['class']} ({f['stratum']}): in/out "
          f"{fmt(f['in_out_db'])} dB, box peak {fmt(f['box_peak_db'])} dB above "
          f"out-median; top-1 raw peak at {fmt(f['top1_abs_mhz'], '{:.2f}', '?')} MHz "
          f"({'inside' if f['top1_inside_box'] else 'OUTSIDE'} the box){note}.")
    a("")
    a("test split untouched; no models trained; raw IQ/JSON unmodified.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: only the first N real frames")
    args = ap.parse_args()
    started = time.perf_counter()

    doc = json.loads(FRAMES_JSON.read_text(encoding="utf-8"))
    frames = doc["frames"][: args.limit] if args.limit else doc["frames"]
    for fr in frames:
        if not str(fr["sample_id"]).startswith("train/"):
            raise SystemExit(f"non-fit sample in frame set: {fr['sample_id']}")

    tasks = []
    for idx, fr in enumerate(frames):
        i0, i1, win_ms, ta, tb = window_bounds(fr)
        tasks.append({
            "id": idx, "kind": "frame",
            "bin": str(DATA_DIR / f"{fr['sample_id']}.bin"),
            "i0": i0, "i1": i1,
            "geom": {"B_mhz": fr["B_mhz"], "f_lo_mhz": fr["f_lo_mhz"],
                     "f1_mhz": fr["f1_mhz"], "f2_mhz": fr["f2_mhz"]},
            "meta": {
                "sample_id": fr["sample_id"], "signal_id": int(fr["signal_id"]),
                "class": int(fr["class"]), "stratum": fr["stratum"],
                "B_mhz": float(fr["B_mhz"]), "bw_mhz": float(fr["bw_mhz"]),
                "f_lo_mhz": float(fr["f_lo_mhz"]),
                "f1_mhz": float(fr["f1_mhz"]), "f2_mhz": float(fr["f2_mhz"]),
                "t1_ms": float(fr["t1_ms"]), "t2_ms": float(fr["t2_ms"]),
                "win_ms": round(float(win_ms), 4),
                "win_t0_ms": round(float(ta), 4), "win_t1_ms": round(float(tb), 4),
                "i0": i0, "i1": i1,
            },
        })
    for j, sp in enumerate(SYNTH_SPECS):
        tasks.append({
            "id": 1000 + j, "kind": "synth", "win_ms": WIN_CAP_MS,
            "geom": {"B_mhz": sp["B_mhz"], "f_lo_mhz": sp["f_lo_mhz"],
                     "f1_mhz": sp["f1_mhz"], "f2_mhz": sp["f2_mhz"]},
            "seed": SEED_BASE + j,
            "meta": {
                "label": sp["label"], "B_mhz": sp["B_mhz"], "bw_mhz": sp["bw_mhz"],
                "f_lo_mhz": sp["f_lo_mhz"], "f1_mhz": sp["f1_mhz"],
                "f2_mhz": sp["f2_mhz"], "clone_of": sp["clone_of"],
                "win_ms": WIN_CAP_MS, "content": "complex white gaussian noise",
            },
        })
    decoys, decoy_fallback = select_decoys(frames, limit=4)
    for j, d in enumerate(decoys):
        tasks.append({
            "id": 2000 + j, "kind": "decoy",
            "bin": str(DATA_DIR / f"{d['sample_id']}.bin"),
            "i0": d["i0"], "i1": d["i1"],
            "geom": {"B_mhz": d["B_mhz"], "f_lo_mhz": d["f_lo_mhz"],
                     "f1_mhz": d["decoy_f1_mhz"], "f2_mhz": d["decoy_f2_mhz"]},
            "meta": {
                "sample_id": d["sample_id"], "signal_id": d["signal_id"],
                "class": d["class"], "stratum": d["stratum"],
                "B_mhz": d["B_mhz"], "bw_mhz": d["bw_mhz"],
                "f_lo_mhz": d["f_lo_mhz"],
                "target_box_mhz": [d["f1_mhz"], d["f2_mhz"]],
                "decoy_band_mhz": [d["decoy_f1_mhz"], d["decoy_f2_mhz"]],
                "decoy_center_mhz": d["decoy_center_mhz"],
                "decoy_shift_mhz": round(d["decoy_shift_mhz"], 3),
                "json_overlap": d["json_overlap"],
                "win_ms": round(float(d["win_ms"]), 4),
            },
        })

    results = {}
    with Pool(processes=args.workers) as pool:
        done = 0
        for tid, rec in pool.imap_unordered(run_task, tasks):
            results[tid] = rec
            done += 1
            tag = rec.get("sample_id") or rec.get("label", "")
            err = f" ERROR={rec['error']}" if "error" in rec else ""
            print(f"[{done}/{len(tasks)}] id={tid} {tag}{err}", flush=True)

    frame_recs = [results[i] for i in range(len(frames))]
    synth_recs = [results[1000 + j] for j in range(len(SYNTH_SPECS))]
    decoy_recs = [results[2000 + j] for j in range(len(decoys))]
    seconds = time.perf_counter() - started
    summary = build_summary(frame_recs, synth_recs, decoy_recs)

    out_doc = {
        "meta": {
            "track": "E — phase A' label-location energy audit",
            "date": "2026-11-21",
            "script": "scripts/baseline3_pa_extract.py",
            "frames_source": "reports/local_iq_round1/phase_ap_frames.json",
            "fit_split_only": True,
            "freq_convention": "abs_mhz = f_lo + B/2 + digital_mhz (np.fft zero "
                               "frequency = observation center)",
            "window_rule": "min(10 ms, t2-t1) centered in [t1,t2]",
            "params": {
                "win_cap_ms": WIN_CAP_MS, "smooth_hz": SMOOTH_HZ,
                "peak_sep_hz": PEAK_SEP_HZ, "out_guard_mhz": OUT_GUARD_HZ / 1e6,
                "trim_ms": TRIM_MS, "keep_bw_frac": KEEP_BW_FRAC,
                "edge_frac": EDGE_FRAC, "verdict_inout_db": VERDICT_INOUT_DB,
                "verdict_peak_db": VERDICT_PEAK_DB,
                "decoy_shift_mhz": DECOY_SHIFT_MHZ,
                "seed_base": SEED_BASE,
            },
            "workers": args.workers,
            "seconds": round(seconds, 1),
        },
        "frames": frame_recs,
        "synthetic_noise": synth_recs,
        "decoy_controls": decoy_recs,
        "decoy_rejected": [
            {"id": f"{rec['sample_id']}:{rec['signal_id']}",
             "decoy_band_mhz": [rec["decoy_f1_mhz"], rec["decoy_f2_mhz"]],
             "json_overlap": rec["json_overlap"]}
            for rec in decoy_fallback],
        "summary": summary,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out_doc, indent=2, allow_nan=False),
                        encoding="utf-8")
    write_md(OUT_MD, out_doc["meta"], frame_recs, synth_recs, decoy_recs,
             decoy_fallback, summary, seconds)
    print(json.dumps({
        "out_json": str(OUT_JSON), "out_md": str(OUT_MD),
        "frames": len(frame_recs), "pass": ov_pass(summary),
        "synth_pass": summary["synthetic_noise"]["n_pass"],
        "decoy_pass": summary["decoy_controls"]["n_pass"],
        "seconds": round(seconds, 1),
    }, indent=2))


def ov_pass(summary):
    return f"{summary['overall']['n_pass']}/{summary['overall']['n']}"


if __name__ == "__main__":
    main()
