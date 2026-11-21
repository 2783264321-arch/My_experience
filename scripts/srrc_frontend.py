#!/usr/bin/env python3
"""SRRC 10/11 local-IQ front-end: GT crop -> matched filter/symbol timing.

All sync parameters are label-blind: fixed SRRC alpha/R search grid,
tone-line gate, no class-dependent thresholds. The same crop is used for
the unsynchronized baseline features and the recovered-symbol features.

This module is used by scripts/baseline3srrc_local_iq.py and the unit
tests. It does not train models and does not touch val/test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from scripts.baseline3_dsp import (
    read_segment, demod_to_center, band_limit_fft, srrc_pulse_taps,
    cubic_sample, tone_line_scan,
)

BAND_HZ = 10e6
BAND_LIMIT_HZ = BAND_HZ * 1.02
CAP_MS = 2.0
TRIM_MS = 0.5
R_GUESS_HZ = BAND_HZ / 1.2       # 8.333 MHz, prior Phase A' alpha=0.2
SCAN_LO = 0.80                   # covers alpha ~0.15..0.45
SCAN_HI = 1.10
GATE_STRENGTH = 0.04             # synthetic/real separation from Phase A'
GATE_PEAK_MED = 8.0
GATE_COHERENCE = 0.50
MIN_SYMBOLS = 256


def _tone_refine_range(y_sq: np.ndarray, fs: float, r_guess: float,
                       scan_lo: float = SCAN_LO,
                       scan_hi: float = SCAN_HI) -> Optional[dict]:
    """tone_line_scan + direct-correlation refinement over a scan range."""
    n = len(y_sq)
    if n < 64 or r_guess <= 0:
        return None
    f_c, s_c, _ = tone_line_scan(y_sq, fs, scan_lo * r_guess,
                                 scan_hi * r_guess)
    if f_c <= 0:
        return None
    pad = int(2 ** math.ceil(math.log2(n)))
    binw = fs / pad
    se = np.abs(np.fft.rfft(y_sq, n=pad))
    k0 = int(round(f_c / binw))
    rad = max(10, k0 // 4)
    floor = max(float(np.median(se[max(0, k0 - rad):min(len(se), k0 + rad + 1)])),
                1e-30)
    m = np.arange(n, dtype=np.float64)
    grid = f_c + np.linspace(-1.5, 1.5, 25) * binw
    grid = grid[grid > 0.5 * binw]
    D = np.empty(len(grid), dtype=np.complex128)
    for i, ff in enumerate(grid):
        D[i] = np.sum(y_sq * np.exp(-2j * np.pi * ff * m / fs))
    j = int(np.argmax(np.abs(D)))
    f_ref = float(grid[j])
    if 0 < j < len(grid) - 1:
        a0, a1, a2 = np.abs(D[j - 1]) ** 2, np.abs(D[j]) ** 2, np.abs(D[j + 1]) ** 2
        den = a0 - 2 * a1 + a2
        if den != 0:
            step = grid[1] - grid[0]
            f_ref = float(np.clip(grid[j] + 0.5 * (a0 - a2) / den * step,
                                  grid[0], grid[-1]))
    D_ref = complex(np.sum(y_sq * np.exp(-2j * np.pi * f_ref * m / fs)))
    strength = abs(D_ref) / floor / math.sqrt(n)
    sps_ref = fs / f_ref
    t0 = (-np.angle(D_ref) / (2.0 * np.pi) * sps_ref) % sps_ref
    h = n // 2
    D1 = np.sum(y_sq[:h] * np.exp(-2j * np.pi * f_ref * m[:h] / fs))
    D2 = np.sum(y_sq[h:] * np.exp(-2j * np.pi * f_ref * m[h:] / fs))
    coh = abs(D1 + D2) / max(abs(D1) + abs(D2), 1e-30)
    return {"f_hz": float(f_ref), "coarse_hz": float(f_c),
            "coarse_strength": float(s_c), "strength": float(strength),
            "peak_med": float(abs(D_ref) / floor), "coherence": float(coh),
            "t0": float(t0), "t0_frac": float(t0 / sps_ref)}


def crop_box(record, sig, cap_ms: float = CAP_MS,
             trim_ms: float = TRIM_MS) -> dict:
    """GT-box central crop, down-converted to the observation center."""
    fs = float(record.B_mhz) * 1e6
    f1 = float(sig["start_frequency"]); f2 = float(sig["end_frequency"])
    t1 = float(sig["start_time"]); t2 = float(sig["end_time"])
    if t2 - t1 < 0.2:
        return {"status": "crop_too_short", "reason": "duration<0.2ms"}
    tc = 0.5 * (t1 + t2)
    cap = min(t2 - t1, cap_ms)
    ta = max(0.0, tc - 0.5 * cap)
    tb = min(float(record.T_ms), tc + 0.5 * cap)
    i0 = max(0, int(round(ta / 1000.0 * fs)))
    i1 = min(int(record.N), int(round(tb / 1000.0 * fs)))
    if i1 - i0 < 256:
        return {"status": "crop_too_short", "reason": "samples<256"}
    x = read_segment(str(record.bin_path), i0, i1)
    fc = 0.5 * (f1 + f2) * 1e6
    xd = demod_to_center(x, fs, fc, float(record.f_lo) * 1e6)
    xb = band_limit_fft(xd, fs, BAND_LIMIT_HZ, edge_frac=0.05)
    nt = int(round(trim_ms * 1e-3 * fs))
    if len(xb) > 2 * nt + 256:
        xb = xb[nt:len(xb) - nt]
    if len(xb) < 256:
        return {"status": "crop_too_short", "reason": "after_trim<256"}
    return {"status": "ok", "x": xb, "fs": fs, "fc": fc,
            "i0": i0, "i1": i1, "dur_ms": t2 - t1,
            "crop_ms": (i1 - i0) / fs * 1000.0}


def sync_srrc(x: np.ndarray, fs: float,
              r_guess: float = R_GUESS_HZ) -> dict:
    """Label-blind SRRC matched filter + symbol timing."""
    if len(x) < MIN_SYMBOLS:
        return {"status": "too_short"}
    taps = srrc_pulse_taps(0.2, fs / r_guess)
    y = np.convolve(x, taps, mode="same")
    t = _tone_refine_range(np.abs(y) ** 2, fs, r_guess)
    if t is None:
        return {"status": "no_tone"}
    r1 = t["f_hz"]
    alpha1 = float(np.clip(BAND_HZ / r1 - 1.0, 0.05, 0.60))
    if not (0.05 <= alpha1 <= 0.60):
        return {"status": "bad_rate", "tone": t}
    taps = srrc_pulse_taps(alpha1, fs / r1)
    y = np.convolve(x, taps, mode="same")
    t2 = _tone_refine_range(np.abs(y) ** 2, fs, r1,
                            scan_lo=0.99, scan_hi=1.01)
    if t2 is not None and abs(t2["f_hz"] - r1) < 0.005 * r1:
        t = t2
        r1 = t["f_hz"]
        alpha1 = float(np.clip(BAND_HZ / r1 - 1.0, 0.05, 0.60))
        taps = srrc_pulse_taps(alpha1, fs / r1)
        y = np.convolve(x, taps, mode="same")

    if t["strength"] < GATE_STRENGTH:
        return {"status": "gate_strength", "tone": t,
                "R_mhz": r1 / 1e6, "alpha": alpha1}
    if t["peak_med"] < GATE_PEAK_MED:
        return {"status": "gate_peak", "tone": t,
                "R_mhz": r1 / 1e6, "alpha": alpha1}
    if t["coherence"] < GATE_COHERENCE:
        return {"status": "gate_coherence", "tone": t,
                "R_mhz": r1 / 1e6, "alpha": alpha1}

    sps = fs / r1
    guard = (len(taps) - 1) // 2 + 2
    n = len(y)
    if n <= guard + 8:
        return {"status": "too_short", "tone": t}
    k_max = int(math.floor((n - 3 - guard - t["t0"]) / sps))
    pos = t["t0"] + sps * np.arange(max(k_max, 0), dtype=np.float64)
    pos = pos[(pos >= guard) & (pos <= n - 3 - guard)]
    syms = cubic_sample(y, pos)
    if len(syms) < MIN_SYMBOLS:
        return {"status": "too_few_symbols", "tone": t, "n_sym": len(syms)}
    p2 = float(np.mean(np.abs(syms) ** 2))
    if not np.isfinite(p2) or p2 <= 0:
        return {"status": "zero_energy", "tone": t}
    return {"status": "ok", "symbols": syms, "indices": np.arange(len(syms)),
            "R_mhz": r1 / 1e6, "alpha": alpha1, "sps": sps,
            "n_sym": len(syms), "tone": t}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _hist01(v: np.ndarray, bins: int, lo: float, hi: float) -> np.ndarray:
    h, _ = np.histogram(v, bins=np.linspace(lo, hi, bins + 1))
    return (h / max(len(v), 1)).astype(np.float64)


def baseline_features(x: np.ndarray, fs: float, bw: float = BAND_HZ) -> dict:
    """Local power + unsynchronized IQ statistics (no symbol timing)."""
    n = len(x)
    p = float(np.mean(np.abs(x) ** 2))
    rms = math.sqrt(max(p, 1e-30))
    spec = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
    edges = np.linspace(-bw / 2, bw / 2, 17)
    band = np.array([
        np.mean(spec[(freqs >= edges[i]) & (freqs < edges[i + 1])])
        if np.any((freqs >= edges[i]) & (freqs < edges[i + 1])) else 0.0
        for i in range(16)], float)
    band = band / max(band.sum(), 1e-30)
    segs = np.array_split(x, 8)
    segp = np.array([np.mean(np.abs(s) ** 2) for s in segs], float)
    segp = segp / max(segp.sum(), 1e-30)
    m4 = float(np.mean(np.abs(x) ** 4) / max(p * p, 1e-30))
    papr = float(np.max(np.abs(x) ** 2) / max(p, 1e-30))
    ps = spec[spec > 0]
    ps = ps / max(ps.sum(), 1e-30)
    entropy = float(-np.sum(ps * np.log(ps)) / math.log(max(len(ps), 2)))
    flatness = float(np.exp(np.mean(np.log(ps))) / max(np.mean(ps), 1e-30))
    amp_hist = _hist01(np.abs(x) / max(rms, 1e-30), 12, 0.0, 3.0)
    dphi = np.angle(x[1:] * np.conj(x[:-1])) if len(x) > 1 else np.zeros(1)
    ph_hist = _hist01(dphi, 12, -np.pi, np.pi)
    names = (["log_power"] +
             [f"band{i}" for i in range(16)] +
             [f"seg{i}" for i in range(8)] +
             ["time_m4", "papr", "spec_entropy", "spec_flatness"] +
             [f"amp_hist{i}" for i in range(12)] +
             [f"unsync_phase_hist{i}" for i in range(12)])
    feat = np.concatenate(([math.log10(max(p, 1e-30))], band, segp,
                           [m4, papr, entropy, flatness], amp_hist, ph_hist))
    return {"features": feat.astype(np.float64), "names": names,
            "log_power": math.log10(max(p, 1e-30))}


def amplitude_features(syms: np.ndarray) -> dict:
    z = np.abs(syms).astype(np.float64)
    z = z / max(math.sqrt(np.mean(z ** 2)), 1e-30)
    hist = _hist01(z, 16, 0.0, 3.0)
    m4 = float(np.mean(z ** 4) / max(np.mean(z ** 2) ** 2, 1e-30))
    m6 = float(np.mean(z ** 6) / max(np.mean(z ** 2) ** 3, 1e-30))
    papr = float(np.max(z ** 2) / max(np.mean(z ** 2), 1e-30))
    qs = np.quantile(z, [0.05, 0.25, 0.50, 0.75, 0.95])
    names = (["amp_mean", "amp_std", "amp_m4", "amp_m6", "amp_papr"] +
             [f"amp_q{int(q*100):02d}" for q in (0.05, 0.25, 0.50, 0.75, 0.95)] +
             [f"amp_hist{i}" for i in range(16)])
    feat = np.concatenate(([z.mean(), z.std(), m4, m6, papr], qs, hist))
    return {"features": feat.astype(np.float64), "names": names,
            "m4": m4, "m6": m6, "n_sym": len(z)}


def phase_diff_features(syms: np.ndarray,
                        indices: Optional[np.ndarray] = None) -> dict:
    """Adjacent-symbol phase difference; only truly consecutive symbols.

    indices[k] is the original symbol index; a pair is used only when
    indices[k+1] == indices[k]+1.
    """
    if indices is None:
        indices = np.arange(len(syms))
    indices = np.asarray(indices, int)
    if len(syms) < 2:
        d = np.zeros(0, float)
        n_pairs = 0
    else:
        mask = indices[1:] == indices[:-1] + 1
        d = np.angle(syms[1:][mask] * np.conj(syms[:-1][mask]))
        n_pairs = int(mask.sum())
    if len(d) == 0:
        hist = np.zeros(16, float)
        mu = 0.0; resultant = 0.0; mean_abs = 0.0
    else:
        mu = float(np.angle(np.mean(np.exp(1j * d))))
        dd = np.angle(np.exp(1j * (d - mu)))
        hist = _hist01(dd, 16, -np.pi, np.pi)
        resultant = float(np.abs(np.mean(np.exp(1j * dd))))
        mean_abs = float(np.mean(np.abs(dd)))
    names = (["phase_resultant", "phase_mean_abs"] +
             [f"phase_hist{i}" for i in range(16)])
    feat = np.concatenate(([resultant, mean_abs], hist))
    return {"features": feat.astype(np.float64), "names": names,
            "n_pairs": n_pairs, "n_consecutive": int(len(d)),
            "mu": mu, "resultant": resultant}


def extract_instance(record, sig) -> dict:
    """Crop + baseline + sync + features for one instance."""
    base = {"sample_id": record.sample_id,
            "signal_id": int(sig["signal_id"]),
            "class": int(sig["class"]),
            "bandwidth_mhz": float(sig["end_frequency"]) - float(sig["start_frequency"]),
            "duration_ms": float(sig["end_time"]) - float(sig["start_time"])}
    crop = crop_box(record, sig)
    if crop["status"] != "ok":
        return {**base, "status": crop["status"],
                "reason": crop.get("reason", crop["status"]),
                "baseline": None, "amp": None, "phase": None}
    x = crop["x"]; fs = crop["fs"]
    b = baseline_features(x, fs)
    s = sync_srrc(x, fs)
    out = {**base, "status": s["status"],
           "reason": s.get("status", "ok"),
           "crop_ms": crop["crop_ms"],
           "baseline": b,
           "sync": {k: v for k, v in s.items()
                    if k not in ("symbols", "indices")},
           "amp": None, "phase": None}
    if s["status"] == "ok":
        out["amp"] = amplitude_features(s["symbols"])
        out["phase"] = phase_diff_features(s["symbols"], s["indices"])
        out["R_mhz"] = s["R_mhz"]
        out["alpha"] = s["alpha"]
        out["n_sym"] = s["n_sym"]
        out["tone"] = s["tone"]
    return out
