#!/usr/bin/env python3
"""Phase B' OFDM frontend: faithful, validated IQ -> OFDM symbols chain.

Scope (task contract):
  - The dataset convention (audited in phase A'): np.fft bin 0 corresponds
    to the OBSERVATION CENTER f_lo + B/2; demod_to_center implements the
    correct shift f_target - (f_lo + fs/2).
  - Issue A fix: synthetic OFDM is generated at the TARGET bandwidth rate
    (20/40 MHz), resampled to the capture rate with exact rational
    resample_poly, and embedded at a legal in-band offset with the correct
    center-referenced up-conversion. No more "generate at capture rate"
    shortcuts.
  - Issue B fix: the CP detector accumulates the conjugate product over CP
    windows for EVERY phase residue mod T_sym, including windows that cross
    period boundaries (a cumsum over the raw product array, not a reshape
    that only tests non-wrapping starts). Symbol starts with any random
    phase are detectable with start-invariant metrics.
  - Numerology candidates are parameters; (Tu=3.2us, CP=0.8us) is the
    phase-A' primary candidate, never an assumption for real data: the
    winning candidate is chosen per frame by the label-blind scan.
  - Class labels are never used for per-instance synchronization or masks.

Reference OFDM layout (physical reference only, verified per-frame):
legacy-style 20 MHz -> 64-point FFT at 3.2us (SCS 312.5 kHz), 52 occupied
subcarriers (48 data + 4 pilots, nulls at DC and band edges); 40 MHz ->
128-point at 3.2us, 104 occupied (98 data + 6 occupied pilots; the ±53
pilot indices listed by legacy_masks lie outside the occupied set). The
masks are configurable and the real-data occupancy is measured, not
assumed.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from scripts.baseline3_dsp import (
    read_segment, demod_to_center, band_limit_fft, qam_symbols,
)

# Pre-registered candidate grids (phase A'), label-blind.
CP_TU_US = (3.2, 6.4, 12.8, 25.6, 51.2)
CP_CP_US = (0.2, 0.4, 0.8, 1.6, 3.2, 6.4)


# ---------------------------------------------------------------------------
# Synthetic OFDM generation at the TARGET rate + legal embedding
# ---------------------------------------------------------------------------

def legacy_masks(n_fft: int):
    """Legacy-style occupied/data/pilot bin indices for n_fft in {64,128}.

    Physical reference (802.11a-style): 64-FFT -> occupied bins -26..-1,
    +1..+26 (52 total, pilots at +-7, +-21); 128-FFT -> -52..-1, +1..+52
    (104 total, pilots at +-11, +-25, +-39, +-53). DC and guard bins null.
    """
    half = n_fft // 2
    if n_fft == 64:
        occ = np.concatenate([np.arange(-26, 0), np.arange(1, 27)])
        pilots = np.array([-21, -7, 7, 21])
    elif n_fft == 128:
        occ = np.concatenate([np.arange(-52, 0), np.arange(1, 53)])
        pilots = np.array([-53, -39, -25, -11, 11, 25, 39, 53])
    elif n_fft == 256:
        occ = np.concatenate([np.arange(-114, 0), np.arange(1, 115)])
        pilots = np.array([-103, -75, -39, -11, 11, 39, 75, 103])
    else:
        # generic: occupy 81% central, every 14th as pilot
        edge = int(round(half * 0.81))
        occ = np.concatenate([np.arange(-edge, 0), np.arange(1, edge + 1)])
        pilots = occ[::14]
    data = np.setdiff1d(occ, pilots)
    return occ, pilots, data


def gen_ofdm_baseband(bw_hz, order, n_sym, rng, tu_us=3.2, cp_us=0.8,
                       n_fft=None, pilots_are_qam=True):
    """Continuous CP-OFDM at the target rate fs=bw_hz. Returns (x, meta).

    Data subcarriers carry unit-power QAM; pilots carry BPSK-like +-1 by
    default or QAM if pilots_are_qam (physical reference only).
    """
    fs = bw_hz
    tu = int(round(tu_us * 1e-6 * fs))
    cp = int(round(cp_us * 1e-6 * fs))
    if n_fft is None:
        n_fft = tu
    if n_fft != tu:
        raise ValueError("phase B' assumes FFT length == T_u samples "
                         f"(got n_fft={n_fft}, tu={tu})")
    occ, pilots, data = legacy_masks(n_fft)
    x = np.zeros((n_fft, n_sym), dtype=np.complex128)
    for b in data:
        x[b] = qam_symbols(order, n_sym, rng)
    for b in pilots:
        if pilots_are_qam:
            x[b] = qam_symbols(order, n_sym, rng)
        else:
            x[b] = rng.choice([-1.0 + 0j, 1.0 + 0j], size=n_sym)
    syms = np.fft.ifft(x, axis=0) * math.sqrt(n_fft)
    wave = np.concatenate([syms[-cp:], syms], axis=0).T.reshape(-1)
    return wave.astype(np.complex64), {
        "fs_hz": fs, "tu": tu, "cp": cp, "n_fft": n_fft, "order": order,
        "occ": occ, "pilots": pilots, "data": data,
    }


def embed_capture(x_bb, fs_sig, fs_cap, offset_hz, rng,
                  start_samples=None, cfo_hz=0.0, snr_db=None,
                  n_cap=None, f16_roundtrip=True):
    """Place a target-rate OFDM baseband into a capture-rate observation.

    The baseband is resampled to the capture rate with exact rational
    resample_poly (anti-aliased), delayed by a random or given symbol-start
    offset, up-converted to `offset_hz` DIGITAL frequency (center-referenced
    convention: the offset is measured from the observation center), and
    mixed with band white noise at the requested in-band SNR if given.
    Float16 round-trip mirrors the real storage path.
    """
    frac = Fraction(fs_cap / fs_sig).limit_denominator(64)
    xr = resample_poly(x_bb, frac.numerator, frac.denominator)
    fs_true = fs_sig * frac.numerator / frac.denominator  # == fs_cap exactly
    if abs(fs_true - fs_cap) > 1e-6 * fs_cap:
        raise RuntimeError(f"resample ratio mismatch: {fs_true} vs {fs_cap}")
    if start_samples is None:
        start_samples = int(rng.integers(0, max(1, len(xr) // 8)))
    x_pad = np.zeros(len(xr) + start_samples, dtype=np.complex64)
    x_pad[start_samples:] = xr
    if n_cap is None:
        n_cap = len(x_pad) + int(0.5e-3 * fs_cap)
    x = np.zeros(n_cap, dtype=np.complex64)
    x[:len(x_pad)] = x_pad[:n_cap]
    t = np.arange(len(x), dtype=np.float64)
    if cfo_hz:
        x = x * np.exp(2j * np.pi * cfo_hz * t / fs_cap)
    x = x * np.exp(2j * np.pi * offset_hz * t / fs_cap)
    if snr_db is not None and snr_db < 200:
        # in-band SNR: band-flat noise relative to the signal's occupied-band
        # average power (signal occupies len(xr) of the capture samples)
        p_sig = float(np.mean(np.abs(x) ** 2))
        sig_band_frac = len(xr) / max(len(x), 1)
        p_sig_band = p_sig / max(sig_band_frac, 1e-9)
        p_noise_hz = p_sig_band / (10 ** (snr_db / 10.0))
        noise = (rng.standard_normal(len(x)) +
                 1j * rng.standard_normal(len(x))) * math.sqrt(p_noise_hz / 2)
        x = x + noise
    if f16_roundtrip:
        raw = np.empty(2 * len(x), dtype="<f2")
        raw[0::2] = x.real
        raw[1::2] = x.imag
        x = np.empty(len(x), dtype=np.complex64)
        x.real = raw[0::2]
        x.imag = raw[1::2]
    return x.astype(np.complex64), {"start_samples": start_samples,
                                    "offset_hz": offset_hz, "cfo_hz": cfo_hz,
                                    "fs_cap": fs_cap}


# ---------------------------------------------------------------------------
# Full-cycle complex CP detector (issue B fix)
# ---------------------------------------------------------------------------

def cp_scan_full(x, fs_hz, tu_us_candidates=CP_TU_US,
                 cp_us_candidates=CP_CP_US, min_sym=50):
    """Complex CP correlation over EVERY phase residue, no reshape gaps.

    c[n] = x[n] conj(x[n+T_u]) is coherently elevated inside CP windows for
    ANY symbol phase. We compute the CP window sum S[i] = sum_{m<cp} c[i+m]
    once over the whole array, then accumulate R(j) = sum_k S[j + k*T_sym]
    for EVERY residue j in [0, T_sym) -- including windows that straddle
    period boundaries (they are contiguous in time on the raw array).

    metric = max_j |R(j)| / (n_sym * cp * pbar), ratio = max/median |R(j)|,
    both label-blind; cfo from the phase of the best window.
    """
    n = len(x)
    pbar = float(np.mean(np.abs(x) ** 2)) if n else 0.0
    if pbar <= 0:
        return {"best": None, "table": []}
    table = []
    for tu_us in tu_us_candidates:
        tau = int(round(tu_us * 1e-6 * fs_hz))
        if abs(tau - tu_us * 1e-6 * fs_hz) > 1e-6 or tau <= 8 or n <= tau + 32:
            continue
        c = x[:-tau] * np.conj(x[tau:])
        nc = len(c)
        for cp_us in cp_us_candidates:
            cp = int(round(cp_us * 1e-6 * fs_hz))
            per = tau + cp
            if cp <= 0 or nc <= per * min_sym + cp:
                continue
            # CP window sums over the RAW product array: S[i] covers
            # [i, i+cp) so windows that straddle period boundaries are
            # included (the phase-A' reshape missed those residues).
            cs = np.concatenate([[0.0 + 0j], np.cumsum(c)])
            S = cs[cp:] - cs[:-cp]           # S[i] = sum_{m<cp} c[i+m]
            # equal window count for EVERY residue: the binding residue is
            # per-1, so m_common windows fit for all residues at once.
            m = (nc - cp - per) // per + 1
            if m < min_sym:
                continue
            R = np.zeros(per, dtype=np.complex128)
            for k in range(m):
                R += S[k * per:(k + 1) * per].astype(np.complex128)
            mag = np.abs(R)
            jbest = int(np.argmax(mag))
            med = float(np.median(mag))
            metric = float(mag[jbest] / (m * cp * pbar))
            ratio = float(mag[jbest] / max(med, 1e-30))
            cfo = float(-np.angle(R[jbest]) / (2.0 * math.pi * tau))
            table.append({"tu_us": tu_us, "cp_us": cp_us, "tau": tau,
                          "cp": cp, "metric": metric, "ratio": ratio,
                          "phi": jbest, "cfo_cycles": cfo, "n_sym": m})
    if not table:
        return {"best": None, "table": []}
    best = max(table, key=lambda t: t["metric"])
    return {"best": best, "table": table}


def strip_symbols(x, fs_hz, tu_us, cp_us, phi, cfo_cycles=0.0, max_sym=None):
    """Strip CP at the detected residue and FFT each symbol.

    Returns [n_sym, n_fft] with n_fft = Tu samples at the capture rate
    (subcarrier spacing = 1/Tu_us, independent of B). Symbols whose CP
    window or FFT window would leave the array are dropped.
    """
    tau = int(round(tu_us * 1e-6 * fs_hz))
    cp = int(round(cp_us * 1e-6 * fs_hz))
    per = tau + cp
    if cfo_cycles:
        n = np.arange(len(x), dtype=np.float64)
        x = (x * np.exp(1j * 2.0 * math.pi * cfo_cycles * n)).astype(np.complex64)
    # phi is the CP-window-start residue mod per; first usable start >= 0
    s0 = phi % per
    starts = np.arange(s0, len(x) - per + 1, per, dtype=np.int64)
    rows = []
    for s in starts:
        a, b = s + cp, s + cp + tau
        if a < 0 or b > len(x):
            continue
        rows.append(np.fft.fft(x[a:b]))
        if max_sym and len(rows) >= max_sym:
            break
    if not rows:
        return np.zeros((0, tau), dtype=np.complex64)
    return np.stack(rows).astype(np.complex64)


def filter_symbol_rows(X, bins, energy_frac=0.5):
    """Drop symbol rows whose in-band energy is below energy_frac * median.

    Pre-onset padding, post-signal tails and transient rows carry no
    modulation content (their residual numerical noise is Rayleigh-like,
    m4 ~ 2) and would bias pooled statistics. Label-blind: energy only.
    Returns (X_filtered, keep_idx, n_kept, n_dropped); keep_idx holds the
    ORIGINAL row indices of the kept symbols so that downstream phase
    differences can be restricted to truly consecutive symbols.
    """
    if X.shape[0] == 0:
        return X, np.zeros(0, dtype=np.int64), 0, 0
    e = np.mean(np.abs(X[:, bins]) ** 2, axis=1)
    med = float(np.median(e[e > 0])) if np.any(e > 0) else 0.0
    keep = e >= energy_frac * med
    return X[keep], np.where(keep)[0], int(keep.sum()), int((~keep).sum())


# ---------------------------------------------------------------------------
# Real-frame helpers
# ---------------------------------------------------------------------------

def extract_band(bin_path, f_lo, B_hz, f1, f2, t1_ms, t2_ms, cap_ms=None,
                 margin_frac=0.02, trim_ms=0.5):
    """read -> demod -> band-limit -> trim, the fixed phase-B' recipe."""
    fs = B_hz
    c = t2_ms - t1_ms
    if cap_ms is not None:
        c = min(cap_ms, c)
    ta, tb = (t1_ms + t2_ms) / 2 - c / 2, (t1_ms + t2_ms) / 2 + c / 2
    i0 = max(0, int(round(ta / 1000.0 * fs)))
    i1 = int(round(min(tb, t2_ms) / 1000.0 * fs))
    x = read_segment(bin_path, i0, i1)
    fc = (f1 + f2) / 2 * 1e6
    xb = demod_to_center(x, fs, fc, f_lo * 1e6)
    bw = (f2 - f1) * 1e6
    xb = band_limit_fft(xb, fs, bw * (1.0 + margin_frac), edge_frac=0.05)
    if trim_ms:
        k = int(trim_ms / 1000.0 * fs)
        if len(xb) > 2 * k:
            xb = xb[k:-k]
    return xb, fs, bw
