#!/usr/bin/env python3
"""Shared DSP helpers for Baseline-3 round 1 (local raw IQ modulation stats).

Everything in this module is class-label blind. The only per-frame inputs are
physical geometry (GT box + margin, capture rate Fs=B) and pre-registered
candidate grids. All statistics are phase-invariant (magnitude moments).

Conventions
  - IQ: float16 interleaved little-endian I/Q, Fs = B MHz, N = B*T samples.
  - Baseband: x[n] * exp(-j2*pi*(f_center - f_lo)*n/Fs); band-limited by an
    FFT mask to the GT box band.
  - WiFi candidates (pre-registered): SCS 78.125 kHz, T_u in {12.8 us} with
    fallbacks {3.2, 6.4, 25.6} us, CP in {0.8, 1.6, 3.2, 6.4} us. All lags are
    integer samples for every B in the manifest (multiples of 5 MHz).
  - SRRC: symbol rate found by a spectral-line scan of |y|^2 inside
    [bw/2, bw] MHz (equivalently alpha in (0, 1)); no class information.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

SCS_KHZ = 78.125
WIFI_TU_US = (12.8, 3.2, 6.4, 25.6)
WIFI_CP_US = (0.8, 1.6, 3.2, 6.4)
SRRC_SPAN_SYMS = 6.0


# ---------------------------------------------------------------------------
# IQ access and baseband conditioning
# ---------------------------------------------------------------------------

def read_segment(bin_path, i0, i1):
    """Read complex64 IQ [i0, i1) from an interleaved float16 .bin file."""
    n = max(0, i1 - i0)
    raw = np.fromfile(bin_path, dtype="<f2", count=2 * n, offset=4 * i0)
    m = len(raw) // 2
    iq = np.empty(m, dtype=np.complex64)
    iq.real = raw[0::2][:m]
    iq.imag = raw[1::2][:m]
    return iq


def demod_to_center(x, fs_hz, f_center_hz, f_lo_hz):
    """Frequency-shift so that f_center sits at DC.

    Dataset frequency convention (audited 2026-11-21, phase A' step 1):
    the complex .bin IQ is CENTER-referenced -- FFT bin 0 (zero digital
    frequency) corresponds to the observation center (f_lo + B/2), where
    Fs = B. Evidence: raw-FFT landmarks match JSON boxes only under this
    mapping, and the production STFT label mapping (data.py _label_boxes /
    infer.py gt_global) is algebraically consistent only with it.

    The required down-conversion is therefore
        shift = f_center - (f_lo + fs/2),   fs = B.
    The old formula (f_center - f_lo) was wrong by half the observation
    bandwidth and invalidated the round-1 raw-IQ census.
    """
    df = 2.0 * np.pi * (f_center_hz - f_lo_hz - fs_hz / 2.0) / fs_hz
    n = np.arange(len(x), dtype=np.float64)
    return (x * np.exp(-1j * df * n)).astype(np.complex64)


def band_limit_fft(x, fs_hz, keep_bw_hz, edge_frac=0.02):
    """FFT mask to +-keep_bw/2 with a small raised-cosine transition."""
    n = len(x)
    spec = np.fft.fft(x)
    freq = np.fft.fftfreq(n, d=1.0 / fs_hz)
    half = keep_bw_hz / 2.0
    edge = max(edge_frac * keep_bw_hz, fs_hz / max(n, 1) * 8)
    m = np.ones(n, dtype=np.float64)
    m[np.abs(freq) > half + edge] = 0.0
    tr = (np.abs(freq) >= half - edge) & (np.abs(freq) <= half + edge)
    m[tr] = 0.5 * (1.0 + np.cos(np.pi * (np.abs(freq)[tr] - half + edge) / (2.0 * edge)))
    return np.fft.ifft(spec * m).astype(np.complex64)


def band_snr_db(x, fs_hz, keep_bw_hz, guard_frac=0.12):
    """Rough in-band vs out-of-band power SNR (scale-free, label-blind)."""
    n = len(x)
    spec = np.fft.fft(x)
    p = np.abs(spec) ** 2
    freq = np.fft.fftfreq(n, d=1.0 / fs_hz)
    half = keep_bw_hz / 2.0
    inb = np.abs(freq) <= half * (1.0 - guard_frac)
    out = np.abs(freq) >= half * (1.0 + guard_frac)
    if not inb.any() or not out.any():
        return -np.inf, 0.0, 0.0
    p_in = float(np.median(p[inb]))
    p_out = float(np.median(p[out]))
    if p_out <= 0:
        return 60.0, p_in, p_out
    snr = 10.0 * math.log10(max(p_in, 1e-30) / max(p_out, 1e-30))
    return snr, p_in, p_out


# ---------------------------------------------------------------------------
# Constellations and magnitude moments
# ---------------------------------------------------------------------------

def qam_symbols(order, count, rng):
    """Unit-average-power M-QAM symbols, shape [count]."""
    if order == 4:
        s = (np.arange(count) % 2) * 2 - 1
        t = (rng.integers(0, 2, count) * 2 - 1)
        u = rng.integers(0, 2, count) * 2 - 1
        return (t.astype(np.float32) + 1j * u.astype(np.float32)) / math.sqrt(2.0)
    if order == 16:
        lv = np.array([-3, -1, 1, 3], dtype=np.float32)
    elif order == 64:
        lv = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float32)
    else:
        raise ValueError(order)
    k = int(round(math.sqrt(order)))
    i = lv[rng.integers(0, k, count)]
    q = lv[rng.integers(0, k, count)]
    return (i + 1j * q) / math.sqrt(float((lv ** 2).mean()) * 2.0)


def mag_moments(mag):
    """(m4, m6) of magnitudes, scale-invariant."""
    mag = np.asarray(mag, dtype=np.float64)
    p2 = float(np.mean(mag ** 2))
    if p2 <= 0:
        return float("nan"), float("nan")
    m4 = float(np.mean(mag ** 4)) / (p2 * p2)
    m6 = float(np.mean(mag ** 6)) / (p2 ** 3)
    return m4, m6


def mag_histogram(mag, bins=None):
    """Normalized |s|/RMS histogram on a fixed grid (label-blind)."""
    if bins is None:
        bins = np.linspace(0.0, 3.0, 9)
    mag = np.asarray(mag, dtype=np.float64)
    p2 = float(np.mean(mag ** 2)) if len(mag) else 1.0
    z = mag / math.sqrt(max(p2, 1e-30))
    h, _ = np.histogram(z, bins=bins)
    return (h / max(len(z), 1)).astype(np.float64)


# ---------------------------------------------------------------------------
# WiFi OFDM: CP comb detection, alignment, subcarrier statistics
# ---------------------------------------------------------------------------

def comb_metric(x, tau, period):
    """DEPRECATED (phase A' audit 2026-11-21): magnitude-only comb metric.
    It discards the CP phase coherence and is ~3 orders of magnitude less
    sensitive than the standard complex accumulation in cp_scan_complex.
    Kept only so the round-1 census remains reproducible; do not use for
    new structure checks.

    Returns (strength, bump_center_residue, complex comb D).
    strength = |sum m[n] e^{-2*pi*i n/period}| / sum m[n] with
    m[n] = |x[n] conj(x[n+tau])|. CP bumps of width cp sit at one residue.
    """
    if tau <= 0 or period <= tau or len(x) <= tau + period:
        return 0.0, -1.0, 0.0 + 0.0j
    c = x[:-tau] * np.conj(x[tau:])
    m = np.abs(c)
    s = float(m.sum())
    if s <= 0:
        return 0.0, -1.0, 0.0 + 0.0j
    n = np.arange(len(m), dtype=np.float64)
    ph = np.exp(-2j * np.pi * n / float(period))
    D = complex((m * ph).sum())
    strength = abs(D) / s
    bump = (-np.angle(D) / (2.0 * np.pi) * period) % period
    return float(strength), float(bump), D


def cp_cfo_estimate(x, tau, period, bump, cp):
    """CFO (cycles/sample) from the CP-region correlation phase."""
    n = np.arange(len(x) - tau, dtype=np.int64)
    residue = (n - round(bump - cp / 2.0)) % period
    mask = residue < cp
    if not mask.any():
        return 0.0
    c = (x[:-tau] * np.conj(x[tau:]))[mask]
    acc = complex(np.sum(c))
    if abs(acc) <= 0:
        return 0.0
    return float(-np.angle(acc) / (2.0 * np.pi * tau))


def wifi_candidates(fs_hz, bw_hz):
    """Integer-sample (tau, cp, period) candidates at the capture rate."""
    out = []
    for tu_us in WIFI_TU_US:
        tau = int(round(tu_us * 1e-6 * fs_hz))
        if abs(tau - tu_us * 1e-6 * fs_hz) > 1e-6:
            continue
        for cp_us in WIFI_CP_US:
            cp = int(round(cp_us * 1e-6 * fs_hz))
            if cp <= 0:
                continue
            out.append((tau, cp, tau + cp, tu_us, cp_us))
    return out


def strip_fft_symbols(x, starts, cp, tau, max_sym):
    """Stack CP-stripped symbols (FFT per symbol), shape [n_sym, tau]."""
    rows = []
    for s in starts:
        a, b = s + cp, s + cp + tau
        if a < 0 or b > len(x):
            continue
        rows.append(np.fft.fft(x[a:b]))
        if len(rows) >= max_sym:
            break
    if not rows:
        return np.zeros((0, tau), dtype=np.complex64)
    return np.stack(rows).astype(np.complex64)


def subcarrier_stats(X, bw_hz, fs_hz):
    """Per-subcarrier and pooled magnitude statistics (label-blind).

    X: [n_sym, tau] with rows = FFT of CP-stripped symbols (bin spacing =
    fs/tau which equals SCS for the standard candidates).
    """
    n_sym, tau = X.shape
    if n_sym == 0:
        return None
    freq = np.fft.fftfreq(tau, d=1.0 / fs_hz)
    half = bw_hz / 2.0
    # central data region: 78% of the band, away from unknown edge/DC nulls
    central = np.abs(freq) <= 0.39 * bw_hz
    # noise region: beyond the signal band
    noise = np.abs(freq) >= 0.60 * bw_hz
    p = np.abs(X) ** 2
    p_noise = float(np.median(p[:, noise])) if noise.any() else 0.0
    p_in = p[:, central]
    strong = p_in.mean(axis=0) >= max(4.0 * p_noise, 1e-30)
    mag = np.abs(X[:, central])
    mags = mag[:, strong] if strong.any() else mag
    if mags.size == 0:
        return None
    # normalize per subcarrier, then pool
    rms = np.sqrt(np.mean(mags ** 2, axis=0))
    z = mags / np.maximum(rms, 1e-30)
    m4, m6 = mag_moments(z.reshape(-1))
    per_sc_m4 = []
    for j in range(mags.shape[1]):
        mm, _ = mag_moments(mags[:, j])
        per_sc_m4.append(mm)
    per_sc_m4 = np.asarray(per_sc_m4, dtype=np.float64)
    snr_sc = 10.0 * math.log10(max(float(np.mean(p_in[:, strong].mean(axis=0))), 1e-30) /
                               max(p_noise, 1e-30)) if strong.any() else -np.inf
    return {
        "m4_pool": m4, "m6_pool": m6,
        "per_sc_m4_median": float(np.median(per_sc_m4)) if len(per_sc_m4) else float("nan"),
        "per_sc_m4_hist": np.histogram(per_sc_m4, bins=np.linspace(0.6, 2.6, 9))[0] /
                          max(len(per_sc_m4), 1),
        "frac_sc_m4_below_1p15": float(np.mean(per_sc_m4 < 1.15)) if len(per_sc_m4) else float("nan"),
        "n_sym": int(n_sym), "n_strong_sc": int(strong.sum()),
        "snr_db_proxy": float(snr_sc),
    }


def wifi_sync(x, fs_hz, bw_hz, max_sym=600):
    """Label-blind WiFi sync: pick the best (T_u, CP) candidate by comb metric.

    Returns a dict with chosen params, quality metrics and per-symbol FFTs.
    """
    best = None
    for tau, cp, period, tu_us, cp_us in wifi_candidates(fs_hz, bw_hz):
        strength, bump, D = comb_metric(x, tau, period)
        if best is None or strength > best[0]:
            best = (strength, bump, D, tau, cp, period, tu_us, cp_us)
    (strength, bump, D, tau, cp, period, tu_us, cp_us) = best
    cfo = cp_cfo_estimate(x, tau, period, bump, cp)
    # CFO correction (usually ~0 in this dataset; still applied uniformly)
    if abs(cfo) > 0:
        n = np.arange(len(x), dtype=np.float64)
        x = (x * np.exp(1j * 2.0 * np.pi * cfo * n)).astype(np.complex64)
    s0 = int(round(bump - cp / 2.0)) % period
    starts = np.arange(s0, len(x) - period + 1, period, dtype=np.int64)
    X = strip_fft_symbols(x, starts, cp, tau, max_sym)
    return {
        "strength": float(strength), "bump": float(bump),
        "tau": int(tau), "cp": int(cp), "period": int(period),
        "tu_us": float(tu_us), "cp_us": float(cp_us),
        "cfo_cycles_per_sample": float(cfo),
        "cfo_subcarriers": float(cfo * tau),
        "n_sym": int(X.shape[0]), "X": X,
    }


# ---------------------------------------------------------------------------
# Standard complex cyclic-prefix detector (phase A')
# ---------------------------------------------------------------------------

CP_TU_US = (3.2, 6.4, 12.8, 25.6, 51.2)
CP_CP_US = (0.2, 0.4, 0.8, 1.6, 3.2, 6.4)


def cp_scan_complex(x, fs_hz, tu_us_candidates=CP_TU_US,
                    cp_us_candidates=CP_CP_US, min_sym=50):
    """Complex CP correlation detector over a pre-registered candidate grid.

    For candidate (T_u, T_sym=T_u+CP): the conjugate product
    c[n] = x[n] * conj(x[n + T_u]) is coherently elevated only inside CP
    windows, with a constant phase set by any CFO. We accumulate the
    COMPLEX sum over the CP window for every phase residue of T_sym and
    take the maximum magnitude, normalized so that a clean CP-OFDM whose
    in-window product matches the average power scores ~1 and noise scores
    ~1/sqrt(n_sym * CP):

        R(phi) = sum_k sum_{m<CP} c[phi + m + k*T_sym]
        metric = max_phi |R(phi)| / (n_sym * CP * pbar),  pbar = mean|x|^2

    Returns dict(best=(metric, tu_us, cp_us, phi, cfo), table=[...]) with
    entries for every candidate whose lags are integer at fs_hz.
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
        for cp_us in cp_us_candidates:
            cp = int(round(cp_us * 1e-6 * fs_hz))
            per = tau + cp
            if cp <= 0 or n <= per * min_sym:
                continue
            c = x[:-tau] * np.conj(x[tau:])
            m = (len(c) - cp) // per
            if m < min_sym:
                continue
            ctrim = c[: m * per + cp - 1]
            # M[k, j] = c[k*per + j]
            M = ctrim[: m * per].reshape(m, per)
            # window sums of length cp starting at phase j
            Cs = np.concatenate([np.zeros((m, 1), dtype=M.dtype),
                                 np.cumsum(M, axis=1)], axis=1)
            W = Cs[:, cp:] - Cs[:, :-cp]          # W[k, j], j in [0, per-cp]
            R = W.sum(axis=0)                      # complex R(phi)
            j = int(np.argmax(np.abs(R)))
            metric = float(np.abs(R[j]) / (m * cp * pbar))
            medR = float(np.median(np.abs(R)))
            ratio = float(np.abs(R[j]) / max(medR, 1e-30))
            cfo = float(-np.angle(R[j]) / (2.0 * np.pi * tau))  # cycles/sample
            table.append({"tu_us": tu_us, "cp_us": cp_us, "tau": tau,
                          "cp": cp, "metric": metric, "ratio": ratio,
                          "phi": j, "cfo_cycles": cfo, "n_sym": m})
    if not table:
        return {"best": None, "table": []}
    best = max(table, key=lambda t: t["metric"])
    return {"best": best, "table": table}


def cp_strip_fft(x, fs_hz, tu_us, cp_us, phi, cfo=0.0, max_sym=600):
    """Strip CP at the detected phase and return FFT'd symbols [n_sym, tau].

    CFO convention (verified synthetically in the phase-A' WiFi track):
    this function multiplies by exp(+j2*pi*cfo*n), so to APPLY a detected
    CFO (from cp_scan_complex, defined with the opposite sign) you must
    pass cfo = -detected. With SCS-aligned content cfo ~= 0 and it is moot.
    """
    tau = int(round(tu_us * 1e-6 * fs_hz))
    cp = int(round(cp_us * 1e-6 * fs_hz))
    per = tau + cp
    if cfo:
        n = np.arange(len(x), dtype=np.float64)
        x = (x * np.exp(1j * 2.0 * np.pi * cfo * n)).astype(np.complex64)
    starts = np.arange(phi, len(x) - per + 1, per, dtype=np.int64)
    rows = []
    for s in starts:
        a, b = s + cp, s + cp + tau
        if a < 0 or b > len(x):
            continue
        rows.append(np.fft.fft(x[a:b]))
        if len(rows) >= max_sym:
            break
    if not rows:
        return np.zeros((0, tau), dtype=np.complex64)
    return np.stack(rows).astype(np.complex64)


# ---------------------------------------------------------------------------
# SRRC: timing-tone rate scan, matched filter, symbol sampling
# ---------------------------------------------------------------------------

def srrc_pulse_taps(alpha, sps, span_syms=SRRC_SPAN_SYMS):
    """Root-raised-cosine taps (standard Proacos/MATLAB rcosdesign form).

    h(0) = 1 - beta + 4*beta/pi
    h(+-1/(4 beta)) = (beta/sqrt(2)) * ((1+2/pi) sin(pi/4b) + (1-2/pi) cos(pi/4b))
    else  h(t) = [sin(pi t (1-b)) + 4 b t cos(pi t (1+b))] / [pi t (1-(4bt)^2)]

    The previous implementation used the sinc(t)cos(pi b t)/(1-(4bt)^2)
    form, which is NOT the standard RRC and has wider spectral skirts.
    Taps are normalized to unit energy; the grid may be fractional (sps
    need not be an integer).
    """
    n_half = max(4, int(round(span_syms / 2.0 * sps)))
    n = np.arange(-n_half, n_half + 1, dtype=np.float64)
    t = n / sps
    beta = float(alpha)
    h = np.zeros_like(t)
    if beta <= 1e-9:
        h = np.sinc(t)
    else:
        t4 = 1.0 / (4.0 * beta)
        tol = 0.5 / sps
        for i, ti in enumerate(t):
            if abs(ti) < 1e-12:
                h[i] = 1.0 - beta + 4.0 * beta / math.pi
            elif abs(abs(ti) - t4) <= tol:
                h[i] = (beta / math.sqrt(2.0)) * (
                    (1.0 + 2.0 / math.pi) * math.sin(math.pi / (4.0 * beta)) +
                    (1.0 - 2.0 / math.pi) * math.cos(math.pi / (4.0 * beta)))
            else:
                h[i] = (math.sin(math.pi * ti * (1.0 - beta)) +
                        4.0 * beta * ti * math.cos(math.pi * ti * (1.0 + beta))) / \
                       (math.pi * ti * (1.0 - (4.0 * beta * ti) ** 2))
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    e = float(np.sum(h ** 2))
    if e <= 0:
        h = np.sinc(t)
        e = float(np.sum(h ** 2))
    return (h / math.sqrt(e)).astype(np.float64)


def tone_line_scan(y_sq, fs_hz, f_lo_hz, f_hi_hz):
    """Spectral line of |y|^2 in [f_lo, f_hi]; returns (f_hz, strength, phase)."""
    n = len(y_sq)
    if n < 64:
        return 0.0, 0.0, 0.0
    pad = int(2 ** math.ceil(math.log2(n)))
    spec = np.fft.rfft(y_sq, n=pad)
    freqs = np.fft.rfftfreq(pad, d=1.0 / fs_hz)
    band = (freqs >= f_lo_hz) & (freqs <= f_hi_hz)
    if not band.any():
        return 0.0, 0.0, 0.0
    mag = np.abs(spec)
    sel = np.where(band)[0]
    k = int(sel[int(np.argmax(mag[sel]))])
    # local floor: median magnitude in a surrounding octave
    lo = max(0, k - max(10, k // 4))
    hi = min(len(mag) - 1, k + max(10, k // 4))
    floor = float(np.median(mag[lo:hi + 1]))
    strength = mag[k] / max(floor, 1e-30)
    # line phase (bin-center convention: sum y_sq * exp(-2 pi i k n / pad))
    return float(freqs[k]), float(strength / math.sqrt(max(n, 1))), complex(spec[k])


def srrc_sync(x, fs_hz, bw_hz, max_sym=6000):
    """Label-blind SRRC sync: rate via |y|^2 line, timing via line phase.

    Returns a dict with rate, alpha, symbol magnitudes and quality metrics.
    """
    # preliminary matched filter with alpha implied by a mid guess is not
    # needed: the tone exists on the raw band-limited signal as well.
    f_line, strength, line = tone_line_scan(np.abs(x) ** 2, fs_hz,
                                            0.5 * bw_hz, 0.98 * bw_hz)
    if f_line <= 0:
        return {"ok": False, "reason": "no_line"}
    r_hat = f_line
    alpha = float(np.clip(bw_hz / r_hat - 1.0, 0.02, 0.98))
    sps = fs_hz / r_hat
    taps = srrc_pulse_taps(alpha, sps)
    y = np.convolve(x, taps, mode="same")
    # refine on the matched-filter output
    f2, strength2, line2 = tone_line_scan(np.abs(y) ** 2, fs_hz,
                                          0.5 * bw_hz, 1.02 * bw_hz)
    if f2 > 0:
        r_hat = f2
        alpha = float(np.clip(bw_hz / r_hat - 1.0, 0.02, 0.98))
        sps = fs_hz / r_hat
        line = line2
        strength = strength2
    # symbol instants from the line phase: |y|^2 ~ P0 + P1 cos(2 pi R (t - t0))
    # sum |y|^2 e^{-2 pi i R n / fs} = (P1 N / 2) e^{-2 pi i R t0 / fs}
    t0 = (-np.angle(line) / (2.0 * np.pi) * fs_hz / r_hat) % sps
    pos = t0 + sps * np.arange(max_sym, dtype=np.float64)
    pos = pos[pos < len(y) - 2]
    syms = cubic_sample(y, pos)
    return {
        "ok": True,
        "r_hz": float(r_hat), "alpha": alpha, "sps": float(sps),
        "tone_strength": float(strength), "t0": float(t0),
        "n_sym": int(len(syms)), "symbols": syms.astype(np.complex64),
    }


def cubic_sample(y, pos):
    """4-point cubic convolution sampling of complex y at fractional pos."""
    n = len(y)
    pos = np.asarray(pos, dtype=np.float64)
    i = np.clip(np.floor(pos).astype(np.int64), 1, n - 3)
    f = pos - i
    ym1 = y[i - 1]
    y0 = y[i]
    y1 = y[i + 1]
    y2 = y[i + 2]
    # cubic Hermite with centered finite-difference tangents
    m0 = 0.5 * (y1 - ym1)
    m1 = 0.5 * (y2 - y0)
    h00 = (1 + 2 * f) * (1 - f) ** 2
    h10 = f * (1 - f) ** 2
    h01 = f * f * (3 - 2 * f)
    h11 = f * f * (f - 1)
    return (h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1).astype(np.complex64)


# ---------------------------------------------------------------------------
# Synthetic generators (for pipeline validation and gate calibration)
# ---------------------------------------------------------------------------

def gen_wifi_ofdm(bw_hz, tu_us, cp_us, n_sym, order, snr_db, cfo_sc=0.0, seed=0):
    """Continuous CP-OFDM at the signal rate; unit average signal power.

    Returns (x, meta) where x is at fs = bw_hz, subcarriers spaced SCS.
    """
    rng = np.random.default_rng(seed)
    fs = bw_hz
    tu = int(round(tu_us * 1e-6 * fs))
    cp = int(round(cp_us * 1e-6 * fs))
    data = qam_symbols(order, tu * n_sym, rng).reshape(tu, n_sym)
    time_syms = np.fft.ifft(data, axis=0) * math.sqrt(tu)
    x = np.concatenate([time_syms[-cp:], time_syms], axis=0).T.reshape(-1)
    n = np.arange(len(x), dtype=np.float64)
    if cfo_sc:
        x = x * np.exp(2j * np.pi * cfo_sc * n)
    if snr_db is not None and snr_db > -10:
        p_n = 10 ** (-snr_db / 10.0)
        x = x + math.sqrt(p_n / 2.0) * (rng.standard_normal(len(x)) +
                                         1j * rng.standard_normal(len(x)))
    return x.astype(np.complex64), {"tu": tu, "cp": cp, "order": order}


def gen_srrc(bw_hz, alpha, n_sym, order, snr_db, fs_hz, seed=0):
    """SRRC pulse-shaped single carrier at fs_hz; band = R*(1+alpha) = bw.

    Returns (x, meta) with the true symbol rate recorded.
    """
    rng = np.random.default_rng(seed)
    r = bw_hz / (1.0 + alpha)
    sps_gen = 8.0
    fs_gen = sps_gen * r
    taps = srrc_pulse_taps(alpha, sps_gen, span_syms=8.0)
    syms = qam_symbols(order, n_sym, rng)
    up = np.zeros(int(math.ceil((n_sym + 8.0) * sps_gen)), dtype=np.complex64)
    up[:: int(sps_gen)] = 0
    pos0 = int(round(4.0 * sps_gen))
    up[pos0:pos0 + n_sym * int(sps_gen): int(sps_gen)] = syms
    x = np.convolve(up, taps.astype(np.complex64), mode="same")
    # resample fs_gen -> fs_hz (both rational)
    frac = Fraction(fs_hz / fs_gen).limit_denominator(2000)
    from scipy.signal import resample_poly
    xr = resample_poly(x, frac.numerator, frac.denominator)
    if snr_db is not None and snr_db > -10:
        p_n = 10 ** (-snr_db / 10.0)
        xr = xr + math.sqrt(p_n / 2.0) * (rng.standard_normal(len(xr)) +
                                           1j * rng.standard_normal(len(xr)))
    return xr.astype(np.complex64), {"r_hz": float(r), "alpha": float(alpha),
                                     "order": order, "fs_gen": fs_gen}
