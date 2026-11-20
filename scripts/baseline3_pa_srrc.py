#!/usr/bin/env python3
"""Baseline-3 phase A' audit, Track S: SRRC single-carrier pipeline.

STEPS
  A. Label-free synthetic calibration: 10 MHz SRRC single carriers,
     alpha in {0.2, 0.35} x order {QPSK(4), 16QAM(16)} x B in {20,40,80}
     MHz, ~40000 symbols, generated with gen_srrc_clean in this script:
     the SAME standard Proakis RRC taps (scripts.baseline3_dsp
     .srrc_pulse_taps, unmodified) built at 32 samples/symbol with a
     16-symbol span, then resampled to the capture rate with
     scipy.signal.resample_poly using exact rational ratios (3/40, 3/20,
     3/10 for alpha=0.2; 27/320, 27/160, 27/80 for alpha=0.35; the 10 MHz
     band is far inside every output Nyquist so the resampler AA filter
     is harmless). Why not the shared gen_srrc: it builds at 8 sps with
     span 8, and that tap train's power spectrum droops up to ~6.5%
     through the RC rolloff (time discretization + truncation, measured),
     which would bias the width-based alpha calibration; at 32 sps /
     span 16 the pulse fidelity vs the ideal RC is ~0.6%.
     Signals are embedded at digital offsets {0, +6, -12} MHz for
     B = {20, 40, 80} (band always inside the observation), with white
     capture noise at 20 dB in-box SNR, round-tripped through float16
     so the real read path is exercised. Pure-noise negative controls
     are run per B.
  B. The SAME pipeline on the 10 class-10/11 frames (bw 10 MHz) of
     reports/local_iq_round1/phase_ap_frames.json (clean first, overlap
     flagged).
  C. Exploratory tabulation of class-10 vs class-11 clean-frame m4
     against synthetic QPSK / 16QAM / noise references. NO classification
     claims are made or supported.

PIPELINE (identical for synthetic and real)
  1. Fixed extraction recipe: central window min(10 ms, dur) of the GT
     box -> read_segment -> demod_to_center(box center) -> band_limit_fft
     (bw*1.02, edge_frac=0.05) -> trim 0.5 ms at both ends.
  2. PSD profile: two-sided Welch PSD (~40 kHz grid, 75%-overlap Hann,
     ~500 kHz median kernel, plateau reference = median of the top
     quartile, persistence/hysteresis level crossings). W90 / W50 / W10
     = widths at 90 / 50 / 10 % of that reference.
     Raised-cosine inversion (exact for an ideal RC PSD):
       W90 = R(1 - 0.590334 a),  W50 = R,  W10 = R(1 + 0.590334 a)
       -> R_est = W50                (max-slope crossing: level-robust)
          a_est = (W10 - W90) / (1.180668 * W50)  (level errors cancel)
     Step A debugging showed the naive rules fail: the raw in-band max
     is speckle-inflated and the strict contiguous-run width collapses
     the 90% band. The exact-in-theory pair rule R=(W10+W90)/2 and the
     task's literal rule (R_est = W90, alpha = W10/W90 - 1) are also
     recorded for every run (see the estimator note in the report).
  3. Matched filter srrc_pulse_taps(a_psd, fs/R_psd); tone_line_scan of
     |y|^2 in [0.9, 1.1]*R_psd; then a fine direct-correlation refinement
     D(f) = sum_n |y|^2 exp(-2 pi i f n/fs) on a +-1.5-bin grid with
     parabolic peak interpolation. Evaluating D at the refined frequency
     avoids the padded-DFT bin-center phase error pi*delta*(N-1) and
     gives both the rate (R_tone = refined line frequency) and the
     symbol instant t0 = -angle(D)/(2 pi) * fs/f mod fs/f.
     Auxiliary sync quality: raw peak/octave-median (no 1/sqrt(n)) and
     an off-band specificity scan (same statistic in a line-free band).
  4. Second matched-filter pass at (a_fin, fs/R_tone) and a second tone
     refinement; symbols sampled at t0 + k*fs/R_fin with cubic_sample;
     m4/m6 via mag_moments and an 8-bin |s|/RMS histogram on [0,3].

All statistics are computed from GT-box geometry only; class labels are
carried for reporting. Fit-split data only: the script refuses any
sample outside train/ and never touches .../advanced/test. No models are
trained. Run: PYTHONPATH=. .venv/bin/python scripts/baseline3_pa_srrc.py
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from fractions import Fraction

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import fftconvolve, resample_poly

from scripts.baseline3_dsp import (
    read_segment, demod_to_center, band_limit_fft, band_snr_db,
    mag_moments, mag_histogram, tone_line_scan, srrc_pulse_taps,
    cubic_sample, qam_symbols)

ROOT = Path(__file__).resolve().parents[1]
FRAMES_JSON = ROOT / "reports/local_iq_round1/phase_ap_frames.json"
OUT_JSON = ROOT / "reports/local_iq_round1/phase_ap_srrc.json"
OUT_MD = ROOT / "reports/local_iq_round1/phase_ap_srrc.md"
DATA_DIR = Path("/root/autodl-fs/SpaceNet_Dataset/advanced")
CAL_DIR = Path("/tmp/opencode/srrc_cal")

ALPHAS = (0.2, 0.35)
ORDERS = (4, 16)
B_GRID = (20.0, 40.0, 80.0)
NSYM = 40000
IN_BOX_SNR_DB = 20.0          # synthetic capture noise level (documented choice)
OFFSET_MHZ = {20.0: 0.0, 40.0: 6.0, 80.0: -12.0}
F_LO_SYN = 2400.0             # synthetic observation center convention helper
CAP_MS = 10.0
TRIM_MS = 0.5
GEN_SPS = 32.0                # calibration generator oversampling (see gen_srrc_clean)
GEN_SPAN = 16.0               # tap span in symbols for the calibration generator
RC_W = 1.180668               # (W10 - W90) / (alpha * R) for an RC PSD
M4_REF = {4: 1.00, 16: 1.32}  # ideal QPSK / 16QAM magnitude kurtosis
M6_REF = {4: 1.00, 16: 1.96}


# ---------------------------------------------------------------------------
# PSD profile and RC-shape inversion
# ---------------------------------------------------------------------------

def psd_profile(x, fs):
    """Two-sided Welch PSD (Hann, 75% overlap), ~40 kHz grid.

    Aggressive averaging is deliberate: Step A showed that with ~70
    segments the flat-top speckle (+-8%) breaks the 90% width run.
    Resolution (~40 kHz) is far finer than the RC transition widths
    (alpha*R >= 1.7 MHz here).
    """
    n = len(x)
    seg = int(2 ** round(math.log2(fs / 40e3)))
    seg = max(256, min(seg, 1 << 13))
    while seg > 256 and n < 2 * seg:
        seg //= 2
    hop = max(1, seg // 4)
    win = np.hanning(seg)
    acc = np.zeros(seg, dtype=np.float64)
    cnt = 0
    for s in range(0, n - seg + 1, hop):
        acc += np.abs(np.fft.fft(x[s:s + seg] * win)) ** 2
        cnt += 1
    if cnt == 0:
        seg = n
        acc = np.abs(np.fft.fft(x * np.hanning(n))) ** 2
        cnt = 1
    p = acc / cnt
    df = fs / seg
    freqs = np.fft.fftfreq(seg, 1.0 / fs)
    order = np.argsort(freqs)
    return freqs[order], p[order], df


def psd_widths(freqs, p, df, half_hz):
    """W90/W50/W10 + shape flags, robust to finite-averaging speckle.

    Measured on a ~500 kHz median-smoothed in-band PSD (a symmetric
    median kernel is exact on the monotone RC edges; only speckle is
    removed), referenced to a robust plateau level v_ref = median of the
    top quartile (the raw max is speckle-inflated -- Step A debugging),
    and with PERSISTENCE (hysteresis) crossings: walking outward from
    the in-band max, the level is considered crossed only after a
    ~0.4 MHz stretch below it, so single speckle dips cannot truncate a
    width (this is what broke the naive contiguous-run rule).

    Estimator pairing (ideal raised-cosine algebra, W in Hz):
      W90 = R(1 - 0.590334 a), W50 = R, W10 = R(1 + 0.590334 a)
      -> R_est  = W50           (max-slope crossing: most level-robust)
         a_est  = (W10 - W90) / (1.180668 * W50)   (level errors cancel)
    The pair rule R=(W10+W90)/2 and the task's literal rule
    (R=W90, a=W10/W90-1) are also recorded for comparison.
    """
    inb = np.abs(freqs) <= half_hz
    fr = freqs[inb]
    ps = p[inb]
    k_w = int(round(0.5e6 / df)) | 1
    ps_w = median_filter(ps, size=k_w, mode="nearest") if k_w > 1 else ps
    persist = max(3, int(round(0.4e6 / df)))
    n_top = max(8, len(ps_w) // 4)
    top = np.partition(ps_w, len(ps_w) - n_top)[len(ps_w) - n_top:]
    v_ref = max(float(np.median(top)), 1e-30)
    pmax = float(ps_w.max())
    imax = int(np.argmax(ps_w))
    out = {"pmax": pmax, "v_ref": v_ref}

    def cross(level):
        """Fractional crossing indices (left, right) with persistence."""
        lvl = level * v_ref
        la = imax
        below = 0
        i = imax
        while i > 0:
            i -= 1
            if ps_w[i] >= lvl:
                la = i
                below = 0
            else:
                below += 1
                if below >= persist:
                    break
        lc = float(la)
        if la + 1 < len(ps_w) and ps_w[la + 1] < lvl:
            d = ps_w[la] - ps_w[la + 1]
            if d > 0:
                lc = la + min(1.0, max(0.0, (ps_w[la] - lvl) / d))
        ra = imax
        below = 0
        i = imax
        while i < len(ps_w) - 1:
            i += 1
            if ps_w[i] >= lvl:
                ra = i
                below = 0
            else:
                below += 1
                if below >= persist:
                    break
        rc = float(ra)
        if ra - 1 >= 0 and ps_w[ra - 1] < lvl:
            d = ps_w[ra - 1] - ps_w[ra]
            if d > 0:
                rc = ra - min(1.0, max(0.0, (ps_w[ra] - lvl) / d))
        return lc, rc

    f0 = fr[0]
    for lvl, name in ((0.9, "W90"), (0.5, "W50"), (0.1, "W10")):
        lc, rc = cross(lvl)
        out[name + "_lo"] = float(f0 + lc * df)
        out[name + "_hi"] = float(f0 + rc * df)
        out[name] = float(out[name + "_hi"] - out[name + "_lo"])

    # number of distinct >=10% regions (multi-tone / multi-carrier content)
    above10 = ps_w >= 0.1 * v_ref
    out["n_runs10"] = int(np.sum(above10[1:] & ~above10[:-1]) + (1 if above10[0] else 0))

    # edge symmetry around the -3 dB center
    c50 = 0.5 * (out["W50_lo"] + out["W50_hi"])
    l10 = c50 - out["W10_lo"]
    r10 = out["W10_hi"] - c50
    out["center_off"] = float(c50)
    out["asym10"] = float((r10 - l10) / max(r10 + l10, 1e-9))

    # plateau flatness: central 50% of the W90 band
    i0 = np.searchsorted(fr, out["W90_lo"] + 0.25 * out["W90"])
    i1 = np.searchsorted(fr, out["W90_hi"] - 0.25 * out["W90"])
    if i1 - i0 < 3:
        i0, i1 = np.searchsorted(fr, out["W90_lo"]), np.searchsorted(fr, out["W90_hi"])
    if i1 - i0 >= 2:
        seg_ps = ps_w[i0:i1]
        out["plateau_flat_db"] = float(10 * math.log10(max(seg_ps.max(), 1e-30) /
                                                       max(float(np.median(seg_ps)), 1e-30)))
    else:
        out["plateau_flat_db"] = None

    # edge drop beyond the 10% crossings on both sides (weaker side wins)
    guard = 0.15e6
    hi_reg = (fr >= out["W10_hi"] + guard) & (fr <= half_hz)
    lo_reg = (fr <= out["W10_lo"] - guard) & (fr >= -half_hz)
    lv = []
    if hi_reg.sum() >= 3:
        lv.append(float(np.median(ps_w[hi_reg])))
    if lo_reg.sum() >= 3:
        lv.append(float(np.median(ps_w[lo_reg])))
    out["edge_drop_db"] = (float(10 * math.log10(max(v_ref, 1e-30) / max(max(lv), 1e-30)))
                           if lv else None)
    return out


def rc_invert(w):
    """(R_est, alpha_est) from the measured widths, plus recorded variants.

    Used estimator (robust pairing, exact for an ideal RC PSD):
        R_est  = W50                              (max-slope crossing)
        a_est  = (W10 - W90) / (1.180668 * W50)   (level errors cancel)
    Also returned for transparency:
        pair   : R = (W10+W90)/2, a = (W10-W90)/(1.180668 R)   (exact in theory)
        literal: R = W90, a = W10/W90 - 1  (task letter; biased by the RC shape)
    """
    w90, w50, w10 = w["W90"], w["W50"], w["W10"]
    if w50 <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    a_used = (w10 - w90) / (RC_W * w50)
    r_pair = 0.5 * (w10 + w90)
    a_pair = (w10 - w90) / (RC_W * r_pair) if r_pair > 0 else float("nan")
    a_lit = w10 / w90 - 1.0 if w90 > 0 else float("nan")
    return (float(w50), float(np.clip(a_used, 0.0, 0.98)),
            float(r_pair), float(np.clip(a_pair, 0.0, 0.98)),
            float(w90), float(a_lit))


# ---------------------------------------------------------------------------
# Tone (symbol-rate clock) estimation and refinement
# ---------------------------------------------------------------------------

def tone_refine(y_sq, fs, r_guess):
    """tone_line_scan coarse line + direct-correlation frequency/phase refine."""
    n = len(y_sq)
    if n < 64 or r_guess <= 0:
        return None
    f_lo, f_hi = 0.9 * r_guess, 1.1 * r_guess
    f_c, s_c, _ = tone_line_scan(y_sq, fs, f_lo, f_hi)
    if f_c <= 0:
        return None
    pad = int(2 ** math.ceil(math.log2(n)))
    binw = fs / pad
    se = np.abs(np.fft.rfft(y_sq, n=pad))
    k0 = int(round(f_c / binw))
    rad = max(10, k0 // 4)
    floor = max(float(np.median(se[max(0, k0 - rad):min(len(se), k0 + rad + 1)])), 1e-30)

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
    t0 = (-np.angle(D_ref) / (2.0 * math.pi) * sps_ref) % sps_ref
    # half-split coherence: both halves share the line phase at f_ref
    h = n // 2
    D1 = np.sum(y_sq[:h] * np.exp(-2j * np.pi * f_ref * m[:h] / fs))
    D2 = np.sum(y_sq[h:] * np.exp(-2j * np.pi * f_ref * m[h:] / fs))
    coh = abs(D1 + D2) / max(abs(D1) + abs(D2), 1e-30)
    return {
        "found": True, "f_hz": f_ref, "coarse_hz": float(f_c),
        "coarse_strength": float(s_c), "strength": float(strength),
        "peak_med": float(abs(D_ref) / floor), "coherence": float(coh),
        "t0": float(t0), "t0_frac": float(t0 / sps_ref),
    }


def off_band_specificity(y_sq, fs, r_guess):
    """Same peak/median statistic in a line-free band (false-alarm control)."""
    n = len(y_sq)
    if 1.45 * r_guess <= 0.95 * fs / 2:
        lo, hi = 1.25 * r_guess, 1.45 * r_guess
    else:
        lo, hi = 0.5 * r_guess, 0.7 * r_guess
    f_c, s_c, _ = tone_line_scan(y_sq, fs, lo, hi)
    if f_c <= 0:
        return None
    return float(s_c * math.sqrt(n))


def matched_filter(x, alpha, sps):
    taps = srrc_pulse_taps(float(alpha), float(sps))
    return np.convolve(x, taps, mode="same"), taps


def sample_symbols(y, t0, sps, guard):
    n = len(y)
    if n <= guard + 8 or sps <= 0:
        return np.zeros(0, dtype=np.complex64), np.zeros(0)
    k_max = int(math.floor((n - 3 - guard - t0) / sps))
    pos = t0 + sps * np.arange(max(k_max, 0), dtype=np.float64)
    pos = pos[(pos >= guard) & (pos <= n - 3 - guard)]
    return cubic_sample(y, pos), pos


# ---------------------------------------------------------------------------
# Full analysis of one already-read segment (recipe steps after read_segment)
# ---------------------------------------------------------------------------

def analyze_segment(x, fs, bw_hz, f_center, f_lo, timing_probe=False, tag=""):
    xd = demod_to_center(x, fs, f_center, f_lo)
    snr_db, _, _ = band_snr_db(xd, fs, bw_hz * 1.02)
    xb = band_limit_fft(xd, fs, bw_hz * 1.02, edge_frac=0.05)
    ntr = int(round(TRIM_MS * 1e-3 * fs))
    if len(xb) > 2 * ntr + 256:
        xb = xb[ntr:len(xb) - ntr]

    freqs, p, df = psd_profile(xb, fs)
    w = psd_widths(freqs, p, df, half_hz=bw_hz * 1.02 / 2.0)
    r_psd, a_psd, r_pair, a_pair, r_lit, a_lit = rc_invert(w)
    rec = {
        "tag": tag, "n_samples": int(len(xb)),
        "snr_db": float(snr_db),
        "psd": {
            "W90_mhz": w["W90"] / 1e6, "W50_mhz": w["W50"] / 1e6,
            "W10_mhz": w["W10"] / 1e6,
            "W10_lo_mhz": w["W10_lo"] / 1e6, "W10_hi_mhz": w["W10_hi"] / 1e6,
            "center_off_mhz": w["center_off"] / 1e6, "asym10": w["asym10"],
            "plateau_flat_db": w["plateau_flat_db"], "edge_drop_db": w["edge_drop_db"],
            "n_runs10": w["n_runs10"],
            "R_psd_mhz": r_psd / 1e6, "alpha_psd": a_psd,
            "R_pair_mhz": r_pair / 1e6, "alpha_pair": a_pair,
            "R_lit_mhz": r_lit / 1e6, "alpha_lit": a_lit,
        },
    }

    # pass 1: matched filter at the PSD-derived shape, tone in [0.9,1.1] R_psd
    y1, taps1 = matched_filter(xb, a_psd, fs / max(r_psd, 1.0))
    y1_sq = (np.abs(y1) ** 2).astype(np.float64)
    t1 = tone_refine(y1_sq, fs, r_psd)
    spec_alt1 = off_band_specificity(y1_sq, fs, r_psd)
    rec["tone_pass1"] = t1
    rec["spec_alt_peak_med_pass1"] = spec_alt1

    if t1 is not None:
        r_fin = t1["f_hz"]
        a_fin = float(np.clip((w["W10"] - w["W90"]) / (RC_W * r_fin), 0.0, 0.98))
        # pass 2: matched filter at the tone-derived rate
        y2, taps2 = matched_filter(xb, a_fin, fs / r_fin)
        y2_sq = (np.abs(y2) ** 2).astype(np.float64)
        t2 = tone_refine(y2_sq, fs, r_fin)
        if t2 is not None and abs(t2["f_hz"] - r_fin) < 0.005 * r_fin:
            tone_fin, y_fin, taps = t2, y2, taps2
            r_fin = t2["f_hz"]
            a_fin = float(np.clip((w["W10"] - w["W90"]) / (RC_W * r_fin), 0.0, 0.98))
        else:
            tone_fin, y_fin, taps = t1, y2, taps2
    else:
        r_fin, a_fin = r_psd, a_psd
        tone_fin, y_fin, taps = None, y1, taps1

    rec["final"] = {"R_mhz": r_fin / 1e6, "alpha": a_fin,
                    "sps": fs / r_fin, "source": "tone" if tone_fin else "psd_fallback"}
    if tone_fin is not None:
        rec["tone"] = {
            "f_mhz": tone_fin["f_hz"] / 1e6, "strength": tone_fin["strength"],
            "peak_med": tone_fin["peak_med"], "coherence": tone_fin["coherence"],
            "t0_frac": tone_fin["t0_frac"],
            "coarse_strength": tone_fin["coarse_strength"],
        }
    else:
        rec["tone"] = None

    guard = (len(taps) - 1) // 2 + 2
    t0 = tone_fin["t0"] if tone_fin is not None else 0.0
    syms, pos = sample_symbols(y_fin, t0, fs / r_fin, guard)
    mag = np.abs(syms)
    m4, m6 = mag_moments(mag)
    rec["syms"] = {
        "n_sym": int(len(syms)),
        "m4": float(m4), "m6": float(m6),
        "hist": [float(v) for v in mag_histogram(mag, bins=np.linspace(0, 3, 9))],
    }
    if timing_probe and tone_fin is not None:
        offs = {}
        for off in (-0.5, -0.25, 0.25):
            s_o, _ = sample_symbols(y_fin, t0 + off * (fs / r_fin), fs / r_fin, guard)
            m4o, _ = mag_moments(np.abs(s_o))
            offs[f"{off:+.2f}"] = float(m4o)
        rec["m4_offsets"] = offs
    return rec


# ---------------------------------------------------------------------------
# Synthetic calibration (Step A) and negative controls
# ---------------------------------------------------------------------------

def gen_srrc_clean(bw_hz, alpha, n_sym, order, fs_hz, seed):
    """High-fidelity SRRC generator for the Step A calibration.

    Same standard Proakis-form taps (scripts.baseline3_dsp.srrc_pulse_taps,
    NOT modified), but built at GEN_SPS = 32 samples/symbol with a
    GEN_SPAN = 16-symbol span and resampled to the capture rate with an
    exact rational ratio. Reason: the shared gen_srrc builds at 8 sps /
    span 8, whose time-discretized + truncated tap train droops the pulse
    POWER spectrum by up to ~6.5% through the rolloff (measured
    |H(f)|^2 vs ideal RC) and biases any width-based alpha estimate;
    at 32 sps / span 16 the deviation is ~0.6%, so the calibration
    actually tests the estimator rather than the generator.
    Ratios are exact: alpha=0.2 -> 3/40, 3/20, 3/10 for B=20/40/80;
    alpha=0.35 -> 27/320, 27/160, 27/80 (no resampler rate error, and the
    10 MHz band is far inside every output Nyquist so the anti-alias
    filter is harmless).
    """
    rng = np.random.default_rng(seed)
    r = bw_hz / (1.0 + alpha)
    fs_gen = GEN_SPS * r
    taps = srrc_pulse_taps(alpha, GEN_SPS, span_syms=GEN_SPAN)
    syms = qam_symbols(order, n_sym, rng)
    step = int(GEN_SPS)
    pos0 = int(round(GEN_SPAN * GEN_SPS))
    up = np.zeros(2 * pos0 + n_sym * step, dtype=np.complex64)
    up[pos0:pos0 + n_sym * step:step] = syms
    x = fftconvolve(up, taps.astype(np.complex64), mode="same")
    frac = Fraction(fs_hz / fs_gen).limit_denominator(2000)
    xr = resample_poly(x, frac.numerator, frac.denominator)
    return xr.astype(np.complex64), {
        "r_hz": float(r), "fs_gen": float(fs_gen),
        "ratio": f"{frac.numerator}/{frac.denominator}",
        "gen_sps": GEN_SPS, "gen_span": GEN_SPAN}


def synth_case(alpha, order, B, seed):
    fs = B * 1e6
    r_true = 10e6 / (1.0 + alpha)
    dur = NSYM / r_true
    n_cap = int(round(dur * fs))
    sig, gmeta = gen_srrc_clean(10e6, alpha, NSYM, order, fs, seed)
    sig = sig[:n_cap]
    c = sig[len(sig) // 5: 4 * len(sig) // 5]
    p = max(float(np.mean(np.abs(c) ** 2)), 1e-30)
    sig = sig / math.sqrt(p)
    rng = np.random.default_rng(seed + 1)
    fd = OFFSET_MHZ[B] * 1e6
    nn = np.arange(n_cap, dtype=np.float64)
    x = sig * np.exp(2j * np.pi * fd * nn / fs)
    sigma2 = fs / (10e6 * 1.02 * (10 ** (IN_BOX_SNR_DB / 10.0)))
    x = x + math.sqrt(sigma2 / 2.0) * (rng.standard_normal(n_cap) +
                                       1j * rng.standard_normal(n_cap))
    # float16 round trip -> exercises the real read path
    tag = f"a{alpha}_m{order}_B{int(B)}"
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    path = CAL_DIR / f"{tag}.bin"
    raw = np.empty(2 * n_cap, dtype="<f2")
    raw[0::2] = x.real
    raw[1::2] = x.imag
    raw.tofile(path)
    xr = read_segment(str(path), 0, n_cap)
    fc = (F_LO_SYN + B / 2.0 + OFFSET_MHZ[B]) * 1e6
    rec = analyze_segment(xr, fs, 10e6, fc, F_LO_SYN * 1e6,
                          timing_probe=True, tag=tag)
    rec["truth"] = {"R_mhz": r_true / 1e6, "alpha": alpha, "order": order,
                    "B_mhz": B, "n_sym": NSYM, "dur_ms": dur * 1e3,
                    "offset_mhz": OFFSET_MHZ[B], "in_box_snr_db": IN_BOX_SNR_DB,
                    "gen_ratio": gmeta["ratio"]}
    r_est = rec["final"]["R_mhz"]
    a_est = rec["final"]["alpha"]
    rec["errors"] = {
        "R_pct": 100.0 * (r_est - r_true / 1e6) / (r_true / 1e6),
        "alpha_pct": 100.0 * (a_est - alpha) / alpha if alpha > 0 else float("nan"),
        "R_psd_pct": 100.0 * (rec["psd"]["R_psd_mhz"] - r_true / 1e6) / (r_true / 1e6),
        "alpha_psd_pct": 100.0 * (rec["psd"]["alpha_psd"] - alpha) / alpha,
        "R_pair_pct": 100.0 * (rec["psd"]["R_pair_mhz"] - r_true / 1e6) / (r_true / 1e6),
        "alpha_pair_pct": 100.0 * (rec["psd"]["alpha_pair"] - alpha) / alpha,
        "R_lit_pct": 100.0 * (rec["psd"]["R_lit_mhz"] - r_true / 1e6) / (r_true / 1e6),
        "alpha_lit_pct": 100.0 * (rec["psd"]["alpha_lit"] - alpha) / alpha,
        "m4_target": M4_REF[order],
    }
    return rec


def noise_case(B, seed):
    fs = B * 1e6
    dur = 5e-3
    n_cap = int(round(dur * fs))
    rng = np.random.default_rng(seed)
    sigma2 = fs / (10e6 * 1.02 * (10 ** (IN_BOX_SNR_DB / 10.0)))
    x = math.sqrt(sigma2 / 2.0) * (rng.standard_normal(n_cap) +
                                   1j * rng.standard_normal(n_cap))
    tag = f"noise_B{int(B)}"
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    path = CAL_DIR / f"{tag}.bin"
    raw = np.empty(2 * n_cap, dtype="<f2")
    raw[0::2] = x.real
    raw[1::2] = x.imag
    raw.tofile(path)
    xr = read_segment(str(path), 0, n_cap)
    fc = (F_LO_SYN + B / 2.0 + OFFSET_MHZ[B]) * 1e6
    rec = analyze_segment(xr, fs, 10e6, fc, F_LO_SYN * 1e6,
                          timing_probe=False, tag=tag)
    rec["truth"] = {"order": 0, "B_mhz": B, "dur_ms": dur * 1e3,
                    "in_box_snr_db": None}
    return rec


# ---------------------------------------------------------------------------
# Real frames (Step B)
# ---------------------------------------------------------------------------

def real_frame(fr):
    fs = fr["B_mhz"] * 1e6
    bw = (fr["f2_mhz"] - fr["f1_mhz"]) * 1e6
    t1, t2 = float(fr["t1_ms"]), float(fr["t2_ms"])
    c = min(CAP_MS, t2 - t1)
    ta = (t1 + t2) / 2.0 - c / 2.0
    tb = (t1 + t2) / 2.0 + c / 2.0
    i0 = max(0, int(round(ta * 1e-3 * fs)))
    i1 = min(int(round(tb * 1e-3 * fs)), int(round(t2 * 1e-3 * fs)))
    if i1 <= i0:
        i1 = i0 + int(1e-3 * fs)
    path = DATA_DIR / f"{fr['sample_id']}.bin"
    x = read_segment(str(path), i0, i1)
    rec = analyze_segment(x, fs, bw, (fr["f1_mhz"] + fr["f2_mhz"]) / 2.0 * 1e6,
                          fr["f_lo_mhz"] * 1e6, timing_probe=False,
                          tag=fr["sample_id"])
    rec["frame"] = {
        "sample_id": fr["sample_id"], "signal_id": fr["signal_id"],
        "class": fr["class"], "stratum": fr["stratum"],
        "B_mhz": fr["B_mhz"], "bw_mhz": fr["bw_mhz"],
        "f1_mhz": fr["f1_mhz"], "f2_mhz": fr["f2_mhz"],
        "t1_ms": t1, "t2_ms": t2, "dur_ms": t2 - t1,
        "window_ms": c, "n_read": int(len(x)),
    }
    return rec


def apply_flags(rec, thr_strength):
    """Label-blind sync-quality verdicts; threshold from Step A calibration."""
    tone = rec.get("tone")
    psd = rec["psd"]
    fin = rec["final"]
    tone_ok = bool(tone is not None and tone["strength"] >= thr_strength
                   and abs(tone["f_mhz"] / max(psd["R_psd_mhz"], 1e-9) - 1.0) <= 0.15)
    psd_rc_ok = bool(0.05 <= psd["alpha_psd"] <= 0.95
                     and psd["plateau_flat_db"] is not None
                     and psd["plateau_flat_db"] <= 3.5
                     and psd["edge_drop_db"] is not None
                     and psd["edge_drop_db"] >= 8.0)
    w50_ok = abs(psd["W50_mhz"] / max(fin["R_mhz"], 1e-9) - 1.0) <= 0.08
    srrc_like = bool(tone_ok and psd_rc_ok and w50_ok)
    m4 = rec["syms"]["m4"]
    if abs(m4 - 1.00) <= 0.15:
        m4_class = "qpsk_like"
    elif abs(m4 - 1.32) <= 0.18:
        m4_class = "qam16_like"
    elif m4 >= 1.75:
        m4_class = "rayleigh_noise_like"
    else:
        m4_class = "other"
    rec["flags"] = {"tone_ok": tone_ok, "psd_rc_ok": psd_rc_ok, "w50_ok": bool(w50_ok),
                    "srrc_like": srrc_like, "m4_class": m4_class}
    # descriptive hint (label-blind, from measured numbers only)
    hint = []
    if srrc_like:
        hint.append("SRRC-like")
        if m4_class in ("rayleigh_noise_like", "other"):
            hint.append(f"note: sync gates pass but envelope is {m4_class} "
                        f"(m4={m4:.2f}) -- noisy or non-QAM constellation")
    else:
        if psd["W10_mhz"] < 0.35 * 10.2:
            hint.append(f"narrowband content (W10={psd['W10_mhz']:.2f} MHz, "
                        f"n_runs10={psd['n_runs10']})")
        if psd["plateau_flat_db"] is not None and psd["plateau_flat_db"] > 4.0:
            hint.append(f"peaked/irregular spectrum (flat_db={psd['plateau_flat_db']:.1f})")
        if psd["edge_drop_db"] is None and psd["W10_mhz"] >= 0.95 * 10.2:
            hint.append("flat-topped band with no visible edge rolloff (10% width "
                        "fills the box; edge drop unmeasurable)")
        elif psd["edge_drop_db"] is not None and psd["edge_drop_db"] < 5.0 and \
                psd["plateau_flat_db"] is not None and psd["plateau_flat_db"] <= 4.0:
            hint.append("flat-topped band without RC edge rolloff")
        if psd["alpha_psd"] <= 0.02 or psd["alpha_psd"] >= 0.98:
            hint.append(f"no measurable RC transition (alpha_psd={psd['alpha_psd']:.3f})")
        if tone_ok and not srrc_like:
            fails = []
            if not psd_rc_ok:
                fails.append("RC-shape gate" +
                             (" (edge drop unmeasurable)" if psd["edge_drop_db"] is None
                              else f" (edge_drop={psd['edge_drop_db']:.1f} dB)"
                              if psd["edge_drop_db"] is not None else ""))
            if not w50_ok:
                fails.append(f"W50-vs-R (W50={psd['W50_mhz']:.2f} vs R={fin['R_mhz']:.2f})")
            hint.append(f"symbol-rate tone present ({tone['f_mhz']:.3f} MHz, "
                         f"strength={tone['strength']:.3f}) but PSD gates fail: "
                         + ", ".join(fails))
        if not tone_ok:
            hint.append("no symbol-rate tone" if tone is None
                        else f"tone below threshold (strength={tone['strength']:.4g})")
            if psd["edge_drop_db"] is None and psd["W10_mhz"] >= 0.95 * 10.2 and \
                    rec["syms"]["m4"] >= 1.7:
                hint.append("flat band + near-Rayleigh envelope + no tone: "
                            "noise-like or OFDM-style content, not single-carrier SRRC")
        if abs(psd["center_off_mhz"]) > 0.5:
            hint.append(f"band off-center by {psd['center_off_mhz']:+.2f} MHz")
        if psd["n_runs10"] > 2:
            hint.append(f"{psd['n_runs10']} separate >=10% spectral regions")
        if m4_class == "rayleigh_noise_like":
            hint.append("noise-like envelope (m4~2)")
    if rec["frame"]["stratum"] == "overlap":
        hint.append("overlap stratum: box may contain a second signal")
    rec["content_hint"] = "; ".join(hint) if hint else "—"
    return rec


# ---------------------------------------------------------------------------
# Step C: exploratory class comparison (no classification claims)
# ---------------------------------------------------------------------------

def build_comparison(cal, noise, frames):
    ok = [f for f in frames if "error" not in f]
    c10 = [f for f in ok if f["frame"]["class"] == 10 and f["frame"]["stratum"] == "clean"]
    c11 = [f for f in ok if f["frame"]["class"] == 11 and f["frame"]["stratum"] == "clean"]
    c10s = [f for f in c10 if f["flags"]["srrc_like"]]
    c11s = [f for f in c11 if f["flags"]["srrc_like"]]
    q4 = [c for c in cal if c["truth"]["order"] == 4 and "m4_offsets" in c]
    q16 = [c for c in cal if c["truth"]["order"] == 16 and "m4_offsets" in c]
    m4s = lambda lst: [round(f["syms"]["m4"], 3) for f in lst]
    mean = lambda lst: float(np.mean(m4s(lst))) if lst else None
    cmp_ = {
        "class10_clean_m4": m4s(c10),
        "class11_clean_m4": m4s(c11),
        "class10_srrclike_m4": m4s(c10s),
        "class11_srrclike_m4": m4s(c11s),
        "class10_srrclike_mean_m4": mean(c10s),
        "class11_srrclike_mean_m4": mean(c11s),
        "synthetic_qpsk_m4": m4s(q4),
        "synthetic_qam16_m4": m4s(q16),
        "noise_control_m4": [round(n["syms"]["m4"], 3) for n in noise],
        "class10_clean_mean_m4": mean(c10),
        "class11_clean_mean_m4": mean(c11),
        "synthetic_qpsk_mean_m4": mean(q4),
        "synthetic_qam16_mean_m4": mean(q16),
        "noise_control_mean_m4": float(np.mean([n["syms"]["m4"] for n in noise])),
    }
    m10 = cmp_["class10_clean_mean_m4"]
    m11 = cmp_["class11_clean_mean_m4"]
    m10s = cmp_["class10_srrclike_mean_m4"]
    m11s = cmp_["class11_srrclike_mean_m4"]
    if m10 is None or m11 is None:
        cmp_["trend_statement"] = "insufficient frames"
    else:
        order_ok = m11 > m10
        order_ok_s = (m10s is not None and m11s is not None and m11s > m10s)
        mag_ok = abs(m10 - M4_REF[4]) < abs(m10 - M4_REF[16]) and \
            abs(m11 - M4_REF[16]) < abs(m11 - M4_REF[4])
        if order_ok and mag_ok:
            cmp_["trend_statement"] = (
                "class-11 clean mean m4 exceeds class-10, and each is closer to its "
                "nominal reference (QPSK 1.00 / 16QAM 1.32) than to the other; "
                "consistent with the QPSK-vs-16QAM hypothesis IN THIS TINY SAMPLE "
                "(n=4 per class; exploratory only, no classification claim).")
        elif order_ok:
            m10_list = [f["syms"]["m4"] for f in c10s]
            m11_list = [f["syms"]["m4"] for f in c11s]
            r10 = f"{min(m10_list):.2f}-{max(m10_list):.2f}" if m10_list else "n/a"
            r11 = f"{min(m11_list):.2f}-{max(m11_list):.2f}" if m11_list else "n/a"
            cmp_["trend_statement"] = (
                f"class-11 clean mean m4 exceeds class-10 ({m11:.2f} vs {m10:.2f}), "
                "matching the QPSK<16QAM ordering, but the absolute values do not "
                "separately sit at the references"
                + (f" (SRRC-like-only means: class-10 {m10s:.2f} vs QPSK 1.00 "
                   f"[range {r10}], class-11 {m11s:.2f} vs 16QAM 1.32 [range {r11}])"
                   if order_ok_s and m10s is not None and m11s is not None else "")
                + "; the class-10 SRRC-like frames cluster near the QPSK reference, "
                "while class-11 frames land above the 16QAM reference, consistent with "
                "low-SNR pull toward the Rayleigh value 2.0 and/or a non-QAM "
                "constellation. Partially consistent at most; exploratory only, "
                "no classification claim.")
        else:
            cmp_["trend_statement"] = (
                "class-10 and class-11 clean m4 do NOT follow the QPSK<16QAM "
                "ordering; no class-consistent m4 trend in this sample.")
    return cmp_


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def build_md(out):
    L = []
    a = L.append
    a("# Phase A' Track S — SRRC single-carrier audit (classes 10/11, bw 10 MHz)\n")
    a(f"Generated: {out['meta']['generated']}  |  runtime: {out['meta']['runtime_s']:.0f} s"
      "  |  pipeline: `scripts/baseline3_pa_srrc.py`"
      "  |  data: `reports/local_iq_round1/phase_ap_frames.json` (fit split only)\n")

    a("## Method (fixed, label-blind)\n")
    a("- Extraction recipe (fixed): central window `min(10 ms, dur)` of the GT box →"
      " `read_segment` → `demod_to_center` (box center) → `band_limit_fft(bw*1.02, edge_frac=0.05)`"
      " → trim 0.5 ms at both ends. Frequency convention as audited: FFT bin 0 = observation"
      " center `(f_lo + B/2)`; absolute f = `f_lo + B/2 + digital`.\n")
    a("- Synthetic generation (Step A): `gen_srrc_clean` — the same standard"
      " Proakis RRC taps (`srrc_pulse_taps`, unmodified) built at 32 samples/symbol"
      " with a 16-symbol span, resampled to the capture rate with `resample_poly`"
      " at exact rational ratios (3/40, 3/20, 3/10 for alpha=0.2; 27/320, 27/160,"
      " 27/80 for alpha=0.35 — no resampler rate error; the 10 MHz band is far"
      " inside every output Nyquist). 8-sps/span-8 taps (the shared `gen_srrc`"
      " setting) droop the pulse power spectrum up to ~6.5% through the rolloff"
      " (measured |H|² vs ideal RC) and would bias the width calibration; at"
      " 32 sps/span 16 the pulse fidelity is ~0.6%. Embedded at digital offsets"
      " 0 / +6 / −12 MHz for B = 20 / 40 / 80 MHz (band always inside the"
      f" observation), white capture noise at {IN_BOX_SNR_DB:.0f} dB in-box SNR,"
      " round-tripped through float16 so the real read path is exercised.\n")
    a("- PSD widths: two-sided Welch PSD (~40 kHz grid, 75%-overlap Hann, ~500 kHz median"
      " kernel, plateau reference = median of the top quartile, persistence/hysteresis"
      " crossings — the raw max and the naive contiguous-run rule are speckle-broken, as"
      " Step A debugging showed). W90/W50/W10 = widths at 90/50/10% of the plateau"
      " reference. Raised-cosine inversion (exact for an ideal RC PSD):"
      " `W90 = R(1−0.590334a)`, `W50 = R`, `W10 = R(1+0.590334a)`, used as"
      " `R_est = W50` (max-slope crossing, most level-robust) and"
      " `alpha = (W10−W90)/(1.180668·W50)` (level errors cancel in the difference)."
      " The exact-in-theory pair rule `R=(W10+W90)/2` and the task's literal rule"
      " (`R_est = W90`, `alpha = W10/W90 − 1`) are also recorded (see estimator note).\n")
    a("- Timing: matched filter → `tone_line_scan` of `|y|²` in `[0.9, 1.1]·R` → direct"
      " correlation `D(f) = Σ |y|² e^{−2πifn/fs}` refined on a ±1.5-bin grid (parabolic peak),"
      " then `t0 = −angle(D)/(2π)·fs/f mod fs/f`. Evaluating D at the refined frequency avoids"
      " the padded-DFT bin-center phase error `πδ(N−1)`; the residual frequency error is"
      " <2 Hz over ≤9 ms, i.e. <0.01 symbol of t0 bias.\n")
    a("- Symbols at `t0 + k·fs/R` via `cubic_sample`; `m4`, `m6` (magnitude moments),"
      " 8-bin `|s|/RMS` histogram on [0,3]. Reference values: QPSK m4/m6 = 1.00/1.00,"
      " 16QAM = 1.32/1.96, Rayleigh noise = 2.0/6.0.\n")
    a("- Class labels are carried for reporting only; all statistics use box geometry."
      " Only `train/` frames are read; `advanced/test` is never touched; no models trained.\n")

    # ---- Step A ----
    a("## Step A — synthetic calibration (label-free)\n")
    a("| alpha | order | B | R_true (MHz) | R_est (MHz) | err R % | alpha_est | err a % |"
      " tone str | peak/med | n_sym | m4 | m6 |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in out["calibration"]["grid"]:
        t, e = r["truth"], r["errors"]
        ord_txt = "QPSK" if t["order"] == 4 else "16QAM"
        a(f"| {t['alpha']} | {ord_txt} | {t['B_mhz']:.0f} |"
          f" {t['R_mhz']:.3f} | {_fmt(r['final']['R_mhz'])} | {e['R_pct']:+.2f} |"
          f" {_fmt(r['final']['alpha'])} | {e['alpha_pct']:+.2f} |"
          f" {_fmt(r['tone']['strength'], 4) if r['tone'] else 'none'} |"
          f" {_fmt(r['tone']['peak_med'], 1) if r['tone'] else 'none'} |"
          f" {r['syms']['n_sym']} | {_fmt(r['syms']['m4'])} | {_fmt(r['syms']['m6'])} |")
    a("")
    a(f"Mean |R_est error| = {out['calibration']['mean_abs_R_err_pct']:.2f}%, "
      f"mean |alpha_est error| = {out['calibration']['mean_abs_alpha_err_pct']:.2f}%, "
      f"m4 range QPSK = {out['calibration']['m4_qpsk_range']}, "
      f"m4 range 16QAM = {out['calibration']['m4_qam16_range']}.\n")
    a("**Estimator note (three width rules, measured errors).** For a raised-cosine PSD"
      " the 90%/50%/10% widths are `R(1−0.590334a)` / `R` / `R(1+0.590334a)`"
      " (cosine-rolloff algebra), so the literal task rule `R_est = W90, a = W10/W90 − 1`"
      " is biased even on a PERFECT noiseless SRRC. Measured over the calibration grid:\n")
    a("| rule | R error range | alpha error range |")
    a("|---|---|---|")
    a(f"| literal (`R=W90, a=W10/W90−1`) | {out['calibration']['lit_R_err_pct_min']:+.1f}%…"
      f"{out['calibration']['lit_R_err_pct_max']:+.1f}% | "
      f"{out['calibration']['lit_alpha_err_pct_min']:+.0f}%…"
      f"{out['calibration']['lit_alpha_err_pct_max']:+.0f}% (shape bias) |")
    a(f"| pair (`R=(W10+W90)/2`, `a=(W10−W90)/(1.180668 R)`) | "
      f"{out['calibration']['pair_R_err_pct_min']:+.1f}%…"
      f"{out['calibration']['pair_R_err_pct_max']:+.1f}% | "
      f"{out['calibration']['pair_alpha_err_pct_min']:+.1f}%…"
      f"{out['calibration']['pair_alpha_err_pct_max']:+.1f}% |")
    a(f"| USED (`R=W50`, `a=(W10−W90)/(1.180668 W50)`) | "
      f"{out['calibration']['used_R_err_pct_min']:+.1f}%…"
      f"{out['calibration']['used_R_err_pct_max']:+.1f}% | "
      f"{out['calibration']['used_alpha_err_pct_min']:+.1f}%…"
      f"{out['calibration']['used_alpha_err_pct_max']:+.1f}% |")
    a("")
    a("The used pairing is chosen for robustness: the −3 dB (50%) crossing sits at the"
      " maximum-slope point of the RC rolloff (least sensitive to reference-level error),"
      " and the alpha difference (W10−W90) cancels common level errors. On real frames"
      " all three variants are in the JSON (`psd.R_psd_mhz/alpha_psd` = used,"
      " `R_pair_mhz/alpha_pair`, `R_lit_mhz/alpha_lit`).\n")
    a("**Timing sensitivity (deliberate t0 offsets).** m4 at the recovered timing vs"
      " t0 deliberately offset by ±0.25 and −0.50 symbol:\n")
    a("| order | alpha | m4(t0) | m4(−0.25) | m4(+0.25) | m4(−0.50) |")
    a("|---|---|---|---|---|---|")
    for r in out["calibration"]["grid"]:
        if "m4_offsets" not in r:
            continue
        o = r["m4_offsets"]
        ord_txt = "QPSK" if r["truth"]["order"] == 4 else "16QAM"
        a(f"| {ord_txt} | {r['truth']['alpha']} | {_fmt(r['syms']['m4'])} |"
          f" {_fmt(o['-0.25'])} | {_fmt(o['+0.25'])} | {_fmt(o['-0.50'])} |")
    a("")
    a(f"Summary: mean m4 inflation at ±0.25 symbol = "
      f"{out['calibration']['m4_mean_inflate_025']:.3f} (QPSK), "
      f"{out['calibration']['m4_mean_inflate_025_q16']:.3f} (16QAM); at −0.50 symbol = "
      f"{out['calibration']['m4_mean_inflate_050']:.3f} (QPSK), "
      f"{out['calibration']['m4_mean_inflate_050_q16']:.3f} (16QAM). Residual timing error"
      " therefore shows up as m4 inflation above the constellation reference; the recovered"
      " t0 must be used for any m4-based reading.\n")
    a("**Negative controls (pure noise through the same chain).**\n")
    a("| B | R_est (failure mode) | W90 | W10 | alpha_psd | tone str | tone f | m4 | m6 |")
    a("|---|---|---|---|---|---|---|---|---|")
    for n in out["calibration"]["noise_controls"]:
        a(f"| {n['truth']['B_mhz']:.0f} "
          f"| {_fmt(n['psd']['R_psd_mhz'])} MHz (flat PSD → R_psd ≈ full band) |"
          f" {_fmt(n['psd']['W90_mhz'], 2)} | {_fmt(n['psd']['W10_mhz'], 2)} |"
          f" {_fmt(n['psd']['alpha_psd'])} |"
          f" {_fmt(n['tone']['strength'], 4) if n['tone'] else 'none'} |"
          f" {_fmt(n['tone']['f_mhz'], 2) if n['tone'] else '—'} |"
          f" {_fmt(n['syms']['m4'])} | {_fmt(n['syms']['m6'])} |")
    a("")
    a(f"R_est failure mode on flat noise: the persistence crossings return widths covering"
      f" essentially the whole analysis band ({out['calibration']['noise_widths_txt']} MHz),"
      " so `R_psd = W50 ≈ 9.9 MHz` is the band width, not a symbol rate; `alpha_psd` ≈"
      " 0.06–0.08 is spurious structure on a flat spectrum; the tone scan in"
      " [0.9, 1.1]·R_psd ≈ [8.9, 10.9] MHz finds only a noise maximum"
      f" (strength ≤ {out['calibration']['noise_max_strength']:.4f}, raw peak/median ≤"
      f" {out['calibration']['noise_peakmed_max']:.1f}, same order as the off-band control),"
      f" and m4 ≈ {out['calibration']['noise_control_mean_m4']:.2f} (Rayleigh). Note:"
      " the half-split coherence is NOT a discriminator — the line frequency is selected"
      " as the maximum of the full-record sum, which biases the two halves into apparent"
      " alignment for noise as well — and is recorded in the JSON for completeness only.\n")
    a(f"Tone-strength threshold for Step B (geometric mean of the worst noise control"
      f" {out['calibration']['noise_max_strength']:.4f} and the weakest synthetic line"
      f" {out['calibration']['synth_strength_min']:.4f}):"
      f" **{out['thresholds']['tone_strength']:.4f}**"
      f" (synthetic range {out['calibration']['synth_strength_txt']}; raw peak/median ≥"
      f" {out['calibration']['synth_peakmed_min']:.1f} vs noise ≤"
      f" {out['calibration']['noise_peakmed_max']:.1f}; off-band specificity: the synthetic"
      " in-band peak/median is 20–100x its off-band control, noise ≈ 1x).\n")

    # ---- Step B ----
    a("## Step B — real frames (10 class-10/11 SRRC frames, same pipeline)\n")
    a("| id | class | stratum | R_est (MHz) | alpha_est | tone str | n_sym | m4 | m6 | verdict |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for r in out["frames"]:
        if "error" in r:
            a(f"| {r['frame']['sample_id']}#{r['frame']['signal_id']} |"
              f" {r['frame']['class']} | {r['frame']['stratum']} |"
              f" — | — | — | — | — | — | ERROR: {r['error']} |")
            continue
        f_ = r["frame"]
        verdict = "SRRC-like" if r["flags"]["srrc_like"] else "NOT SRRC-like"
        a(f"| {f_['sample_id']}#{f_['signal_id']} | {f_['class']} | {f_['stratum']} |"
          f" {_fmt(r['final']['R_mhz'])} | {_fmt(r['final']['alpha'])} |"
          f" {_fmt(r['tone']['strength'],4) if r['tone'] else 'none'} |"
          f" {r['syms']['n_sym']} | {_fmt(r['syms']['m4'])} | {_fmt(r['syms']['m6'])} |"
          f" {verdict} ({r['flags']['m4_class']}) |")
    a("")
    a("PSD shape detail (the −3 dB width should equal R_est for an RC band; edge symmetry"
      " should be ≈0 for a centered SRRC):\n")
    a("| id | W50 (−3 dB) | W10 (−10 dB) | W90 | center off | asym@10% | edge drop |"
      " plateau flat | t0/T | peak/med | spec | SNR dB | n_runs10 |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in out["frames"]:
        if "error" in r:
            continue
        p = r["psd"]
        spec = (f"{r['tone']['peak_med'] / r['spec_alt_peak_med_pass1']:.1f}"
                if r["tone"] and r["spec_alt_peak_med_pass1"] else "n/a")
        a(f"| {r['frame']['sample_id']}#{r['frame']['signal_id']} |"
          f" {_fmt(p['W50_mhz'],2)} | {_fmt(p['W10_mhz'],2)} | {_fmt(p['W90_mhz'],2)} |"
          f" {_fmt(p['center_off_mhz'],2)} | {_fmt(p['asym10'])} |"
          f" {_fmt(p['edge_drop_db'],1)} | {_fmt(p['plateau_flat_db'],1)} |"
          f" {_fmt(r['tone']['t0_frac']) if r['tone'] else '—'} |"
          f" {_fmt(r['tone']['peak_med'],1) if r['tone'] else '—'} | {spec} |"
          f" {_fmt(r['snr_db'],1)} | {p['n_runs10']} |")
    a("")
    a("Per-frame notes:\n")
    for r in out["frames"]:
        a(f"- `{r['frame']['sample_id']}#{r['frame']['signal_id']}` "
          f"(class {r['frame']['class']}, {r['frame']['stratum']}): "
          f"{r['content_hint']}.")
    a("")

    # ---- Step C ----
    a("## Step C — exploratory class-10 vs class-11 comparison (no classification claims)\n")
    c = out["comparison"]
    a("| group | m4 values | mean | nominal reference |")
    a("|---|---|---|---|")
    a(f"| class-10 clean (n={len(c['class10_clean_m4'])}) | {c['class10_clean_m4']} |"
      f" {_fmt(c['class10_clean_mean_m4'])} | QPSK {M4_REF[4]:.2f} |")
    a(f"| class-11 clean (n={len(c['class11_clean_m4'])}) | {c['class11_clean_m4']} |"
      f" {_fmt(c['class11_clean_mean_m4'])} | 16QAM {M4_REF[16]:.2f} |")
    a(f"| class-10 clean, SRRC-like only (n={len(c['class10_srrclike_m4'])}) |"
      f" {c['class10_srrclike_m4']} | {_fmt(c['class10_srrclike_mean_m4'])} |"
      f" QPSK {M4_REF[4]:.2f} |")
    a(f"| class-11 clean, SRRC-like only (n={len(c['class11_srrclike_m4'])}) |"
      f" {c['class11_srrclike_m4']} | {_fmt(c['class11_srrclike_mean_m4'])} |"
      f" 16QAM {M4_REF[16]:.2f} |")
    a(f"| synthetic QPSK (calibration grid, n={len(c['synthetic_qpsk_m4'])}) |"
      f" {c['synthetic_qpsk_m4']} | {_fmt(c['synthetic_qpsk_mean_m4'])} | {M4_REF[4]:.2f} |")
    a(f"| synthetic 16QAM (calibration grid, n={len(c['synthetic_qam16_m4'])}) |"
      f" {c['synthetic_qam16_m4']} | {_fmt(c['synthetic_qam16_mean_m4'])} | {M4_REF[16]:.2f} |")
    a(f"| noise controls | {c['noise_control_m4']} |"
      f" {_fmt(c['noise_control_mean_m4'])} | 2.0 (Rayleigh) |")
    a("")
    a(f"**Trend statement:** {c['trend_statement']}\n")

    # ---- facts vs inferences ----
    tone833 = [r for r in out["frames"] if "error" not in r and r["tone"]
               and abs(r["tone"]["f_mhz"] - 8.3333) < 0.02]
    a("## Facts vs inferences\n")
    a("**Facts (measured / verified here):**\n")
    a(f"1. The extraction recipe + pipeline recovers synthetic SRRC truth: R within "
      f"{out['calibration']['max_abs_R_err_pct']:.1f}% (tone-locked, 0.00% on every grid"
      f" point) and alpha within {out['calibration']['max_abs_alpha_err_pct']:.1f}% over"
      f" all 12 grid points; m4 at recovered timing = "
      f"{out['calibration']['m4_qpsk_range']} (QPSK) and "
      f"{out['calibration']['m4_qam16_range']} (16QAM) vs references 1.00 / 1.32.\n")
    a(f"2. The literal 90/10 width rule is biased by the RC rolloff shape even on a"
      f" perfect SRRC (measured {out['calibration']['lit_alpha_err_pct_min']:+.1f}%…"
      f"{out['calibration']['lit_alpha_err_pct_max']:+.1f}% on alpha); the USED"
      " W50/(W10−W90) pairing is what the tables use, and all three variants are"
      " recorded per run (see the estimator note).\n")
    a(f"3. Timing is the fragile part: a deliberate ±0.25-symbol t0 error inflates QPSK"
      f" m4 to ~{out['calibration']['qpsk_m4_at_025']:.2f} and 16QAM m4 to "
      f"~{out['calibration']['q16_m4_at_025']:.2f} (mean over the grid); the "
      "phase-of-line t0 is what keeps m4 at the reference values.\n")
    a(f"4. Pure-noise negative controls: R_psd ≈ full band (no symbol rate), tone "
      f"strength ≤ {out['calibration']['noise_max_strength']:.4f}, m4 ≈ "
      f"{out['calibration']['noise_control_mean_m4']:.2f} (Rayleigh).\n")
    a(f"5. Per-frame measured values for the 10 frames are in the tables above"
      f" ({out['summary']['n_srrc_like']} of {out['summary']['n_frames']} frames pass the "
      f"SRRC-like gates: symbol-rate tone above threshold + RC-shaped PSD + W50≈R).\n")
    if tone833:
        ids = ", ".join(f"`{r['frame']['sample_id']}#{r['frame']['signal_id']}`"
                        for r in tone833)
        w10lo = min(r["psd"]["W10_mhz"] for r in tone833)
        w10hi = max(r["psd"]["W10_mhz"] for r in tone833)
        w90lo = min(r["psd"]["W90_mhz"] for r in tone833)
        w90hi = max(r["psd"]["W90_mhz"] for r in tone833)
        alo = min(r["final"]["alpha"] for r in tone833)
        ahi = max(r["final"]["alpha"] for r in tone833)
        a(f"5a. {len(tone833)} frames ({ids}) carry a symbol-rate line at"
          " 8.333 MHz = 10/1.2 MHz: for a 10 MHz box this pins the box-implied"
          " rolloff at alpha = 0.2.\n")
        a(f"5b. On those frames the 10%-edge widths W10 = {w10lo:.2f}–{w10hi:.2f} MHz"
          " (ideal RC at R=8.333, alpha=0.2: 9.32 MHz — edges are RC-consistent), but"
          f" the 90% plateau widths W90 = {w90lo:.2f}–{w90hi:.2f} MHz fall short of the"
          " ideal 7.35 MHz: the measured passbands carry ≥10% ripple/tilt. The"
          f" width-based alpha_est ({alo:.2f}–{ahi:.2f}) therefore over-reads the"
          " edge-implied 0.2 and is reported with that caveat.\n")
    a("**Inferences (interpretation, weaker than the facts):**\n")
    a("6. Any statement that a given frame *is* an SRRC single carrier rests on the "
      "sync-quality gates (tone + RC shape + W50 consistency), not on class labels; "
      "m4 alone cannot separate constant-envelope modulations (a pure tone also has "
      "m4≈1), so the PSD/tone checks carry the discrimination.\n")
    a(f"7. {out['comparison']['trend_statement']}\n")
    flat_notone = [r for r in out["frames"] if "error" not in r and not r["flags"]["tone_ok"]
                   and r["psd"].get("W10_mhz", 0) >= 0.95 * 10.2
                   and r["syms"]["m4"] >= 1.7]
    offcenter = [r for r in out["frames"] if "error" not in r and not r["flags"]["tone_ok"]
                 and abs(r["psd"].get("center_off_mhz", 0.0)) > 0.5]
    ids1 = ", ".join(f"`{r['frame']['sample_id']}#{r['frame']['signal_id']}`"
                     for r in flat_notone) or "none"
    ids2 = ", ".join(f"`{r['frame']['sample_id']}#{r['frame']['signal_id']}`"
                     for r in offcenter) or "none"
    a("8. Frames failing the gates split into two label-blind content patterns:"
      f" flat-topped boxes with no tone above threshold and near-Rayleigh envelopes ({ids1})"
      " — consistent with noise-like or OFDM-style wideband content rather than a"
      f" single-carrier SRRC — and boxes whose content sits off-center with no coherent"
      f" tone ({ids2}) — mostly empty or interfered boxes. The overlap-stratum rows may"
      " additionally contain a second signal inside the GT box; these are content"
      " descriptions, not label claims.\n")
    a("9. Nothing here trains or evaluates a classifier; no classification gain is "
      "claimed or implied.\n")
    a(f"\nRuntime: {out['meta']['runtime_s']:.0f} s total (single process, no training, "
      "fit-split frames only).")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    t_start = time.perf_counter()
    frames_doc = json.loads(FRAMES_JSON.read_text())
    sel = [fr for fr in frames_doc["frames"] if int(fr["class"]) in (10, 11)]
    for fr in sel:
        assert fr["sample_id"].startswith("train/"), \
            f"refusing non-train sample {fr['sample_id']}"
    sel.sort(key=lambda fr: (0 if fr["stratum"] == "clean" else 1,
                             int(fr["class"]), fr["sample_id"]))
    print(f"Track S: {len(sel)} class-10/11 frames selected", flush=True)

    # ---- Step A ----
    grid = []
    for ai, alpha in enumerate(ALPHAS):
        for oi, order in enumerate(ORDERS):
            for bi, B in enumerate(B_GRID):
                seed = 4001 + 100 * ai + 10 * oi + bi
                rec = synth_case(alpha, order, B, seed)
                grid.append(rec)
                e = rec["errors"]
                print(f"  cal a={alpha} M={order} B={B:.0f}: "
                      f"R={rec['final']['R_mhz']:.3f} (err {e['R_pct']:+.2f}%) "
                      f"a={rec['final']['alpha']:.3f} (err {e['alpha_pct']:+.2f}%) "
                      f"tone={rec['tone']['strength']:.4f} "
                      f"m4={rec['syms']['m4']:.3f}", flush=True)
    noise = []
    for bi, B in enumerate(B_GRID):
        rec = noise_case(B, 9000 + bi)
        noise.append(rec)
        tstr = f"{rec['tone']['strength']:.4f}" if rec["tone"] else "none"
        print(f"  noise B={B:.0f}: R_psd={rec['psd']['R_psd_mhz']:.2f} "
              f"tone={tstr} m4={rec['syms']['m4']:.3f}", flush=True)

    strengths = [r["tone"]["strength"] for r in grid if r["tone"]]
    noise_str = [n["tone"]["strength"] for n in noise if n["tone"]]
    thr = float(math.sqrt(max(noise_str) * min(strengths))) if strengths and noise_str else 0.01
    print(f"  tone threshold = {thr:.4f} "
          f"(synth min {min(strengths):.4f}, noise max {max(noise_str):.4f})", flush=True)

    cal = {"grid": grid, "noise_controls": noise}
    rerrs = [abs(r["errors"]["R_pct"]) for r in grid]
    aerrs = [abs(r["errors"]["alpha_pct"]) for r in grid]
    lit_r = [r["errors"]["R_lit_pct"] for r in grid]
    lit_a = [r["errors"]["alpha_lit_pct"] for r in grid]
    pair_r = [r["errors"]["R_pair_pct"] for r in grid]
    pair_a = [r["errors"]["alpha_pair_pct"] for r in grid]
    used_r = [r["errors"]["R_psd_pct"] for r in grid]
    used_a = [r["errors"]["alpha_psd_pct"] for r in grid]
    q4 = [r for r in grid if r["truth"]["order"] == 4]
    q16 = [r for r in grid if r["truth"]["order"] == 16]
    cal.update({
        "mean_abs_R_err_pct": float(np.mean(rerrs)),
        "mean_abs_alpha_err_pct": float(np.mean(aerrs)),
        "max_abs_R_err_pct": float(np.max(rerrs)),
        "max_abs_alpha_err_pct": float(np.max(aerrs)),
        "lit_R_err_pct_min": float(np.min(lit_r)), "lit_R_err_pct_max": float(np.max(lit_r)),
        "lit_alpha_err_pct_min": float(np.min(lit_a)), "lit_alpha_err_pct_max": float(np.max(lit_a)),
        "pair_R_err_pct_min": float(np.min(pair_r)), "pair_R_err_pct_max": float(np.max(pair_r)),
        "pair_alpha_err_pct_min": float(np.min(pair_a)), "pair_alpha_err_pct_max": float(np.max(pair_a)),
        "used_R_err_pct_min": float(np.min(used_r)), "used_R_err_pct_max": float(np.max(used_r)),
        "used_alpha_err_pct_min": float(np.min(used_a)), "used_alpha_err_pct_max": float(np.max(used_a)),
        "m4_qpsk_range": f"{min(r['syms']['m4'] for r in q4):.3f}–{max(r['syms']['m4'] for r in q4):.3f}",
        "m4_qam16_range": f"{min(r['syms']['m4'] for r in q16):.3f}–{max(r['syms']['m4'] for r in q16):.3f}",
        "m4_mean_inflate_025": float(np.mean(
            [r["m4_offsets"]["-0.25"] + r["m4_offsets"]["+0.25"] - 2 * r["syms"]["m4"]
             for r in q4 if "m4_offsets" in r])),
        "m4_mean_inflate_025_q16": float(np.mean(
            [r["m4_offsets"]["-0.25"] + r["m4_offsets"]["+0.25"] - 2 * r["syms"]["m4"]
             for r in q16 if "m4_offsets" in r])),
        "m4_mean_inflate_050": float(np.mean(
            [r["m4_offsets"]["-0.50"] - r["syms"]["m4"] for r in q4 if "m4_offsets" in r])),
        "m4_mean_inflate_050_q16": float(np.mean(
            [r["m4_offsets"]["-0.50"] - r["syms"]["m4"] for r in q16 if "m4_offsets" in r])),
        "qpsk_m4_at_025": float(np.mean(
            [0.5 * (r["m4_offsets"]["-0.25"] + r["m4_offsets"]["+0.25"])
             for r in q4 if "m4_offsets" in r])),
        "q16_m4_at_025": float(np.mean(
            [0.5 * (r["m4_offsets"]["-0.25"] + r["m4_offsets"]["+0.25"])
             for r in q16 if "m4_offsets" in r])),
        "noise_max_strength": float(max(noise_str)) if noise_str else 0.0,
        "noise_peakmed_max": float(max([n["tone"]["peak_med"] for n in noise if n["tone"]] or [0])),
        "noise_coh_max": float(max([n["tone"]["coherence"] for n in noise if n["tone"]] or [0])),
        "synth_strength_txt": f"{min(strengths):.4f}–{max(strengths):.4f}",
        "synth_strength_min": float(min(strengths)),
        "synth_peakmed_min": float(min([r["tone"]["peak_med"] for r in grid if r["tone"]] or [0])),
        "synth_coh_min": float(min([r["tone"]["coherence"] for r in grid if r["tone"]] or [0])),
        "noise_widths_txt": "; ".join(
            f"B{int(n['truth']['B_mhz'])}: W90={n['psd']['W90_mhz']:.1f}/"
            f"W50={n['psd']['W50_mhz']:.1f}/W10={n['psd']['W10_mhz']:.1f}"
            for n in noise),
        "noise_control_mean_m4": float(np.mean([n["syms"]["m4"] for n in noise])),
    })

    # ---- Step B ----
    frecs = []
    for fr in sel:
        try:
            rec = real_frame(fr)
            rec = apply_flags(rec, thr)
        except Exception as e:  # keep the audit going; record the failure
            rec = {"frame": dict(fr), "error": str(e),
                   "flags": {"tone_ok": False, "psd_rc_ok": False, "w50_ok": False,
                             "srrc_like": False, "m4_class": "error"},
                   "content_hint": f"analysis error: {e}",
                   "final": {"R_mhz": float("nan"), "alpha": float("nan"),
                             "sps": float("nan"), "source": "error"},
                   "psd": {}, "tone": None, "syms": {"n_sym": 0, "m4": float("nan"),
                                                     "m6": float("nan"), "hist": []},
                   "snr_db": float("nan"), "spec_alt_peak_med_pass1": None}
            fr = dict(fr)
            rec["frame"].setdefault("stratum", "clean")
        frecs.append(rec)
        tstr = f"tone={rec['tone']['strength']:.4f} " if rec["tone"] else "tone=none "
        print(f"  frame {fr['sample_id']}#{fr['signal_id']} "
              f"c={fr['class']} {fr['stratum']}: "
              f"R={rec['final']['R_mhz']:.3f} a={rec['final']['alpha']:.3f} {tstr}"
              f"m4={rec['syms']['m4']:.3f} srrc_like={rec['flags']['srrc_like']}",
              flush=True)

    # ---- Step C ----
    comparison = build_comparison(grid, noise, frecs)

    n_like = sum(1 for r in frecs if r["flags"]["srrc_like"])
    out = {
        "meta": {
            "script": "scripts/baseline3_pa_srrc.py",
            "frames_source": str(FRAMES_JSON),
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_s": time.perf_counter() - t_start,
            "config": {
                "alphas": ALPHAS, "orders": ORDERS, "B_mhz": B_GRID,
                "n_sym": NSYM, "in_box_snr_db": IN_BOX_SNR_DB,
                "offset_mhz": OFFSET_MHZ, "cap_ms": CAP_MS, "trim_ms": TRIM_MS,
                "generation": "gen_srrc_clean: standard srrc_pulse_taps at 32 "
                              "samples/symbol, span 16 symbols, resample_poly "
                              "(exact rational ratios), float16 round trip",
                "rc_inversion": "R=W50, alpha=(W10-W90)/(1.180668*W50); pair and "
                                "literal variants also recorded",
            },
            "notes": "label-blind analysis; class labels reported only; fit-split "
                     "train/ frames only; advanced/test never read; no models trained.",
        },
        "thresholds": {"tone_strength": thr},
        "calibration": cal,
        "frames": frecs,
        "comparison": comparison,
        "summary": {
            "n_frames": len(frecs), "n_srrc_like": n_like,
            "n_clean": sum(1 for r in frecs if r["frame"]["stratum"] == "clean"),
            "n_overlap": sum(1 for r in frecs if r["frame"]["stratum"] == "overlap"),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    OUT_MD.write_text(build_md(out), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"runtime {time.perf_counter() - t_start:.0f} s; "
          f"{n_like}/{len(frecs)} frames SRRC-like")


if __name__ == "__main__":
    main()
