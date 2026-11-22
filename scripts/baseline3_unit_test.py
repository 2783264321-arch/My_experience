#!/usr/bin/env python3
"""Phase A' step 1: unit tests for the local IQ extraction chain.

Validates, with synthetic captures of known content:
  T1  full chain via a float16 .bin file (write -> read_segment -> fixed
      demod_to_center -> band_limit_fft) for center / off-center / negative /
      near-edge target frequencies;
  T2  B-independence: the same absolute tone extracted in B in {20,30,40,50,80};
  T3  band-limited modulated signal occupying exactly its box: energy
      preservation and out-of-band suppression;
  T4  changing observation bandwidth introduces no systematic shift;
  T5  complex CP detector calibration: synthetic OFDM positives (12.8/0.8 and
      3.2/0.8 numerologies) across SNR, plus pure-noise negatives;
  T6  0.2 ms IQ crop context and windows confined to the GT time interval.

Run:  PYTHONPATH=. python scripts/baseline3_unit_test.py
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np

from scripts.baseline3_dsp import (
    read_segment, demod_to_center, band_limit_fft,
    cp_scan_complex, gen_wifi_ofdm, qam_symbols,
)

from scripts.baseline3srrc_phase0_verify import crop_and_resample, windows_from_crop

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}  {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def synth_capture(n, fs, comps):
    """Complex baseband capture, zero frequency = observation center."""
    t = np.arange(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.complex128)
    for f_hz, amp in comps:
        x += amp * np.exp(2j * np.pi * f_hz * t / fs)
    return (x / np.max(np.abs(x))).astype(np.complex64)


def peak_offset_hz(xb, fs):
    spec = np.abs(np.fft.fft(xb)) ** 2
    k = int(np.argmax(spec))
    fr = np.fft.fftfreq(len(xb), 1.0 / fs)
    return fr[k]


def energy_ratio(x_in, xb):
    return float(np.mean(np.abs(xb) ** 2) / np.mean(np.abs(x_in) ** 2))


def main():
    fs = 40e6
    flo = 2432.0
    obs_c = flo + 20.0
    n = int(5e-3 * fs)

    # ---- T1: full chain through a real float16 .bin ----
    print("T1: full chain via .bin, center / off-center / negative / edge")
    cases = [
        ("center", 2447.0, 1.0),        # +15 MHz from obs center
        ("off+", 2457.0, 1.0),          # +25
        ("neg", 2437.0, 1.0),           # -15
        ("edge_lo", 2432.6, 1.0),       # -19.4, near low edge
        ("edge_hi", 2471.4, 1.0),       # +19.4, near high edge
    ]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "synth.bin"
        x = synth_capture(n, fs, [((f - obs_c) * 1e6, a) for _nm, f, a in cases])
        # interleaved float16 I/Q like the dataset
        raw = np.empty(2 * n, dtype="<f2")
        raw[0::2] = x.real
        raw[1::2] = x.imag
        raw.tofile(path)
        for name, f_abs, _amp in cases:
            xseg = read_segment(str(path), 0, n)
            fc = f_abs * 1e6
            xb = demod_to_center(xseg, fs, fc, flo * 1e6)
            xb = band_limit_fft(xb, fs, 1.0e6 * 1.05, edge_frac=0.05)
            off = abs(peak_offset_hz(xb, fs))
            keep = energy_ratio(demod_to_center(xseg, fs, fc, flo * 1e6), xb)
            check(f"{name} f={f_abs}", off < 30e3,
                  f"peakOff={off/1e3:.1f} kHz keep={keep:.2f}")

    # ---- T2: B-independence ----
    print("T2: same absolute tone, B in {20,30,40,50,80}")
    f_abs = 2447.0
    for B in (20.0, 30.0, 40.0, 50.0, 80.0):
        fs2 = B * 1e6
        flo2 = 2432.0
        c2 = flo2 + B / 2
        n2 = int(3e-3 * fs2)
        x2 = synth_capture(n2, fs2, [((f_abs - c2) * 1e6, 1.0)])
        xb = demod_to_center(x2, fs2, f_abs * 1e6, flo2 * 1e6)
        xb = band_limit_fft(xb, fs2, 1.0e6 * 1.05, edge_frac=0.05)
        off = abs(peak_offset_hz(xb, fs2))
        check(f"B={B:.0f}", off < 30e3, f"peakOff={off/1e3:.1f} kHz")

    # ---- T3: band-limited modulated signal filling its box ----
    print("T3: FFT-bandlimited QPSK (9 MHz of a 10 MHz box), B=40")
    rng = np.random.default_rng(3)
    n_sym = 40000
    syms = qam_symbols(4, n_sym, rng)
    up = np.zeros(4 * n_sym, dtype=np.complex64)
    up[::4] = syms
    base = np.fft.ifft(np.fft.fft(up) * (np.abs(np.fft.fftfreq(len(up), 1.0 / fs)) <= 4.5e6))
    base = (base / np.abs(base).max()).astype(np.complex64)
    # place at abs 2447 (= -5 MHz from obs center 2452): up-convert by -5 MHz
    t = np.arange(len(base), dtype=np.float64)
    x3 = (base * np.exp(-2j * np.pi * 5e6 * t / fs)).astype(np.complex64)
    x3 = x3[:n]
    x_demod = demod_to_center(x3, fs, 2447.0 * 1e6, flo * 1e6)
    xb = band_limit_fft(x_demod, fs, 10e6 * 1.05, edge_frac=0.05)
    keep = energy_ratio(x_demod, xb)
    spec = np.abs(np.fft.fft(xb)) ** 2
    fr = np.fft.fftfreq(len(xb), 1.0 / fs)
    inb = np.abs(fr) <= 4.5e6
    out = np.abs(fr) > 5.3e6
    ratio = 10 * math.log10(np.mean(spec[inb]) / max(np.mean(spec[out]), 1e-30))
    check("keep>=0.95", keep >= 0.95, f"keep={keep:.3f}")
    check("out-of-band suppressed >20dB", ratio > 20, f"in/out={ratio:.1f} dB")

    # ---- T4: extraction centroid does not shift with B ----
    print("T4: extracted centroid stays at DC across B")
    f_abs = 2447.0
    for B in (20.0, 40.0, 80.0):
        fs2 = B * 1e6
        flo2 = 2432.0
        c2 = flo2 + B / 2
        n2 = int(2e-3 * fs2)
        tone = synth_capture(n2, fs2, [((f_abs - c2) * 1e6, 1.0)])
        xb = band_limit_fft(demod_to_center(tone, fs2, f_abs * 1e6, flo2 * 1e6),
                            fs2, 10e6 * 1.05, edge_frac=0.05)
        spec = np.abs(np.fft.fft(xb)) ** 2
        fr = np.fft.fftfreq(len(xb), 1.0 / fs2)
        cen = float(np.sum(fr * spec) / np.sum(spec))
        check(f"B={B:.0f} centroid", abs(cen) < 50e3, f"centroid={cen/1e3:.1f} kHz")

    # ---- T5: complex CP detector calibration ----
    print("T5: cp_scan_complex calibration (12.8/0.8 and 3.2/0.8, SNR sweep, noise)")
    rng = np.random.default_rng(11)
    for tu_us, cp_us in [(12.8, 0.8), (3.2, 0.8)]:
        # generate at the CAPTURE rate fs=40 MHz so T_u is tu_us at this rate;
        # place at abs 2447 (= -5 MHz digital), then demod to 2447.
        sig, meta = gen_wifi_ofdm(fs, tu_us, cp_us, 4000, 16, None, seed=1)
        t = np.arange(len(sig), dtype=np.float64)
        placed = (sig * np.exp(-2j * np.pi * 5e6 * t / fs)).astype(np.complex64)
        xb = band_limit_fft(demod_to_center(placed, fs, 2447e6, flo * 1e6),
                            fs, 20e6 * 1.02, edge_frac=0.05)
        for snr in (None, 10, 0, -5):
            if snr is None:
                xn = xb
            else:
                pn = 10 ** (-snr / 10.0) * float(np.mean(np.abs(xb) ** 2))
                xn = (xb + math.sqrt(pn / 2) *
                      (rng.standard_normal(len(xb)) +
                       1j * rng.standard_normal(len(xb)))).astype(np.complex64)
            res = cp_scan_complex(xn, fs)
            b = res["best"]
            ok = b is not None and b["tu_us"] == tu_us and b["cp_us"] == cp_us
            # candidates sharing the true T_u legitimately co-score (larger CP
            # windows contain the true CP); the discriminator is other T_u.
            others = max(r["metric"] for r in res["table"]
                         if r["tu_us"] != tu_us)
            check(f"Tu={tu_us} SNR={'inf' if snr is None else snr}",
                  ok and b["metric"] > 3 * others and b["ratio"] > 15,
                  f"best={b['metric']:.3f} bestOtherTu={others:.3f} "
                  f"ratio={b['ratio']:.0f} cfo={b['cfo_cycles']*fs/1e3:+.1f} kHz")
    # noise negatives: the self-calibrated max/median phase ratio must stay
    # near the noise floor (the absolute metric is inflated by max-over-phi
    # selection at small CP * n_sym, so the ratio is the discriminator)
    for trial in range(3):
        xn = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
        res = cp_scan_complex(xn, fs)
        b = res["best"]
        check(f"noise trial {trial}",
              b is not None and b["ratio"] < 8,
              f"metric={b['metric']:.4f} ratio={b['ratio']:.1f} at Tu={b['tu_us']}us")

    print("T6: SRRC crop bounds on a 40 MS/s capture")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "srrc.bin"
        np.ones(2 * n, dtype="<f2").tofile(path)
        rec = SimpleNamespace(bin_path=path, B_mhz=40, T_ms=5, N=n, f_lo=flo)
        sig = {"start_time": 0.4, "end_time": 1.4,
               "start_frequency": flo + 15, "end_frequency": flo + 25}
        crop = crop_and_resample(rec, sig)
        check("0.2 ms context per side",
              (crop["i0"], crop["i1"], crop["out_points"]) == (8000, 64000, 28000))
        windows = windows_from_crop(crop, sig)
        check("training windows stay in GT",
              windows["windows"][0]["start"] == 4000
              and windows["windows"][-1]["stop"] == 24000)
        short = windows_from_crop(crop, dict(sig, end_time=0.41))
        check("short GT clips to its end",
              short["n_windows"] == 1 and short["windows"][0]["stop"] == short["gt_end"])

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
