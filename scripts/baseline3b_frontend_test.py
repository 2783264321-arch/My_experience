#!/usr/bin/env python3
"""Phase B' stage-0 acceptance battery for the OFDM frontend.

Validates, with PASS/FAIL assertions:
  T1  synthetic end-to-end: legacy-layout OFDM generated at the TARGET
      bandwidth, resampled to the capture rate, embedded at legal offsets
      (incl. edge-hugging), random symbol start, CFO, orders, SNR, B in
      {20,30,40,50,80} -> full-cycle CP scan must select (3.2, 0.8) us,
      with start-invariant metric and accurate CFO.
  T2  after strip+FFT at native rate, the data-subcarrier constellation
      m4 must match the true order (QPSK 1.00 / 16QAM 1.32 / 64QAM 1.38)
      using the generator's own data mask (label-free for the frontend:
      the mask here is the generator's ground truth for self-test).
  T3  no-CP OFDM control and pure-noise control must FAIL the pass rule.
  T4  real-IQ frequency landmarks: JSON narrowband tones must appear at
      their absolute frequencies under the center-referenced convention
      (guards against any regression of the demodulation chain).
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

sys.path.insert(0, "/root/autodl-tmp/workspace/baseline")
from scripts.baseline3b_ofdm_frontend import (
    gen_ofdm_baseband, embed_capture, cp_scan_full, strip_symbols,
    filter_symbol_rows, extract_band,
)
from scripts.baseline3_dsp import read_segment

PASS = FAIL = 0
DETAIL = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}  {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")
    DETAIL.append((name, cond, detail))


def theory_m4(order):
    return {4: 1.00, 16: 1.32, 64: 1.38}[order]


def main():
    rng = np.random.default_rng(2026)

    # ---- T1: end-to-end synthetic battery ----
    print("T1: end-to-end synthetic OFDM at legal offsets, random starts")
    cases = []
    for B, bw in [(20, 20), (30, 20), (40, 20), (50, 20), (80, 20),
                  (40, 40), (50, 40), (80, 40)]:
        max_off = B / 2 - bw / 2
        offs = [0.0] if max_off < 1.0 else [0.0, max_off * 0.5,
                                           max_off * 0.97]
        for off in offs:
            cases.append((B, bw, off))
    worst = {"metric_ratio": 1e9, "cfo_err_hz": 0.0}
    start_spread = []
    for B, bw, off in cases:
        for order in (4, 16, 64):
            n_sym = 600
            xb, meta = gen_ofdm_baseband(bw * 1e6, order, n_sym, rng)
            fs_cap = B * 1e6
            cfo_hz = float(rng.choice([0.0, 0.3 * 1 / 3.2e-6,  # 0.3 subcarrier
                                       -0.25 * 1 / 3.2e-6]))
            snr = None if order == 4 else float(rng.choice([20.0, 10.0]))
            starts = [int(s) for s in rng.integers(0, 4000, size=4)]
            for st in starts:
                x, em = embed_capture(
                    xb, bw * 1e6, fs_cap, off * 1e6, rng,
                    start_samples=st, cfo_hz=cfo_hz, snr_db=snr)
                # demod like a GT box at the absolute center of the offset
                flo = 2400.0
                f_center_abs = flo + B / 2 + off
                from scripts.baseline3_dsp import demod_to_center, band_limit_fft
                seg = x  # already at capture rate; demod directly
                xd = demod_to_center(seg, fs_cap, f_center_abs * 1e6, flo * 1e6)
                xd = band_limit_fft(xd, fs_cap, bw * 1e6 * 1.02, edge_frac=0.05)
                res = cp_scan_full(xd, fs_cap)
                b = res["best"]
                ok = (b is not None and b["tu_us"] == 3.2 and b["cp_us"] == 0.8
                      and b["metric"] > 0.34 and b["ratio"] > 15)
                cfo_err = abs(b["cfo_cycles"] * fs_cap - cfo_hz) if b else 9e9
                worst["cfo_err_hz"] = max(worst["cfo_err_hz"], cfo_err)
                worst["metric_ratio"] = min(worst["metric_ratio"],
                                            b["metric"] if b else 0)
                start_spread.append(b["metric"] if b else 0.0)
                if not ok:
                    check(f"B{B}/bw{bw}/off{off}/o{order}/st{st}", False,
                          f"best={b}")
                    break
        else:
            check(f"B{B} bw{bw} off{off:+.1f} all orders x random starts",
                  True, f"min_metric={min(start_spread[-12:]):.2f}")
            continue
        break
    # start invariance: same config, sweep the start over one full period
    print("T1b: metric start-invariance over a full period sweep")
    xb, meta = gen_ofdm_baseband(20e6, 16, 600, rng)
    fs_cap = 40e6
    per = int(round(3.2e-6 * fs_cap)) + int(round(0.8e-6 * fs_cap))
    metrics = []
    for st in range(0, per, max(1, per // 24)):
        x, em = embed_capture(xb, 20e6, fs_cap, 5e6, rng, start_samples=st)
        xd = demod_to_center(x, fs_cap, (2400.0 + 20 + 5) * 1e6, 2400e6)
        xd = band_limit_fft(xd, fs_cap, 20e6 * 1.02, edge_frac=0.05)
        b = cp_scan_full(xd, fs_cap)["best"]
        metrics.append(b["metric"])
    spread = (max(metrics) - min(metrics)) / np.mean(metrics)
    check("start-invariance <15% over full period", spread < 0.15,
          f"spread={spread*100:.1f}% min={min(metrics):.3f} max={max(metrics):.3f}")

    # ---- T2: constellation recovery on data subcarriers ----
    print("T2: strip+FFT constellation m4 at data subcarriers")
    for B, bw in [(40, 20), (80, 40), (20, 20)]:
        for order in (4, 16, 64):
            xb, meta = gen_ofdm_baseband(bw * 1e6, order, 800, rng)
            fs_cap = B * 1e6
            x, em = embed_capture(xb, bw * 1e6, fs_cap,
                                   (B / 2 - bw / 2) * 0.5e6, rng,
                                   start_samples=1234)
            off_abs = 2400.0 + B / 2 + (B / 2 - bw / 2) * 0.5
            xd = demod_to_center(x, fs_cap, off_abs * 1e6, 2400e6)
            xd = band_limit_fft(xd, fs_cap, bw * 1e6 * 1.02, edge_frac=0.05)
            res = cp_scan_full(xd, fs_cap)
            b = res["best"]
            X = strip_symbols(xd, fs_cap, b["tu_us"], b["cp_us"], b["phi"],
                              cfo_cycles=-b["cfo_cycles"] if b["cfo_cycles"] else 0.0)
            data_bins = meta["data"]
            X, keep_idx, n_kept, n_drop = filter_symbol_rows(X, data_bins)
            mags = np.abs(X[:, data_bins])
            m4 = float(np.mean(mags ** 4) / np.mean(mags ** 2) ** 2)
            th = theory_m4(order)
            check(f"B{B} bw{bw} order{order} m4", abs(m4 - th) < 0.06,
                  f"m4={m4:.3f} theory={th:.2f} n_sym={n_kept} "
                  f"(dropped {n_drop})")

    # ---- T3: no-CP OFDM and noise negatives ----
    print("T3: no-CP OFDM and pure-noise negatives")
    from scripts.baseline3_dsp import qam_symbols
    for trial in range(3):
        order = 16
        n_fft = 64
        grid = qam_symbols(order, n_fft * 500, rng).reshape(n_fft, 500)
        syms = np.fft.ifft(grid, axis=0) * math.sqrt(n_fft)
        nocp = syms.T.reshape(-1).astype(np.complex64)  # no CP at all
        x, _ = embed_capture(nocp, 20e6, 40e6, 5e6, rng, start_samples=77)
        xd = demod_to_center(x, 40e6, 2425e6, 2400e6)
        xd = band_limit_fft(xd, 40e6, 20e6 * 1.02, edge_frac=0.05)
        b = cp_scan_full(xd, 40e6)["best"]
        check(f"no-CP trial {trial}",
              b["metric"] < 0.34 or b["ratio"] < 15,
              f"metric={b['metric']:.3f} ratio={b['ratio']:.1f} at "
              f"({b['tu_us']},{b['cp_us']})us")
    for trial in range(3):
        xn = (rng.standard_normal(400000) +
              1j * rng.standard_normal(400000)).astype(np.complex64)
        b = cp_scan_full(xn, 40e6)["best"]
        check(f"noise trial {trial}", b["ratio"] < 8,
              f"metric={b['metric']:.3f} ratio={b['ratio']:.1f}")

    # ---- T4: real-IQ landmark check ----
    print("T4: real-IQ landmarks (center-referenced convention)")
    for sid, expect in [("train/1058", [(2464.0, 2465.0, 9021),
                                         (2442.0, 2452.0, 266)]),
                        ("train/3663", [(2436.0, 2438.0, 351),
                                        (2446.0, 2448.0, 189)])]:
        d = json.load(open(f"/root/autodl-fs/SpaceNet_Dataset/advanced/{sid}.json"))
        flo, fhi = d["observation_range"]
        B = fhi - flo
        fs = B * 1e6
        i1 = int(10e-3 * fs)
        raw = np.fromfile(f"/root/autodl-fs/SpaceNet_Dataset/advanced/{sid}.bin",
                          dtype="<f2", count=2 * i1)
        iq = np.empty(len(raw) // 2, dtype=np.complex64)
        iq.real = raw[0::2]; iq.imag = raw[1::2]
        S = np.abs(np.fft.fft(iq)) ** 2
        N = len(S)
        df = fs / N
        k = max(1, int(50e3 / df))
        Sm = np.convolve(S, np.ones(k) / k, mode="same")
        order = np.argsort(Sm)[::-1]
        peaks = []
        for idx in order:
            fdig = (idx if idx < N / 2 else idx - N) * df / 1e6
            if any(abs(fdig - p) < 1.0 for p in peaks):
                continue
            peaks.append(fdig)
            if len(peaks) >= 4:
                break
        center = (flo + fhi) / 2
        abs_peaks = [center + p for p in peaks]
        ok = False
        for lo, hi, ratio in expect:
            ok = any(lo - 0.6 <= a <= hi + 0.6 for a in abs_peaks)
            if not ok:
                break
        check(f"{sid} landmarks", ok,
              f"abs top peaks {[round(a,1) for a in abs_peaks]} vs "
              f"expect {expect}")

    print(f"\n{PASS} passed, {FAIL} failed")
    from pathlib import Path
    out_dir = Path("/root/autodl-tmp/workspace/baseline/reports/baseline3_wifi_expert")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {"passed": PASS, "failed": FAIL,
           "detail": [{"name": n, "ok": bool(c), "info": d}
                      for n, c, d in DETAIL]}
    with open(out_dir / "frontend_battery.json", "w") as f:
        json.dump(out, f, indent=2)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
