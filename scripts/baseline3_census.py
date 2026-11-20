#!/usr/bin/env python3
"""Baseline-3 phase A: per-frame physical census of GT-box IQ content.

For every frame (GT box + central analysis window) this script computes
label-blind physical descriptors:

  fill10       fraction of in-box spectrum within 10 dB of the in-box max
  edge_db      in-box edge (85-98%) median power vs in-box max
  sym_db       |P(f) - P(-f)| median over the box -> DSB mirror test
  peak_off     |argmax freq| (MHz) -> dominant component offset from center
  r32 / r128   normalized autocorr at T_u = 3.2 / 12.8 us (CP signature)
  env_line     top |x|^2 spectral line (MHz) and strength (peak/median)
  x2_line      top x^2 spectral line (MHz) and strength (2 f_c for real tone)
  env_at_srrc  |x|^2 line strengths at R = bw/(1+alpha), alpha = .2/.35/.5
  kurt_time    m4 of band-limited time samples (2 ~ Gaussian, 1 ~ tone)
  box_to_cap   in-box median power vs whole-capture median power (dB)

The census never uses the class label for computation; labels are carried
only so the summary can be grouped for reporting.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts.baseline3_dsp import read_segment, demod_to_center, mag_moments

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "reports/local_iq_round1/frames.npz"
SRRC_ALPHAS = (0.2, 0.35, 0.5)


def load_frames():
    z = np.load(FRAMES)
    cols = {k: z[k] for k in z.files}
    return cols


def seg_and_band(cols, i, cap_ms=10.0):
    sid = str(cols["sample_id"][i])
    fs = float(cols["B_mhz"][i]) * 1e6
    flo = float(cols["f_lo_mhz"][i]) * 1e6
    f1 = float(cols["f1_mhz"][i]) * 1e6
    f2 = float(cols["f2_mhz"][i]) * 1e6
    t1 = float(cols["t1_ms"][i])
    t2 = float(cols["t2_ms"][i])
    c = min(cap_ms, t2 - t1)
    ta = (t1 + t2) / 2 - c / 2
    tb = (t1 + t2) / 2 + c / 2
    i0 = max(0, int(round(ta / 1000 * fs)))
    i1 = int(round(min(tb, t2) / 1000 * fs))
    x = read_segment(f"/root/autodl-fs/SpaceNet_Dataset/advanced/{sid}.bin", i0, i1)
    fc = (f1 + f2) / 2
    xb = demod_to_center(x, fs, fc, flo)
    bw = f2 - f1
    return x, xb, fs, bw


def census_one(cols, i):
    x_raw, xb, fs, bw = seg_and_band(cols, i)
    n = len(xb)
    out = {"n": int(n)}
    if n < 256:
        return None
    # band-limit (FFT mask, box + 2%)
    spec = np.fft.fft(xb)
    fr = np.fft.fftfreq(n, 1.0 / fs)
    half = bw / 2 * 1.02
    m = np.abs(fr) <= half
    spec = spec * m
    xb = np.fft.ifft(spec).astype(np.complex64)
    p = np.abs(spec) ** 2
    inb = np.abs(fr) <= bw / 2 * 0.98
    if not inb.any():
        return None
    pmax = float(p[inb].max())
    out["fill10"] = float(np.mean(p[inb] >= pmax / 10.0))
    edge = (np.abs(fr) >= bw / 2 * 0.85) & inb
    out["edge_db"] = 10.0 * float(np.log10(max(float(np.median(p[edge])), 1e-30) / max(pmax, 1e-30)))
    out["peak_off_mhz"] = float(abs(fr[np.argmax(p * inb)])) / 1e6
    # mirror symmetry on a coarse grid
    nb = int(bw / 2 * 0.98)
    step = max(1, int(fs / n * 20))  # ~50 kHz grid
    sym = []
    for k in range(step, nb, step * 40):
        a = np.median(p[(np.abs(fr - k) <= step * 20)])
        b = np.median(p[(np.abs(fr + k) <= step * 20)])
        if a > 0 and b > 0:
            sym.append(abs(10 * np.log10(a / b)))
    out["sym_db"] = float(np.median(sym)) if sym else 0.0
    # autocorr CP signature
    r = np.fft.ifft(np.abs(np.fft.fft(xb)) ** 2)
    r0 = max(abs(r[0]), 1e-30)
    for tu_us, key in [(3.2, "r32"), (6.4, "r64"), (12.8, "r128")]:
        lag = int(round(tu_us * 1e-6 * fs))
        out[key] = float(abs(r[lag]) / r0) if lag < len(r) else 0.0
    # envelope and squared-signal lines
    pad = int(2 ** np.ceil(np.log2(n)))
    env = np.abs(xb) ** 2
    se = np.abs(np.fft.rfft(env, n=pad))
    fe = np.fft.rfftfreq(pad, 1.0 / fs)
    bm = (fe > 0.02 * bw) & (fe < 1.1 * bw)
    idx = np.where(bm)[0]
    if len(idx):
        med = max(float(np.median(se[idx])), 1e-30)
        k = idx[int(np.argmax(se[idx]))]
        out["env_line_mhz"] = float(fe[k]) / 1e6
        out["env_line_str"] = float(se[k] / med)
    else:
        out["env_line_mhz"] = 0.0
        out["env_line_str"] = 0.0
    sq = xb ** 2
    ss = np.abs(np.fft.fft(sq, n=pad))
    fsq = np.fft.fftfreq(pad, 1.0 / fs)
    bm2 = (fsq > 0.02 * bw) & (fsq < 2.2 * bw)
    idx2 = np.where(bm2)[0]
    if len(idx2):
        med2 = max(float(np.median(ss[idx2])), 1e-30)
        k2 = idx2[int(np.argmax(ss[idx2]))]
        out["x2_line_mhz"] = float(fsq[k2]) / 1e6
        out["x2_line_str"] = float(ss[k2] / med2)
    else:
        out["x2_line_mhz"] = 0.0
        out["x2_line_str"] = 0.0
    # SRRC rate candidates: line strength at R = bw/(1+alpha)
    for a in SRRC_ALPHAS:
        f = bw / (1.0 + a)
        kk = int(round(f / fs * pad))
        if 0 < kk < len(se):
            meda = max(float(np.median(se[idx])), 1e-30) if len(idx) else 1.0
            out[f"env_at_a{int(a*100)}"] = float(se[kk] / meda)
        else:
            out[f"env_at_a{int(a*100)}"] = 0.0
    # time-domain kurtosis of band-limited samples
    m4, _ = mag_moments(np.abs(xb))
    out["kurt_time"] = float(m4)
    # box vs whole-capture strength
    p_raw = np.abs(np.fft.fft(x_raw)) ** 2
    fr_raw = np.fft.fftfreq(len(x_raw), 1.0 / fs)
    med_all = float(np.median(p_raw))
    med_box = float(np.median(p[inb]))
    out["box_to_cap_db"] = 10.0 * float(np.log10(max(med_box, 1e-30) / max(med_all, 1e-30)))
    # wideband plateau test: in-box lower-40% median vs out-of-box floor
    outb = (np.abs(fr) > bw / 2 * 1.05) & (np.abs(fr) < fs / 2 * 0.98)
    p_in_sorted = np.sort(p[inb])
    k40 = max(1, int(len(p_in_sorted) * 0.4))
    floor_in = float(np.median(p_in_sorted[:k40]))
    floor_out = float(np.median(p_raw[outb])) if outb.any() else floor_in
    out["floor_ratio_db"] = 10.0 * float(np.log10(max(floor_in, 1e-30) / max(floor_out, 1e-30)))
    # occupancy at 20 dB (wider plateau than fill10)
    out["fill20"] = float(np.mean(p[inb] >= pmax / 100.0))
    # narrow-component layout: 16-bin normalized in-box energy profile
    idx_in = np.where(inb)[0]
    prof16 = []
    for q in np.array_split(idx_in, 16):
        prof16.append(float(np.mean(p[q])))
    prof16 = np.asarray(prof16) / max(np.sum(prof16), 1e-30)
    for q in range(16):
        out[f"band{q:02d}"] = float(prof16[q])
    # peak counting and -6 dB width of the strongest component
    nbin = max(8, int(0.05e6 / (fs / n)))  # ~50 kHz smoothing
    ps = np.convolve(p[idx_in], np.ones(nbin) / nbin, mode="same")
    thr = ps.max() / 10.0
    above = ps >= thr
    runs = []
    st = None
    for j, a in enumerate(above):
        if a and st is None:
            st = j
        elif not a and st is not None:
            runs.append((st, j)); st = None
    if st is not None:
        runs.append((st, len(above)))
    out["n_peaks"] = float(len(runs))
    if runs:
        w = max(runs, key=lambda r: ps[r[0]:r[1]].max())
        df = fs / n
        out["tone_width_mhz"] = float((w[1] - w[0]) * df) / 1e6
        fr_in = fr[idx_in]
        out["tone_center_mhz"] = float(fr_in[(w[0] + w[1]) // 2]) / 1e6
    else:
        out["tone_width_mhz"] = 0.0
        out["tone_center_mhz"] = 0.0
    # second |x|^2 line
    if len(idx):
        order = np.argsort(se[idx])[::-1]
        if len(order) > 8:
            k2 = idx[order[8]]  # skip the 8 bins around the top line
            out["env_line2_mhz"] = float(fe[k2]) / 1e6
            out["env_line2_str"] = float(se[k2] / med)
        else:
            out["env_line2_mhz"] = 0.0
            out["env_line2_str"] = 0.0
    else:
        out["env_line2_mhz"] = 0.0
        out["env_line2_str"] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=30,
                    help="frames per class (0 = all)")
    ap.add_argument("--classes", default="",
                    help="comma-separated classes; default = all frame classes; "
                         "'controls' = 6,7,8,9,12,13 from the manifest directly")
    ap.add_argument("--cap-ms", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--out", default="reports/local_iq_round1/census.npz")
    ap.add_argument("--summary", default="reports/local_iq_round1/census_summary.json")
    args = ap.parse_args()

    cols = load_frames()
    labels = cols["label"].astype(np.int64)
    n_all = len(labels)
    rng = np.random.default_rng(args.seed)
    if args.classes == "controls":
        from src.signals.data import SPLIT_MANIFEST, load_manifest
        want = [6, 7, 8, 9, 12, 13]
        fit = load_manifest(SPLIT_MANIFEST, split="fit")
        per = {}
        for rec in fit:
            for gi, sig in enumerate(rec.signals):
                c = int(sig["class"])
                if c not in want:
                    continue
                bw = float(sig["end_frequency"]) - float(sig["start_frequency"])
                dur = float(sig["end_time"]) - float(sig["start_time"])
                if dur < 2.0 or bw <= 0:
                    continue
                per.setdefault(c, []).append({
                    "sample_id": rec.sample_id, "signal_id": int(sig["signal_id"]),
                    "label": c, "B_mhz": float(rec.B_mhz), "bw_mhz": bw,
                    "dur_ms": dur, "overlap_any": 0, "overlap_wifi": 0,
                    "overlap_srrc": 0,
                    "f1_mhz": float(sig["start_frequency"]),
                    "f2_mhz": float(sig["end_frequency"]),
                    "t1_ms": float(sig["start_time"]),
                    "t2_ms": float(sig["end_time"]),
                    "f_lo_mhz": float(rec.f_lo),
                })
        sel_rows = []
        for c in want:
            rows = per.get(c, [])
            if args.per_class and len(rows) > args.per_class:
                rows = [rows[i] for i in rng.choice(len(rows), args.per_class, replace=False)]
            sel_rows.extend(rows)
        # census_one works off the frames npz; build a compatible dict
        merged = {k: np.array([r.get(k, "") for r in sel_rows]) for k in
                  ["sample_id", "signal_id", "label", "B_mhz", "bw_mhz", "dur_ms",
                   "overlap_any", "overlap_wifi", "overlap_srrc",
                   "f1_mhz", "f2_mhz", "t1_ms", "t2_ms", "f_lo_mhz"]}
        cols = merged
        sel = list(range(len(sel_rows)))
    else:
        sel = []
        for c in sorted(set(labels.tolist())):
            idx = np.where(labels == c)[0]
            if args.per_class and len(idx) > args.per_class:
                idx = rng.choice(idx, args.per_class, replace=False)
            sel.extend(int(i) for i in idx)
        sel = sorted(sel)
    print(f"census frames: {len(sel)} of {n_all}", flush=True)

    rows = []
    started = time.perf_counter()
    for j, i in enumerate(sel, 1):
        try:
            r = census_one(cols, i)
        except Exception as e:  # keep the census going; record the failure
            r = {"error": str(e)}
        if r is None:
            continue
        base = {k: cols[k][i] for k in
                ["sample_id", "signal_id", "label", "B_mhz", "bw_mhz",
                 "dur_ms", "overlap_any", "overlap_wifi", "overlap_srrc"]}
        base = {k: (str(v) if k == "sample_id" else float(v)) for k, v in base.items()}
        base.update(r)
        rows.append(base)
        if j % 25 == 0:
            print(f"{j}/{len(sel)} {time.perf_counter()-started:.0f}s", flush=True)

    keys = list(rows[0].keys())
    out = {k: np.array([r.get(k, np.nan) for r in rows]) for k in keys}
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)

    # summary grouped by class (labels used for reporting only)
    lab = out["label"].astype(np.int64)
    metrics = ["fill10", "fill20", "edge_db", "sym_db", "peak_off_mhz", "r32", "r64", "r128",
               "env_line_mhz", "env_line_str", "env_line2_mhz", "env_line2_str",
               "x2_line_mhz", "x2_line_str",
               "env_at_a20", "env_at_a35", "env_at_a50", "kurt_time", "box_to_cap_db",
               "floor_ratio_db", "n_peaks", "tone_width_mhz", "tone_center_mhz"] + \
              [f"band{q:02d}" for q in range(16)]
    summary = {"n_frames": len(rows), "per_class": {}}
    for c in sorted(set(lab.tolist())):
        m = lab == c
        cp = {"n": int(m.sum())}
        for k in metrics:
            v = out[k][m]
            v = v[~np.isnan(v)]
            if len(v):
                cp[k] = {
                    "q25": float(np.quantile(v, 0.25)),
                    "q50": float(np.quantile(v, 0.50)),
                    "q75": float(np.quantile(v, 0.75)),
                }
        summary["per_class"][str(c)] = cp
    sPath = ROOT / args.summary
    sPath.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path} and {sPath}")


if __name__ == "__main__":
    main()
