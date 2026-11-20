#!/usr/bin/env python3
"""Build the Baseline-3 round-1 frame lists (WiFi same batch; SRRC 10 MHz tier).

Frames are (sample_id, GT signal_id) pairs taken from ground-truth JSON boxes:

  - WiFi 0-5: the exact same frames as
    `reports/local_rep/proposal_local_win0100.npz` (fit proposals, one per GT
    box, IoU >= 0.75). Geometry is re-read from the GT JSON so that every
    representation in round 1 uses the same GT box + 10% physical margin.
  - SRRC 10/11: every fit GT signal whose clipped box bandwidth falls in
    9.5-10.5 MHz, the bandwidth tier both classes share.

Overlap flags follow the separability-study convention and are recomputed
uniformly for both families (agreement with the stored WiFi flags is
verified and reported). SRRC frames additionally get SRRC-specific flags.

No class label is used for any selection in this script; labels are only
carried for later aggregate evaluation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.signals.data import SPLIT_MANIFEST, load_manifest

ROOT = Path(__file__).resolve().parents[1]
WIFI_CLASSES = tuple(range(6))
SRRC_CLASSES = (10, 11)


def clip_box(sig, rec):
    """Clip a GT box to the observation; return None if the box misses it."""
    f1 = max(rec.f_lo, float(sig["start_frequency"]))
    f2 = min(rec.f_hi, float(sig["end_frequency"]))
    t1 = max(0.0, float(sig["start_time"]))
    t2 = min(float(rec.T_ms), float(sig["end_time"]))
    if f2 - f1 <= 0 or t2 - t1 <= 0:
        return None
    return f1, f2, t1, t2


def overlap_fields(rec, gi):
    """Overlap flags of signal gi against all other signals (GT coordinates)."""
    s = rec.signals[gi]
    area = max(0.0, float(s["end_time"]) - float(s["start_time"])) * \
        max(0.0, float(s["end_frequency"]) - float(s["start_frequency"]))
    any_ov = wifi_ov = srrc_ov = 0
    wifi_cross = wifi_same = srrc_cross = srrc_same = 0
    any_frac = wifi_frac = srrc_frac = 0.0
    for j, o in enumerate(rec.signals):
        if j == gi:
            continue
        inter_t = max(0.0, min(float(s["end_time"]), float(o["end_time"])) -
                      max(float(s["start_time"]), float(o["start_time"])))
        inter_f = max(0.0, min(float(s["end_frequency"]), float(o["end_frequency"])) -
                      max(float(s["start_frequency"]), float(o["start_frequency"])))
        inter = inter_t * inter_f
        if inter <= 0.0 or area <= 0.0:
            continue
        frac = inter / area
        any_ov = 1
        any_frac = max(any_frac, frac)
        oc = int(o["class"])
        if oc in WIFI_CLASSES:
            wifi_ov = 1
            wifi_frac = max(wifi_frac, frac)
            if oc == int(s["class"]):
                wifi_same = 1
            else:
                wifi_cross = 1
        if oc in SRRC_CLASSES:
            srrc_ov = 1
            srrc_frac = max(srrc_frac, frac)
            if oc == int(s["class"]):
                srrc_same = 1
            else:
                srrc_cross = 1
    return (any_ov, wifi_ov, wifi_cross, wifi_same, wifi_frac, any_frac,
            srrc_ov, srrc_cross, srrc_same, srrc_frac)


def frame_row(rec, sig, gi):
    geom = clip_box(sig, rec)
    if geom is None:
        return None
    f1, f2, t1, t2 = geom
    (oa, ow, oc, os_, owf, oaf,
     so, sc, ss, sof) = overlap_fields(rec, gi)
    return {
        "sample_id": rec.sample_id,
        "signal_id": int(sig["signal_id"]),
        "label": int(sig["class"]),
        "f1_mhz": f1, "f2_mhz": f2, "t1_ms": t1, "t2_ms": t2,
        "bw_mhz": f2 - f1, "dur_ms": t2 - t1,
        "f_lo_mhz": float(rec.f_lo), "f_hi_mhz": float(rec.f_hi),
        "B_mhz": float(rec.B_mhz), "T_ms": float(rec.T_ms), "N": int(rec.N),
        "overlap_any": oa, "overlap_wifi": ow,
        "overlap_wifi_cross": oc, "overlap_wifi_same": os_,
        "overlap_wifi_frac_max": owf, "overlap_any_frac_max": oaf,
        "overlap_srrc": so, "overlap_srrc_cross": sc,
        "overlap_srrc_same": ss, "overlap_srrc_frac_max": sof,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wifi-source", default="reports/local_rep/proposal_local_win0100.npz")
    ap.add_argument("--srrc-bw-min", type=float, default=9.5)
    ap.add_argument("--srrc-bw-max", type=float, default=10.5)
    ap.add_argument("--out", default="reports/local_iq_round1/frames.npz")
    ap.add_argument("--counts", default="reports/local_iq_round1/frame_counts.json")
    args = ap.parse_args()

    t0 = time.perf_counter()
    fit = load_manifest(SPLIT_MANIFEST, split="fit")
    by_id = {r.sample_id: r for r in fit}
    print(f"fit records: {len(fit)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    rows = []
    mismatch = 0

    # ---- WiFi 0-5: same frames as the separability study ----
    src = ROOT / args.wifi_source
    with np.load(src) as zf:
        sid = zf["sample_ids"].astype(str)
        gid = zf["gt_signal_ids"].astype(np.int64)
        lab = zf["labels"].astype(np.int64)
        ov_any = zf["overlap_any"].astype(np.int8)
        ov_wifi = zf["overlap_wifi"].astype(np.int8)
        ov_cross = zf["overlap_wifi_cross"].astype(np.int8)
        ov_same = zf["overlap_wifi_same"].astype(np.int8)
    seen = set()
    wifi_missing = 0
    for i in range(len(sid)):
        rec = by_id.get(str(sid[i]))
        if rec is None:
            wifi_missing += 1
            continue
        gi = int(gid[i])
        if gi < 0 or gi >= len(rec.signals):
            wifi_missing += 1
            continue
        sig = rec.signals[gi]
        if int(sig["class"]) != int(lab[i]):
            wifi_missing += 1
            continue
        key = (str(sid[i]), int(sig["signal_id"]))
        if key in seen:
            continue
        seen.add(key)
        row = frame_row(rec, sig, gi)
        if row is None:
            wifi_missing += 1
            continue
        # verify recomputed flags agree with the stored ones
        if (row["overlap_any"] != int(ov_any[i]) or
                row["overlap_wifi"] != int(ov_wifi[i]) or
                row["overlap_wifi_cross"] != int(ov_cross[i]) or
                row["overlap_wifi_same"] != int(ov_same[i])):
            mismatch += 1
        rows.append(row)
    n_wifi = sum(1 for r in rows if r["label"] in WIFI_CLASSES)

    # ---- SRRC 10/11: every fit GT signal in the shared 10 MHz tier ----
    srrc_all = {10: [], 11: []}
    for rec in fit:
        for gi, sig in enumerate(rec.signals):
            if int(sig["class"]) not in SRRC_CLASSES:
                continue
            bw = max(0.0, float(sig["end_frequency"]) - float(sig["start_frequency"]))
            srrc_all[int(sig["class"])].append(bw)
            if not (args.srrc_bw_min <= bw <= args.srrc_bw_max):
                continue
            key = (rec.sample_id, int(sig["signal_id"]))
            if key in seen:
                continue
            row = frame_row(rec, sig, gi)
            if row is None:
                continue
            seen.add(key)
            rows.append(row)
    n_srrc = sum(1 for r in rows if r["label"] in SRRC_CLASSES)

    if not rows:
        raise SystemExit("no frames")

    rows.sort(key=lambda r: (r["sample_id"], r["signal_id"]))
    cols = list(rows[0].keys())
    out = {c: np.array([r[c] for r in rows]) for c in cols}
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)

    labels = out["label"].astype(np.int64)
    bw = out["bw_mhz"]
    dur = out["dur_ms"]
    B = out["B_mhz"]

    def q(x, p):
        x = np.sort(np.asarray(x, dtype=float))
        return float(x[int(round((len(x) - 1) * p))]) if len(x) else None

    counts = {
        "frames_total": len(rows),
        "wifi_frames": n_wifi,
        "srrc_frames": n_srrc,
        "wifi_missing_or_class_mismatch": wifi_missing,
        "wifi_overlap_flag_mismatch_vs_npz": mismatch,
        "samples": int(len(set(out["sample_id"].astype(str)))),
        "class_counts": {str(c): int((labels == c).sum()) for c in range(14)
                         if (labels == c).sum() > 0},
        "srrc_box_bw_hist": {
            c: {
                "n": len(srrc_all[c]),
                "bw_hist_mhz": {
                    f"{lo}-{hi}": int(np.sum((np.array(srrc_all[c]) >= lo) &
                                              (np.array(srrc_all[c]) < hi)))
                    for lo, hi in [(0.0, 0.5), (0.5, 2.0), (2.0, 5.0),
                                   (5.0, 9.5), (9.5, 10.5), (10.5, 99.0)]
                },
            } for c in SRRC_CLASSES
        },
        "per_class": {},
    }
    for c in sorted(set(labels.tolist())):
        m = labels == c
        counts["per_class"][str(c)] = {
            "n": int(m.sum()),
            "bw_q05_q50_q95_mhz": [q(bw[m], 0.05), q(bw[m], 0.5), q(bw[m], 0.95)],
            "dur_q05_q50_q95_ms": [q(dur[m], 0.05), q(dur[m], 0.5), q(dur[m], 0.95)],
            "B_hist_mhz": {str(b): int((B[m] == b).sum()) for b in sorted(set(B[m].tolist()))},
        }
    counts["seconds"] = round(time.perf_counter() - t0, 1)

    counts_path = ROOT / args.counts
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    counts_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print(f"wrote {out_path} and {counts_path}")


if __name__ == "__main__":
    main()
