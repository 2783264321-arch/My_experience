#!/usr/bin/env python3
"""Supplementary: local contrast of thin narrowband targets in the coarse mean
map vs the intra-band peak map (80 MHz focal case).

Motivation: diagnose_coarse_grid_peak.py showed the coarse mean map already
contains enough rank information to separate 95% of coarse-missed (c_only)
narrow targets from matched background. The remaining argument for a peak
channel is *learnability*: a thin 1-2 row stripe may be much more salient
(higher local contrast) in the peak map than in the area-averaged mean map,
even without new rank information.

For every 80 MHz val sample the same grids as the main diagnostic are
computed; for each narrow GT (grouped via the frozen B/C matches) the contrast
is

    max cell inside the GT region  -  median cell in a local ring around it

in dB (log10 power x 10), separately for the mean and the peak map. If the
c_only stripes are already as salient as the detected (both) group in the mean
map, the miss is not a contrast/visibility problem and a peak channel has no
mechanism to fix it.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from src.signals.data import SPLIT_MANIFEST, load_iq, load_manifest  # noqa: E402
from src.signals.stft_fast import stft_power_fast  # noqa: E402
from diagnose_coarse_grid_peak import (  # noqa: E402
    C_ALL, C_COARSE, COARSE_H, NARROW, N_TIME, bbox_cells, grids_from_power,
    load_preds, log10_power, match_map,
)


def ring_mask(r0, r1, c0, c1, pad_r=3, pad_c=2):
    rr0 = max(0, r0 - pad_r)
    rr1 = min(COARSE_H - 1, r1 + pad_r)
    cc0 = max(0, c0 - pad_c)
    cc1 = min(N_TIME - 1, c1 + pad_c)
    rows, cols = np.meshgrid(np.arange(rr0, rr1 + 1), np.arange(cc0, cc1 + 1), indexing="ij")
    m = ~((rows >= r0) & (rows <= r1) & (cols >= c0) & (cols <= c1))
    return rows[m], cols[m]


def main():
    val = [r for r in load_manifest(SPLIT_MANIFEST, "val") if float(r.B_mhz) == 80.0]
    pred_c = load_preds(C_ALL)
    pred_b = load_preds(C_COARSE)
    rows = []
    t0 = time.perf_counter()
    for i, rec in enumerate(val):
        sid = rec.sample_id
        if sid not in pred_c or sid not in pred_b:
            continue
        pc, pb = pred_c[sid], pred_b[sid]
        mc = match_map(pc["gt_boxes"], pc["gt_labels"], *pc["pred"])
        mb = match_map(pc["gt_boxes"], pc["gt_labels"], *pb["pred"])
        p_time, _win, h, L, F = stft_power_fast(load_iq(rec.bin_path), rec.B_mhz, fft_threads=1)
        mean, peak, _cent, _rc, _hn, _bounds = grids_from_power(p_time, F)
        lm, lp = log10_power(mean), log10_power(peak)
        for gi, (box, lab) in enumerate(zip(pc["gt_boxes"], pc["gt_labels"])):
            if int(lab) not in NARROW:
                continue
            r0, r1, c0, c1 = bbox_cells(box)
            rr, cc = ring_mask(r0, r1, c0, c1)
            if rr.size == 0:
                continue
            group = "both" if gi in mb else ("c_only" if gi in mc else
                                             ("b_only" if gi in mb else "neither"))
            rows.append({
                "sid": sid, "cls": int(lab), "group": group,
                "h": r1 - r0 + 1,
                "c_mean": float(lm[c0:c1 + 1, r0:r1 + 1].max() - np.median(lm[cc, rr])),
                "c_peak": float(lp[c0:c1 + 1, r0:r1 + 1].max() - np.median(lp[cc, rr])),
            })
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(val)} {time.perf_counter()-t0:.0f}s", flush=True)

    out = {"n_samples": len(val), "rows": rows}
    (ROOT / "reports/coarse_contrast_80.json").write_text(json.dumps(out, indent=1))

    lines = ["# Local contrast: coarse mean vs intra-band peak (80 MHz)", "",
             "Contrast = max cell inside the GT region - median cell in a local ring",
             "(rows +-3, cols +-2, region excluded), dB. Same grids as",
             "diagnose_coarse_grid_peak.py. Read-only.", ""]
    lines += ["| group | n | mean contrast med | peak contrast med | peak-mean med | peak >= mean+3dB | h rows med |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    summ = {}
    for g in ("both", "c_only", "neither"):
        r = [x for x in rows if x["group"] == g]
        if not r:
            continue
        summ[g] = {
            "n": len(r),
            "c_mean_med": float(np.median([x["c_mean"] for x in r])),
            "c_peak_med": float(np.median([x["c_peak"] for x in r])),
            "diff_med": float(np.median([x["c_peak"] - x["c_mean"] for x in r])),
            "frac_peak_ge_mean3": float(np.mean([x["c_peak"] >= x["c_mean"] + 3 for x in r])),
            "h_med": float(np.median([x["h"] for x in r])),
        }
        s = summ[g]
        lines.append(f"| {g} | {s['n']} | {s['c_mean_med']:.1f} | {s['c_peak_med']:.1f} | "
                     f"{s['diff_med']:+.1f} | {s['frac_peak_ge_mean3']:.2f} | {s['h_med']:.0f} |")
    # per class for c_only
    lines += ["", "c_only per class:", "",
              "| class | n | mean contrast med | peak contrast med |", "| --- | ---: | ---: | ---: |"]
    for c in NARROW:
        r = [x for x in rows if x["group"] == "c_only" and x["cls"] == c]
        if not r:
            continue
        lines.append(f"| {c} | {len(r)} | {np.median([x['c_mean'] for x in r]):.1f} | "
                     f"{np.median([x['c_peak'] for x in r]):.1f} |")
    (ROOT / "reports/coarse_contrast_80.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
