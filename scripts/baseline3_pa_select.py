#!/usr/bin/env python3
"""Phase A' step 2: deterministic small-scale frame selection.

Selects from fit-only frames (frames.npz for target classes 0-5/10/11;
manifest scan for BLE/Zigbee controls) a fixed, recorded sample set:

  - per target class: up to 4 clean (overlap_any==0) frames with varied
    observation bandwidth B > target bandwidth and duration >= 10 ms,
    plus 1 overlap frame as a hard case;
  - BLE (6) and Zigbee (8) controls: 3 clean frames each, B well above
    the signal bandwidth.

Writes reports/local_iq_round1/phase_ap_frames.json as the single source
of truth for the parallel phase-A' audit tracks.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.signals.data import SPLIT_MANIFEST, load_manifest

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "reports/local_iq_round1/frames.npz"


def pick_varied(idx, B, want=4):
    """Spread picks across distinct B values, deterministic order."""
    out = []
    seen_b = set()
    rest = []
    for i in idx:
        b = float(B[i])
        if b not in seen_b:
            out.append(int(i))
            seen_b.add(b)
        else:
            rest.append(int(i))
        if len(out) >= want:
            break
    while len(out) < want and rest:
        out.append(rest.pop(0))
    return out


def main():
    z = np.load(FRAMES)
    sid = z["sample_id"].astype(str)
    lab = z["label"].astype(np.int64)
    B = z["B_mhz"].astype(float)
    dur = z["dur_ms"].astype(float)
    clean = z["overlap_any"] == 0
    ov = z["overlap_any"] == 1

    def row(i):
        return {
            "sample_id": str(sid[i]), "signal_id": int(z["signal_id"][i]),
            "class": int(lab[i]), "B_mhz": float(B[i]),
            "T_ms": float(z["T_ms"][i]), "bw_mhz": float(z["bw_mhz"][i]),
            "f1_mhz": float(z["f1_mhz"][i]), "f2_mhz": float(z["f2_mhz"][i]),
            "t1_ms": float(z["t1_ms"][i]), "t2_ms": float(z["t2_ms"][i]),
            "f_lo_mhz": float(z["f_lo_mhz"][i]), "f_hi_mhz": float(z["f_hi_mhz"][i]),
        }

    sel = []
    plan = [
        (0, 4, 1), (1, 4, 1), (2, 4, 1),
        (3, 4, 1), (4, 4, 1), (5, 4, 1),
        (10, 4, 1), (11, 4, 1),
    ]
    for c, n_clean, n_ov in plan:
        m = (lab == c) & clean & (B > z["bw_mhz"]) & (dur >= 10.0)
        for i in pick_varied(np.where(m)[0], B, n_clean):
            r = row(i); r["stratum"] = "clean"; sel.append(r)
        mo = (lab == c) & ov & (dur >= 10.0)
        for i in pick_varied(np.where(mo)[0], B, n_ov):
            r = row(i); r["stratum"] = "overlap"; sel.append(r)

    # controls: BLE 6 / Zigbee 8 from the manifest, clean-ish, B >> bw
    fit = load_manifest(SPLIT_MANIFEST, split="fit")
    ctrl = {6: [], 8: []}
    for rec in fit:
        boxes = []
        for gi, s in enumerate(rec.signals):
            boxes.append((float(s["start_time"]), float(s["end_time"]),
                          float(s["start_frequency"]), float(s["end_frequency"])))
        for gi, s in enumerate(rec.signals):
            c = int(s["class"])
            if c not in ctrl:
                continue
            t1, t2 = float(s["start_time"]), float(s["end_time"])
            f1, f2 = float(s["start_frequency"]), float(s["end_frequency"])
            bw, dur_ = f2 - f1, t2 - t1
            if dur_ < 8.0 or rec.B_mhz < 5.0 * bw:
                continue
            # overlap check against all other signals
            ovf = False
            for gj, (a, b, c1, d1) in enumerate(boxes):
                if gj == gi:
                    continue
                it = max(0.0, min(t2, b) - max(t1, a))
                iff = max(0.0, min(f2, d1) - max(f1, c1))
                if it * iff > 0:
                    ovf = True
                    break
            if ovf:
                continue
            ctrl[c].append({
                "sample_id": rec.sample_id, "signal_id": int(s["signal_id"]),
                "class": c, "B_mhz": float(rec.B_mhz), "T_ms": float(rec.T_ms),
                "bw_mhz": bw, "f1_mhz": f1, "f2_mhz": f2,
                "t1_ms": t1, "t2_ms": t2,
                "f_lo_mhz": float(rec.f_lo), "f_hi_mhz": float(rec.f_hi),
            })
    for c, rows in ctrl.items():
        seen = set()
        for r in rows:
            key = r["B_mhz"]
            if key in seen:
                continue
            seen.add(key)
            r["stratum"] = "control"
            sel.append(r)
            if len(seen) >= 3:
                break

    out = {"seed_note": "deterministic first-match per B bucket, fit only",
           "n": len(sel), "frames": sel}
    counts = {}
    for r in sel:
        counts.setdefault(f"{r['class']}/{r['stratum']}", 0)
        counts[f"{r['class']}/{r['stratum']}"] += 1
    out["counts"] = counts
    path = ROOT / "reports/local_iq_round1/phase_ap_frames.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print(f"wrote {path} ({len(sel)} frames)")


if __name__ == "__main__":
    main()
