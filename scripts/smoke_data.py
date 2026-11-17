#!/usr/bin/env python3
"""Stage-1 data/STFT/label/coordinate smoke checks on real samples."""
import json
import math
from pathlib import Path

from src.signals.data import (
    N_TIME, FINE_H, COARSE_H, load_manifest, build_sample_views, frequency_starts,
)

normalizer = json.load(open("configs/normalizer.json"))
norm = (normalizer["lo"], normalizer["hi"])
fit = load_manifest(split="fit")
val = load_manifest(split="val")
all_recs = fit + val

# cover both B extremes and T extremes
picks = [
    next(r for r in all_recs if r.B_mhz == 5.0 and r.T_ms == 20.0),
    next(r for r in all_recs if r.B_mhz == 20.0 and r.T_ms == 20.0),
    next(r for r in all_recs if r.B_mhz == 20.0 and r.T_ms == 150.0),
    next(r for r in all_recs if r.B_mhz == 40.0 and r.T_ms == 80.0),
    next(r for r in all_recs if r.B_mhz == 80.0 and r.T_ms == 20.0),
    next(r for r in all_recs if r.B_mhz == 80.0 and r.T_ms == 150.0),
]

def inv_time(x):
    # x normalized by 512; pixel center convention
    return (x * N_TIME - 0.5) * (h / (B * 1e6)) * 1000.0

def inv_freq(y, start=None):
    if start is None:
        return f_lo + y * B
    return f_lo + (start + y * FINE_H) / F * B

out = []
for r in picks:
    n_fine = len(frequency_starts(int(round(r.B_mhz * 700))))
    coarse, fines, meta = build_sample_views(r, norm, list(range(n_fine)))
    h, B, F, f_lo = meta["h"], r.B_mhz, meta["F"], r.f_lo
    assert coarse.tensor.shape == (1, COARSE_H, N_TIME)
    assert len(fines) == meta["n_fine"]
    assert all(v.tensor.shape == (1, FINE_H, N_TIME) for v in fines)
    # coarse has all original signals (positive time/frequency intersection)
    assert len(coarse.labels) == len(r.signals), (r.sample_id, len(coarse.labels), len(r.signals))
    # every fine view has a valid shape and only in-range labels
    for v in fines:
        if len(v.boxes):
            assert ((v.boxes >= -1e-6) & (v.boxes <= 1 + 1e-6)).all()
            assert (v.boxes[:, 2] > v.boxes[:, 0]).all()
            assert (v.boxes[:, 3] > v.boxes[:, 1]).all()
    # coarse coordinate inverse for each GT center
    errs = []
    for sig in r.signals:
        t_mid = (sig["start_time"] + sig["end_time"]) / 2.0
        f_mid = (sig["start_frequency"] + sig["end_frequency"]) / 2.0
        x = ((t_mid / 1000.0) * (B * 1e6) / h + 0.5) / N_TIME
        y = (f_mid - f_lo) / B
        errs.append((abs(inv_time(x) - t_mid), abs(inv_freq(y) - f_mid)))
    assert max(e[0] for e in errs) < 1e-3
    assert max(e[1] for e in errs) < 1e-4
    out.append({
        "sample_id": r.sample_id, "B": B, "T": r.T_ms, "N": r.N,
        "F": F, "h": h, "n_fine": len(fines),
        "coarse_labels": len(coarse.labels),
        "fine_labels": [len(v.labels) for v in fines],
        "fine_nonempty": sum(len(v.labels) > 0 for v in fines),
        "max_time_err_ms": max(e[0] for e in errs),
        "max_freq_err_mhz": max(e[1] for e in errs),
    })

Path("reports").mkdir(exist_ok=True)
Path("reports/smoke_data.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(json.dumps(out, ensure_ascii=False, indent=2))
print("SMOKE DATA OK")
