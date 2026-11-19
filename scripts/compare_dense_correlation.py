#!/usr/bin/env python3
"""Compare the current dense correlation estimator with aggregate-first.

Current production:
  per adjacent-frame pair: C / sqrt(P_j P_j+1), then average per column.
Alternative:
  sum phase-corrected C and pair energies per column, then divide once.

Only front-end diagnostics on a few samples; the training cache and production
operator are not changed. Focus: weak 3/4/5 signals and noise regions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.signals.c3_frontend import (
    COARSE_NFFT, dense_correlation, dense_correlation_aggregate,
)
from src.signals.data import (
    COARSE_H, N_TIME, SPLIT_MANIFEST, build_sample_views, coarse_targets,
    load_iq, load_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDS = ["train/29", "train/14", "train/1697", "train/1205",
               "train/7401", "train/3441"]


def region_stats(corr, power, rec):
    """corr [2,H,W]; power [H,W] coarse log power; GT 3/4/5 regions."""
    mag = corr.norm(dim=0)
    re = corr[0]
    im = corr[1]
    boxes, labels, _, _ = coarse_targets(rec)
    gt_rows = []
    for b, l in zip(boxes.tolist(), labels.tolist()):
        if int(l) not in (3, 4, 5):
            continue
        x1, y1, x2, y2 = b
        r1 = max(0, min(COARSE_H - 1, int(round(y1 * COARSE_H))))
        r2 = max(r1 + 1, min(COARSE_H, int(round(y2 * COARSE_H))))
        c1 = max(0, min(N_TIME - 1, int(round(x1 * N_TIME))))
        c2 = max(c1 + 1, min(N_TIME, int(round(x2 * N_TIME))))
        gt_rows.append((r1, r2, c1, c2, int(l), float(power[r1:r2, c1:c2].mean())))
    # noise = outside all GT boxes, with 2-cell margin
    mask = torch.ones((COARSE_H, N_TIME), dtype=torch.bool)
    for b in boxes.tolist():
        x1, y1, x2, y2 = b
        r1 = max(0, int(round(y1 * COARSE_H)) - 2)
        r2 = min(COARSE_H, int(round(y2 * COARSE_H)) + 2)
        c1 = max(0, int(round(x1 * N_TIME)) - 2)
        c2 = min(N_TIME, int(round(x2 * N_TIME)) + 2)
        mask[r1:r2, c1:c2] = False
    noise_mag = float(mag[mask].mean()) if mask.any() else None
    out = {"noise_mean_mag": noise_mag, "gt": []}
    for r1, r2, c1, c2, cls, pwr in gt_rows:
        m = mag[r1:r2, c1:c2]
        rea = re[r1:r2, c1:c2]
        ima = im[r1:r2, c1:c2]
        mean = torch.complex(rea.mean(), ima.mean())
        out["gt"].append({
            "class": cls, "rows": [r1, r2], "cols": [c1, c2],
            "power": pwr,
            "mean_mag": float(m.mean()),
            "coherence": float(mean.abs() / max(float(m.mean()), 1e-12)),
            "mean_re": float(rea.mean()), "mean_im": float(ima.mean()),
        })
    return out


def synthetic(fs_mhz=80.0, T_ms=20.0, row=1500, snr_db=-10.0):
    fs = fs_mhz * 1e6
    N = int(fs * T_ms / 1000.0)
    h = math.ceil(N / 511)
    n = torch.arange(N, dtype=torch.float64)
    f0 = (row - COARSE_NFFT / 2.0 + 0.5) * fs / COARSE_NFFT
    tone = np.exp(1j * 2 * np.pi * f0 * n.numpy() / fs + 0.3)
    var = 10.0 ** (-snr_db / 10.0)
    rng = np.random.default_rng(1729)
    noise = np.sqrt(var / 2.0) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    x = torch.from_numpy((tone + noise).astype(np.complex64))
    rec = SimpleNamespace(B_mhz=fs_mhz, T_ms=T_ms, N=N)
    out = {}
    for name, fn in [("current", dense_correlation),
                     ("aggregate", dense_correlation_aggregate)]:
        c, diag = fn(rec, x)
        mag = c.norm(dim=0)
        out[name] = {
            "target_mean_mag": float(mag[row].mean()),
            "target_phase": float(torch.atan2(c[1, row].mean(), c[0, row].mean())),
            "noise_rows_mean_mag": float(mag[[100, 500, 1000, 1900]].mean()),
            "last_column_pairs": diag.get("last_column_pairs"),
        }
    out["snr_db"] = snr_db
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=",".join(DEFAULT_IDS))
    ap.add_argument("--out", default="reports/c3_dense_correlation_comparison.json")
    ap.add_argument("--md", default="reports/c3_dense_correlation_comparison.md")
    args = ap.parse_args()
    norm_cfg = json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm_cfg["lo"]), float(norm_cfg["hi"]))
    fit = load_manifest(SPLIT_MANIFEST, "fit")
    by_id = {r.sample_id: r for r in fit}
    ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    rows = []
    for sid in ids:
        rec = by_id[sid]
        iq = load_iq(rec.bin_path)
        coarse, _, _ = build_sample_views(rec, norm, [])
        power = coarse.tensor[0]
        cur, dcur = dense_correlation(rec, iq)
        agg, dagg = dense_correlation_aggregate(rec, iq)
        sc = region_stats(cur, power, rec)
        sa = region_stats(agg, power, rec)
        rows.append({"sample_id": sid, "B_mhz": rec.B_mhz, "T_ms": rec.T_ms,
                     "current": sc, "aggregate": sa,
                     "diag_current": dcur, "diag_aggregate": dagg})
        print(f"{sid}: GT 3/4/5={len(sc['gt'])}", flush=True)
    synth = [synthetic(snr_db=s) for s in (-10.0, -5.0, 0.0)]

    # aggregate weak/strong GT instances across samples (median power split)
    flat = []
    for r in rows:
        for side in ("current", "aggregate"):
            for g in r[side]["gt"]:
                flat.append({"sample_id": r["sample_id"], "side": side, **g})
    pw = [x["power"] for x in flat if x["side"] == "current"]
    med = float(np.median(pw)) if pw else 0.0
    weak = {}
    strong = {}
    for side in ("current", "aggregate"):
        for bucket, pred in ((weak, lambda x: x["power"] <= med),
                             (strong, lambda x: x["power"] > med)):
            vals = [x["mean_mag"] for x in flat if x["side"] == side and pred(x)]
            coh = [x["coherence"] for x in flat if x["side"] == side and pred(x)]
            bucket[side] = {"n": len(vals),
                            "mean_mag": float(np.mean(vals)) if vals else None,
                            "mean_coherence": float(np.mean(coh)) if coh else None}

    out = {"rows": rows, "synthetic": synth,
           "power_median_3_4_5": med, "weak": weak, "strong": strong,
           "note": "current is production per-pair normalize then average; aggregate is sum numerator / sum pair energy"}
    p = ROOT / args.out
    p.write_text(json.dumps(out, indent=2))

    lines = ["# Dense correlation estimator comparison", "",
             "Current = per-pair normalized correlation, then average per column.",
             "Aggregate = sum phase-corrected correlation and pair energy per column, then divide once.",
             "No cache/training operator was changed.", "",
             "## Synthetic tone + noise", "",
             "| SNR dB | current target | aggregate target | current noise | aggregate noise | current phase | aggregate phase |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in synth:
        c, a = s["current"], s["aggregate"]
        lines.append(f"| {s['snr_db']} | {c['target_mean_mag']:.4f} | {a['target_mean_mag']:.4f} | "
                     f"{c['noise_rows_mean_mag']:.4f} | {a['noise_rows_mean_mag']:.4f} | "
                     f"{c['target_phase']:.4f} | {a['target_phase']:.4f} |")
    lines += ["", "## Real samples, GT 3/4/5 and noise", "",
              "| sample | B/T | GT n | current GT | aggregate GT | current noise | aggregate noise |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        c, a = r["current"], r["aggregate"]
        cm = np.mean([g["mean_mag"] for g in c["gt"]]) if c["gt"] else None
        am = np.mean([g["mean_mag"] for g in a["gt"]]) if a["gt"] else None
        lines.append(f"| {r['sample_id']} | {r['B_mhz']:g}/{r['T_ms']:g} | {len(c['gt'])} | "
                     f"{cm:.4f} | {am:.4f} | {c['noise_mean_mag']:.4f} | {a['noise_mean_mag']:.4f} |" if cm is not None and am is not None and c['noise_mean_mag'] is not None and a['noise_mean_mag'] is not None else
                     f"| {r['sample_id']} | {r['B_mhz']:g}/{r['T_ms']:g} | {len(c['gt'])} | n/a | n/a | n/a | n/a |")
    lines += ["", "## Weak/strong 3/4/5 split", "",
              f"Median GT coarse power = {med:.6g}.", "",
              "| bucket | current mean mag | aggregate mean mag | current coherence | aggregate coherence |",
              "| --- | ---: | ---: | ---: | ---: |"]
    def f4(x):
        return f"{x:.4f}" if x is not None else "n/a"
    for name, b in (("weak", weak), ("strong", strong)):
        c, a = b["current"], b["aggregate"]
        lines.append(f"| {name} (n={c['n']}) | {f4(c['mean_mag'])} | {f4(a['mean_mag'])} | "
                     f"{f4(c['mean_coherence'])} | {f4(a['mean_coherence'])} |")
    lines += ["", "Interpretation: aggregate-first should suppress independent noise more (lower noise mean), "
              "but on weak signals it can be dominated by a few strong pairs; compare both GT magnitude and coherence. "
              "This is a diagnostic, not a training result.", ""]
    md = ROOT / args.md
    md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", p, "and", md)


if __name__ == "__main__":
    main()
