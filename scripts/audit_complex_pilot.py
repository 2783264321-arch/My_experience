#!/usr/bin/env python3
"""Audit cache, channel/box construction, C3 weights, and 90-degree rotation."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from src.signals.complex_frontend import (
    complex_stft, fine_window_len, normalize_complex, normalize_log_power,
    overlap_bands,
)
from src.signals.data import (
    COARSE_H, FINE_H, N_TIME, SPLIT_MANIFEST, _label_boxes,
    h_for_n, load_iq, load_manifest,
)
from src.signals.stft_fast import normalize_slice, stft_power_fast
from scripts.build_complex_pilot_cache import build_one, pick_fine_start
from scripts.train_complex_pilot import CachedPairs, collate_pairs

ROOT = Path(__file__).resolve().parents[1]


def first_conv_stats(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    w = sd["model.0.conv.weight"].float()
    return {
        "shape": list(w.shape),
        "channel_norm": [float(x) for x in w.square().sum(dim=(0, 2, 3)).sqrt()],
        "channel_std": [float(x) for x in w.permute(1, 0, 2, 3).flatten(1).std(dim=1)],
        "total_norm": float(w.norm()),
    }


def main():
    cache = ROOT / "reports/complex_pilot_cache"
    fit_meta = json.loads((cache / "fit_meta.json").read_text())
    val_meta = json.loads((cache / "val_meta.json").read_text())
    scale_c = fit_meta["scale_coarse"]; scale_f = fit_meta["scale_fine"]
    norm = fit_meta["normalizer"]; lo, hi = norm["lo"], norm["hi"]
    fit = {r.sample_id: r for r in load_manifest(SPLIT_MANIFEST, "fit")}
    fit_coarse = np.load(cache / "fit_coarse.npy", mmap_mode="r")
    fit_fine = np.load(cache / "fit_fine.npy", mmap_mode="r")
    # Pick four cached fit samples across B/T.
    wanted = []
    for target in [(20, 20), (40, 40), (20, 150), (80, 150)]:
        for i, m in enumerate(fit_meta["samples"]):
            if abs(m["B_mhz"] - target[0]) < 1e-6 and abs(m["T_ms"] - target[1]) < 0.01:
                wanted.append((i, m)); break
    out = {"samples": [], "weights": {}, "rotation": []}
    for i, m in wanted:
        rec = fit[m["sample_id"]]
        meta2, c2, f2 = build_one(rec, lo, hi, scale_c, scale_f)
        c_cache = np.asarray(fit_coarse[i]); f_cache = np.asarray(fit_fine[i])
        # Direct fine power from the existing STFT path.
        z = complex_stft(load_iq(rec.bin_path), rec.B_mhz * 1e6,
                         fine_window_len(rec.B_mhz * 1e6),
                         h_for_n(rec.N), (N_TIME - 1) * h_for_n(rec.N))
        p, _, _, _, _ = stft_power_fast(load_iq(rec.bin_path), rec.B_mhz, fft_threads=1)
        p_ref = normalize_slice(p[:, m["fine_start"]:m["fine_start"] + FINE_H], lo, hi).numpy()
        zs = z[m["fine_start"]:m["fine_start"] + FINE_H]
        cb, cl, _, _ = _label_boxes(rec.signals, rec, m["F"], m["h"], None)
        fb, fl, _, _ = _label_boxes(rec.signals, rec, m["F"], m["h"], m["fine_start"])
        out["samples"].append({
            "sample_id": m["sample_id"], "B_mhz": m["B_mhz"], "T_ms": m["T_ms"],
            "cache_index": int(i), "fine_start": int(m["fine_start"]), "F": int(m["F"]),
            "coarse_max_abs_diff": float(np.max(np.abs(c_cache - c2))),
            "fine_max_abs_diff": float(np.max(np.abs(f_cache - f2))),
            "coarse_bitwise": bool(np.array_equal(c_cache, c2)),
            "fine_bitwise": bool(np.array_equal(f_cache, f2)),
            "fine_power_vs_stft_fast_max_abs": float(np.max(np.abs(f_cache[0].astype(np.float32) - p_ref))),
            "fine_power_vs_complex_max_abs": float(np.max(np.abs(f_cache[0].astype(np.float32) - normalize_log_power(zs.abs().square(), lo, hi).numpy()))),
            "fine_complex_vs_recompute_max_abs": float(np.max(np.abs(f_cache[1:3].astype(np.float32) - normalize_complex(zs, scale_f).numpy()))),
            "coarse_boxes_max_abs_diff": float(np.max(np.abs(np.asarray(m["coarse_boxes"]) - cb.numpy()))),
            "fine_boxes_max_abs_diff": float(np.max(np.abs(np.asarray(m["fine_boxes"]) - fb.numpy()))),
            "coarse_labels_equal": bool(np.array_equal(np.asarray(m["coarse_labels"]), cl.numpy())),
            "fine_labels_equal": bool(np.array_equal(np.asarray(m["fine_labels"]), fl.numpy())),
            "power_identical_ch0": bool(np.array_equal(c_cache[0], c_cache[0])),
            "coarse_power_clip": m["coarse_power_clip"],
            "fine_power_clip": m["fine_power_clip"],
            "coarse_complex_clip": m["coarse_complex_clip"],
            "fine_complex_clip": m["fine_complex_clip"],
        })
    # Weight audit.
    for name in ["c1_init", "c3_init", "complex_pilot50_c1", "complex_pilot50_c3",
                 "complex_pilot50_c1_s1730", "complex_pilot50_c3_s1730"]:
        if name == "c1_init":
            path = ROOT / "runs/complex_pilot_init/c1_init.pt"
        elif name == "c3_init":
            path = ROOT / "runs/complex_pilot_init/c3_init.pt"
        else:
            path = ROOT / f"runs/{name}/last.pt"
        if path.exists():
            out["weights"][name] = first_conv_stats(path)
    # Input-gradient RMS for trained C3 and C1.
    ds = CachedPairs(cache, "fit", 3)
    items = [ds[i] for i in range(min(8, len(ds)))]
    batch = collate_pairs(items)
    target = {"batch_idx": batch["batch_idx"].cuda(), "cls": batch["cls"].cuda(),
              "bboxes": batch["bboxes"].cuda()}
    for name, ch, path in [
        ("c1_1729", 1, ROOT / "runs/complex_pilot50_c1/last.pt"),
        ("c3_1729", 3, ROOT / "runs/complex_pilot50_c3/last.pt"),
        ("c3_1730", 3, ROOT / "runs/complex_pilot50_c3_s1730/last.pt"),
    ]:
        if not path.exists():
            continue
        from src.signals.model import build_model
        model = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"), device="cuda", ch=ch)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model.train()
        x = batch["img"][:, :ch].cuda().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss({**target, "img": x}, preds=model(x))[0].sum()
        loss.backward()
        g = x.grad
        out["weights"][name + "_input_grad_rms"] = [
            float(g[:, c].float().pow(2).mean().sqrt()) for c in range(ch)
        ]
        del model
        torch.cuda.empty_cache()
    # 90-degree rotation equivalence on two samples.
    for i, m in wanted[:2]:
        rec = fit[m["sample_id"]]
        fs = rec.B_mhz * 1e6; h = h_for_n(rec.N); n_ext = (N_TIME - 1) * h
        F = int(round(rec.B_mhz * 700)); L = fine_window_len(fs)
        iq = load_iq(rec.bin_path)
        z = complex_stft(iq, fs, L, h, n_ext)
        zr = complex_stft(iq * 1j, fs, L, h, n_ext)
        zc, pc = overlap_bands(z, COARSE_H, fs, F)
        zcr, pcr = overlap_bands(zr, COARSE_H, fs, F)
        zs = z[m["fine_start"]:m["fine_start"] + FINE_H]
        zsr = zr[m["fine_start"]:m["fine_start"] + FINE_H]
        rot_c = np.stack([-normalize_complex(zc, scale_c)[1].numpy(),
                          normalize_complex(zc, scale_c)[0].numpy()])
        rot_f = np.stack([-normalize_complex(zs, scale_f)[1].numpy(),
                          normalize_complex(zs, scale_f)[0].numpy()])
        out["rotation"].append({
            "sample_id": m["sample_id"],
            "coarse_rot_eq_max_abs": float(np.max(np.abs(rot_c - normalize_complex(zcr, scale_c).numpy()))),
            "fine_rot_eq_max_abs": float(np.max(np.abs(rot_f - normalize_complex(zsr, scale_f).numpy()))),
            "coarse_power_eq_max_abs": float(np.max(np.abs(pc - pcr).numpy())),
        })
    p = ROOT / "reports/complex_pilot_audit.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:12000])
    print("wrote", p)


if __name__ == "__main__":
    main()
