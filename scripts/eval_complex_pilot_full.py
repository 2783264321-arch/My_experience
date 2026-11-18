#!/usr/bin/env python3
"""Evaluate C1/C3 pilot weights on all native fine slices of the fixed val subset.

Builds the coarse 50%-overlap power/complex view and all native fine power/
complex slices for 128 val samples once, then runs each checkpoint over the
same views. Reports no-filter and fine-0-5-filtered metrics with the standard
mapping, boundary drop and global class-wise NMS.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.signals.complex_frontend import (
    complex_stft, fine_window_len, normalize_complex, normalize_log_power,
    overlap_bands,
)
from src.signals.data import (
    COARSE_H, FINE_H, N_TIME, SPLIT_MANIFEST, _label_boxes,
    frequency_starts, h_for_n, load_iq, load_manifest,
)
from src.signals.infer import (
    Prediction, _boundary_mask, _map_view_dets, _nms_view_batch,
    filter_fine_classes, global_classwise_nms, gt_global,
)
from src.signals.metrics import ImagePredictions, evaluate
from src.signals.model import build_model
from scripts.view_source_attribution import CLASS_GROUPS, GROUPS, evaluate_group

ROOT = Path(__file__).resolve().parents[1]


def build_views(records, lo, hi, scale_c, scale_f):
    out = []
    t0 = time.perf_counter()
    for ni, rec in enumerate(records):
        fs = rec.B_mhz * 1e6
        h = h_for_n(rec.N)
        n_ext = (N_TIME - 1) * h
        F = int(round(rec.B_mhz * 700))
        L = fine_window_len(fs)
        z = complex_stft(load_iq(rec.bin_path), fs, L, h, n_ext)
        zc, pc = overlap_bands(z, COARSE_H, fs, F)
        coarse = np.empty((3, COARSE_H, N_TIME), dtype=np.float16)
        coarse[0] = normalize_log_power(pc, lo, hi).numpy().astype(np.float16)
        coarse[1:3] = normalize_complex(zc, scale_c).numpy().astype(np.float16)
        cb, cl, _, _ = _label_boxes(rec.signals, rec, F, h, None)
        fines = []
        for start in frequency_starts(F):
            zf = z[start:start + FINE_H]
            fine = np.empty((3, FINE_H, N_TIME), dtype=np.float16)
            fine[0] = normalize_log_power(zf.abs().square(), lo, hi).numpy().astype(np.float16)
            fine[1:3] = normalize_complex(zf, scale_f).numpy().astype(np.float16)
            fb, fl, _, _ = _label_boxes(rec.signals, rec, F, h, start)
            fines.append({"tensor": fine, "boxes": fb, "labels": fl, "start": int(start)})
        out.append({"sample_id": rec.sample_id, "F": F, "coarse": coarse,
                    "coarse_boxes": cb, "coarse_labels": cl,
                    "fine": fines, "record": rec})
        if (ni + 1) % 20 == 0 or ni + 1 == len(records):
            print(f"views {ni+1}/{len(records)} {time.perf_counter()-t0:.0f}s", flush=True)
    return out


@torch.no_grad()
def predict_views(model, sample_views, channels, batch_views=8, mode="normal"):
    """Return per-sample list of (Prediction, view_type, start).

    mode is a diagnostic transform: coarse_complex_zero, fine_complex_zero,
    coarse_view_zero, fine_view_zero, rotate90.
    """
    flat = []
    for i, sv in enumerate(sample_views):
        flat.append((i, "coarse", 0, sv["coarse"], sv["F"]))
        for fv in sv["fine"]:
            flat.append((i, "fine", fv["start"], fv["tensor"], sv["F"]))
    out = [[] for _ in sample_views]
    for b in range(0, len(flat), batch_views):
        chunk = flat[b:b + batch_views]
        images = torch.stack([torch.from_numpy(np.asarray(x[3][:channels]).copy()).float()
                              for x in chunk], 0).cuda()
        if mode != "normal":
            for j, (_, vt, _, _, _) in enumerate(chunk):
                if mode == "coarse_complex_zero" and vt == "coarse":
                    images[j, 1:3] = 0
                elif mode == "fine_complex_zero" and vt == "fine":
                    images[j, 1:3] = 0
                elif mode == "coarse_view_zero" and vt == "coarse":
                    images[j] = 0
                elif mode == "fine_view_zero" and vt == "fine":
                    images[j] = 0
                elif mode == "all_complex_zero" and channels == 3:
                    images[j, 1:3] = 0
                elif mode == "rotate90" and channels == 3:
                    re = images[j, 1].clone(); im = images[j, 2].clone()
                    images[j, 1] = -im; images[j, 2] = re
        dets = _nms_view_batch(model, images, conf=0.001, iou=0.5,
                               max_det=300, multi_label=False)
        for (i, vt, st, _, F), det in zip(chunk, dets):
            pred = _map_view_dets(det, vt, st, F)
            bm = _boundary_mask(pred, vt, st, F)
            if bm.any():
                pred = Prediction(pred.boxes[~bm], pred.scores[~bm], pred.labels[~bm])
            out[i].append((pred, vt, st))
    return out


def evaluate_parts(sample_views, parts, gts, records, sample_ids, drop_fine):
    preds = []
    for i, plist in enumerate(parts):
        pp = []
        for pred, vt, st in plist:
            if drop_fine and vt == "fine":
                pred = filter_fine_classes(pred, vt, list(range(6)))
            pp.append(pred)
        cat = Prediction(torch.cat([p.boxes for p in pp]),
                         torch.cat([p.scores for p in pp]),
                         torch.cat([p.labels for p in pp]))
        final = global_classwise_nms(cat, iou=0.5, max_det=100)
        preds.append(ImagePredictions(final.boxes.numpy(), final.scores.numpy(),
                                      final.labels.numpy()))
    ev = evaluate(gts, preds, iou_thresholds=(0.5,), max_dets=100)
    pi = ev["per_iou"]["0.5"]
    out = {"mAP50": pi["map_class_aware"],
           "map50_agnostic": pi["ap_class_agnostic"],
           "localization_recall": ev["classification"]["localization_recall"],
           "classification_accuracy": ev["classification"]["classification_accuracy"],
           "n_pred": sum(len(p.boxes) for p in preds),
           "per_class": pi["per_class"], "groups": {},
           "confusion": ev["classification"]["confusion_gt_x_pred"]}
    for g in GROUPS:
        gm = evaluate_group(g, gts, preds, CLASS_GROUPS[g], records, sample_ids)
        aps = [pi["per_class"].get(str(c), {}).get("ap") for c in CLASS_GROUPS[g]]
        gm["group_class_ap_mean"] = float(np.mean([a for a in aps if a is not None])) if any(a is not None for a in aps) else None
        out["groups"][g] = gm
    return out


def main():
    cache = ROOT / "reports/complex_pilot_cache"
    fit_meta = json.loads((cache / "fit_meta.json").read_text())
    subset = json.loads((cache / "subset.json").read_text())
    norm = fit_meta["normalizer"]; lo, hi = norm["lo"], norm["hi"]
    scale_c = fit_meta["scale_coarse"]; scale_f = fit_meta["scale_fine"]
    val_all = load_manifest(SPLIT_MANIFEST, "val")
    by_id = {r.sample_id: r for r in val_all}
    recs = [by_id[s] for s in subset["val"]]
    print(f"val {len(recs)}", flush=True)
    sample_views = build_views(recs, lo, hi, scale_c, scale_f)
    gts = [gt_global(r, h_for_n(r.N)) for r in recs]
    sample_ids = [r.sample_id for r in recs]
    out = {"val_samples": len(recs), "weights": {}}
    for name, ch, path in [
        ("C1_s1729", 1, ROOT / "runs/complex_pilot50_c1/last.pt"),
        ("C3_s1729", 3, ROOT / "runs/complex_pilot50_c3/last.pt"),
        ("C1_s1730", 1, ROOT / "runs/complex_pilot50_c1_s1730/last.pt"),
        ("C3_s1730", 3, ROOT / "runs/complex_pilot50_c3_s1730/last.pt"),
    ]:
        print("eval", name, flush=True)
        t0 = time.perf_counter()
        model = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"),
                            device="cuda", ch=ch)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        parts = predict_views(model, sample_views, ch, batch_views=8)
        entry = {"channels": ch, "elapsed_s": time.perf_counter() - t0,
                 "no_filter": evaluate_parts(sample_views, parts, gts, recs, sample_ids, False),
                 "fine_0_5_filter": evaluate_parts(sample_views, parts, gts, recs, sample_ids, True)}
        out["weights"][name] = entry
        for cond in ["no_filter", "fine_0_5_filter"]:
            e = entry[cond]
            print(f" {name} {cond}: mAP {e['mAP50']:.4f} agn {e['map50_agnostic']:.4f} "
                  f"loc {e['localization_recall']:.4f} 012 {e['groups']['0/1/2']['group_class_ap_mean']:.4f} "
                  f"345 {e['groups']['3/4/5']['group_class_ap_mean']:.4f} "
                  f"time {entry['elapsed_s']:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    p = ROOT / "reports/complex_pilot_full_fine_eval.json"
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
