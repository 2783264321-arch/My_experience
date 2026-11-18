#!/usr/bin/env python3
"""C3 inference diagnostics: coarse/fine masking and global 90-degree rotation.

Uses all native fine slices of the fixed 128-sample val subset, the same
mapping/boundary/global-NMS rules, and no fine-class filter (the filter effect
is reported separately by eval_complex_pilot_full.py).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from src.signals.data import SPLIT_MANIFEST, h_for_n, load_manifest
from src.signals.infer import gt_global
from src.signals.model import build_model
from scripts.eval_complex_pilot_full import (
    build_views, evaluate_parts, predict_views,
)

ROOT = Path(__file__).resolve().parents[1]
MODES = ["normal", "all_complex_zero", "coarse_complex_zero",
         "fine_complex_zero", "coarse_view_zero", "fine_view_zero",
         "rotate90"]


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
    t0 = time.perf_counter()
    views = build_views(recs, lo, hi, scale_c, scale_f)
    gts = [gt_global(r, h_for_n(r.N)) for r in recs]
    sids = [r.sample_id for r in recs]
    print(f"views built {time.perf_counter()-t0:.0f}s", flush=True)
    out = {"val_samples": len(recs), "models": {}}
    for name, path in [
        ("C3_s1729", ROOT / "runs/complex_pilot50_c3/last.pt"),
        ("C3_s1730", ROOT / "runs/complex_pilot50_c3_s1730/last.pt"),
    ]:
        model = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"),
                            device="cuda", ch=3)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); model.eval()
        out["models"][name] = {}
        for mode in MODES:
            t = time.perf_counter()
            parts = predict_views(model, views, 3, batch_views=8, mode=mode)
            e = evaluate_parts(views, parts, gts, recs, sids, False)
            e["elapsed_s"] = time.perf_counter() - t
            out["models"][name][mode] = e
            print(f"{name} {mode}: mAP {e['mAP50']:.4f} agn {e['map50_agnostic']:.4f} "
                  f"loc {e['localization_recall']:.4f} "
                  f"012 {e['groups']['0/1/2']['group_class_ap_mean']:.4f} "
                  f"345 {e['groups']['3/4/5']['group_class_ap_mean']:.4f} "
                  f"narrow {e['groups']['narrow_9/12/13']['group_class_ap_mean']:.4f} "
                  f"{e['elapsed_s']:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    # C1 rotation control: power unchanged, so predictions should be identical.
    model = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"),
                        device="cuda", ch=1)
    ck = torch.load(ROOT / "runs/complex_pilot50_c1/last.pt",
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    out["C1_s1729_rotation_control"] = {}
    for mode in ["normal", "rotate90"]:
        parts = predict_views(model, views, 1, batch_views=8, mode=mode)
        e = evaluate_parts(views, parts, gts, recs, sids, False)
        out["C1_s1729_rotation_control"][mode] = e
        print(f"C1 rotation control {mode}: mAP {e['mAP50']:.4f}", flush=True)
    del model
    p = ROOT / "reports/complex_pilot_diagnostics.json"
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
