"""End-to-end val-subset benchmark: build views -> model -> decode -> merge."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/detector_upgrade"))

from src.signals.data import (SPLIT_MANIFEST, build_sample_views, frequency_starts,
                              load_manifest)
from src.signals.infer import (_boundary_mask, _map_view_dets, global_classwise_nms,
                               Prediction)
from ultralytics.utils.nms import non_max_suppression

OUT = ROOT / "reports/detector_upgrade"; OUT.mkdir(parents=True, exist_ok=True)


def pick_val(n_per_b=5):
    val = load_manifest(SPLIT_MANIFEST, split="val")
    by_b = {}
    for r in val:
        by_b.setdefault(round(float(r.B_mhz)), []).append(r)
    recs = []
    for b in sorted(by_b):
        recs += by_b[b][:n_per_b]
    return recs


def merge_view_dets(dets, types, starts, F, drop_fine_0_5=False):
    allb, alls, alll = [], [], []
    n_boundary = 0
    for det, vt, st in zip(dets, types, starts):
        pred = _map_view_dets(det, vt, st, F)
        bm = _boundary_mask(pred, vt, st, F)
        n_boundary += int(bm.sum())
        if bm.any():
            pred = Prediction(pred.boxes[~bm], pred.scores[~bm], pred.labels[~bm])
        if drop_fine_0_5 and vt == "fine" and len(pred.boxes):
            keep = pred.labels > 5
            pred = Prediction(pred.boxes[keep], pred.scores[keep], pred.labels[keep])
        allb.append(pred.boxes); alls.append(pred.scores); alll.append(pred.labels)
    pred = Prediction(torch.cat(allb), torch.cat(alls), torch.cat(alll))
    return global_classwise_nms(pred, 0.5, 100), n_boundary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["yolo26", "dfine"], required=True)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=8)
    args = ap.parse_args()
    import adapters

    if args.kind == "yolo26":
        model, info = adapters.build_yolo26(); model = model.cuda().eval()
        model.model[-1].end2end = False
        def infer(imgs):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                preds = model(imgs)
                dets = non_max_suppression(preds, 0.001, 0.5, nc=14, max_det=300,
                                           max_nms=30000, max_time_img=20)
            return [d.float() for d in dets]
    else:
        model, info = adapters.build_dfine(); model = model.cuda().eval()
        post = model.postprocessor
        def infer(imgs):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(imgs)
                sizes = torch.tensor([[512, 2048]] * imgs.shape[0], device=imgs.device)
                res = post(out, sizes)
            dets = []
            for r in res:
                dets.append(torch.cat([r["boxes"], r["scores"][:, None], r["labels"][:, None].float()], 1).float())
            return dets

    import json as _json
    norm = _json.loads((ROOT / "configs/normalizer.json").read_text())
    norm = (float(norm["lo"]), float(norm["hi"]))
    recs = pick_val()[: args.samples]
    torch.cuda.reset_peak_memory_stats()
    t_build = t_model = t_merge = 0.0
    n_views = 0
    n_gt = 0
    out_counts = []
    for r in recs:
        F = round(r.B_mhz * 700)
        starts = frequency_starts(F)
        t0 = time.perf_counter()
        coarse, fines, _ = build_sample_views(r, norm, list(range(len(starts))))
        views = [coarse] + fines
        t1 = time.perf_counter()
        dets, types, istarts = [], [], []
        for b in range(0, len(views), args.chunk):
            chunk = views[b:b + args.chunk]
            imgs = torch.stack([v.tensor for v in chunk]).cuda()
            d = infer(imgs)
            dets += d; types += [v.view_type for v in chunk]; istarts += [v.start for v in chunk]
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        merged, nb = merge_view_dets(dets, types, istarts, F)
        t3 = time.perf_counter()
        t_build += t1 - t0; t_model += t2 - t1; t_merge += t3 - t2
        n_views += len(views); n_gt += len(r.signals)
        out_counts.append({"id": r.sample_id, "B": float(r.B_mhz), "T": float(r.T_ms),
                           "views": len(views), "dets": len(merged.boxes)})
    total = t_build + t_model + t_merge
    res = {"kind": args.kind, "samples": len(recs), "views": n_views, "gt": n_gt,
           "build_s": t_build, "model_s": t_model, "merge_s": t_merge, "total_s": total,
           "per_view_model_ms": 1000 * t_model / max(n_views, 1),
           "per_view_total_ms": 1000 * total / max(n_views, 1),
           "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
           "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
           "build": info, "sample_detail": out_counts}
    print(json.dumps(res, ensure_ascii=False))
    (OUT / f"bench_val_{args.kind}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
