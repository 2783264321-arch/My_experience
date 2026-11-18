#!/usr/bin/env python3
"""Single-sample inference trace: IQ -> STFT -> views -> model -> NMS -> boxes.

This is a diagnostic/reproducibility tool, not a training script. It uses the
same functions as src/signals/infer.py and validate.py, but prints and saves
every intermediate count/shape. The sample is taken from the frozen val split
by default; the test split is sealed.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.signals.data import (
    COARSE_H, FINE_H, N_TIME, SPLIT_MANIFEST, WINDOW_MS, _label_boxes,
    frequency_starts, load_iq, load_manifest,
)
from src.signals.infer import (
    Prediction, _boundary_mask, _map_view_dets, _nms_view_batch,
    filter_fine_classes, global_classwise_nms, gt_global, parse_class_set,
)
from src.signals.metrics import ImagePredictions, ImageTargets, evaluate
from src.signals.model import build_model
from src.signals.stft_fast import (
    aggregate_frequency_fast, normalize_slice, stft_power_fast,
)

ROOT = Path(__file__).resolve().parents[1]


def to_physical(box, B, T, f_lo, f_hi, h):
    x1, y1, x2, y2 = [float(v) for v in box]
    fs = B * 1e6
    t1 = ((x1 * N_TIME - 0.5) * h / fs) * 1000.0
    t2 = ((x2 * N_TIME - 0.5) * h / fs) * 1000.0
    f1 = f_lo + y1 * B
    f2 = f_lo + y2 * B
    t1 = max(0.0, min(T, t1))
    t2 = max(0.0, min(T, t2))
    f1 = max(f_lo, min(f_hi, f1))
    f2 = max(f_lo, min(f_hi, f2))
    return t1, t2, f1, f2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="train/872")
    ap.add_argument("--split", choices=["fit", "val"], default="val")
    ap.add_argument("--checkpoint", default="runs/yolo11s_c1_directreg_2048x512/best_corrected.pt")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--max-det-view", type=int, default=300)
    ap.add_argument("--batch-views", type=int, default=8)
    ap.add_argument("--fft-threads", type=int, default=2)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--drop-fine-classes", default="")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()

    t_start = time.perf_counter()
    records = load_manifest(SPLIT_MANIFEST, split=args.split)
    by_id = {r.sample_id: r for r in records}
    if args.sample not in by_id:
        raise SystemExit(f"sample {args.sample} not in {args.split} ({len(records)} records)")
    r = by_id[args.sample]
    h = math.ceil(r.N / (N_TIME - 1))
    norm_cfg = json.loads((ROOT / "configs/normalizer.json").read_text())
    lo, hi = float(norm_cfg["lo"]), float(norm_cfg["hi"])
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    trace = {
        "sample": {
            "sample_id": r.sample_id, "split": args.split,
            "B_mhz": r.B_mhz, "T_ms": r.T_ms, "N": r.N,
            "f_lo_mhz": r.f_lo, "f_hi_mhz": r.f_hi,
            "bin_path": str(r.bin_path), "bin_bytes": r.bin_path.stat().st_size,
            "json_path": str(r.json_path), "n_signals": len(r.signals),
            "signal_classes": [int(s["class"]) for s in r.signals],
            "h_formula": h,
        },
        "normalizer": {"lo": lo, "hi": hi},
        "checkpoint": str(ckpt_path),
        "stages": {},
        "views": [],
        "per_view_nms": [],
        "final_boxes_physical": [],
        "gt_boxes_physical": [],
    }
    print(f"sample {r.sample_id} split={args.split} B={r.B_mhz} T={r.T_ms} N={r.N}", flush=True)

    # 1. IQ
    t = time.perf_counter()
    iq = load_iq(r.bin_path)
    iq_s = time.perf_counter() - t
    iq_stats = {
        "n": int(iq.numel()), "dtype": str(iq.dtype),
        "real_min": float(iq.real.min()), "real_max": float(iq.real.max()),
        "imag_min": float(iq.imag.min()), "imag_max": float(iq.imag.max()),
        "first3": [[float(v.real), float(v.imag)] for v in iq[:3]],
        "seconds": iq_s,
    }
    trace["stages"]["iq"] = iq_stats
    print("IQ", json.dumps(iq_stats, ensure_ascii=False), flush=True)

    # 2. STFT
    t = time.perf_counter()
    p, win, h_stft, L, F = stft_power_fast(iq, r.B_mhz, fft_threads=args.fft_threads)
    stft_s = time.perf_counter() - t
    n_ext = (N_TIME - 1) * h_stft
    pad = n_ext - r.N
    delta_f_khz = (r.B_mhz * 1e6 / L) / 1000.0
    delta_t_ms = h_stft / (r.B_mhz * 1e6) * 1000.0
    stft_stats = {
        "L": int(L), "h": int(h_stft), "h_formula": h,
        "N_ext": int(n_ext), "right_pad_samples": int(pad),
        "F": int(F), "power_shape": list(p.shape),
        "delta_f_khz": delta_f_khz, "delta_t_ms": delta_t_ms,
        "power_min": float(p.min()), "power_max": float(p.max()),
        "power_mean": float(p.mean()),
        "window_samples": int(win.size),
        "window_sum_sq": float(np.sum(win.astype(np.float64) ** 2)),
        "backend": "scipy.fft (time-major, 512xF)",
        "seconds": stft_s,
    }
    if int(h_stft) != h:
        raise SystemExit(f"h mismatch: stft {h_stft} vs ceil {h}")
    trace["stages"]["stft"] = stft_stats
    print("STFT", json.dumps(stft_stats, ensure_ascii=False), flush=True)

    # 3. views
    t = time.perf_counter()
    coarse_p = aggregate_frequency_fast(p)
    coarse_t = normalize_slice(coarse_p, lo, hi).unsqueeze(0)
    c_boxes, c_labels, c_ids, c_trunc = _label_boxes(r.signals, r, F, h, None)
    coarse = SimpleNamespace(
        tensor=coarse_t, view_type="coarse", start=0,
        boxes=c_boxes, labels=c_labels, ids=c_ids, truncated=c_trunc,
    )
    starts = frequency_starts(F)
    views = [coarse]
    for idx, start in enumerate(starts):
        f_t = normalize_slice(p[:, start:start + FINE_H], lo, hi).unsqueeze(0)
        f_boxes, f_labels, f_ids, f_trunc = _label_boxes(r.signals, r, F, h, start)
        views.append(SimpleNamespace(
            tensor=f_t, view_type="fine", start=start,
            boxes=f_boxes, labels=f_labels, ids=f_ids, truncated=f_trunc,
        ))
    views_s = time.perf_counter() - t
    for i, v in enumerate(views):
        rec = {
            "index": i, "type": v.view_type, "start": int(v.start),
            "shape": list(v.tensor.shape),
            "norm_min": float(v.tensor.min()), "norm_max": float(v.tensor.max()),
            "norm_mean": float(v.tensor.mean()),
            "gt_count": int(len(v.labels)),
            "gt_classes": dict(Counter(int(x) for x in v.labels.tolist())),
            "gt_truncated": int(v.truncated.sum()) if v.truncated is not None else 0,
        }
        trace["views"].append(rec)
    print(f"views: {len(views)} ({len(starts)} fine + 1 coarse), "
          f"built in {views_s:.2f}s", flush=True)
    for rec in trace["views"][:6]:
        print("  view", json.dumps(rec, ensure_ascii=False), flush=True)
    if len(trace["views"]) > 6:
        print(f"  ... {len(trace['views']) - 6} more views", flush=True)

    # 4. model
    model = build_model(str(ROOT / "configs/yolo11s_c1_directreg.yaml"), device="cuda")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["ema"] if ck.get("ema") is not None else ck["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    head = model.model[-1]
    model.eval()
    model_stats = {
        "checkpoint": str(ckpt_path), "weights": "ema" if ck.get("ema") is not None else "model",
        "epoch": ck.get("epoch"), "best_fitness": ck.get("best_fitness"),
        "nc": int(head.nc), "reg_max": int(head.reg_max),
        "end2end": bool(head.end2end),
        "stride": [int(x) for x in head.stride.tolist()],
        "first_conv": list(model.model[0].conv.weight.shape),
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "input": [1, 1, FINE_H, N_TIME],
        "precision": "fp32 (no autocast)",
    }
    trace["stages"]["model"] = model_stats
    print("MODEL", json.dumps(model_stats, ensure_ascii=False), flush=True)

    # 5. per-view NMS + boundary + global merge
    t = time.perf_counter()
    parts = []
    n_raw = n_boundary = 0
    for begin in range(0, len(views), args.batch_views):
        chunk = views[begin:begin + args.batch_views]
        images = torch.stack([v.tensor for v in chunk], 0).cuda()
        dets = _nms_view_batch(
            model, images, conf=args.conf, iou=args.iou,
            max_det=args.max_det_view, multi_label=False,
        )
        for v, det in zip(chunk, dets):
            pred = _map_view_dets(det, v.view_type, v.start, F)
            bm = _boundary_mask(pred, v.view_type, v.start, F)
            nb = int(bm.sum())
            if nb:
                pred = Prediction(pred.boxes[~bm], pred.scores[~bm], pred.labels[~bm])
            pred = filter_fine_classes(pred, v.view_type, parse_class_set(args.drop_fine_classes))
            n_raw += int(len(det))
            n_boundary += nb
            parts.append(pred)
            top_score = float(det[:, 4].max()) if len(det) else None
            top_class = int(det[int(det[:, 4].argmax()), 5]) if len(det) else None
            rec = {
                "index": v.view_type + (f"@{v.start}" if v.view_type == "fine" else ""),
                "raw": int(len(det)), "boundary": nb,
                "after_boundary": int(len(pred.boxes)),
                "top_score": top_score, "top_class": top_class,
            }
            trace["per_view_nms"].append(rec)
            print("NMS", json.dumps(rec, ensure_ascii=False), flush=True)
    if parts:
        all_pred = Prediction(
            torch.cat([x.boxes for x in parts]),
            torch.cat([x.scores for x in parts]),
            torch.cat([x.labels for x in parts]),
        )
    else:
        all_pred = Prediction(torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long))
    merged = global_classwise_nms(all_pred, iou=args.iou, max_det=args.max_det)
    merge_s = time.perf_counter() - t
    scores = merged.scores.numpy()
    labels = merged.labels.numpy()
    merge_stats = {
        "n_views": len(views),
        "n_raw_dets": n_raw,
        "n_boundary_dropped": n_boundary,
        "n_after_boundary": int(len(all_pred.boxes)),
        "n_final": int(len(merged.boxes)),
        "class_counts": {str(i): int((labels == i).sum()) for i in range(14) if (labels == i).any()},
        "score_min": float(scores.min()) if len(scores) else None,
        "score_max": float(scores.max()) if len(scores) else None,
        "score_mean": float(scores.mean()) if len(scores) else None,
        "n_score_ge_0.05": int((scores >= 0.05).sum()),
        "n_score_ge_0.25": int((scores >= 0.25).sum()),
        "n_score_ge_0.50": int((scores >= 0.50).sum()),
        "seconds": merge_s,
        "conf": args.conf, "iou": args.iou,
        "max_det_view": args.max_det_view, "max_det": args.max_det,
    }
    trace["stages"]["merge"] = merge_stats
    print("MERGE", json.dumps(merge_stats, ensure_ascii=False), flush=True)

    # 6. physical final boxes
    order = np.argsort(-scores) if len(scores) else np.array([], dtype=int)
    final_phys = []
    for rank, j in enumerate(order[: args.top]):
        t1, t2, f1, f2 = to_physical(merged.boxes[j].numpy(), r.B_mhz, r.T_ms, r.f_lo, r.f_hi, h)
        rec = {
            "rank": int(rank + 1), "class": int(labels[j]), "score": float(scores[j]),
            "t1_ms": t1, "t2_ms": t2, "f1_mhz": f1, "f2_mhz": f2,
            "bw_mhz": f2 - f1, "dur_ms": t2 - t1,
            "box_norm": [round(float(x), 6) for x in merged.boxes[j].numpy()],
        }
        final_phys.append(rec)
    trace["final_boxes_physical"] = final_phys
    all_phys = []
    for j in order:
        t1, t2, f1, f2 = to_physical(merged.boxes[j].numpy(), r.B_mhz, r.T_ms, r.f_lo, r.f_hi, h)
        all_phys.append({
            "class": int(labels[j]), "score": float(scores[j]),
            "t1_ms": t1, "t2_ms": t2, "f1_mhz": f1, "f2_mhz": f2,
            "bw_mhz": f2 - f1, "dur_ms": t2 - t1,
            "box_norm": [round(float(x), 6) for x in merged.boxes[j].numpy()],
        })
    trace["final_boxes_all_physical"] = all_phys
    print(f"FINAL boxes (top {len(final_phys)})", flush=True)
    for rec in final_phys:
        print("  ", json.dumps(rec, ensure_ascii=False), flush=True)

    # 7. GT physical (val only)
    gt = gt_global(r, h)
    gt_phys = []
    for sig, b, lab in zip(r.signals, gt.boxes.numpy(), gt.labels.numpy()):
        t1, t2, f1, f2 = to_physical(b, r.B_mhz, r.T_ms, r.f_lo, r.f_hi, h)
        gt_phys.append({
            "signal_id": int(sig["signal_id"]), "class": int(lab),
            "t1_ms": t1, "t2_ms": t2, "f1_mhz": f1, "f2_mhz": f2,
            "bw_mhz": f2 - f1, "dur_ms": t2 - t1,
        })
    trace["gt_boxes_physical"] = gt_phys
    print("GT", json.dumps(gt_phys, ensure_ascii=False), flush=True)

    # 8. one-sample evaluator
    ev = evaluate(
        [ImageTargets(gt.boxes.numpy(), gt.labels.numpy())],
        [ImagePredictions(merged.boxes.numpy(), merged.scores.numpy(), merged.labels.numpy())],
        iou_thresholds=(0.5,), max_dets=args.max_det,
    )
    trace["sample_eval"] = ev
    pi = ev["per_iou"]["0.5"]
    print("SAMPLE EVAL", json.dumps({
        "n_gt": pi["n_gt"], "n_pred": int(len(merged.boxes)),
        "map_class_aware": pi["map_class_aware"],
        "ap_class_agnostic": pi["ap_class_agnostic"],
        "per_class": pi["per_class"],
        "classification": ev["classification"],
    }, ensure_ascii=False, indent=2), flush=True)

    # 9. save
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.sample.replace("/", "_")
    json_path = out_dir / f"inference_trace_{stem}.json"
    md_path = out_dir / f"inference_trace_{stem}.md"
    trace["total_seconds"] = time.perf_counter() - t_start
    json_path.write_text(json.dumps(trace, indent=2))
    write_markdown(trace, md_path)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {md_path}", flush=True)
    print(f"total {trace['total_seconds']:.2f}s", flush=True)


def write_markdown(tr, path: Path):
    s = tr["sample"]
    st = tr["stages"]
    lines = []
    lines.append(f"# 单样本推理 trace: {s['sample_id']} ({s['split']})")
    lines.append("")
    lines.append(f"- B={s['B_mhz']:g} MHz, T={s['T_ms']:g} ms, N={s['N']}, "
                 f"f_lo={s['f_lo_mhz']:g}, f_hi={s['f_hi_mhz']:g}, GT={s['n_signals']}")
    lines.append(f"- checkpoint: `{tr['checkpoint']}`")
    lines.append(f"- normalizer: lo={tr['normalizer']['lo']:.6f}, hi={tr['normalizer']['hi']:.6f}")
    lines.append("")
    lines.append("## 0. 输入")
    lines.append("")
    lines.append(f"| 项 | 值 |")
    lines.append(f"| --- | --- |")
    lines.append(f"| bin 文件 | `{s['bin_path']}` |")
    lines.append(f"| bin 字节 | {s['bin_bytes']} |")
    lines.append(f"| IQ 复采样 | {st['iq']['n']} |")
    lines.append(f"| I 范围 | [{st['iq']['real_min']:.6g}, {st['iq']['real_max']:.6g}] |")
    lines.append(f"| Q 范围 | [{st['iq']['imag_min']:.6g}, {st['iq']['imag_max']:.6g}] |")
    lines.append("")
    lines.append("## 1. STFT")
    lines.append("")
    lines.append(f"| 项 | 值 |")
    lines.append(f"| --- | --- |")
    lines.append(f"| L (窗长) | {st['stft']['L']} |")
    lines.append(f"| h (跳步) | {st['stft']['h']} (ceil(N/511)) |")
    lines.append(f"| N_ext | {st['stft']['N_ext']} |")
    lines.append(f"| 右补零 | {st['stft']['right_pad_samples']} |")
    lines.append(f"| F | {st['stft']['F']} |")
    lines.append(f"| 功率图 shape | {st['stft']['power_shape']} |")
    lines.append(f"| Δf | {st['stft']['delta_f_khz']:.6f} kHz |")
    lines.append(f"| Δt | {st['stft']['delta_t_ms']:.6f} ms |")
    lines.append(f"| 功率 min/max/mean | {st['stft']['power_min']:.6g} / {st['stft']['power_max']:.6g} / {st['stft']['power_mean']:.6g} |")
    lines.append(f"| STFT 耗时 | {st['stft']['seconds']*1000:.1f} ms |")
    lines.append("")
    lines.append("## 2. 视图")
    lines.append("")
    lines.append("| # | type | start | shape | norm mean | GT | GT classes | truncated |")
    lines.append("| ---: | --- | ---: | --- | ---: | ---: | --- | ---: |")
    for v in tr["views"]:
        cls = ",".join(f"{k}:{n}" for k, n in v["gt_classes"].items())
        lines.append(f"| {v['index']} | {v['type']} | {v['start']} | {v['shape']} | "
                     f"{v['norm_mean']:.4f} | {v['gt_count']} | {cls} | {v['gt_truncated']} |")
    lines.append("")
    lines.append("## 3. 模型")
    lines.append("")
    m = st["model"]
    lines.append(f"| 项 | 值 |")
    lines.append(f"| --- | --- |")
    lines.append(f"| checkpoint | `{m['checkpoint']}` ({m['weights']}) |")
    lines.append(f"| nc / reg_max / end2end | {m['nc']} / {m['reg_max']} / {m['end2end']} |")
    lines.append(f"| stride | {m['stride']} |")
    lines.append(f"| 首层卷积 | {m['first_conv']} |")
    lines.append(f"| 参数数 | {m['param_count']} |")
    lines.append("")
    lines.append("## 4. 每视图 NMS")
    lines.append("")
    lines.append("| view | raw | boundary | after | top score | top class |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for v in tr["per_view_nms"]:
        ts = f"{v['top_score']:.4f}" if v["top_score"] is not None else "—"
        tc = str(v["top_class"]) if v["top_class"] is not None else "—"
        lines.append(f"| {v['index']} | {v['raw']} | {v['boundary']} | {v['after_boundary']} | {ts} | {tc} |")
    lines.append("")
    lines.append("## 5. 全局合并")
    lines.append("")
    mg = st["merge"]
    lines.append(f"| 项 | 值 |")
    lines.append(f"| --- | --- |")
    lines.append(f"| 视图数 | {mg['n_views']} |")
    lines.append(f"| 原始检测 | {mg['n_raw_dets']} |")
    lines.append(f"| 边界扣留 | {mg['n_boundary_dropped']} |")
    lines.append(f"| 边界后 | {mg['n_after_boundary']} |")
    lines.append(f"| 最终 | {mg['n_final']} |")
    lines.append(f"| score min/max/mean | {mg['score_min']} / {mg['score_max']} / {mg['score_mean']} |")
    lines.append(f"| score>=0.05/0.25/0.50 | {mg['n_score_ge_0.05']} / {mg['n_score_ge_0.25']} / {mg['n_score_ge_0.50']} |")
    lines.append(f"| class counts | {mg['class_counts']} |")
    lines.append(f"| 合并耗时 | {mg['seconds']*1000:.1f} ms |")
    lines.append("")
    lines.append("## 6. 最终物理框（top）")
    lines.append("")
    lines.append("| # | class | score | t1 ms | t2 ms | f1 MHz | f2 MHz | BW MHz | dur ms |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for b in tr["final_boxes_physical"]:
        lines.append(f"| {b['rank']} | {b['class']} | {b['score']:.4f} | "
                     f"{b['t1_ms']:.3f} | {b['t2_ms']:.3f} | {b['f1_mhz']:.4f} | "
                     f"{b['f2_mhz']:.4f} | {b['bw_mhz']:.4f} | {b['dur_ms']:.3f} |")
    lines.append("")
    lines.append("## 7. GT 物理框")
    lines.append("")
    lines.append("| signal | class | t1 ms | t2 ms | f1 MHz | f2 MHz | BW MHz | dur ms |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for b in tr["gt_boxes_physical"]:
        lines.append(f"| {b['signal_id']} | {b['class']} | {b['t1_ms']:.3f} | "
                     f"{b['t2_ms']:.3f} | {b['f1_mhz']:.4f} | {b['f2_mhz']:.4f} | "
                     f"{b['bw_mhz']:.4f} | {b['dur_ms']:.3f} |")
    lines.append("")
    lines.append("## 8. 单样本评估（IoU=0.5）")
    lines.append("")
    ev = tr["sample_eval"]
    pi = ev["per_iou"]["0.5"]
    lines.append(f"- n_gt={pi['n_gt']}, n_pred={mg['n_final']}")
    lines.append(f"- class-aware mAP@0.5: {pi['map_class_aware']}")
    lines.append(f"- class-agnostic AP@0.5: {pi['ap_class_agnostic']}")
    lines.append(f"- localization recall: {ev['classification']['localization_recall']}")
    lines.append(f"- classification accuracy: {ev['classification']['classification_accuracy']}")
    lines.append("")
    lines.append("> 单样本 AP 只用于核对链路，不是数据集指标；正式指标见 `reports/stage5_results.md`。")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
