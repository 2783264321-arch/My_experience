#!/usr/bin/env python3
"""V2 SRRC expert: real fit YOLO proposals, 3-way keep/10/11 output.

Subcommands:
  prepare  build a parallel candidate-window cache from fit/val YOLO predictions
  train    fine-tune the V1 amplitude+phase backbone with a 3-class head
  eval     score the frozen YOLO val with V2 and report AP

No detector training, no test use.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn

from scripts.baseline3srrc_gt_pilot import INPUT_CHANNELS, TinyResNet, batches, predict
from scripts.baseline3srrc_phase0_verify import FS_OUT, N_WIN
from scripts.baseline3srrc_yolo_expert import local_box, report_ap
from src.signals.data import N_TIME, SPLIT_MANIFEST, h_for_n, load_manifest
from src.signals.infer import gt_global
from src.signals.metrics import ImagePredictions, ImageTargets, _match_single

ROOT = Path(__file__).resolve().parents[1]
PILOT = Path("/root/autodl-fs/runs_local/srrc-gt-pilot")
RUN = PILOT / "yolo26-label-only-v2"
SRRC = (10, 11)
IOU = 0.5


def load_predictions(manifest, pred_dir, split):
    ids = json.loads((pred_dir / f"{split}_sample_ids.json").read_text())
    records = {r.sample_id: r for r in load_manifest(manifest, split=split)}
    if len(ids) != len(records) or set(ids) != records.keys():
        raise ValueError("prediction IDs do not cover the split")
    with np.load(pred_dir / f"{split}_predictions_filtered_0_5.npz",
                 allow_pickle=False) as z:
        boxes = z["pred_boxes"]; scores = z["pred_scores"]
        labels = z["pred_labels"]; sample_index = z["pred_sample_index"]
    preds = []
    for i in range(len(ids)):
        m = sample_index == i
        preds.append(ImagePredictions(boxes[m], scores[m], labels[m]))
    return [records[s] for s in ids], preds


def group_map(manifest):
    with manifest.open(newline="", encoding="utf-8") as f:
        return {r["sample_id"]: r["group_id"] or r["sample_id"]
                for r in csv.DictReader(f)}


def fit_dev_groups(cache_path):
    with np.load(cache_path, allow_pickle=False) as z:
        groups = z["groups"]; is_dev = z["is_dev"]
    return set(groups[is_dev].tolist()), set(groups.tolist())


def crop_job(job):
    rec = SimpleNamespace(sample_id=job["sample_id"], B_mhz=job["B_mhz"],
                          T_ms=job["T_ms"], N=job["N"], f_lo=job["f_lo"],
                          bin_path=job["bin_path"])
    try:
        from scripts.baseline3srrc_gt_pilot import instance_windows
        w = instance_windows(rec, job["sig"], N_WIN, allow_short=True)
        return job["index"], w, None
    except Exception as exc:
        return job["index"], None, str(exc)[:200]


def prepare(args):
    records, preds = load_predictions(args.manifest, args.pred_dir, args.split)
    gmap = group_map(args.manifest)
    dev_groups, known_groups = (fit_dev_groups(args.fit_cache)
                                if args.split == "fit" else (set(), set()))
    targets = []
    for r in records:
        gt = gt_global(r, h_for_n(r.N))
        targets.append(ImageTargets(gt.boxes.numpy().reshape(-1, 4),
                                    gt.labels.numpy()))
    jobs = []; meta = []
    for image, (rec, pred) in enumerate(zip(records, preds)):
        width = (pred.boxes[:, 3] - pred.boxes[:, 1]) * rec.B_mhz
        routed = (np.isin(pred.labels, SRRC) & (pred.scores >= args.min_yolo_score)
                  & (width >= 9.5) & (width <= 10.5))
        matches, _ = _match_single(targets[image], pred, IOU, class_aware=False)
        truth = {int(pi): int(targets[image].labels[gi]) for pi, gi in matches}
        for idx in np.flatnonzero(routed):
            box = local_box(pred.boxes[idx], rec)
            if box["end_time"] - box["start_time"] < 1000 / FS_OUT:
                continue
            t = truth.get(int(idx))
            target = (t - 10) if t in SRRC else 2
            group = gmap[rec.sample_id]
            if args.split == "fit":
                if group in known_groups:
                    part = "dev" if group in dev_groups else "train"
                else:
                    part = ("dev" if int(hashlib.sha256(
                        f"v2:{group}".encode()).hexdigest()[:8], 16) % 5 == 0
                        else "train")
            else:
                part = "val"
            jobs.append({
                "index": len(jobs), "sample_id": rec.sample_id,
                "pred_idx": int(idx), "group": group, "part": part,
                "target": int(target), "yolo_class": int(pred.labels[idx]),
                "yolo_score": float(pred.scores[idx]), "matched": int(t is not None),
                "B_mhz": float(rec.B_mhz), "T_ms": float(rec.T_ms),
                "N": int(rec.N), "f_lo": float(rec.f_lo),
                "bin_path": str(rec.bin_path), "sig": box,
            })
            meta.append(jobs[-1])
    print(f"routed/scored candidates: {len(jobs)}", flush=True)
    t0 = time.perf_counter()
    windows = np.zeros((len(jobs), N_WIN, 2, 1024), dtype=np.float32)
    errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, (idx, w, err) in enumerate(ex.map(crop_job, jobs, chunksize=8)):
            if w is None:
                errors += 1
            else:
                windows[idx] = w
            if (k + 1) % 50 == 0 or k + 1 == len(jobs):
                print(f"  cropped {k+1}/{len(jobs)} {time.perf_counter()-t0:.0f}s",
                      flush=True)
    if errors:
        raise RuntimeError(f"{errors} crop failures; inspect before training")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, windows=windows,
        target=np.array([m["target"] for m in meta], np.int64),
        yolo_class=np.array([m["yolo_class"] for m in meta], np.int64),
        yolo_score=np.array([m["yolo_score"] for m in meta], np.float32),
        matched=np.array([m["matched"] for m in meta], np.int8),
        sample_ids=np.array([m["sample_id"] for m in meta]),
        pred_idx=np.array([m["pred_idx"] for m in meta], np.int64),
        groups=np.array([m["group"] for m in meta]),
        part=np.array([m["part"] for m in meta]),
        metadata=json.dumps({
            "split": args.split, "pred_dir": str(args.pred_dir),
            "n_candidates": len(meta), "n_windows": N_WIN,
            "window_points": 1024, "crop_seconds": time.perf_counter() - t0,
            "target_rule": "0/1 matched GT SRRC, 2 keep original",
        }))
    print(json.dumps({"candidates": len(meta),
                      "targets": {str(k): int(v) for k, v in
                                  zip(*np.unique([m["target"] for m in meta],
                                                 return_counts=True))},
                      "parts": {str(k): int(v) for k, v in
                                zip(*np.unique([m["part"] for m in meta],
                                               return_counts=True))},
                      "seconds": round(time.perf_counter() - t0, 2)}), flush=True)


def make_v2_model(v1_checkpoint, device):
    model = TinyResNet(in_channels=INPUT_CHANNELS["amplitude_phase_change"]).to(device)
    state = torch.load(v1_checkpoint, map_location=device, weights_only=True)
    old_w = state.pop("head.weight"); old_b = state.pop("head.bias")
    model.load_state_dict(state, strict=False)
    model.head = nn.Linear(model.head.in_features, 3).to(device)
    with torch.no_grad():
        model.head.weight[:2].copy_(old_w)
        model.head.bias[:2].copy_(old_b)
        model.head.weight[2].zero_()
        model.head.bias[2].zero_()
    return model


def metrics3(y, pred):
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    rec = [float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(3)]
    return {"n": int(len(y)), "accuracy": float(cm.trace() / max(cm.sum(), 1)),
            "balanced_accuracy": float(np.mean(rec)), "recall_0_1_2": rec,
            "confusion": cm.tolist()}


def train(args):
    with np.load(args.cache, allow_pickle=False) as z:
        cache = {k: z[k] for k in z.files}
    windows = torch.from_numpy(cache.pop("windows"))
    target = np.asarray(cache["target"], int)
    part = np.asarray(cache["part"])
    train_ids = np.flatnonzero(part == "train")
    dev_ids = np.flatnonzero(part == "dev")
    counts = np.bincount(target[train_ids], minlength=3).astype(float)
    weights = torch.tensor((counts.sum() / (3 * np.maximum(counts, 1))),
                           dtype=torch.float32, device=args.device)
    model = make_v2_model(args.v1, args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss(weight=weights)
    y = torch.from_numpy(target).long().to(args.device)
    rng = np.random.default_rng(args.seed)
    history = []
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for ids in batches(rng.permutation(train_ids), args.batch):
            opt.zero_grad(set_to_none=True)
            logits = predict(model, windows[ids].to(args.device),
                             "amplitude_phase_change")
            loss = lossf(logits, y[ids])
            loss.backward(); opt.step()
            total += float(loss) * len(ids)
        model.eval()
        with torch.inference_mode():
            dev_logits = torch.cat([
                predict(model, windows[ids].to(args.device),
                        "amplitude_phase_change").cpu()
                for ids in batches(dev_ids, args.batch)
            ]).numpy()
        m = metrics3(target[dev_ids], dev_logits.argmax(1))
        history.append({"epoch": epoch + 1, "loss": total / len(train_ids), "dev": m})
        print(f"v2 epoch {epoch+1}: loss={total/len(train_ids):.4f} "
              f"dev acc={m['accuracy']:.4f} bal={m['balanced_accuracy']:.4f} "
              f"rec={m['recall_0_1_2']}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "amplitude_phase_change_3class_last.pt")
    (args.out / "summary.json").write_text(json.dumps({
        "cache": str(args.cache), "v1": str(args.v1), "seed": args.seed,
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "class_counts_train": counts.tolist(), "class_weights": weights.tolist(),
        "n_train": int(len(train_ids)), "n_dev": int(len(dev_ids)),
        "seconds": time.perf_counter() - t0, "history": history,
    }, indent=2) + "\n", encoding="utf-8")


def apply_rule(preds, keys, decisions, threshold, keep_label=2):
    updated = [ImagePredictions(p.boxes, p.scores, p.labels.copy()) for p in preds]
    changed = 0
    for (image, index), (action, conf) in zip(keys, decisions):
        label = action + 10
        if action == keep_label or conf < threshold or label == preds[image].labels[index]:
            continue
        updated[image].labels[index] = label
        changed += 1
    return updated, changed


def eval_v2(args):
    with np.load(args.cache, allow_pickle=False) as z:
        cache = {k: z[k] for k in z.files}
    windows = torch.from_numpy(cache["windows"])
    target = np.asarray(cache["target"], int)
    keys = list(zip([str(x) for x in cache["sample_ids"]],
                    [int(x) for x in cache["pred_idx"]]))
    records, preds = load_predictions(args.manifest, args.pred_dir, "val")
    targets = []
    for r in records:
        gt = gt_global(r, h_for_n(r.N))
        targets.append(ImageTargets(gt.boxes.numpy().reshape(-1, 4),
                                    gt.labels.numpy()))
    baseline = report_ap(targets, preds)
    model = make_v2_model(args.v1, args.device)  # architecture
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device,
                                     weights_only=True)); model.eval()
    with torch.inference_mode():
        logits = torch.cat([
            predict(model, windows[i:i + 32].to(args.device),
                    "amplitude_phase_change").cpu()
            for i in range(0, len(windows), 32)
        ]).numpy()
    prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    action = logits.argmax(1); conf = prob.max(1)
    # Map cache rows to (image, pred_idx) by sample id lookup.
    by_id = {r.sample_id: i for i, r in enumerate(records)}
    keyed = [((by_id[str(sid)], int(pid)), int(a), float(c))
             for (sid, pid), a, c in zip(keys, action, conf)]
    known = {}
    for image, (gt, pred) in enumerate(zip(targets, preds)):
        matches, _ = _match_single(gt, pred, IOU, class_aware=False)
        known[image] = {pi: int(gt.labels[gi]) for pi, gi in matches}
    result = {"baseline": baseline, "routed": len(keyed)}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, thr in (("argmax", 0.0), ("conf_ge_0.70", 0.70)):
        updated, n_changed = apply_rule(preds, [k for k, _, _ in keyed],
                                        [(a, c) for _, a, c in keyed], thr)
        ap = report_ap(targets, updated)
        corrected = broken = broken_class10 = unmatched = other = 0
        matched_up = [ImagePredictions(p.boxes, p.scores, p.labels.copy()) for p in preds]
        unmatched_up = [ImagePredictions(p.boxes, p.scores, p.labels.copy()) for p in preds]
        rows = []
        for (image, idx), a, c in keyed:
            truth = known[image].get(idx)
            prev = int(preds[image].labels[idx])
            new = int(updated[image].labels[idx])
            rows.append((image, idx, a, c, truth, int(new != prev)))
            if new == prev:
                continue
            if truth is None:
                unmatched += 1
                unmatched_up[image].labels[idx] = new
            elif truth not in SRRC:
                other += 1
            else:
                corrected += int(new == truth)
                broken += int(prev == truth)
                broken_class10 += int(prev == truth == 10)
                matched_up[image].labels[idx] = new
        map_matched = report_ap(targets, matched_up)
        map_unmatched = report_ap(targets, unmatched_up)
        result[name] = {
            "threshold": thr, "n_changed": n_changed,
            "corrected": corrected, "broken": broken, "broken_class10": broken_class10,
            "unmatched_changed": unmatched, "other_matched_class_changed": other,
            "net": corrected - broken, "map50": ap["map50"],
            "map50_95": ap["map50_95"], "ap50": ap["ap50"],
            "delta_map50_pp": (ap["map50"] - baseline["map50"]) * 100,
            "matched_only_map50": map_matched["map50"],
            "unmatched_only_map50": map_unmatched["map50"],
            "matched_only_delta_pp": (map_matched["map50"] - baseline["map50"]) * 100,
            "unmatched_only_delta_pp": (map_unmatched["map50"] - baseline["map50"]) * 100,
        }
        path = args.out / f"{name}_candidates.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "pred_idx", "yolo_class", "expert_action",
                        "expert_confidence", "matched_gt_class", "changed"])
            for image, idx, action, confidence, truth, changed in rows:
                w.writerow([records[image].sample_id, idx,
                            int(preds[image].labels[idx]), action, confidence,
                            "" if truth is None else truth, changed])
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps({"baseline_map50": baseline["map50"],
                      "routed": len(keyed),
                      "rules": {k: {x: result[k][x] for x in
                                    ("n_changed", "corrected", "broken",
                                     "unmatched_changed", "map50",
                                     "delta_map50_pp", "ap50")}
                                for k in ("argmax", "conf_ge_0.70")}},
                     indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--split", choices=("fit", "val"), required=True)
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fit-cache", type=Path, default=PILOT / "fit_8win.npz")
    p.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--min-yolo-score", type=float, default=0.01)
    p.set_defaults(fn=prepare)
    t = sub.add_parser("train")
    t.add_argument("--cache", type=Path, required=True)
    t.add_argument("--v1", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=1729)
    t.set_defaults(fn=train)
    e = sub.add_parser("eval")
    e.add_argument("--cache", type=Path, required=True)
    e.add_argument("--v1", type=Path, required=True)
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--pred-dir", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST)
    e.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    e.set_defaults(fn=eval_v2)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
