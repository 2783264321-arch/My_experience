#!/usr/bin/env python3
"""Offline SRRC 10/11 YOLO-box correction audit.

Stages
  prepare  : GT-crop baseline features + reproduce frozen B0_unsync
  features : YOLO class-10/11 candidate features (fit/val)
  diagnose : predicted-only candidate/confidence counts (fit; no GT)
  eval     : matched GT-vs-Pred comparison, full-candidate changes, AP

The production detector, boxes, scores and NMS are never changed. Only
class labels of adopted candidate predictions are changed offline.

No val tuning: rule_lock.json is written from fit predicted-only
diagnostics; val is evaluated once after it is locked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.signals.data import SPLIT_MANIFEST, load_manifest
from src.signals.metrics import (
    ImagePredictions, ImageTargets, _match_single, evaluate,
)
from scripts.srrc_frontend import crop_box, baseline_features

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs_local/baseline3c"
OUT = ROOT / "reports/baseline3_srrc_yolo_offline"
SPLIT_JSON = ROOT / "reports/baseline3_srrc_local_iq/srrc_local_iq_split.json"
CLASSES = (10, 11)
BW_LO, BW_HI = 9.5, 10.5
FEAT_NAME = "B0_unsync"
MODEL = CACHE / "srrc_b0_unsync.pkl"
MODEL_OK = CACHE / "srrc_b0_success.pkl"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_classifier(X, y):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced",
                           max_iter=5000, solver="lbfgs"))
    clf.fit(np.nan_to_num(X), np.asarray(y, int))
    return clf


def gt_worker(args):
    rec = SimpleNamespace(**args["rec"])
    try:
        r = crop_box(rec, args["sig"])
        if r["status"] != "ok":
            return {"sample_id": rec.sample_id, "signal_id": args["sig"]["signal_id"],
                    "class": args["sig"]["class"], "status": r["status"],
                    "reason": r.get("reason"), "baseline": None}
        b = baseline_features(r["x"], r["fs"])
        # sync status is needed only to reproduce the B0_success model
        from scripts.srrc_frontend import sync_srrc
        s = sync_srrc(r["x"], r["fs"])
        return {"sample_id": rec.sample_id, "signal_id": args["sig"]["signal_id"],
                "class": args["sig"]["class"], "status": s["status"],
                "reason": s.get("status"), "baseline": b}
    except Exception as e:
        return {"sample_id": rec.sample_id, "signal_id": args["sig"]["signal_id"],
                "class": args["sig"]["class"], "status": "error",
                "reason": str(e)[:200], "baseline": None}


def prepare(args):
    split = json.loads(SPLIT_JSON.read_text())
    assign = split["assignment"]
    recs = load_manifest(SPLIT_MANIFEST, split="fit")
    jobs = []
    meta = []
    for r in recs:
        for s in r.signals:
            bw = float(s["end_frequency"]) - float(s["start_frequency"])
            if int(s["class"]) in CLASSES and BW_LO <= bw <= BW_HI:
                jobs.append({"rec": {
                    "sample_id": r.sample_id, "bin_path": str(r.bin_path),
                    "B_mhz": r.B_mhz, "T_ms": r.T_ms, "N": r.N,
                    "f_lo": r.f_lo}, "sig": s})
                meta.append({"sample_id": r.sample_id,
                             "signal_id": int(s["signal_id"]),
                             "class": int(s["class"]),
                             "split": assign.get(r.sample_id)})
    print(f"GT instances: {len(jobs)}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, x in enumerate(ex.map(gt_worker, jobs, chunksize=8)):
            results.append(x)
            if (k + 1) % 300 == 0:
                print(f"  {k+1}/{len(jobs)}", flush=True)
    names = next(x["baseline"]["names"] for x in results if x["baseline"])
    X = np.full((len(results), len(names)), np.nan, np.float64)
    for i, x in enumerate(results):
        if x["baseline"] is not None:
            X[i] = x["baseline"]["features"]
    y = np.array([x["class"] for x in results])
    split_vec = np.array([m["split"] for m in meta])
    status = np.array([x["status"] for x in results])
    sample_ids = np.array([x["sample_id"] for x in results])
    signal_ids = np.array([x["signal_id"] for x in results])
    np.savez_compressed(
        CACHE / "srrc_gt_baseline_features.npz",
        X=X, y=y, split=split_vec, status=status,
        sample_ids=sample_ids, signal_ids=signal_ids,
        names=np.array(names))
    train = (split_vec == "train") & np.all(np.isfinite(X), axis=1)
    hold = (split_vec == "holdout") & np.all(np.isfinite(X), axis=1)
    clf = fit_classifier(X[train], y[train])
    joblib.dump(clf, MODEL)
    # reproduce the success-only model too
    ok_train = train & (status == "ok")
    clf_ok = fit_classifier(X[ok_train], y[ok_train])
    joblib.dump(clf_ok, MODEL_OK)
    hold_all_pred = clf.predict(X[hold])
    hold_ok_mask = hold & (status == "ok")
    hold_ok_pred = clf_ok.predict(X[hold_ok_mask])
    out = {
        "n": len(results), "n_train": int(train.sum()), "n_holdout": int(hold.sum()),
        "n_train_ok": int(ok_train.sum()), "n_holdout_ok": int(hold_ok_mask.sum()),
        "holdout_all_acc": float(np.mean(hold_all_pred == y[hold])),
        "holdout_ok_acc": float(np.mean(hold_ok_pred == y[hold_ok_mask])),
        "model_sha256": sha256_file(MODEL),
        "model_ok_sha256": sha256_file(MODEL_OK),
        "names": names,
    }
    (OUT / "reproduction.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


def expected_gt_boxes(meta: dict, sigs) -> np.ndarray:
    B = float(meta["B_mhz"]); N = int(meta["N"]); f_lo = float(meta["f_lo"])
    h = (N + 510) // 511
    out = []
    for s in sigs:
        x1 = ((float(s["start_time"]) / 1000.0) * (B * 1e6) / h + 0.5) / 512.0
        x2 = ((float(s["end_time"]) / 1000.0) * (B * 1e6) / h + 0.5) / 512.0
        y1 = (float(s["start_frequency"]) - f_lo) / B
        y2 = (float(s["end_frequency"]) - f_lo) / B
        out.append([min(1.0, max(0.0, x1)), min(1.0, max(0.0, y1)),
                    min(1.0, max(0.0, x2)), min(1.0, max(0.0, y2))])
    return np.asarray(out, float)



def prepare_val_features(args):
    recs = load_manifest(SPLIT_MANIFEST, split="val")
    jobs = []
    for r in recs:
        for s in r.signals:
            bw = float(s["end_frequency"]) - float(s["start_frequency"])
            if int(s["class"]) in CLASSES and BW_LO <= bw <= BW_HI:
                jobs.append({"rec": {
                    "sample_id": r.sample_id, "bin_path": str(r.bin_path),
                    "B_mhz": r.B_mhz, "T_ms": r.T_ms, "N": r.N,
                    "f_lo": r.f_lo}, "sig": s})
    print(f"val GT instances: {len(jobs)}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, x in enumerate(ex.map(gt_worker, jobs, chunksize=8)):
            results.append(x)
            if (k + 1) % 300 == 0:
                print(f"  {k+1}/{len(jobs)}", flush=True)
    names = next(x["baseline"]["names"] for x in results if x["baseline"])
    X = np.full((len(results), len(names)), np.nan, np.float64)
    for i, x in enumerate(results):
        if x["baseline"] is not None:
            X[i] = x["baseline"]["features"]
    np.savez_compressed(
        CACHE / "srrc_gt_baseline_features_val.npz",
        X=X, y=np.array([x["class"] for x in results]),
        status=np.array([x["status"] for x in results]),
        sample_ids=np.array([x["sample_id"] for x in results]),
        signal_ids=np.array([x["signal_id"] for x in results]),
        names=np.array(names))
    print(f"wrote val GT features: {sum(x['baseline'] is not None for x in results)}/{len(results)}")

def pred_worker(args):
    rec = SimpleNamespace(**args["rec"])
    try:
        r = crop_box(rec, args["sig"])
        if r["status"] != "ok":
            return {**args["key"], "status": r["status"],
                    "reason": r.get("reason"), "features": None}
        b = baseline_features(r["x"], r["fs"])
        return {**args["key"], "status": "ok", "reason": "ok",
                "features": b["features"]}
    except Exception as e:
        return {**args["key"], "status": "error", "reason": str(e)[:200],
                "features": None}


def pred_features(args, cache_path: Path, out_path: Path):
    z = np.load(cache_path, allow_pickle=True)
    z = {k: z[k] for k in z.files}
    metas = [json.loads(str(m)) for m in z["meta"]]
    off = np.concatenate([[0], np.cumsum(z["c0_counts"])])
    jobs = []
    meta = []
    for si, m in enumerate(metas):
        lo, hi = int(off[si]), int(off[si + 1])
        for pi in range(lo, hi):
            lab = int(z["c0_labels"][pi])
            if lab not in CLASSES:
                continue
            x1, y1, x2, y2 = z["c0_boxes"][pi]
            B = float(m["B_mhz"])
            width = (float(y2) - float(y1)) * B
            if not (BW_LO <= width <= BW_HI):
                continue
            # physical box, prediction geometry only
            h = (int(m["N"]) + 510) // 511
            fs = B * 1e6
            t1 = (float(x1) * 512 - 0.5) * h / fs * 1000.0
            t2 = (float(x2) * 512 - 0.5) * h / fs * 1000.0
            f1 = float(m["f_lo"]) + float(y1) * B
            f2 = float(m["f_lo"]) + float(y2) * B
            fsig = {"signal_id": -1, "class": lab,
                    "start_frequency": f1, "end_frequency": f2,
                    "start_time": max(0.0, min(float(m["T_ms"]), t1)),
                    "end_time": max(0.0, min(float(m["T_ms"]), t2))}
            if fsig["end_time"] - fsig["start_time"] < 0.1:
                continue
            stem = m["sample_id"].split("/")[1]
            jobs.append({"rec": {
                "sample_id": m["sample_id"],
                "bin_path": f"/root/autodl-fs/SpaceNet_Dataset/advanced/train/{stem}.bin",
                "B_mhz": B, "T_ms": float(m["T_ms"]), "N": int(m["N"]),
                "f_lo": float(m["f_lo"])}, "sig": fsig,
                "key": {"c0_idx": int(pi), "sample_id": m["sample_id"],
                        "score": float(z["c0_scores"][pi]),
                        "yolo_label": lab, "width_mhz": width}})
            meta.append(jobs[-1]["key"])
    print(f"{cache_path.name}: candidate-width predictions {len(jobs)}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, x in enumerate(ex.map(pred_worker, jobs, chunksize=8)):
            results.append(x)
            if (k + 1) % 500 == 0:
                print(f"  {k+1}/{len(jobs)}", flush=True)
    names = next(x["features"] for x in results if x["features"] is not None)
    n_names = len(names)
    X = np.full((len(results), n_names), np.nan, np.float64)
    for i, x in enumerate(results):
        if x["features"] is not None:
            X[i] = x["features"]
    np.savez_compressed(
        out_path, X=X,
        c0_idx=np.array([x["c0_idx"] for x in results]),
        sample_id=np.array([x["sample_id"] for x in results]),
        score=np.array([x["score"] for x in results], float),
        yolo_label=np.array([x["yolo_label"] for x in results], int),
        width_mhz=np.array([x["width_mhz"] for x in results], float),
        status=np.array([x["status"] for x in results]),
        reason=np.array([x["reason"] for x in results]),
        names=np.array(names))
    print(f"wrote {out_path}: feature-ok {sum(x['features'] is not None for x in results)}/{len(results)}")


def load_pred(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def diagnose(args):
    z = load_pred(CACHE / "srrc_pred_features_fit.npz")
    clf = joblib.load(MODEL)
    ok = np.all(np.isfinite(z["X"]), axis=1)
    X = z["X"][ok]
    p = clf.predict_proba(X)
    pred = clf.classes_[np.argmax(p, axis=1)]
    conf = p.max(axis=1)
    yolo = z["yolo_label"][ok]
    score = z["score"][ok]
    changes = pred != yolo
    out = {"n_feature_ok": int(ok.sum()),
           "score_quantiles": {str(q): float(np.quantile(score, q))
                               for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
           "by_score": {}, "by_conf": {}}
    for s in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1):
        m = score >= s
        out["by_score"][str(s)] = {
            "n": int(m.sum()), "n_flip": int((changes & m).sum()),
            "conf_q": {str(q): float(np.quantile(conf[m], q))
                       for q in (0.5, 0.75, 0.9, 0.95)} if m.any() else None}
    for c in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        m = changes & (conf >= c)
        out["by_conf"][str(c)] = {
            "n_flip": int(m.sum()),
            "n_flip_score01": int((m & (score >= 0.01)).sum()),
        }
    (OUT / "fit_candidate_diagnostics.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


def eval_stage(args, split: str):
    rule = json.loads((OUT / "rule_lock.json").read_text())
    score_thr = float(rule["score_threshold"])
    prob_thr = float(rule["prob_threshold"])
    clf = joblib.load(MODEL)
    gt_path = (CACHE / "srrc_gt_baseline_features_val.npz" if split == "val"
               else CACHE / "srrc_gt_baseline_features.npz")
    gt = np.load(gt_path, allow_pickle=True)
    gt = {k: gt[k] for k in gt.files}
    gt_map = {(str(s), int(i)): k for k, (s, i) in
              enumerate(zip(gt["sample_ids"], gt["signal_ids"]))}
    pz = load_pred(CACHE / f"srrc_pred_features_{split}.npz")
    pz_map = {int(pi): k for k, pi in enumerate(pz["c0_idx"])}
    z = np.load(CACHE / f"yolo_{split}_cache.npz", allow_pickle=True)
    z = {k: z[k] for k in z.files}
    metas = [json.loads(str(m)) for m in z["meta"]]
    off = np.concatenate([[0], np.cumsum(z["c0_counts"])])
    goff = np.concatenate([[0], np.cumsum(z["gt_counts"])])
    # JSON signal mapping for GT bandwidth/signal_id; validate count/class order
    sigs_by_sid = {}
    for r in load_manifest(SPLIT_MANIFEST, split=split):
        sigs_by_sid[r.sample_id] = r.signals
    rows = []
    for si, m in enumerate(metas):
        sid = m["sample_id"]
        sigs = sigs_by_sid[sid]
        g_lo, g_hi = int(goff[si]), int(goff[si + 1])
        gb = z["gt_boxes"][g_lo:g_hi]
        gl = z["gt_labels"][g_lo:g_hi].astype(int)
        if len(sigs) != len(gl) or not np.array_equal(
                np.array([int(s["class"]) for s in sigs]), gl):
            raise RuntimeError(f"{sid}: GT/JSON order mismatch")
        if not np.allclose(expected_gt_boxes(m, sigs), gb, atol=2e-4, rtol=0.0):
            raise RuntimeError(f"{sid}: GT/JSON geometry mismatch")
        srrc_gt = [j for j in range(len(gl)) if gl[j] in CLASSES]
        gt_boxes = gb[srrc_gt]
        gt_labels = gl[srrc_gt]
        lo, hi = int(off[si]), int(off[si + 1])
        pidx = [pi for pi in range(lo, hi) if int(z["c0_labels"][pi]) in CLASSES]
        if not pidx:
            continue
        pred = ImagePredictions(z["c0_boxes"][pidx], z["c0_scores"][pidx],
                                z["c0_labels"][pidx])
        target = ImageTargets(gt_boxes, gt_labels)
        matches, unmatched = _match_single(target, pred, 0.5,
                                           class_aware=False, max_dets=100)
        matched_pred = {m[0]: m[1] for m in matches}
        # best IoU to any GT (for unmatched category)
        ious = np.zeros((len(gt_boxes), len(pidx)))
        for a in range(len(gt_boxes)):
            ax1, ay1, ax2, ay2 = gt_boxes[a]
            for b in range(len(pidx)):
                bx1, by1, bx2, by2 = pred.boxes[b]
                iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
                ih = max(0.0, min(ay2, by2) - max(ay1, by1))
                inter = iw * ih
                aa = max((ax2 - ax1) * (ay2 - ay1), 1e-12)
                bb = max((bx2 - bx1) * (by2 - by1), 1e-12)
                ious[a, b] = inter / max(aa + bb - inter, 1e-12)
        for b, pi in enumerate(pidx):
            lab = int(z["c0_labels"][pi])
            x1, y1, x2, y2 = z["c0_boxes"][pi]
            B = float(m["B_mhz"])
            width = (float(y2) - float(y1)) * B
            score = float(z["c0_scores"][pi])
            gi = matched_pred.get(b)
            true_cls = int(gt_labels[gi]) if gi is not None else None
            gt_sig_idx = srrc_gt[gi] if gi is not None else None
            best_j = int(np.argmax(ious[:, b])) if len(gt_boxes) else None
            best_iou = float(ious[best_j, b]) if best_j is not None else 0.0
            # candidate features/prob
            k = pz_map.get(pi)
            feats_ok = k is not None and np.all(np.isfinite(pz["X"][k]))
            pred_cls = None; conf = None; changed = False
            if feats_ok:
                pp = clf.predict_proba(pz["X"][k][None, :])[0]
                pred_cls = int(clf.classes_[int(np.argmax(pp))])
                conf = float(np.max(pp))
                changed = (pred_cls != lab and conf >= prob_thr and
                           score >= score_thr and BW_LO <= width <= BW_HI)
            # GT feature prediction
            gt_pred = None
            if gi is not None:
                gsid = str(sid); gsig = int(sigs[gt_sig_idx]["signal_id"])
                gk = gt_map.get((gsid, gsig))
                if gk is not None and np.all(np.isfinite(gt["X"][gk])):
                    gp = clf.predict(gt["X"][gk][None, :])[0]
                    gt_pred = int(gp)
            rows.append({
                "sample_id": sid, "c0_idx": int(pi), "yolo_label": lab,
                "score": score, "width_mhz": width, "true_class": true_cls,
                "matched": gi is not None, "best_iou": best_iou,
                "best_gt_label": int(gt_labels[best_j]) if best_j is not None else None,
                "pred_class": pred_cls, "conf": conf, "changed": changed,
                "gt_pred_class": gt_pred,
            })
    rule_mask = [r for r in rows if r["score"] >= score_thr and
                 BW_LO <= r["width_mhz"] <= BW_HI]
    cand = [r for r in rule_mask if r["pred_class"] is not None]
    matched = [r for r in cand if r["matched"]]
    unmatched = [r for r in cand if not r["matched"]]
    def acc(rs, key):
        if not rs:
            return None
        return float(np.mean([int(r[key]) == int(r["true_class"]) for r in rs]))
    changed = [r for r in cand if r["changed"]]
    res = {
        "split": split, "n_predictions_srrc": len(rows),
        "n_candidates_pre_feature": len(rule_mask),
        "n_feature_failed": len(rule_mask) - len(cand),
        "n_candidates": len(cand), "n_matched": len(matched),
        "n_unmatched": len(unmatched),
        "matched_yolo_acc": acc(matched, "yolo_label"),
        "matched_pred_acc": acc(matched, "pred_class"),
        "matched_gt_acc": acc([r for r in matched if r["gt_pred_class"] is not None],
                              "gt_pred_class"),
        "n_changed": len(changed),
        "n_corrected": sum(1 for r in matched if r["changed"] and
                           int(r["pred_class"]) == int(r["true_class"])),
        "n_worsened": sum(1 for r in matched if r["changed"] and
                          int(r["pred_class"]) != int(r["true_class"])),
        "n_unmatched_changed": sum(1 for r in unmatched if r["changed"]),
        "unmatched_changed_categories": dict(Counter(
            ("duplicate_srrc_gt" if r["best_gt_label"] in CLASSES and r["best_iou"] >= 0.5
             else "non_srrc_gt" if r["best_gt_label"] is not None and r["best_iou"] >= 0.5
             else "low_iou_srrc" if r["best_gt_label"] in CLASSES
             else "no_gt") for r in unmatched if r["changed"])) if unmatched else {},
        "changed_by_class": dict(Counter(r["yolo_label"] for r in changed)),
        "candidate_pred_class": dict(Counter(r["pred_class"] for r in cand)),
        "conf_quantiles": {str(q): float(np.quantile([r["conf"] for r in cand], q))
                           for q in (0.5, 0.75, 0.9, 0.95, 0.99)} if cand else None,
    }
    # GT vs Pred on matched candidate subset
    both = [r for r in matched if r["gt_pred_class"] is not None and r["pred_class"] is not None]
    if both:
        res["same_target"] = {
            "n": len(both),
            "gt_acc": float(np.mean([int(r["gt_pred_class"]) == int(r["true_class"]) for r in both])),
            "pred_acc": float(np.mean([int(r["pred_class"]) == int(r["true_class"]) for r in both])),
            "agreement": float(np.mean([int(r["gt_pred_class"]) == int(r["pred_class"]) for r in both])),
            "gt_conf": np.array([[sum(1 for r in both if r["true_class"] == a and r["gt_pred_class"] == b) for b in CLASSES] for a in CLASSES]).tolist(),
            "pred_conf": np.array([[sum(1 for r in both if r["true_class"] == a and r["pred_class"] == b) for b in CLASSES] for a in CLASSES]).tolist(),
        }
    # AP for val: apply changes to cache labels
    if split == "val":
        labels0 = z["c0_labels"].copy()
        targets = []
        for si in range(len(metas)):
            glo, ghi = int(goff[si]), int(goff[si + 1])
            targets.append(ImageTargets(z["gt_boxes"][glo:ghi],
                                        z["gt_labels"][glo:ghi]))

        def eval_labels(lab):
            preds = []
            for si in range(len(metas)):
                lo, hi = int(off[si]), int(off[si + 1])
                preds.append(ImagePredictions(z["c0_boxes"][lo:hi],
                                              z["c0_scores"][lo:hi],
                                              lab[lo:hi]))
            return evaluate(targets, preds, iou_thresholds=(0.5,), max_dets=100)

        changed = [r for r in cand if r["changed"]]
        labels = labels0.copy()
        for r in changed:
            labels[r["c0_idx"]] = int(r["pred_class"])
        labels_m = labels0.copy()
        for r in changed:
            if r["matched"]:
                labels_m[r["c0_idx"]] = int(r["pred_class"])
        labels_u = labels0.copy()
        for r in changed:
            if not r["matched"]:
                labels_u[r["c0_idx"]] = int(r["pred_class"])
        evb = eval_labels(labels0)
        ev0 = eval_labels(labels)
        evm = eval_labels(labels_m)
        evu = eval_labels(labels_u)
        pcb = evb["per_iou"]["0.5"]["per_class"]
        pc0 = ev0["per_iou"]["0.5"]["per_class"]
        pcm = evm["per_iou"]["0.5"]["per_class"]
        pcu = evu["per_iou"]["0.5"]["per_class"]
        res["ap"] = {
            "C0_map50": evb["per_iou"]["0.5"]["map_class_aware"],
            "new_map50": ev0["per_iou"]["0.5"]["map_class_aware"],
            "delta_pp": 100 * (ev0["per_iou"]["0.5"]["map_class_aware"] -
                               evb["per_iou"]["0.5"]["map_class_aware"]),
            "C0_class10_ap50": pcb.get("10", {}).get("ap"),
            "C0_class11_ap50": pcb.get("11", {}).get("ap"),
            "new_class10_ap50": pc0.get("10", {}).get("ap"),
            "new_class11_ap50": pc0.get("11", {}).get("ap"),
            "n_changed": len(changed),
            "oracle_matched_only_map50": evm["per_iou"]["0.5"]["map_class_aware"],
            "oracle_matched_only_delta_pp": 100 * (
                evm["per_iou"]["0.5"]["map_class_aware"] -
                evb["per_iou"]["0.5"]["map_class_aware"]),
            "oracle_unmatched_only_map50": evu["per_iou"]["0.5"]["map_class_aware"],
            "oracle_unmatched_only_delta_pp": 100 * (
                evu["per_iou"]["0.5"]["map_class_aware"] -
                evb["per_iou"]["0.5"]["map_class_aware"]),
            "oracle_note": ("matched/unmatched split is an oracle diagnostic "
                            "using GT matching; not deployable."),
        }
    csv_path = OUT / f"{split}_changed_candidates.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["sample_id", "c0_idx", "yolo_label",
                           "pred_class", "conf", "score", "width_mhz",
                           "matched", "true_class", "best_iou",
                           "best_gt_label", "corrected", "worsened"],
            lineterminator="\n")
        w.writeheader()
        for r in changed:
            w.writerow({
                "sample_id": r["sample_id"], "c0_idx": r["c0_idx"],
                "yolo_label": r["yolo_label"], "pred_class": r["pred_class"],
                "conf": r["conf"], "score": r["score"],
                "width_mhz": r["width_mhz"], "matched": r["matched"],
                "true_class": r["true_class"], "best_iou": r["best_iou"],
                "best_gt_label": r["best_gt_label"],
                "corrected": bool(r["matched"] and
                                  int(r["pred_class"]) == int(r["true_class"])),
                "worsened": bool(r["matched"] and
                                 int(r["pred_class"]) != int(r["true_class"])),
            })
    (OUT / f"fit_{split}_eval.json" if split == "fit" else OUT / "val_eval.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("prepare", "prepare_val", "features", "diagnose", "fit", "val"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "prepare_val":
        prepare_val_features(args)
    elif args.stage == "features":
        # fit first (rule lock); val only after rule_lock exists
        pred_features(args, CACHE / "yolo_fit_cache.npz",
                      CACHE / "srrc_pred_features_fit.npz")
        if (OUT / "rule_lock.json").exists():
            pred_features(args, CACHE / "yolo_val_cache.npz",
                          CACHE / "srrc_pred_features_val.npz")
    elif args.stage == "diagnose":
        diagnose(args)
    elif args.stage in ("fit", "val"):
        eval_stage(args, args.stage)


if __name__ == "__main__":
    main()
