#!/usr/bin/env python3
"""Phase C'-Adaptation audit v2: fixes for overlap, routing sets, split.

Fixes over v1 (per review of e17289b):
  1. overlap flags computed from VAL JSON annotations directly (v1 looked
     them up in the fit-split manifest and silently defaulted to clean);
  2. the file-level split now actually uses a seeded permutation and a
     documented stratification rule (+ class coverage promotion), and
     records algorithm/seed/assignment SHA;
  3. physical matching, actual expert routing and same-family routing are
     reported as three separate sets; non-WiFi predictions are skipped
     explicitly (the old NOMINAL_BW.get(pred_label, 20) silent default is
     removed).

No fine-tuning in this round. Reads adaptation_pairs.json (v1 ledger,
predicted-box sync results) and outputs adaptation_audit_v2.json +
adaptation_split_v2.json.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.signals.data import SPLIT_MANIFEST, load_manifest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports/baseline3_wifi_cascade"
WIFI = [0, 1, 2, 3, 4, 5]
W20 = [0, 1, 2]
W40 = [3, 4, 5]
SPLIT_SEED = 20261121


def overlap_fields(rec, gi):
    """Same semantics as the fit-split overlap_flags: raw GT boxes."""
    s = rec.signals[gi]
    area = max(0.0, float(s["end_time"]) - float(s["start_time"])) * \
        max(0.0, float(s["end_frequency"]) - float(s["start_frequency"]))
    any_ov = wifi_ov = 0
    wifi_cross = wifi_same = 0
    any_frac = wifi_frac = 0.0
    for j, o in enumerate(rec.signals):
        if j == gi:
            continue
        inter_t = max(0.0, min(float(s["end_time"]), float(o["end_time"])) -
                      max(float(s["start_time"]), float(o["start_time"])))
        inter_f = max(0.0, min(float(s["end_frequency"]), float(o["end_frequency"])) -
                      max(float(s["start_frequency"]), float(o["start_frequency"])))
        inter = inter_t * inter_f
        if inter <= 0.0 or area <= 0.0:
            continue
        frac = inter / area
        any_ov = 1
        any_frac = max(any_frac, frac)
        oc = int(o["class"])
        if oc in WIFI:
            wifi_ov = 1
            wifi_frac = max(wifi_frac, frac)
            if oc == int(s["class"]):
                wifi_same = 1
            else:
                wifi_cross = 1
    return {"overlap_any": any_ov, "overlap_wifi": wifi_ov,
            "overlap_wifi_cross": wifi_cross,
            "overlap_wifi_same": wifi_same,
            "overlap_any_frac_max": round(any_frac, 4),
            "overlap_wifi_frac_max": round(wifi_frac, 4),
            "strong_overlap": int(any_frac >= 0.25)}


def main():
    pairs = json.loads((EXP / "adaptation_pairs.json").read_text())
    recs = {r.sample_id: r for r in
            load_manifest(SPLIT_MANIFEST, split="val")}
    print(f"pairs: {len(pairs)}, val samples: {len(recs)}")

    # ---- 1. overlap flags from val JSONs + signal-id resolution ----
    unmatched = 0
    for p in pairs:
        rec = recs[p["sample_id"]]
        best, best_d = None, 1e9
        for gi, s in enumerate(rec.signals):
            if int(s["class"]) != p["gt_label"]:
                continue
            d = (abs(float(s["start_frequency"]) - (p["gt_f_center"] -
                     (p["gt_f_center"] - p.get("gt_f_center", p["gt_f_center"]))))
                 if False else
                 abs(float(s["start_frequency"]) + float(s["end_frequency"]) -
                     2 * p["gt_f_center"]) +
                 abs(float(s["start_time"]) - p["gt_t1"]) +
                 abs(float(s["end_time"]) - p["gt_t2"]))
            if d < best_d:
                best_d, best = d, gi
        if best is None or best_d > 1.0:
            unmatched += 1
            p.update({"overlap_any": None, "overlap_wifi": None,
                      "overlap_wifi_cross": None, "overlap_wifi_same": None,
                      "overlap_any_frac_max": None,
                      "overlap_wifi_frac_max": None,
                      "strong_overlap": None})
            continue
        p["gt_signal_index"] = int(best)
        p.update(overlap_fields(rec, best))
    print(f"overlap resolved: {len(pairs)-unmatched}, unmatched: {unmatched}")

    # ---- 2. three routing sets ----
    def fam(classes):
        cs = set(int(c) for c in classes)
        if cs <= set(W20):
            return "W20"
        if cs <= set(W40):
            return "W40"
        return "mixed"

    sets = {
        "matched_wifi_gt": pairs,
        "routed_wifi_candidate": [p for p in pairs
                                  if 0 <= p["pred_label"] <= 5],
        "correct_family_candidate": [
            p for p in pairs
            if 0 <= p["pred_label"] <= 5 and
            fam([p["pred_label"]]) == fam([p["gt_label"]])],
    }
    set_stats = {}
    for name, plist in sets.items():
        stat = {"n": len(plist)}
        stat["n_sync_ok"] = sum(1 for p in plist
                                if p["sync_status"] == "ok")
        stat["sync_rate"] = round(stat["n_sync_ok"] / max(stat["n"], 1), 4)
        stat["class_counts"] = dict(sorted(Counter(
            p["gt_label"] for p in plist).items()))
        stat["pred_label_counts"] = dict(sorted(Counter(
            p["pred_label"] for p in plist).items()))
        stat["family_mismatch"] = sum(
            1 for p in plist
            if fam([p["pred_label"]]) != fam([p["gt_label"]]))
        set_stats[name] = stat
    print(json.dumps(set_stats, indent=1))

    # cross-family / non-WiFi pairs detail
    odd = [p for p in pairs if not (0 <= p["pred_label"] <= 5) or
           fam([p["pred_label"]]) != fam([p["gt_label"]])]
    odd_detail = [{"sample_id": p["sample_id"],
                   "gt_label": p["gt_label"], "pred_label": p["pred_label"],
                   "iou": p["iou"], "sync_status": p["sync_status"]}
                  for p in odd]

    # ---- 3. split v2: stratified + coverage promotion ----
    # per-sample summary from the CORRECT-FAMILY routed set (training
    # samples must satisfy the inference routing conditions)
    routed = sets["correct_family_candidate"]
    by_sample = defaultdict(list)
    for p in routed:
        by_sample[p["sample_id"]].append(p)
    samples = sorted(by_sample)
    print(f"samples with correct-family matched pairs: {len(samples)}")

    def sample_key(sid):
        return hashlib.sha256(f"{SPLIT_SEED}:{sid}".encode()).hexdigest()

    strata = defaultdict(list)
    for sid in samples:
        plist = by_sample[sid]
        has_sync = any(p["sync_status"] == "ok" for p in plist)
        f = fam([p["gt_label"] for p in plist])
        strata[(has_sync, f)].append(sid)
    for k in strata:
        strata[k].sort(key=sample_key)  # deterministic within stratum

    def allocate(n, fracs=(0.70, 0.15, 0.15)):
        raw = [n * f for f in fracs]
        base = [int(np.floor(x)) for x in raw]
        rem = n - sum(base)
        order = np.argsort([-(raw[i] - base[i]) for i in range(3)])
        for i in order[:rem]:
            base[i] += 1
        return base

    assignment = {}
    for k, sids in strata.items():
        n_tr, n_ca, n_fi = allocate(len(sids))
        for i, sid in enumerate(sids):
            assignment[sid] = ("adaptation-train" if i < n_tr else
                               "calibration-dev" if i < n_tr + n_ca else
                               "final-dev")

    def split_counts(assign):
        out = {}
        for name in ("adaptation-train", "calibration-dev", "final-dev"):
            sids = [s for s, v in assign.items() if v == name]
            plist = [p for s in sids for p in by_sample.get(s, [])]
            ok = [p for p in plist if p["sync_status"] == "ok"]
            out[name] = {
                "n_samples": len(sids),
                "n_pairs": len(plist),
                "n_sync_ok_pairs": len(ok),
                # v1 cap-bug note: the v1 sync loop could over-produce
                # blocks per pair; Phase B' convention caps at 4 per pair
                "n_hist_blocks_est": int(sum(
                    min(p.get("n_blocks") or 0, 4) for p in ok)),
                "class_sync_ok": dict(sorted(Counter(
                    p["gt_label"] for p in ok).items())),
                "class_pairs": dict(sorted(Counter(
                    p["gt_label"] for p in plist).items())),
            }
        return out

    counts_before = split_counts(assignment)

    # coverage promotion: every class with >=1 sync-ok sample overall must
    # have >=1 sync-ok sample in calibration-dev AND final-dev
    promotions = []
    for cls in WIFI:
        cls_sids = [s for s in samples
                    if any(p["gt_label"] == cls and
                           p["sync_status"] == "ok"
                           for p in by_sample[s])]
        if not cls_sids:
            continue
        cls_sids.sort(key=sample_key)
        for dev in ("calibration-dev", "final-dev"):
            has = any(assignment[s] == dev and
                      any(p["gt_label"] == cls and
                          p["sync_status"] == "ok"
                          for p in by_sample[s])
                      for s in cls_sids)
            if has:
                continue
            # promote the first train sample of this class (deterministic)
            cand = [s for s in cls_sids
                    if assignment[s] == "adaptation-train"]
            if not cand:
                promotions.append({"class": cls, "dev": dev,
                                   "result": "impossible_no_train_sample"})
                continue
            src = cand[0]
            # swap with a final/cal sample from the SAME stratum, not
            # containing this class's sync instances (balance preserving)
            has_sync, f = None, None
            for kk in strata:
                if src in strata[kk]:
                    has_sync, f = kk
                    break
            dst_pool = [s for s in strata[(has_sync, f)]
                        if assignment[s] == dev
                        and not any(p["gt_label"] == cls and
                                    p["sync_status"] == "ok"
                                    for p in by_sample.get(s, []))]
            if not dst_pool:
                promotions.append({"class": cls, "dev": dev,
                                   "result": "no_swap_candidate",
                                   "promoted": src})
                assignment[src] = dev
                continue
            dst = dst_pool[0]
            assignment[src] = dev
            assignment[dst] = "adaptation-train"
            promotions.append({"class": cls, "dev": dev,
                               "result": "swapped",
                               "promoted_from_train": src,
                               "moved_to_train": dst})

    counts_after = split_counts(assignment)

    # histogram block estimates per class for training pools
    def blocks_by_class(sids):
        out = Counter()
        for s in sids:
            for p in by_sample.get(s, []):
                if p["sync_status"] == "ok":
                    out[p["gt_label"]] += min(
                        int(p.get("n_blocks") or 0), 4)
        return dict(sorted(out.items()))

    train_sids = [s for s, v in assignment.items()
                  if v == "adaptation-train"]
    cal_sids = [s for s, v in assignment.items()
                if v == "calibration-dev"]
    final_sids = [s for s, v in assignment.items()
                  if v == "final-dev"]

    # GT-crop replay volume: same samples, GT-box sync results are computed
    # by the paired-sync script; here report sample/pair counts only.
    split_doc = {
        "algorithm": "group split by IQ sample_id, stratified by "
                     "(has_sync_ok, family-mix of correct-family pairs); "
                     "within-stratum order = sha256(seed:sample_id); "
                     "70/15/15 largest-remainder allocation; then a "
                     "deterministic per-class coverage promotion so every "
                     "class with any sync-ok sample has >=1 such sample in "
                     "calibration-dev and final-dev (swap with a "
                     "same-stratum sample lacking that class).",
        "seed": SPLIT_SEED,
        "assignment_sha256": hashlib.sha256(
            json.dumps(sorted(assignment.items())).encode()).hexdigest(),
        "strata": {f"{k[0]}_{k[1]}": len(v) for k, v in strata.items()},
        "counts_before_promotion": counts_before,
        "counts_after_promotion": counts_after,
        "promotions": promotions,
        "train_blocks_by_class": blocks_by_class(train_sids),
        "cal_blocks_by_class": blocks_by_class(cal_sids),
        "final_blocks_by_class": blocks_by_class(final_sids),
        "n_independent_iq_train": len(train_sids),
        "n_independent_iq_cal": len(cal_sids),
        "n_independent_iq_final": len(final_sids),
        "assignment": {s: v for s, v in sorted(assignment.items())},
    }
    (EXP / "adaptation_split_v2.json").write_text(
        json.dumps(split_doc, indent=2), encoding="utf-8")

    # ---- 4. audit v2 doc ----
    audit = {
        "n_val_samples": len(recs),
        "n_pairs_matched_wifi_gt": len(pairs),
        "overlap_unmatched": unmatched,
        "routing_sets": set_stats,
        "odd_routing_pairs": odd_detail,
        "overlap_stats": {
            k: dict(sorted(Counter(
                p[k] for p in pairs if p.get(k) is not None).items()))
            for k in ("overlap_any", "overlap_wifi", "overlap_wifi_cross",
                      "overlap_wifi_same", "strong_overlap")},
        "sync_by_overlap": {},
        "sync_by_class_family": {},
        "split_v2": {k: v for k, v in split_doc.items()
                     if k != "assignment"},
    }
    for ov in (0, 1):
        sel = [p for p in pairs if p.get("overlap_any") == ov]
        audit["sync_by_overlap"][f"overlap_any={ov}"] = {
            "n": len(sel),
            "sync_ok": sum(1 for p in sel if p["sync_status"] == "ok"),
            "rate": round(sum(1 for p in sel
                              if p["sync_status"] == "ok") /
                          max(len(sel), 1), 4)}
    for f in ("W20", "W40"):
        for c in (W20 if f == "W20" else W40):
            sel = [p for p in pairs if p["gt_label"] == c]
            audit["sync_by_class_family"][f"{f}/c{c}"] = {
                "n": len(sel),
                "sync_ok": sum(1 for p in sel if p["sync_status"] == "ok"),
                "rate": round(sum(1 for p in sel
                                  if p["sync_status"] == "ok") /
                              max(len(sel), 1), 4)}
    (EXP / "adaptation_audit_v2.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")

    # ---- manual verification print (2 samples) ----
    print("\n=== manual overlap verification ===")
    shown = 0
    for p in pairs:
        if shown >= 2 or p.get("overlap_any") is None:
            continue
        rec = recs[p["sample_id"]]
        s = rec.signals[p["gt_signal_index"]]
        print(f"{p['sample_id']} GT cls{p['gt_label']} "
              f"f[{s['start_frequency']:.1f},{s['end_frequency']:.1f}] "
              f"t[{s['start_time']:.1f},{s['end_time']:.1f}] -> "
              f"any={p['overlap_any']} wifi={p['overlap_wifi']} "
              f"cross={p['overlap_wifi_cross']} strong={p['strong_overlap']} "
              f"frac={p['overlap_any_frac_max']}")
        shown += 1

    print("\n=== split v2 ===")
    for name, c in counts_after.items():
        print(f"{name}: samples={c['n_samples']} pairs={c['n_pairs']} "
              f"sync_ok={c['n_sync_ok_pairs']} "
              f"blocks={c['n_hist_blocks_est']} "
              f"class_sync_ok={c['class_sync_ok']}")
    print(f"\nassignment sha256: {split_doc['assignment_sha256']}")
    print(f"wrote adaptation_audit_v2.json / adaptation_split_v2.json")


if __name__ == "__main__":
    main()
