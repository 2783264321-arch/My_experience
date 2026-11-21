#!/usr/bin/env python3
"""Phase C'-Adaptation E3: GT-box jitter augmentation cache.

Source: fit-split WiFi GT instances that were sync-ok in Phase B' (the
symbol cache), i.e. the original training distribution. Jitter deltas are
sampled from the empirical pred-vs-GT geometry error distribution of the
adaptation-train dual-sync pairs ONLY (calibration-dev/final-dev geometry
never used). The jittered boxes go through the identical extraction +
sync + histogram pipeline.

Output: runs_local/baseline3ca/jitter_hist_cache.npz (blocks, labels,
status; local only).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from concurrent.futures import ProcessPoolExecutor

from scripts.baseline3ca_build_hist_cache import sync_box


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports/baseline3_wifi_cascade"
CACHE = ROOT / "runs_local/baseline3c"
OUT = ROOT / "runs_local/baseline3ca"
SHARED_MAP = {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2}
NOMINAL_BW = {0: 20, 1: 20, 2: 20, 3: 40, 4: 40, 5: 40}
MAX_INSTANCES = 900
MAX_BLOCKS = 4


def main():
    frames = np.load(ROOT / "reports/local_iq_round1/frames.npz")
    lab = frames["label"].astype(int)
    sid = frames["sample_id"].astype(str)
    b_cache = json.loads((ROOT / "reports/baseline3_wifi_expert/"
                          "symbol_cache_summary.json").read_text())
    ok_fit = {r["instance_id"] for r in b_cache["results"]
              if r.get("status") == "ok"}
    # map B' manifest instance ids back to frames.npz rows via
    # (sample_id, signal_id) -- the two index spaces are not the same
    man = json.loads((ROOT / "reports/baseline3_wifi_expert/"
                      "dataset_manifest.json").read_text())
    key2row = {(str(sid[i]), int(frames["signal_id"][i])): i
               for i in range(len(sid))}
    ok_rows = []
    for m in man["instances"]:
        if m["instance_id"] in ok_fit:
            r = key2row.get((m["sample_id"], m["signal_id"]))
            if r is not None:
                ok_rows.append(int(r))
    wifi = np.array(sorted(ok_rows), dtype=np.int64)
    print(f"fit sync-ok WiFi frames (mapped): {len(wifi)}")

    # empirical geometry error distribution from adaptation-train dual-sync
    pairs = json.loads((EXP / "adaptation_pairs.json").read_text())
    split = json.loads((EXP / "adaptation_split_v2.json").read_text())
    assign = split["assignment"]
    paired = json.loads((EXP / "paired_gt_pred_sync.json").read_text())
    dual_ok = {r["sample_id"] for r in paired["pairs"]
               if r["gt_sync"] == "ok" and r["pred_sync"] == "ok"}
    deltas = []
    for p, q in zip(pairs, paired["pairs"]):
        psid = p["sample_id"]
        if psid not in assign or assign[psid] != "adaptation-train":
            continue
        # include ALL adaptation-train matched pairs (not only dual-sync):
        # the pred-failure geometry errors are part of the training-domain
        # error distribution the jitter must represent
        gt_w = q.get("gt_width_mhz")
        if not gt_w or gt_w <= 0:
            continue
        gt_dur = max(p["gt_t2"] - p["gt_t1"], 1e-6)
        deltas.append({
            "rel_f_center": q["f_center_delta"] / gt_w,
            "rel_t_center": p["t_center_delta"] / gt_dur,
            "rel_width": (q["nominal_bw"] - gt_w) / gt_w,
        })
    print(f"jitter delta samples: {len(deltas)}")

    rng = np.random.default_rng(20261121)
    # deterministic selection of up to MAX_INSTANCES fit sync-ok WiFi frames
    cand = [int(i) for i in wifi]
    cand = sorted(cand, key=lambda i: (sid[i], int(frames["signal_id"][i])))
    cand = cand[:MAX_INSTANCES]
    print(f"jitter source instances: {len(cand)}")

    jobs = []
    meta = []
    for i in cand:
        d = deltas[int(rng.integers(0, len(deltas)))]
        B = float(frames["B_mhz"][i])
        f_lo = float(frames["f_lo_mhz"][i])
        N = int(frames["N"][i])
        f1, f2 = float(frames["f1_mhz"][i]), float(frames["f2_mhz"][i])
        t1, t2 = float(frames["t1_ms"][i]), float(frames["t2_ms"][i])
        w = max(f2 - f1, 1e-3)
        dur = max(t2 - t1, 1e-3)
        fc = (f1 + f2) / 2 + d["rel_f_center"] * w
        tc = (t1 + t2) / 2 + d["rel_t_center"] * dur
        w2 = max(w * (1.0 + d["rel_width"]), 1e-3)
        f1n = max(f_lo, fc - w2 / 2)
        f2n = min(f_lo + B, fc + w2 / 2)
        t1n = max(0.0, tc - dur / 2)
        t2n = min(float(frames["T_ms"][i]), tc + dur / 2)
        if f2n - f1n < 1.0 or t2n - t1n < 1.0:
            continue
        bw_nom = NOMINAL_BW[int(lab[i])]
        jobs.append((f"/root/autodl-fs/SpaceNet_Dataset/advanced/"
                     f"{sid[i]}.bin", f_lo, B, N,
                     (f1n + f2n) / 2, t1n, t2n, bw_nom))
        meta.append({"instance_id": int(i), "sample_id": str(sid[i]),
                     "gt_label": int(lab[i]),
                     "y": SHARED_MAP[int(lab[i])], "job": jobs[-1]})

    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for r in ex.map(sync_box, [m["job"] for m in meta], chunksize=16):
            results.append(r)
    dt = time.perf_counter() - t0
    blocks, labels, status = [], [], []
    offs = [0]
    for m, r in zip(meta, results):
        status.append(r["status"])
        if r["status"] == "ok":
            blocks.append(r["blocks"][:MAX_BLOCKS])
            labels.append(m["y"])
            offs.append(offs[-1] + len(blocks[-1]))
        else:
            labels.append(-1)
    flat = (np.concatenate(blocks) if blocks else
            np.zeros((0, 1, 50, 50), np.float32))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "jitter_hist_cache.npz",
        blocks=flat.astype(np.float32),
        offsets=np.array(offs, np.int64),
        labels=np.array(labels, np.int64),
        status=np.array(status),
        meta=np.array([json.dumps({k: v for k, v in m.items()
                                   if k != "job"}) for m in meta]),
    )
    from collections import Counter
    print(f"jitter sync done {dt:.0f}s, status={dict(Counter(status))}, "
          f"blocks={int(offs[-1])}")
    print(f"wrote {OUT/'jitter_hist_cache.npz'}")


if __name__ == "__main__":
    main()
