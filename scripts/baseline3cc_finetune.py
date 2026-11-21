#!/usr/bin/env python3
"""Phase C'-Adaptation E1/E2: controlled W-Shared fine-tuning.

E1: Pred-only fine-tuning on the 169 dual-sync pairs (pred histograms).
E2: GT+Pred mixed fine-tuning, instance-balanced 1:1 (same 169 pairs).

Protocol (pre-registered, identical across arms and seeds):
- init from the matching-shared GT-only weights SHARED_seed{s}.pt
- freeze the first two conv blocks; train conv3 + fc1 + fc2
- Adam lr 1e-4, weight decay 1e-4
- fixed update budget: 600 optimizer steps (adam steps), batch of 8
  instances per step; each instance contributes all its blocks (<=4 per
  side); loss = inverse-file-frequency-weighted mean over instances of
  the per-instance CE (E2 balances GT and Pred sub-losses 1:1)
- early stopping on calibration-dev Pred-block loss, evaluated every 50
  steps, patience 4 evaluations; best checkpoint by that loss
- final-dev never used for selection
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.baseline3b_train_experts import HistCNN

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs_local/baseline3c/hist_cache.npz"
WEIGHTS_IN = ROOT / "runs_local/baseline3b/weights"
WEIGHTS_OUT = ROOT / "runs_local/baseline3ca/weights"
EXP = ROOT / "reports/baseline3_wifi_cascade"
SHARED_MAP = {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2}
SEEDS = (1729, 1730, 1731)
MAX_UPDATES = 600
EVAL_EVERY = 50
PATIENCE = 4
BATCH_INST = 8
LR = 1e-4
WD = 1e-4


class FineTuneNet(HistCNN):
    """HistCNN with the first two conv blocks frozen."""

    def __init__(self):
        super().__init__(3)
        for idx in range(0, 8):  # conv1..., conv2..., their BN/ReLU/pools
            for p in self.features[idx].parameters():
                p.requires_grad = False


def load_pairs():
    z = np.load(CACHE, allow_pickle=True)
    pairs = json.loads((EXP / "adaptation_pairs.json").read_text())
    split = json.loads((EXP / "adaptation_split_v2.json").read_text())
    assign = split["assignment"]
    pred_pos = {int(pi): k for k, pi in enumerate(z["pred_idx"])}
    gt_off = z["gt_offsets"]; pred_off = z["pred_offsets"]
    rows = []
    for i, p in enumerate(pairs):
        sid = p["sample_id"]
        if sid not in assign:
            continue
        pp = pred_pos.get(int(p["pred_idx"]))
        if pp is None:
            continue
        g_off = int(gt_off[i]); g_n = int(z["gt_n_blocks"][i])
        p_off = int(pred_off[pp]); p_n = int(z["pred_n_blocks"][pp])
        if g_n == 0 or p_n == 0:
            continue
        if not (0 <= p["pred_label"] <= 5):
            continue
        fam_pred = "W20" if p["pred_label"] <= 2 else "W40"
        fam_gt = "W20" if p["gt_label"] <= 2 else "W40"
        if fam_pred != fam_gt:
            continue
        rows.append({
            "sample_id": sid, "split": assign[sid],
            "gt_label": p["gt_label"], "y": SHARED_MAP[p["gt_label"]],
            "gt_blocks": z["gt_blocks"][g_off:g_off + g_n],
            "pred_blocks": z["pred_blocks"][p_off:p_off + p_n],
        })
    return rows


def per_instance_loss(model, blocks, y, device):
    x = torch.from_numpy(blocks).to(device)
    logits = model(x)
    yy = torch.full((len(blocks),), int(y), dtype=torch.long,
                    device=device)
    return F.cross_entropy(logits, yy, reduction="mean")


def evaluate_cal(model, cal_rows, device, mode="pred"):
    model.eval()
    tot = 0.0
    n = 0
    with torch.no_grad():
        for r in cal_rows:
            blocks = r["pred_blocks"] if mode == "pred" else r["gt_blocks"]
            if len(blocks) == 0:
                continue
            tot += float(per_instance_loss(model, blocks, r["y"], device))
            n += 1
    return tot / max(n, 1)


def sample_batch(rng, pool, n_inst):
    idx = rng.choice(len(pool), size=n_inst, replace=True)
    return [pool[i] for i in idx]


def file_weights(pool):
    cnt = Counter(r["sample_id"] for r in pool)
    return {r["sample_id"]: 1.0 / cnt[r["sample_id"]] for r in pool}


def load_e3_rows(target_counts):
    """Class-stratified deterministic subset of the jitter cache.

    target_counts is the per-class instance count of the paired E1/E2
    training set, so E3 sees the same class mix (and the same number of
    instances); blocks are capped at 4 per instance exactly as in E1/E2.
    """
    z = np.load(ROOT / "runs_local/baseline3ca/jitter_hist_cache.npz",
                allow_pickle=True)
    blocks = z["blocks"]; offs = z["offsets"]
    labels = z["labels"]; status = z["status"]
    meta = [json.loads(str(m)) for m in z["meta"]]
    by_class = {}
    for i in range(len(offs) - 1):
        if status[i] != "ok":
            continue
        by_class.setdefault(int(meta[i]["gt_label"]), []).append(i)
    chosen = []
    for c, n in sorted(target_counts.items()):
        cand = sorted(by_class.get(c, []),
                      key=lambda i: (meta[i]["sample_id"],
                                     meta[i]["instance_id"]))
        chosen.extend(cand[:n])
    chosen.sort(key=lambda i: (meta[i]["sample_id"], meta[i]["instance_id"]))
    rows = []
    for i in chosen:
        b = blocks[offs[i]:offs[i + 1]][:4]
        if len(b) == 0:
            continue
        rows.append({"sample_id": meta[i]["sample_id"],
                     "split": "train", "gt_label": meta[i]["gt_label"],
                     "y": int(labels[i]), "gt_blocks": b,
                     "pred_blocks": b})
    return rows


def train_arm(arm, train_rows, cal_rows, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FineTuneNet().to(device)
    sd = torch.load(WEIGHTS_IN / f"SHARED_seed{seed}.pt",
                    map_location=device)
    model.load_state_dict(sd)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR, weight_decay=WD)
    rng = np.random.default_rng(seed)
    fw = file_weights(train_rows)
    best = {"loss": float("inf"), "state": None, "step": -1}
    bad = 0
    history = []
    for step in range(1, MAX_UPDATES + 1):
        model.train()
        batch = sample_batch(rng, train_rows, BATCH_INST)
        losses = []
        weights = []
        for r in batch:
            if arm == "E1":
                loss = per_instance_loss(model, r["pred_blocks"], r["y"],
                                         device)
            elif arm == "E3":
                loss = per_instance_loss(model, r["gt_blocks"], r["y"],
                                         device)
            else:  # E2 instance-balanced 1:1
                k = min(4, len(r["gt_blocks"]), len(r["pred_blocks"]))
                if k == 0:
                    continue
                lg = per_instance_loss(model, r["gt_blocks"][:k], r["y"],
                                       device)
                lp = per_instance_loss(model, r["pred_blocks"][:k], r["y"],
                                       device)
                loss = 0.5 * (lg + lp)
            losses.append(loss)
            weights.append(fw[r["sample_id"]])
        w = torch.tensor(weights, dtype=torch.float32, device=device)
        loss = (torch.stack(losses) * w).sum() / w.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            cal_pred = evaluate_cal(model, cal_rows, device, "pred")
            cal_gt = evaluate_cal(model, cal_rows, device, "gt")
            history.append({"step": step, "train_loss": float(loss),
                            "cal_pred_loss": cal_pred,
                            "cal_gt_loss": cal_gt})
            if cal_pred < best["loss"] - 1e-6:
                best = {"loss": cal_pred, "step": step,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}}
                bad = 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    model.load_state_dict(best["state"])
    out_path = WEIGHTS_OUT / f"{arm}_seed{seed}.pt"
    WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    return {"arm": arm, "seed": seed, "best_step": best["step"],
            "best_cal_pred_loss": round(best["loss"], 5),
            "history": history, "weights": str(out_path),
            "n_train_instances": len(train_rows),
            "train_blocks": int(sum(len(r["pred_blocks"]) if arm in ("E1", "E3")
                                    else 2 * min(4, len(r["gt_blocks"]),
                                                 len(r["pred_blocks"]))
                                    for r in train_rows))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="E1,E2")
    ap.add_argument("--seeds", default="1729,1730,1731")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_pairs()
    cal_rows = [r for r in rows if r["split"] == "calibration-dev"]
    final_rows = [r for r in rows if r["split"] == "final-dev"]
    print(f"cal {len(cal_rows)} final {len(final_rows)}", flush=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    log = {"cal_instances": len(cal_rows),
           "final_instances": len(final_rows),
           "protocol": {"lr": LR, "wd": WD, "batch_inst": BATCH_INST,
                        "max_updates": MAX_UPDATES,
                        "eval_every": EVAL_EVERY, "patience": PATIENCE,
                        "freeze": "conv blocks 1-2; train conv3+fc",
                        "early_stop": "cal-dev pred loss"},
           "runs": []}
    paired_train = [r for r in rows if r["split"] == "adaptation-train"]
    e3_target_counts = Counter(r["gt_label"] for r in paired_train)
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        train_rows = (load_e3_rows(e3_target_counts) if arm == "E3"
                      else paired_train)
        print(f"{arm}: {len(train_rows)} instances, "
              f"{sum(len(r['pred_blocks']) if arm == 'E1' else len(r['gt_blocks']) for r in train_rows)} blocks",
              f"class={dict(sorted(Counter(r['gt_label'] for r in train_rows).items()))}",
              flush=True)
        for seed in seeds:
            t0 = time.perf_counter()
            r = train_arm(arm, train_rows, cal_rows, seed, device)
            r["seconds"] = round(time.perf_counter() - t0, 1)
            log["runs"].append(r)
            print(f"{arm} seed{seed}: best_step={r['best_step']} "
                  f"cal_pred={r['best_cal_pred_loss']} "
                  f"({r['seconds']}s)", flush=True)
    (EXP / "finetune_log.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8")
    print("wrote finetune_log.json")


if __name__ == "__main__":
    main()
