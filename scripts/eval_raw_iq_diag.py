#!/usr/bin/env python3
"""Evaluate raw-IQ 1D and local-power baselines on the same val query subset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/detector_upgrade"))
from eval_local_iq_abc import load_model as load_power, predict_group
from train_raw_iq_diag import IQDataset, IQNet, GROUPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path(
        "/root/autodl-fs/runs_local/dfine_local_iq/val_raw_iq.npz"))
    ap.add_argument("--power", type=Path, default=Path(
        "/root/autodl-fs/runs_local/dfine_local_iq/val_local_crops.npz"))
    ap.add_argument("--raw-models", type=Path, default=Path(
        "/root/autodl-fs/runs_local/dfine_local_iq/models_raw_iq_w128"))
    ap.add_argument("--power-models", type=Path, default=Path(
        "/root/autodl-fs/runs_local/dfine_local_iq/models"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    z = np.load(args.raw, allow_pickle=True)
    pz = np.load(args.power, allow_pickle=True)
    labels = z["label"].astype(np.int64); pred = z["pred_label"].astype(np.int64)
    keep = z["keep"].astype(np.int64)
    src = z["source_index"].astype(np.int64)
    gt_label = pz["gt_label"][src].astype(np.int64)
    gt_iou = pz["gt_iou"][src].astype(np.float32)
    out = {"format": "raw-iq-vs-power-val-v1", "groups": {}}
    for g, cls in GROUPS.items():
        cls = np.asarray(cls, dtype=np.int64)
        pos = (gt_iou >= 0.5) & np.isin(gt_label, cls)
        neg = ~pos & np.isin(pred, cls)
        sel = pos | neg
        idx = np.flatnonzero(sel)
        y = np.where(pos[idx], np.searchsorted(cls, gt_label[idx]), len(cls))
        # Raw IQ model.
        ck = torch.load(args.raw_models / g / "best.pt", map_location="cpu",
                        weights_only=False)
        net = IQNet(int(ck["num_classes"]), width=int(ck.get("width", 128))).to(args.device)
        net.load_state_dict(ck["state"]); net.eval()
        ds = IQDataset(z["iq"][idx], z["lengths"][idx], y.astype(np.int64))
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
        pr = []
        with torch.no_grad():
            for x, _ in dl:
                pr.append(net(x.to(args.device)).argmax(1).cpu().numpy())
        pr = np.concatenate(pr)
        pos_m = y < len(cls); keep_m = y == len(cls)
        raw_acc = float((pr == y).mean())
        raw_pos = float((pr[pos_m] == y[pos_m]).mean()) if pos_m.any() else None
        raw_keep = float((pr[keep_m] == len(cls)).mean()) if keep_m.any() else None
        # Local power baseline on exactly the same source rows.
        pck = torch.load(args.power_models / g / "features/best.pt",
                         map_location="cpu", weights_only=False)
        pmod, pck = load_power(args.power_models / g / "features/best.pt", args.device)
        pcls = np.asarray(pck["classes"], dtype=np.int64)
        meta = np.concatenate([
            pz["class_scores"][src][:, pcls].astype(np.float32),
            pz["score"][src][:, None].astype(np.float32),
            np.log1p(pz["bandwidth_mhz"][src])[:, None].astype(np.float32),
            np.log1p(pz["duration_ms"][src])[:, None].astype(np.float32),
            pz["view_type"][src][:, None].astype(np.float32),
        ], axis=1).astype(np.float32)
        probs = predict_group(pmod, pck, pz["local_power"][src],
                              pz["features"][src], meta, True, args.device)
        # Restrict to the same selected rows.
        probs = probs[sel]
        pp = probs[:, :-1].argmax(1)
        kp = probs[:, -1]
        # "predicted class" for a positive/negative row: group argmax, or keep.
        p_pred = np.where(kp > probs[:, :-1].max(1), len(cls), pp)
        power_acc = float((p_pred == y).mean())
        power_pos = float((p_pred[pos_m] == y[pos_m]).mean()) if pos_m.any() else None
        power_keep = float((p_pred[keep_m] == len(cls)).mean()) if keep_m.any() else None
        out["groups"][g] = {
            "queries": int(sel.sum()), "pos": int(pos_m.sum()), "keep": int(keep_m.sum()),
            "raw": {"acc": raw_acc, "pos_acc": raw_pos, "keep_recall": raw_keep},
            "power": {"acc": power_acc, "pos_acc": power_pos, "keep_recall": power_keep},
        }
        print(g, json.dumps(out["groups"][g]), flush=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
