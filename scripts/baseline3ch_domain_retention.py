#!/usr/bin/env python3
"""GT-domain retention after fine-tuning (final-dev + calibration-dev).

Evaluates every arm/seed on GT-box histograms for all pairs whose GT
crop synced (not only dual-sync), so E1/E2/E3 can be compared with E0
on the domain they were originally trained on. No model selection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.baseline3cb_e0_eval import bootstrap_ci, conf_matrix, \
    metrics_from_conf, predict
from scripts.baseline3ce_evaluate import build_pair_rows, load_all, seed_model

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports/baseline3_wifi_cascade"
SEEDS = (1729, 1730, 1731)
ARMS = ("E0", "E1", "E2", "E3")


def main():
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    hz, yz, pairs, split = load_all()
    assign = split["assignment"]
    out = {}
    for sname in ("calibration-dev", "final-dev"):
        rows = [r for r in build_pair_rows(hz, pairs, assign, sname)
                if len(r["gt_blocks"]) > 0]
        print(f"{sname}: GT-sync pairs {len(rows)}, "
              f"IQ files {len({r['sample_id'] for r in rows})}")
        out[sname] = {"n_pairs": len(rows),
                      "n_iq": len({r["sample_id"] for r in rows}),
                      "arms": {}}
        for arm in ARMS:
            out[sname]["arms"][arm] = {}
            for seed in SEEDS:
                model = seed_model(arm, seed, device)
                ys, ps = [], []
                for r in rows:
                    p = int(np.argmax(predict(model, r["gt_blocks"], device)))
                    ys.append(r["y"]); ps.append(p)
                C = conf_matrix(ys, ps)
                out[sname]["arms"][arm][str(seed)] = {
                    **metrics_from_conf(C),
                    "ci": bootstrap_ci([(r["sample_id"], int(p == r["y"]))
                                        for r, p in zip(rows, ps)], seed=seed),
                }
            accs = [v["accuracy"] for v in
                    out[sname]["arms"][arm].values()]
            out[sname]["arms"][arm]["mean_accuracy"] = float(np.mean(accs))
            print(f"  {arm}: {np.mean(accs):.4f} "
                  f"({', '.join(f'{a:.3f}' for a in accs)})")
    (EXP / "domain_retention.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("wrote domain_retention.json")


if __name__ == "__main__":
    main()
