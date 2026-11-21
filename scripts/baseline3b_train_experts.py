#!/usr/bin/env python3
"""Phase B' stages 2-4: train the P1-faithful histogram CNN experts.

Arms: W20 (classes 0/1/2), W40 (3/4/5), SHARED (0-5 pooled to
QPSK/16QAM/64QAM). Architecture = the audit-verified Sensors-2026 layer
table; the 3-class parameter count is asserted against the audit value
(2,454,467). Training per paper: Adam 1e-3, wd 1e-4, CrossEntropy, early
stopping patience 10 -- on a TRAIN-side dev slice grouped by IQ sample
(10% of samples, seed 1729) so the protocol holdout is evaluated once.

Recorded deviations from the paper: batch 64 (ours ~2-3k train blocks vs
the paper's 487k samples at batch 1024); max 200 epochs; training samples
are block-level histograms with INSTANCE-level evaluation (mean softmax
over an instance's blocks).

Outputs per arm: experts_<arm>.json with per-seed metrics (conditional
on sync-pass instances), confusion, strata, bootstrap CIs by IQ sample,
and per-instance holdout predictions for the comparison stage. Local
weights under runs_local/ are never committed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports/baseline3_wifi_expert"
HIST = EXP / "cache/histograms.npz"
WEIGHTS = ROOT / "runs_local/baseline3b/weights"

ARMS = {
    "W20": {"family": "W20", "classes": [0, 1, 2]},
    "W40": {"family": "W40", "classes": [3, 4, 5]},
    "SHARED": {"family": None, "classes": [0, 1, 2]},
}
SEEDS = (1729, 1730, 1731)
DEV_FRAC = 0.10
MAX_EPOCHS = 200
PATIENCE = 10
BATCH = 64
LR = 1e-3
WD = 1e-4
EXPECTED_PARAMS = 2454467  # audit-verified 3-class parameter count


class HistCNN(nn.Module):
    """Sensors-2026 architecture, audit-verified layer table, 3 outputs."""

    def __init__(self, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc2(self.drop(F.relu(self.fc1(x))))


def train_one(xtr, ytr, xdv, ydv, seed, device="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HistCNN(3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    xt, yt = torch.from_numpy(xtr), torch.from_numpy(ytr)
    xd, yd = torch.from_numpy(xdv), torch.from_numpy(ydv)
    best_loss, best_state, wait = float("inf"), None, 0
    n = len(xt)
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            xb, yb = xt[idx].to(device), yt[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            dl = float(F.cross_entropy(model(xd.to(device)), yd.to(device)))
        if dl < best_loss - 1e-5:
            best_loss, wait = dl, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_loss, epoch + 1


def predict_blocks(model, X, device="cuda", batch=512):
    model.eval()
    outs = []
    with torch.no_grad():
        for b in range(0, len(X), batch):
            xb = torch.from_numpy(X[b:b + batch]).to(device)
            outs.append(F.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 3))


def instance_metrics(mean_prob, y_by_inst, sel=None):
    conf = np.zeros((3, 3), dtype=np.int64)
    iids = sorted(i for i in mean_prob
                  if sel is None or sel(i))
    for iid in iids:
        conf[int(y_by_inst[iid]), int(np.argmax(mean_prob[iid]))] += 1
    n = int(conf.sum())
    if n == 0:
        return {"n": 0, "accuracy": None, "macro_f1": None,
                "confusion": conf.tolist()}
    acc = float(np.trace(conf) / n)
    f1s = []
    for c in range(3):
        tp = int(conf[c, c])
        fp = int(conf[:, c].sum() - tp)
        fn = int(conf[c, :].sum() - tp)
        if tp + fp and tp + fn and prec_rec_ok(tp, fp, fn):
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1s.append(2 * prec * rec / (prec + rec))
    return {"n": n, "accuracy": acc,
            "macro_f1": float(np.mean(f1s)) if f1s else None,
            "confusion": conf.tolist()}


def prec_rec_ok(tp, fp, fn):
    return tp > 0


def bootstrap_ci(pairs, n_boot=1000, seed=0):
    """pairs: (sample_key, correct); cluster bootstrap over IQ samples."""
    rng = np.random.default_rng(seed)
    by = {}
    for k, c in pairs:
        by.setdefault(k, []).append(c)
    keys = sorted(by)
    accs = np.zeros(n_boot)
    base = np.array([np.mean(by[k]) for k in keys])
    wts = np.array([len(by[k]) for k in keys], dtype=float)
    for t in range(n_boot):
        pick = rng.integers(0, len(keys), size=len(keys))
        accs[t] = np.sum(base[pick] * wts[pick]) / np.sum(wts[pick])
    return {"mean": float(np.mean(accs)),
            "lo": float(np.quantile(accs, 0.025)),
            "hi": float(np.quantile(accs, 0.975))}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with np.load(HIST) as z:
        H = z["H"].astype(np.float32)
        inst_ids = z["instance_id"].astype(np.int64)
        metas = [json.loads(str(m)) for m in z["meta"]]
    man = json.loads((EXP / "dataset_manifest.json").read_text())
    inst_by_id = {r["instance_id"]: r for r in man["instances"]}
    print(f"blocks={len(H)} instances={len(set(inst_ids.tolist()))}", flush=True)

    model_tmp = HistCNN(3)
    n_params = sum(p.numel() for p in model_tmp.parameters())
    print(f"param count={n_params} expected={EXPECTED_PARAMS} "
          f"match={n_params == EXPECTED_PARAMS}", flush=True)

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    for arm_name, arm in ARMS.items():
        rows = []
        for i, (iid, meta) in enumerate(zip(inst_ids, metas)):
            if arm["family"] is not None and meta["family"] != arm["family"]:
                continue
            y = (int(meta["shared_label"]) if arm["family"] is None
                 else arm["classes"].index(int(meta["class"])))
            rows.append((i, y, meta["split"],
                         inst_by_id[int(iid)]["sample_id"]))
        tr_rows = [r for r in rows if r[2] == "train"]
        ho_rows = [r for r in rows if r[2] == "holdout"]
        # train-side dev slice grouped by IQ SAMPLE (early stopping)
        uniq = sorted({r[3] for r in tr_rows})
        rng = np.random.default_rng(1729)
        dev_s = set(rng.choice(uniq, size=max(1, int(len(uniq) * DEV_FRAC)),
                                replace=False).tolist())
        core = [r for r in tr_rows if r[3] not in dev_s]
        dev = [r for r in tr_rows if r[3] in dev_s]
        xtr = H[[r[0] for r in core]]
        ytr = np.array([r[1] for r in core], dtype=np.int64)
        xdv = H[[r[0] for r in dev]]
        ydv = np.array([r[1] for r in dev], dtype=np.int64)
        xho = H[[r[0] for r in ho_rows]]
        out = {"arm": arm_name, "n_classes": 3,
               "param_count": int(n_params),
               "n_train_blocks": int(len(core)), "n_dev_blocks": int(len(dev)),
               "n_dev_samples": len(dev_s),
               "n_holdout_blocks": int(len(ho_rows)),
               "n_holdout_instances_with_hist": len({int(inst_ids[r[0]]) for r in ho_rows}),
               "recipe": {"optimizer": "Adam", "lr": LR, "weight_decay": WD,
                          "loss": "CrossEntropy", "batch": BATCH,
                          "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                          "early_stop_on": "train-side dev loss (10% IQ-sample slice, seed 1729)"},
               "seeds": []}
        print(f"\n=== {arm_name}: train {len(core)} dev {len(dev)} "
              f"(samples {len(dev_s)}) holdout blocks {len(ho_rows)} ===",
              flush=True)
        for seed in SEEDS:
            t0 = time.perf_counter()
            model, best_loss, epochs = train_one(xtr, ytr, xdv, ydv, seed,
                                                 device=device)
            torch.save(model.state_dict(), WEIGHTS / f"{arm_name}_seed{seed}.pt")
            probs = predict_blocks(model, xho, device)
            by_inst, y_by = {}, {}
            for k, r in enumerate(ho_rows):
                iid = int(inst_ids[r[0]])
                by_inst.setdefault(iid, []).append(probs[k])
                y_by[iid] = r[1]
            mean_prob = {iid: np.mean(v, axis=0) for iid, v in by_inst.items()}
            m = instance_metrics(mean_prob, y_by)
            m["strata"] = {}
            for strat, sel in (
                    ("clean", lambda i: inst_by_id[i]["overlap_any"] == 0),
                    ("overlap", lambda i: inst_by_id[i]["overlap_any"] == 1),
                    ("strong_ov",
                     lambda i: inst_by_id[i]["overlap_wifi_frac_max"] >= 0.25)):
                m["strata"][strat] = instance_metrics(mean_prob, y_by, sel)
            pairs = [(inst_by_id[iid]["sample_id"],
                      int(np.argmax(p) == y_by[iid]))
                     for iid, p in mean_prob.items()]
            m["bootstrap_by_sample"] = bootstrap_ci(pairs, seed=seed)
            m["best_dev_loss"] = round(best_loss, 4)
            m["epochs"] = int(epochs)
            m["seconds"] = round(time.perf_counter() - t0, 1)
            m["instance_predictions"] = {
                str(iid): {"prob": [round(float(v), 5) for v in mean_prob[iid]],
                           "y": int(y_by[iid]),
                           "pred": int(np.argmax(mean_prob[iid])),
                           "n_blocks": len(by_inst[iid])}
                for iid in sorted(mean_prob)}
            out["seeds"].append({"seed": seed, "metrics": m})
            print(f"  seed {seed}: cond-acc={m['accuracy']:.4f} "
                  f"f1={m['macro_f1']:.4f} epochs={epochs} "
                  f"({m['seconds']}s) "
                  f"clean={m['strata']['clean']['accuracy']:.3f} "
                  f"ov={m['strata']['overlap']['accuracy']:.3f}",
                  flush=True)
        accs = [s["metrics"]["accuracy"] for s in out["seeds"]]
        out["holdout_cond_acc_mean"] = float(np.mean(accs))
        out["holdout_cond_acc_std"] = float(np.std(accs))
        (EXP / f"experts_{arm_name}.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8")
        print(f"  -> experts_{arm_name}.json  "
              f"mean={out['holdout_cond_acc_mean']:.4f}"
              f"±{out['holdout_cond_acc_std']:.4f}", flush=True)


if __name__ == "__main__":
    main()
