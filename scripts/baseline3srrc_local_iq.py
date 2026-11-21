#!/usr/bin/env python3
"""SRRC 10/11 local-IQ classification experiment (fit only, no YOLO).

Stages:
  1. GT-box central crop (2 ms) from fit IQ; label-blind SRRC matched
     filter / symbol timing; all sync gates fixed, no class labels.
  2. Features:
       B0_power  : local spectral power only
       B0_unsync : power + unsynchronized IQ amplitude/phase stats
       R_amp     : recovered-symbol amplitude stats
       R_amp_ph  : recovered amplitude + adjacent-symbol phase difference
  3. File-grouped 70/30 dev/holdout split (no val/test).
  4. Paired comparison on the same recovery-success subset; net all-instance
     comparison with fallback to B0 for recovery failures.
  5. Logistic regression primary; small MLP sensitivity; file-level
     bootstrap and McNemar tests.

No model is trained on val/test; no YOLO; no Wi-Fi weights touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import binomtest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.signals.data import SPLIT_MANIFEST, load_manifest
from scripts.srrc_frontend import extract_instance, BAND_HZ

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/baseline3_srrc_local_iq"
SEED = 20261121
CAP_MS = 2.0
R_GUESS = BAND_HZ / 1.2
BOOT = 2000

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def overlap_flags(rec, sig) -> dict:
    a = (float(sig["start_frequency"]), float(sig["end_frequency"]),
         float(sig["start_time"]), float(sig["end_time"]))
    area = max((a[1] - a[0]) * (a[3] - a[2]), 1e-9)
    anyf = srrcf = wifif = crossf = samef = 0.0
    for o in rec.signals:
        if o is sig:
            continue
        b = (float(o["start_frequency"]), float(o["end_frequency"]),
             float(o["start_time"]), float(o["end_time"]))
        iw = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
        ih = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
        inter = iw * ih
        if inter <= 0:
            continue
        f = inter / area
        anyf = max(anyf, f)
        oc = int(o["class"])
        if oc in (10, 11):
            srrcf = max(srrcf, f)
            if oc == int(sig["class"]):
                samef = max(samef, f)
            else:
                crossf = max(crossf, f)
        if 0 <= oc <= 5:
            wifif = max(wifif, f)
    return {"any": int(anyf > 0), "strong": int(anyf >= 0.25),
            "srrc": int(srrcf > 0), "srrc_cross": int(crossf > 0),
            "srrc_same": int(samef > 0), "wifi": int(wifif > 0),
            "any_frac": float(anyf), "srrc_frac": float(srrcf)}


def build_file_table(recs):
    table = {}
    instances = []
    for r in recs:
        # load_manifest(split="fit") already selects split_new=fit;
        # record.split remains the train/val source folder.
        hits = []
        for s in r.signals:
            bw = float(s["end_frequency"]) - float(s["start_frequency"])
            if int(s["class"]) in (10, 11) and 9.5 <= bw <= 10.5:
                ov = overlap_flags(r, s)
                rec = {
                    "sample_id": r.sample_id,
                    "signal_id": int(s["signal_id"]),
                    "class": int(s["class"]),
                    "bandwidth_mhz": bw,
                    "duration_ms": float(s["end_time"]) - float(s["start_time"]),
                    "f_center_mhz": 0.5 * (float(s["start_frequency"]) +
                                           float(s["end_frequency"])),
                    **ov,
                }
                instances.append(rec)
                hits.append(rec)
        if hits:
            table[r.sample_id] = {
                "c10": sum(1 for h in hits if h["class"] == 10),
                "c11": sum(1 for h in hits if h["class"] == 11),
                "strong": int(any(h["strong"] for h in hits)),
                "any": int(any(h["any"] for h in hits)),
            }
    return table, instances


def split_files(table, seed: int = SEED):
    strata = defaultdict(list)
    for sid, info in table.items():
        strata[(info["c10"] > 0, info["c11"] > 0, bool(info["strong"]))].append(sid)
    assign = {}
    for key in sorted(strata):
        sids = sorted(strata[key], key=lambda s: sha256(f"{seed}:{s}"))
        n = len(sids)
        n_train = n if n <= 1 else max(1, min(n - 1, int(round(0.70 * n))))
        for i, sid in enumerate(sids):
            assign[sid] = "train" if i < n_train else "holdout"
    # ensure every class has holdout instances; swap if needed
    for cls in ("c10", "c11"):
        hold = [s for s in assign if assign[s] == "holdout"]
        if sum(table[s][cls] for s in hold) == 0:
            train_with = [s for s in assign if assign[s] == "train" and table[s][cls] > 0]
            hold_without = [s for s in hold if table[s][cls] == 0]
            if train_with and hold_without:
                a, b = train_with[0], hold_without[0]
                assign[a], assign[b] = "holdout", "train"
    return assign


def worker(args):
    rec = SimpleNamespace(**args["rec"])
    try:
        return extract_instance(rec, args["sig"])
    except Exception as e:  # keep failure in denominator
        return {"sample_id": rec.sample_id,
                "signal_id": int(args["sig"]["signal_id"]),
                "class": int(args["sig"]["class"]),
                "status": "error", "reason": str(e)[:200],
                "baseline": None, "amp": None, "phase": None}


def clean_X(X):
    X = np.asarray(X, float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def fit_logreg(X, y):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced",
                           max_iter=5000, solver="lbfgs"))
    clf.fit(clean_X(X), np.asarray(y, int))
    return clf


def fit_mlp(X, y, seed):
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2,
                      max_iter=2000, random_state=seed))
    clf.fit(clean_X(X), np.asarray(y, int))
    return clf


def acc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return float(np.mean(y == p)) if len(y) else None


def confusion(y, p):
    y = np.asarray(y); p = np.asarray(p)
    C = np.zeros((2, 2), int)
    for a, b in zip(y, p):
        C[0 if a == 10 else 1, 0 if b == 10 else 1] += 1
    return C


def prf(C):
    out = {}
    for i, c in enumerate((10, 11)):
        tp = C[i, i]; fp = C[:, i].sum() - tp; fn = C[i, :].sum() - tp
        out[str(c)] = {
            "precision": float(tp / (tp + fp)) if tp + fp else None,
            "recall": float(tp / (tp + fn)) if tp + fn else None,
            "support": int(C[i, :].sum()),
        }
    return out


def mcnemar(y, p0, p1):
    y = np.asarray(y); p0 = np.asarray(p0); p1 = np.asarray(p1)
    b = int(np.sum((p0 != y) & (p1 == y)))
    c = int(np.sum((p0 == y) & (p1 != y)))
    p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    return {"b_0wrong_1right": b, "c_0right_1wrong": c, "net": b - c,
            "p_exact": p}


def bootstrap_diff(y, p0, p1, groups, n_boot=BOOT, seed=SEED):
    """File-cluster bootstrap for accuracy(p1)-accuracy(p0)."""
    y = np.asarray(y); p0 = np.asarray(p0); p1 = np.asarray(p1)
    groups = np.asarray(groups)
    keys = sorted(set(groups.tolist()))
    idx_by = {k: np.where(groups == k)[0] for k in keys}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), size=len(keys))
        idx = np.concatenate([idx_by[keys[j]] for j in pick])
        if len(idx) == 0:
            continue
        vals.append(acc(y[idx], p1[idx]) - acc(y[idx], p0[idx]))
    if not vals:
        return None
    v = np.asarray(vals)
    return {"mean": float(v.mean()), "lo": float(np.quantile(v, .025)),
            "hi": float(np.quantile(v, .975)), "sd": float(v.std()),
            "n_boot": len(vals)}


def synthetic_validation():
    """Small known-parameter check of matched filter/timing."""
    from scripts.baseline3_pa_srrc import gen_srrc_clean
    from scripts.baseline3_dsp import band_limit_fft
    from scripts.srrc_frontend import sync_srrc, amplitude_features
    out = []
    for alpha, order, ref in ((0.2, 4, 1.0), (0.2, 16, 1.32),
                              (0.35, 4, 1.0), (0.35, 16, 1.32)):
        fs = 50e6
        x, meta = gen_srrc_clean(BAND_HZ, alpha, 20000, order, fs,
                                 1000 + order + int(alpha * 100))
        x = x[:int(2e-3 * fs)]
        x = band_limit_fft(x, fs, BAND_HZ * 1.02, edge_frac=0.05)
        n = np.arange(len(x), dtype=np.float64)
        x = x * np.exp(2j * np.pi * 3e3 * n / fs)
        rng = np.random.default_rng(2000 + order + int(alpha * 100))
        p = float(np.mean(np.abs(x) ** 2))
        x = x + math.sqrt(p / 10 ** 2.0 / 2) * (
            rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
        r = sync_srrc(x.astype(np.complex64), fs)
        rec = {"alpha_true": alpha, "order": order, "status": r["status"]}
        if r["status"] == "ok":
            amp = amplitude_features(r["symbols"])
            rec.update({
                "R_true_mhz": meta["r_hz"] / 1e6,
                "R_est_mhz": r["R_mhz"], "alpha_est": r["alpha"],
                "m4": amp["m4"], "m4_ref": ref,
                "R_err_pct": 100 * (r["R_mhz"] - meta["r_hz"] / 1e6) /
                (meta["r_hz"] / 1e6),
            })
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    recs = load_manifest(SPLIT_MANIFEST, split="fit")
    table, instances = build_file_table(recs)
    print(f"fit files with 10MHz SRRC: {len(table)}, instances: {len(instances)}")
    assign = split_files(table)
    train_sids = {s for s, v in assign.items() if v == "train"}
    hold_sids = {s for s, v in assign.items() if v == "holdout"}
    assignment_sha = sha256(json.dumps(sorted(assign.items())))
    split_info = {
        "seed": SEED, "assignment_sha256": assignment_sha,
        "n_files": len(table), "n_instances": len(instances),
        "files_train": len(train_sids), "files_holdout": len(hold_sids),
        "instances_train": sum(1 for x in instances if x["sample_id"] in train_sids),
        "instances_holdout": sum(1 for x in instances if x["sample_id"] in hold_sids),
        "class_counts": {
            "train": dict(Counter(x["class"] for x in instances
                                  if x["sample_id"] in train_sids)),
            "holdout": dict(Counter(x["class"] for x in instances
                                    if x["sample_id"] in hold_sids)),
        },
        "overlap_counts": {
            "train": dict(Counter(
                "clean" if x["any"] == 0 else
                "strong" if x["strong"] else "any"
                for x in instances if x["sample_id"] in train_sids)),
            "holdout": dict(Counter(
                "clean" if x["any"] == 0 else
                "strong" if x["strong"] else "any"
                for x in instances if x["sample_id"] in hold_sids)),
        },
        "assignment": assign,
    }
    (OUT / "srrc_local_iq_split.json").write_text(
        json.dumps(split_info, indent=2), encoding="utf-8")
    print(json.dumps({k: split_info[k] for k in split_info
                      if k != "assignment"}, indent=2))

    # process instances
    by_key = {(r.sample_id, int(s["signal_id"])): (r, s)
              for r in recs for s in r.signals
              if int(s["class"]) in (10, 11) and
              9.5 <= float(s["end_frequency"]) - float(s["start_frequency"]) <= 10.5}
    jobs = []
    for x in instances:
        r, s = by_key[(x["sample_id"], x["signal_id"])]
        jobs.append({"rec": {
            "sample_id": r.sample_id, "bin_path": str(r.bin_path),
            "B_mhz": r.B_mhz, "T_ms": r.T_ms, "N": r.N,
            "f_lo": r.f_lo}, "sig": s})
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"processing {len(jobs)} instances ...", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, r in enumerate(ex.map(worker, jobs, chunksize=8)):
            results.append(r)
            if (k + 1) % 200 == 0:
                print(f"  {k+1}/{len(jobs)} {time.perf_counter()-t0:.0f}s",
                      flush=True)
    print(f"processed in {time.perf_counter()-t0:.0f}s")

    # assemble
    rows = []
    for r in results:
        key = (r["sample_id"], r["signal_id"])
        meta = next(x for x in instances if (x["sample_id"], x["signal_id"]) == key)
        row = {**meta, **{k: r.get(k) for k in
                          ("status", "reason", "crop_ms", "R_mhz", "alpha",
                           "n_sym", "tone", "baseline", "amp", "phase")}}
        row["split"] = assign[row["sample_id"]]
        rows.append(row)
    first_base = next((r for r in rows if r["baseline"] is not None), None)
    first_amp = next((r for r in rows if r["amp"] is not None), None)
    if first_base is None or first_amp is None:
        raise RuntimeError("no baseline/amp features extracted")
    names_b = first_base["baseline"]["names"]
    names_a = first_amp["amp"]["names"]
    names_p = first_amp["phase"]["names"]
    for r in rows:
        if r["baseline"] is not None:
            assert r["baseline"]["names"] == names_b
        if r["amp"] is not None:
            assert r["amp"]["names"] == names_a
        if r["phase"] is not None:
            assert r["phase"]["names"] == names_p
    # names checked above

    # feature matrix
    def get_X(rows_sub, kind):
        if kind == "b0_power":
            idx = [i for i, n in enumerate(names_b)
                   if n == "log_power" or n.startswith("band") or
                   n.startswith("seg") or n in ("spec_entropy", "spec_flatness")]
            return np.stack([r["baseline"]["features"][idx] for r in rows_sub])
        if kind == "b0_unsync":
            return np.stack([r["baseline"]["features"] for r in rows_sub])
        if kind == "b0_nopow":
            idx = [i for i, n in enumerate(names_b) if n != "log_power"]
            return np.stack([r["baseline"]["features"][idx] for r in rows_sub])
        if kind == "b0_no_unsync_phase":
            idx = [i for i, n in enumerate(names_b)
                   if not n.startswith("unsync_phase_hist")]
            return np.stack([r["baseline"]["features"][idx] for r in rows_sub])
        if kind == "b0_no_unsync_amp":
            drop = {"time_m4", "papr"}
            idx = [i for i, n in enumerate(names_b)
                   if not n.startswith("amp_hist") and n not in drop]
            return np.stack([r["baseline"]["features"][idx] for r in rows_sub])
        if kind == "amp":
            return np.stack([r["amp"]["features"] for r in rows_sub])
        if kind == "amp_phase":
            return np.stack([np.concatenate([r["amp"]["features"],
                                             r["phase"]["features"]])
                             for r in rows_sub])
        raise ValueError(kind)

    train_base = [r for r in rows if r["split"] == "train" and
                  r["baseline"] is not None]
    hold_eval = [r for r in rows if r["split"] == "holdout"]
    hold_base = [r for r in hold_eval if r["baseline"] is not None]
    train_ok = [r for r in train_base if r["status"] == "ok"]
    hold_ok = [r for r in hold_base if r["status"] == "ok"]
    y_train_base = np.array([r["class"] for r in train_base])
    y_hold_eval = np.array([r["class"] for r in hold_eval])
    y_hold_base = np.array([r["class"] for r in hold_base])
    y_train_ok = np.array([r["class"] for r in train_ok])
    y_hold_ok = np.array([r["class"] for r in hold_ok])
    majority = Counter(y_train_base).most_common(1)[0][0]
    print(f"train base {len(train_base)} (ok {len(train_ok)}), "
          f"holdout all {len(hold_eval)} (base {len(hold_base)}, ok {len(hold_ok)})")

    models = {}
    base_kinds = ("b0_power", "b0_unsync", "b0_nopow",
                 "b0_no_unsync_phase", "b0_no_unsync_amp")
    for kind in base_kinds:
        models[kind] = fit_logreg(get_X(train_base, kind), y_train_base)
    for kind in base_kinds + ("amp", "amp_phase"):
        models[f"{kind}_ok"] = fit_logreg(get_X(train_ok, kind), y_train_ok)

    # predictions on holdout with a baseline feature vector
    pred_base = {}
    for kind in base_kinds:
        pred_base[kind] = models[kind].predict(get_X(hold_base, kind))
    # all-instance vectors: majority fallback for crop/baseline failures
    b0_eval = {kind: np.full(len(hold_eval), majority, dtype=int)
               for kind in base_kinds}
    base_pos = {id(r): i for i, r in enumerate(hold_base)}
    for i, r in enumerate(hold_eval):
        j = base_pos.get(id(r))
        if j is not None:
            for kind in base_kinds:
                b0_eval[kind][i] = pred_base[kind][j]

    pred = {}
    for kind in base_kinds:
        pred[kind] = b0_eval[kind]
    for kind in base_kinds + ("amp", "amp_phase"):
        pred[f"{kind}_ok"] = models[f"{kind}_ok"].predict(get_X(hold_ok, kind))

    # m4 threshold on train successes
    m4_i = names_a.index("amp_m4")
    m4_train = np.array([r["amp"]["features"][m4_i] for r in train_ok])
    m4_hold = np.array([r["amp"]["features"][m4_i] for r in hold_ok])
    best = None
    for th in np.quantile(m4_train, np.linspace(0.05, 0.95, 91)):
        p = np.where(m4_train >= th, 11, 10)
        a = acc(y_train_ok, p)
        if best is None or a > best[0]:
            best = (a, float(th))
    m4_th = best[1]
    m4_pred = np.where(m4_hold >= m4_th, 11, 10)

    def eval_pair(y, p0, p1, groups, label):
        C0 = confusion(y, p0); C1 = confusion(y, p1)
        b = int(np.sum((p0 != y) & (p1 == y)))
        c = int(np.sum((p0 == y) & (p1 != y)))
        return {
            "n": int(len(y)),
            "acc_0": acc(y, p0), "acc_1": acc(y, p1),
            "delta_pp": 100 * (acc(y, p1) - acc(y, p0)),
            "corrected": b, "broken": c, "net": b - c,
            "mcnemar": mcnemar(y, p0, p1),
            "bootstrap_delta": bootstrap_diff(y, p0, p1, groups),
            "conf_0": C0.tolist(), "conf_1": C1.tolist(),
            "prf_0": prf(C0), "prf_1": prf(C1),
        }

    hold_ok_groups = [r["sample_id"] for r in hold_ok]
    hold_eval_groups = [r["sample_id"] for r in hold_eval]
    res = {
        "split": {k: v for k, v in split_info.items() if k != "assignment"},
        "n_rows": len(rows),
        "recovery": {
            "n": len(rows),
            "ok": sum(r["status"] == "ok" for r in rows),
            "rate": sum(r["status"] == "ok" for r in rows) / max(len(rows), 1),
            "by_class": {},
            "by_overlap": {},
            "failure_reasons": dict(Counter(
                r["status"] for r in rows if r["status"] != "ok")),
        },
        "comparisons": {},
        "classifier": "logistic_regression_C1_balanced",
        "synthetic": synthetic_validation(),
    }
    for c in (10, 11):
        sub = [r for r in rows if r["class"] == c]
        res["recovery"]["by_class"][str(c)] = {
            "n": len(sub), "ok": sum(r["status"] == "ok" for r in sub),
            "rate": sum(r["status"] == "ok" for r in sub) / max(len(sub), 1),
        }
    for tag, fn in (("clean", lambda r: r["any"] == 0),
                    ("any", lambda r: r["any"] == 1),
                    ("strong", lambda r: r["strong"] == 1)):
        sub = [r for r in rows if fn(r)]
        res["recovery"]["by_overlap"][tag] = {
            "n": len(sub), "ok": sum(r["status"] == "ok" for r in sub),
            "rate": sum(r["status"] == "ok" for r in sub) / max(len(sub), 1),
        }

    res["comparisons"]["b0_unsync_vs_amp_success"] = eval_pair(
        y_hold_ok, pred["b0_unsync_ok"], pred["amp_ok"],
        hold_ok_groups, "b0_unsync -> amp (success subset)")
    res["comparisons"]["amp_vs_amp_phase_success"] = eval_pair(
        y_hold_ok, pred["amp_ok"], pred["amp_phase_ok"],
        hold_ok_groups, "amp -> amp+phase (success subset)")
    res["comparisons"]["b0_power_vs_b0_unsync_success"] = eval_pair(
        y_hold_ok, pred["b0_power_ok"], pred["b0_unsync_ok"],
        hold_ok_groups, "power -> unsync IQ (success subset)")
    res["comparisons"]["b0_nopow_vs_amp_success"] = eval_pair(
        y_hold_ok, pred["b0_nopow_ok"], pred["amp_ok"],
        hold_ok_groups, "power removed -> amp (success subset)")
    res["comparisons"]["b0_no_unsync_phase_vs_amp_success"] = eval_pair(
        y_hold_ok, pred["b0_no_unsync_phase_ok"], pred["amp_ok"],
        hold_ok_groups, "no unsync phase -> amp (success subset)")
    res["comparisons"]["b0_no_unsync_amp_vs_amp_success"] = eval_pair(
        y_hold_ok, pred["b0_no_unsync_amp_ok"], pred["amp_ok"],
        hold_ok_groups, "no unsync amp -> amp (success subset)")

    # net all-instance: b0_unsync fallback for recovery failures
    b0_all = np.asarray(b0_eval["b0_unsync"])
    net = b0_all.copy()
    hold_ok_index = {id(r): i for i, r in enumerate(hold_ok)}
    for i, r in enumerate(hold_eval):
        if r["status"] == "ok":
            j = hold_ok_index[id(r)]
            net[i] = pred["amp_ok"][j]
    res["comparisons"]["all_instances_net_amp"] = eval_pair(
        y_hold_eval, b0_all, net, hold_eval_groups,
        "B0 all -> amp+fallback all instances")
    net2 = b0_all.copy()
    for i, r in enumerate(hold_eval):
        if r["status"] == "ok":
            j = hold_ok_index[id(r)]
            net2[i] = pred["amp_phase_ok"][j]
    res["comparisons"]["all_instances_net_amp_phase"] = eval_pair(
        y_hold_eval, b0_all, net2, hold_eval_groups,
        "B0 all -> amp+phase+fallback all instances")
    res["baseline_all"] = {
        "b0_power": acc(y_hold_eval, b0_eval["b0_power"]),
        "b0_unsync": acc(y_hold_eval, b0_all),
        "b0_nopow": acc(y_hold_eval, b0_eval["b0_nopow"]),
        "b0_no_unsync_phase": acc(y_hold_eval, b0_eval["b0_no_unsync_phase"]),
        "b0_no_unsync_amp": acc(y_hold_eval, b0_eval["b0_no_unsync_amp"]),
    }
    res["n_success"] = {"train": len(train_ok), "holdout": len(hold_ok)}
    res["metadata"] = {}
    for split in ("train", "holdout"):
        res["metadata"][split] = {}
        for c in (10, 11):
            sub = [r for r in rows if r["split"] == split and r["class"] == c
                   and r["baseline"] is not None]
            if not sub:
                continue
            lp = np.array([r["baseline"]["log_power"] for r in sub], float)
            du = np.array([r["duration_ms"] for r in sub], float)
            res["metadata"][split][str(c)] = {
                "n": len(sub),
                "log_power_mean": float(lp.mean()),
                "log_power_median": float(np.median(lp)),
                "duration_median_ms": float(np.median(du)),
                "any_rate": float(np.mean([r["any"] for r in sub])),
                "strong_rate": float(np.mean([r["strong"] for r in sub])),
            }

    # strata on success subset
    def strata_for(preds, name):
        out = {}
        for tag, fn in (("clean", lambda r: r["any"] == 0),
                        ("any", lambda r: r["any"] == 1),
                        ("strong", lambda r: r["strong"] == 1),
                        ("class10", lambda r: r["class"] == 10),
                        ("class11", lambda r: r["class"] == 11)):
            ii = [i for i, r in enumerate(hold_ok) if fn(r)]
            if not ii:
                continue
            out[tag] = {"n": len(ii),
                        "acc": acc(y_hold_ok[ii], preds[ii])}
        return out
    res["strata"] = {
        "b0_unsync": strata_for(pred["b0_unsync_ok"], "b0"),
        "amp": strata_for(pred["amp_ok"], "amp"),
        "amp_phase": strata_for(pred["amp_phase_ok"], "amp+phase"),
        "m4_threshold": {
            "threshold": m4_th,
            "acc_success": acc(y_hold_ok, m4_pred),
            "m4_train": {"10": float(np.mean(m4_train[y_train_ok == 10])),
                         "11": float(np.mean(m4_train[y_train_ok == 11]))},
            "m4_hold": {"10": float(np.mean(m4_hold[y_hold_ok == 10])),
                        "11": float(np.mean(m4_hold[y_hold_ok == 11]))},
        },
    }
    res["m4_success"] = {
        "train": {str(c): {"n": int(np.sum(y_train_ok == c)),
                           "mean": float(np.mean(m4_train[y_train_ok == c])),
                           "median": float(np.median(m4_train[y_train_ok == c]))}
                  for c in (10, 11) if np.any(y_train_ok == c)},
        "holdout": {str(c): {"n": int(np.sum(y_hold_ok == c)),
                             "mean": float(np.mean(m4_hold[y_hold_ok == c])),
                             "median": float(np.median(m4_hold[y_hold_ok == c]))}
                    for c in (10, 11) if np.any(y_hold_ok == c)},
    }

    # MLP sensitivity (3 seeds)
    mlp = {}
    for kind, Xtr, Xho, ytr, yho in (
            ("b0_unsync", get_X(train_ok, "b0_unsync"),
             get_X(hold_ok, "b0_unsync"), y_train_ok, y_hold_ok),
            ("amp", get_X(train_ok, "amp"),
             get_X(hold_ok, "amp"), y_train_ok, y_hold_ok),
            ("amp_phase", get_X(train_ok, "amp_phase"),
             get_X(hold_ok, "amp_phase"), y_train_ok, y_hold_ok)):
        vals = []
        for s in (1729, 1730, 1731):
            m = fit_mlp(Xtr, ytr, s)
            vals.append(acc(yho, m.predict(Xho)))
        mlp[kind] = {"mean": float(np.mean(vals)),
                     "std": float(np.std(vals)), "seeds": vals}
    res["mlp_sensitivity"] = mlp

    # predictions CSV (holdout)
    pred_rows = []
    fields = ["sample_id", "signal_id", "class", "split",
              "any_overlap", "strong_overlap", "status", "reason",
              "R_mhz", "alpha", "n_sym", "m4",
              "b0_power", "b0_unsync", "b0_nopow", "amp", "amp_phase",
              "net_amp", "net_amp_phase"]
    for i, r in enumerate(hold_eval):
        row = {"sample_id": r["sample_id"], "signal_id": r["signal_id"],
               "class": r["class"], "split": r["split"],
               "any_overlap": r["any"], "strong_overlap": r["strong"],
               "status": r["status"], "reason": r.get("reason"),
               "R_mhz": r.get("R_mhz"), "alpha": r.get("alpha"),
               "n_sym": r.get("n_sym"),
               "b0_power": int(b0_eval["b0_power"][i]),
               "b0_unsync": int(b0_all[i]),
               "b0_nopow": int(b0_eval["b0_nopow"][i]),
               "amp": None, "amp_phase": None,
               "net_amp": int(net[i]), "net_amp_phase": int(net2[i]),
               "m4": None}
        if r["status"] == "ok":
            j = hold_ok_index[id(r)]
            row.update({"amp": int(pred["amp_ok"][j]),
                        "amp_phase": int(pred["amp_phase_ok"][j]),
                        "m4": float(r["amp"]["features"][m4_i])})
        pred_rows.append(row)
    with (OUT / "srrc_local_iq_predictions.csv").open("w", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in pred_rows:
            w.writerow(row)

    res["note"] = ("Fit only; GT-box crop; no YOLO; no val/test. "
                   "B0_success models are trained on the same success-only "
                   "train instances as R_amp/R_amp_phase. All-instance net "
                   "uses B0_unsync fallback for failures.")
    (OUT / "srrc_local_iq_results.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in
                      ("split", "recovery", "comparisons", "strata",
                       "mlp_sensitivity", "synthetic")}, indent=2)[:12000])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
