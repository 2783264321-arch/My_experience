#!/usr/bin/env python3
"""Phase B' core comparison: histogram experts vs power baselines.

Protocol (pre-registered):
  - Same instance set, same ONE-split, same GT-box geometry for all arms.
  - Conditional accuracy: sync-pass holdout instances with >=1 histogram.
  - All-instance accuracy: every holdout instance; instances without a
    histogram fall back to the per-family power baseline prediction
    (fallback rep = the higher 3-seed-mean of coarse vs local_power,
    locked from power_baselines_bprime.json BEFORE looking at expert
    results; fallback predictions use power seed 1729, recorded).
  - Paired statistics per expert seed: fix/brake counts vs the power
    baseline on the same instances, exact McNemar binomial p,
    cluster bootstrap CIs over IQ samples.
  - W-Shared is additionally evaluated on each family subset and compared
    with the family expert on the identical instance set.

Outputs: wifi_expert_comparison.json (+ .md written separately).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports/baseline3_wifi_expert"


def binom_p_two_sided(k, n):
    if n == 0:
        return 1.0
    from math import comb
    p = 0.0
    thr = comb(n, min(k, n - k))
    for i in range(n + 1):
        if comb(n, i) <= thr * (1 + 1e-9):
            p += comb(n, i) * 0.5 ** n
    return min(1.0, p)


def bootstrap_ci(pairs, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    by = {}
    for k, c in pairs:
        by.setdefault(k, []).append(c)
    keys = sorted(by)
    base = np.array([np.mean(by[k]) for k in keys])
    wts = np.array([len(by[k]) for k in keys], dtype=float)
    accs = np.zeros(n_boot)
    for t in range(n_boot):
        pick = rng.integers(0, len(keys), size=len(keys))
        accs[t] = np.sum(base[pick] * wts[pick]) / np.sum(wts[pick])
    return {"mean": float(np.mean(accs)),
            "lo": float(np.quantile(accs, 0.025)),
            "hi": float(np.quantile(accs, 0.975))}


def confusion(ys, preds, n=3):
    C = np.zeros((n, n), dtype=np.int64)
    for y, p in zip(ys, preds):
        C[int(y), int(p)] += 1
    return C


def main():
    man = json.loads((EXP / "dataset_manifest.json").read_text())
    insts = man["instances"]
    by_id = {r["instance_id"]: r for r in insts}
    hold = [r for r in insts if r["split"] == "holdout"]
    hold_ids = [r["instance_id"] for r in hold]
    print(f"holdout instances: {len(hold)}")

    cache = json.loads((EXP / "symbol_cache_summary.json").read_text())
    sync_ok = {r["instance_id"] for r in cache["results"]
               if r.get("status") == "ok"}

    pw = json.loads((EXP / "power_baselines_bprime.json").read_text())

    # ---- lock the fallback rule BEFORE expert comparison ----
    fallback = {}
    for arm in ("pw_012", "pw_345", "pw_shared"):
        means = {rep: pw["arms"][arm]["arms"][rep]["holdout_acc_mean"]
                 for rep in ("coarse", "local_power")}
        fallback[arm] = max(means, key=means.get)
    print("fallback rule (3-seed mean, locked):", fallback)

    # power baseline per-instance predictions (seed 1729 rows stored in
    # power_baselines_bprime.json during training); rebuild by arm
    def power_preds(arm, rep, cls_filter=None):
        rows = pw["arms"][arm]["arms"][rep]["seeds"]
        row = next(r for r in rows if r["seed"] == 1729)
        preds = row["pred_holdout"]
        # the baseline script stored predictions aligned to the arm's own
        # holdout subset in manifest order
        ids = [r["instance_id"] for r in hold
               if cls_filter is None or cls_filter(r)]
        if len(preds) != len(ids):
            raise RuntimeError(f"pred/ids mismatch {len(preds)} vs {len(ids)}")
        return {iid: preds[i] for i, iid in enumerate(ids)}, len(preds)

    experts = {}
    for arm in ("W20", "W40", "SHARED"):
        d = json.loads((EXP / f"experts_{arm}.json").read_text())
        experts[arm] = d

    fam_of = {"W20": "pw_012", "W40": "pw_345"}

    out = {"protocol": {
        "geometry": man["geometry"], "split": man["protocol"],
        "fallback_rule": fallback, "fallback_seed": 1729,
        "power_source": "power_baselines_bprime.json (m0 crops, same split)"},
        "sync_availability": {}, "arms": {}}

    # ---- sync availability stratified ----
    for grp, sel in (("W20_all", lambda r: r["family"] == "W20"),
                    ("W40_all", lambda r: r["family"] == "W40"),
                    ("all", lambda r: True)):
        ids = [r["instance_id"] for r in hold if sel(r)]
        ok = [i for i in ids if i in sync_ok]
        out["sync_availability"][grp] = {
            "n": len(ids), "n_sync_ok": len(ok),
            "rate": round(len(ok) / len(ids), 4)}
        for strat, ssel in (("clean", lambda r: r["overlap_any"] == 0),
                            ("overlap", lambda r: r["overlap_any"] == 1),
                            ("strong_ov", lambda r: r["overlap_wifi_frac_max"] >= 0.25)):
            sub = [r["instance_id"] for r in hold if sel(r) and ssel(r)]
            subok = [i for i in sub if i in sync_ok]
            out["sync_availability"][f"{grp}/{strat}"] = {
                "n": len(sub), "n_sync_ok": len(subok),
                "rate": round(len(subok) / len(sub), 4) if sub else None}

    def compare(expert_arm, family, pw_arm):
        """Per-seed paired comparison for one expert vs its power baseline."""
        res = {"conditional": [], "all_instance": []}
        fam_ids = [r["instance_id"] for r in hold
                   if (family is None or r["family"] == family)]
        cls_filter = None
        if pw_arm == "pw_012":
            cls_filter = lambda r: r["family"] == "W20"
        elif pw_arm == "pw_345":
            cls_filter = lambda r: r["family"] == "W40"
        pwp, _ = power_preds(pw_arm, fallback[pw_arm], cls_filter)
        pw_all_ok = dict(pwp)
        for srow in experts[expert_arm]["seeds"]:
            preds = srow["metrics"]["instance_predictions"]
            # conditional subset = expert's own evaluation set
            cond_ids = sorted(int(k) for k in preds)
            ys = {}
            for i in cond_ids:
                c = by_id[i]["class"]
                ys[i] = {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2}[c]
            e_pred = {i: preds[str(i)]["pred"] for i in cond_ids}
            y_true = {i: ys[i] for i in cond_ids}
            # expert vs power on the SAME conditional subset
            pw_on = {i: pw_all_ok[i] for i in cond_ids}
            n_fix = sum(1 for i in cond_ids
                        if pw_on[i] != y_true[i] and e_pred[i] == y_true[i])
            n_brake = sum(1 for i in cond_ids
                          if pw_on[i] == y_true[i] and e_pred[i] != y_true[i])
            e_acc = float(np.mean([e_pred[i] == y_true[i] for i in cond_ids]))
            p_acc = float(np.mean([pw_on[i] == y_true[i] for i in cond_ids]))
            ci_pairs = [(by_id[i]["sample_id"],
                          int(e_pred[i] == y_true[i])) for i in cond_ids]
            res["conditional"].append({
                "seed": srow["seed"], "n": len(cond_ids),
                "expert_acc": round(e_acc, 4),
                "power_acc_on_same": round(p_acc, 4),
                "delta_pp": round(100 * (e_acc - p_acc), 2),
                "power_wrong_expert_right": n_fix,
                "power_right_expert_wrong": n_brake,
                "mcnemar_p": round(binom_p_two_sided(min(n_fix, n_brake),
                                                     n_fix + n_brake), 6),
                "expert_confusion": confusion([y_true[i] for i in cond_ids],
                                               [e_pred[i] for i in cond_ids]).tolist(),
                "bootstrap_ci": bootstrap_ci(ci_pairs, seed=srow["seed"]),
            })
            # all-instance: expert where sync-pass, fallback elsewhere
            fam_set = set(fam_ids)
            a_pred, a_true = {}, {}
            for i in fam_ids:
                if i in cond_ids:
                    a_pred[i] = e_pred[i]
                else:
                    a_pred[i] = pw_all_ok[i]
                a_true[i] = ys[i] if i in ys else \
                    {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2}[by_id[i]["class"]]
            acc_all = float(np.mean([a_pred[i] == a_true[i] for i in fam_ids]))
            pw_all = float(np.mean([pw_all_ok[i] == a_true[i] for i in fam_ids]))
            ap = [(by_id[i]["sample_id"], int(a_pred[i] == a_true[i]))
                  for i in fam_ids]
            res["all_instance"].append({
                "seed": srow["seed"], "n": len(fam_ids),
                "expert_with_fallback_acc": round(acc_all, 4),
                "power_all_instance_acc": round(pw_all, 4),
                "delta_pp": round(100 * (acc_all - pw_all), 2),
                "bootstrap_ci": bootstrap_ci(ap, seed=srow["seed"]),
            })
        return res

    out["arms"]["W20_vs_pw012"] = compare("W20", "W20", "pw_012")
    out["arms"]["W40_vs_pw345"] = compare("W40", "W40", "pw_345")
    out["arms"]["SHARED_vs_pwshared"] = compare("SHARED", None, "pw_shared")

    # ---- W-Shared on family subsets vs family experts (same instances) ----
    shared_breakdown = {}
    for fam, pw_arm in (("W20", "pw_012"), ("W40", "pw_345")):
        e = experts["SHARED"]["seeds"]
        fe = experts[fam]["seeds"]
        rows = []
        for s_sh, s_f in zip(e, fe):
            sh_preds = s_sh["metrics"]["instance_predictions"]
            f_preds = s_f["metrics"]["instance_predictions"]
            common = sorted(set(int(k) for k in sh_preds)
                            & set(int(k) for k in f_preds))
            if not common:
                continue
            sh_ok = 0
            f_ok = 0
            for i in common:
                c = by_id[i]["class"]
                y = {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2}[c]
                sh_ok += int(sh_preds[str(i)]["pred"] == y)
                f_ok += int(f_preds[str(i)]["pred"] == y)
            rows.append({
                "seed_pair": [s_sh["seed"], s_f["seed"]],
                "n": len(common),
                "shared_acc": round(sh_ok / len(common), 4),
                "family_acc": round(f_ok / len(common), 4),
            })
        shared_breakdown[fam] = rows
    out["shared_vs_family_experts"] = shared_breakdown

    # ---- costs ----
    cache_secs = cache_pass = None
    t_total = 0.0
    # symbol cache runtime is in the summary: 839 stored but the run
    # processed all 3754 in ~180s
    out["costs"] = {
        "cnn_params": 2454467,
        "frontend_per_instance_sec": round(180.0 / 3754, 3),
        "histogram_per_instance_sec": round(2.0 / 839, 3),
        "cnn_forward_ms": "measured separately (single instance ~1ms-class)",
    }

    (EXP / "wifi_expert_comparison.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("fallback_rule", "sync_availability")},
                     indent=2)[:2000])
    for arm in ("W20_vs_pw012", "W40_vs_pw345", "SHARED_vs_pwshared"):
        c = out["arms"][arm]["conditional"]
        a = out["arms"][arm]["all_instance"]
        print(f"\n{arm}:")
        for r in c:
            print(f"  cond s{r['seed']}: exp {r['expert_acc']:.3f} vs pw "
                  f"{r['power_acc_on_same']:.3f} d={r['delta_pp']:+.1f}pp "
                  f"fix/brake {r['power_wrong_expert_right']}/"
                  f"{r['power_right_expert_wrong']} p={r['mcnemar_p']:.1e}")
        for r in a:
            print(f"  all  s{r['seed']}: exp+fb {r['expert_with_fallback_acc']:.3f} "
                  f"vs pw {r['power_all_instance_acc']:.3f} "
                  f"d={r['delta_pp']:+.1f}pp")
    print(f"\nwrote {EXP / 'wifi_expert_comparison.json'}")


if __name__ == "__main__":
    main()
