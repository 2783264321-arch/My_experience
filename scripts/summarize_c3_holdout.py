#!/usr/bin/env python3
"""Summarize the frozen-weight 1,159-sample holdout evaluation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [
    ("s1729 best", "c3_pilot_dense", "c3_pilot_zero3_s1729", "best", "s1729_best"),
    ("s1729 last", "c3_pilot_dense", "c3_pilot_zero3_s1729", "last", "s1729_last"),
    ("s1730 best", "c3_pilot_dense_s1730", "c3_pilot_zero3_s1730", "best", "s1730_best"),
    ("s1730 last", "c3_pilot_dense_s1730", "c3_pilot_zero3_s1730", "last", "s1730_last"),
]

def load(p):
    p=Path(p)
    return json.loads(p.read_text()) if p.exists() else None

def fmt(x, nd=4):
    if x is None: return "n/a"
    if isinstance(x,int): return str(x)
    return f"{float(x):.{nd}f}"

def main():
    rows=[]
    for label, dr, zr, ck, tag in PAIRS:
        d=load(ROOT/f"runs/{dr}/eval_holdout_{ck}.json")
        z=load(ROOT/f"runs/{zr}/eval_holdout_{ck}.json")
        m=load(ROOT/f"reports/c3_holdout_matched_{tag}.json")
        rows.append({"label":label,"dense":d,"zero3":z,"matched":m,
                     "dr":dr,"zr":zr,"ck":ck,"tag":tag})
    lines=["# C3 holdout replication (1,159 samples)", "",
           "Frozen best/last checkpoints; no re-selection or threshold tuning.",
           "Holdout = full val 1,543 minus the 384-sample C3 subset.", "",
           "## Overall / groups", "",
           "| pair | model | mAP | AP_agn | R_agn | 0/1/2 | 3/4/5 | 10/11 | 9/12/13 | n_pred |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        for side,key in [("dense","dense"),("zero3","zero3")]:
            e=r[key]
            if not e:
                lines.append(f"| {r['label']} | {side} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            g=e.get("groups",{})
            lines.append(f"| {r['label']} | {side} | {fmt(e.get('mAP50'))} | "
                         f"{fmt(e.get('map50_agnostic'))} | {fmt(e.get('localization_recall'))} | "
                         f"{fmt(g.get('0/1/2',{}).get('map50'))} | {fmt(g.get('3/4/5',{}).get('map50'))} | "
                         f"{fmt(g.get('10/11',{}).get('map50'))} | {fmt(g.get('9/12/13',{}).get('map50'))} | "
                         f"{e.get('n_pred','n/a')} |")
    lines += ["", "## Common-box 3/4/5 classification", "",
              "| pair | common n | dense acc | zero3 acc | Δ | dense R | zero3 R | dense-only correct | zero3-only correct | McNemar p |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        m=r["matched"]
        if not m:
            lines.append(f"| {r['label']} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        mc=m.get("mcnemar",{})
        lines.append(f"| {r['label']} | {m['common']} | {fmt(m['common_accuracy_a'])} | "
                     f"{fmt(m['common_accuracy_b'])} | "
                     f"{fmt((m['common_accuracy_a']-m['common_accuracy_b'])*100,2)} pp | "
                     f"{fmt(m['recall_a'])} | {fmt(m['recall_b'])} | "
                     f"{mc.get('dense_only_correct','n/a')} | {mc.get('zero3_only_correct','n/a')} | "
                     f"{fmt(mc.get('p_two_sided'),3)} |")
    # means over best/last
    def mean(metric_fn, side):
        vals=[metric_fn(r[side]) for r in rows if r[side]]
        return float(np.mean(vals)) if vals else None
    lines += ["", "## Means", "",
              "| metric | dense mean | zero3 mean | Δ |", "| --- | ---: | ---: | ---: |"]
    for name, fn in [
        ("mAP", lambda e:e.get("mAP50")),
        ("0/1/2", lambda e:e.get("groups",{}).get("0/1/2",{}).get("map50")),
        ("3/4/5", lambda e:e.get("groups",{}).get("3/4/5",{}).get("map50")),
        ("10/11", lambda e:e.get("groups",{}).get("10/11",{}).get("map50")),
        ("9/12/13", lambda e:e.get("groups",{}).get("9/12/13",{}).get("map50")),
        ("R_agn", lambda e:e.get("localization_recall")),
    ]:
        d=mean(fn,"dense"); z=mean(fn,"zero3")
        lines.append(f"| {name} | {fmt(d)} | {fmt(z)} | {fmt((d-z)*100,2) if d is not None and z is not None else 'n/a'} pp |")
    lines += ["", "## Notes", "",
              "- Common boxes require both models to match the same 3/4/5 GT at IoU>=0.5.",
              "- McNemar is exact two-sided on discordant common boxes.",
              "- No weight selection or threshold tuning on the holdout.", ""]
    out=ROOT/"reports/c3_holdout_replication.md"
    out.write_text("\n".join(lines)+"\n")
    (ROOT/"reports/c3_holdout_replication.json").write_text(json.dumps({"rows":rows},indent=2))
    print("\n".join(lines))
    print("wrote",out)

if __name__=="__main__": main()
