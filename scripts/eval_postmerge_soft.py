#!/usr/bin/env python3
"""Soft multi-class re-scoring of final detections with local class probs.

For a final detection (box, score s, label old), split s across the group
classes: old gets s*((1-beta)+beta*p_loc[old]), each other class gets
s*beta*p_loc[c].  beta=0 reproduces the baseline; beta=1 trusts the local
class distribution.  Top-100 per sample is applied after splitting.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import torch
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'scripts/detector_upgrade'))
from src.signals.data import N_TIME, SPLIT_MANIFEST, load_manifest
from src.signals.infer import gt_global
from src.signals.metrics import ImagePredictions
from validate_dual import _eval_set

GROUPS = {"012":[0,1,2], "345":[3,4,5], "1011":[10,11]}

def tracked_soft_nms(boxes, scores, labels, iou=0.5, sigma=0.5, score_thr=1e-3, max_det=100):
    kb, ks, kl, ki = [], [], [], []
    for cls in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == cls)
        b = torch.from_numpy(boxes[idx]).float(); s = torch.from_numpy(scores[idx]).float()
        alive = torch.ones(len(b), dtype=torch.bool)
        while bool(alive.any()):
            j = int(torch.argmax(s.masked_fill(~alive, float('-inf'))))
            kb.append(b[j].numpy()); ks.append(float(s[j])); kl.append(int(cls)); ki.append(int(idx[j]))
            alive[j] = False
            if not bool(alive.any()): break
            ov = box_iou(b[j].unsqueeze(0), b[alive])[0]
            s[alive] = s[alive] * torch.exp(-(ov ** 2) / max(sigma, 1e-6))
            alive &= s >= score_thr
    if not kb:
        return (np.zeros((0,4),np.float32), np.zeros(0,np.float32), np.zeros(0,np.int64), np.zeros(0,np.int64))
    bo = np.stack(kb); so = np.asarray(ks); lo = np.asarray(kl); io = np.asarray(ki)
    o = np.argsort(-so)[:max_det]
    return bo[o], so[o], lo[o], io[o]

def main():
    with np.load('/root/autodl-fs/runs_local/dfine_local_iq/val_candidates.npz', allow_pickle=True) as zf:
        c = {k: zf[k] for k in ('boxes','scores','labels','sample_offsets','class_scores')}
    with np.load('/root/autodl-fs/runs_local/dfine_local_iq/val_local_crops.npz', allow_pickle=True) as zf:
        r = {k: zf[k] for k in ('candidate_index',)}
    with np.load('/root/autodl-fs/runs_local/dfine_local_iq/val_local_probs.npz', allow_pickle=True) as pf:
        pc = {k: pf[k] for k in pf.files}
    meta = json.loads(Path('/root/autodl-fs/runs_local/dfine_local_iq/val_candidates.json').read_text())
    n = int(meta['samples'])
    records = load_manifest(SPLIT_MANIFEST, split='val')[:n]
    gts = [gt_global(rr, math.ceil(rr.N / (N_TIME - 1))) for rr in records]
    offsets = c['sample_offsets']; ci = r['candidate_index'].astype(np.int64)
    row_of = {int(j): i for i, j in enumerate(ci)}
    group_of = {}
    for g in GROUPS:
        idx = pc[f'idx_{g}'].astype(np.int64); cls = pc[f'cls_{g}'].astype(np.int64); pr = pc[f'probs_{g}']
        for k, row in enumerate(idx): group_of[int(row)] = (g, cls, pr[k])

    merged = []
    for i in range(n):
        a, b = int(offsets[i]), int(offsets[i+1])
        mb, ms, ml, mi = tracked_soft_nms(c['boxes'][a:b], c['scores'][a:b], c['labels'][a:b].astype(np.int64))
        merged.append({'boxes': mb, 'scores': ms, 'labels': ml, 'src': mi + a})
    base = [ImagePredictions(m['boxes'], m['scores'], m['labels']) for m in merged]
    bm = _eval_set(gts, base); bm['n_pred'] = int(sum(len(m['boxes']) for m in merged))
    print('baseline', bm['map50'], bm['map50_95'], flush=True)

    results = {'baseline': bm, 'variants': []}
    for beta in (0.5, 0.75, 1.0):
        for sel in ('wifi', 'all'):
            preds = []
            for m in merged:
                nb, ns, nl = [], [], []
                for k in range(len(m['labels'])):
                    j = int(m['src'][k]); row = row_of.get(j)
                    old = int(m['labels'][k]); s = float(m['scores'][k]); box = m['boxes'][k]
                    use = row is not None and row in group_of
                    if use:
                        g, cls, pr = group_of[row]
                        if sel == 'wifi' and g not in ('012','345'): use = False
                        if sel == 'srrc' and g != '1011': use = False
                    if not use:
                        nb.append(box); ns.append(s); nl.append(old); continue
                    ploc = pr[:-1].astype(np.float64); ploc = ploc / max(ploc.sum(), 1e-12)
                    jold = int(np.where(cls == old)[0][0]) if old in cls else -1
                    nb.append(box); ns.append(s * ((1-beta) + beta * (ploc[jold] if jold >= 0 else 0.0))); nl.append(old)
                    for ci_, cglob in enumerate(cls):
                        if int(cglob) == old: continue
                        if ploc[ci_] * beta * s < 1e-4: continue
                        nb.append(box); ns.append(float(s * beta * ploc[ci_])); nl.append(int(cglob))
                nb = np.asarray(nb, np.float32); ns = np.asarray(ns, np.float32); nl = np.asarray(nl, np.int64)
                if len(ns) > 100:
                    o = np.argsort(-ns)[:100]; nb, ns, nl = nb[o], ns[o], nl[o]
                preds.append(ImagePredictions(nb, ns, nl))
            mm = _eval_set(gts, preds)
            mm.update(beta=beta, sel=sel)
            results['variants'].append(mm)
            print(f'beta {beta} {sel}: {mm["map50"]:.6f} {mm["map50_95"]:.6f}', flush=True)
    Path('/root/autodl-fs/runs_local/dfine_local_iq/postmerge_soft.json').write_text(json.dumps(results, indent=2)+'\n')
    print('wrote soft')

if __name__ == '__main__':
    main()
