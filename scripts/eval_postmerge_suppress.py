#!/usr/bin/env python3
"""Suppress final detections whose class the local model confidently rejects."""
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
NEED = ('label','gt_label','gt_iou','keep','score','class_scores','candidate_index',
        'sample_index','boxes','bandwidth_mhz','duration_ms','view_type')

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
        r = {k: zf[k] for k in NEED}
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
        merged.append({'sample': i, 'boxes': mb, 'scores': ms, 'labels': ml, 'src': mi + a})
    base = [ImagePredictions(m['boxes'], m['scores'], m['labels']) for m in merged]
    bm = _eval_set(gts, base); bm['n_pred'] = int(sum(len(m['boxes']) for m in merged))
    print('baseline', bm['map50'], bm['map50_95'], flush=True)

    flat = []
    for si, m in enumerate(merged):
        for k in range(len(m['labels'])):
            j = int(m['src'][k]); row = row_of.get(j)
            if row is None or row not in group_of:
                flat.append(None); continue
            g, cls, pr = group_of[row]
            pred = int(pr[:-1].argmax()); conf = float(pr[:-1].max()); pk = float(pr[-1])
            old = int(m['labels'][k]); new = int(cls[pred])
            p_old = float(c['class_scores'][j, old]); p_new = float(c['class_scores'][j, new])
            flat.append((si, k, old, new, conf, pk, p_old, p_new,
                         p_new / max(p_old + p_new, 1e-9), float(m['scores'][k])))
    results = {'baseline': bm, 'variants': []}
    for tau, ratio, pold in ((0.9,0.7,0.3),(0.9,0.8,0.3),(0.9,0.6,0.3),(0.95,0.7,0.3)):
        for mode, mult in (('remove', 0.0), ('decay', 0.1)):
            by_sample = {}
            changed = 0
            for p in flat:
                if p is None: continue
                si,k,old,new,conf,pk,po,pn,rr,sc = p
                if conf > tau and pk < 0.5 and new != old and rr > ratio and po < pold:
                    by_sample.setdefault(si, {})[k] = mult; changed += 1
            out = []
            for si, m in enumerate(merged):
                mask = np.ones(len(m['scores']), dtype=bool)
                sco = m['scores'].copy()
                for k, mult in by_sample.get(si, {}).items():
                    if mult == 0.0: mask[k] = False
                    else: sco[k] *= mult
                out.append(ImagePredictions(m['boxes'][mask], sco[mask], m['labels'][mask]))
            mm = _eval_set(gts, out)
            mm.update(changed=changed, tau=tau, ratio=ratio, pold=pold, mode=mode)
            results['variants'].append(mm)
            print(f'{mode} tau{tau} r{ratio} p{pold} changed {changed} {mm["map50"]:.6f} {mm["map50_95"]:.6f}', flush=True)
    Path('/root/autodl-fs/runs_local/dfine_local_iq/postmerge_suppress.json').write_text(json.dumps(results, indent=2)+'\n')

if __name__ == '__main__':
    main()
