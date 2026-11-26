#!/usr/bin/env python3
"""Diagnose local-vs-detector class accuracy on final Soft-NMS detections."""
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

    rows = []
    for i in range(n):
        a, b = int(offsets[i]), int(offsets[i+1])
        mb, ms, ml, mi = tracked_soft_nms(c['boxes'][a:b], c['scores'][a:b], c['labels'][a:b].astype(np.int64))
        gb = gts[i].boxes.numpy(); gl = gts[i].labels.numpy()
        for k in range(len(ml)):
            j = int(mi[k] + a); row = row_of.get(j)
            if row is None or row not in group_of: continue
            g, cls, pr = group_of[row]
            pred = int(pr[:-1].argmax()); conf = float(pr[:-1].max()); pk = float(pr[-1]); new = int(cls[pred])
            old = int(ml[k])
            # class-agnostic match to best GT
            if len(gb):
                iou = box_iou(torch.from_numpy(mb[k:k+1]).float(), torch.from_numpy(gb).float())[0].numpy()
                gi = int(np.argmax(iou)); best = float(iou[gi])
            else:
                gi, best = -1, 0.0
            matched = best >= 0.5
            gt = int(gl[gi]) if matched else -1
            p_old = float(c['class_scores'][j, old]); p_new = float(c['class_scores'][j, new])
            rows.append(dict(sample=i, score=float(ms[k]), label=old, new=new,
                             conf=conf, pk=pk, matched=matched, gt=gt,
                             det_ok=matched and old == gt, loc_ok=matched and new == gt,
                             p_old=p_old, p_new=p_new,
                             ratio=p_new / max(p_old + p_new, 1e-9), group=g))
    print('final dets with local', len(rows), 'matched', sum(x['matched'] for x in rows), flush=True)
    for g in GROUPS:
        rs = [x for x in rows if x['group'] == g]
        m = [x for x in rs if x['matched']]
        if not m: continue
        det = np.mean([x['det_ok'] for x in m]); loc = np.mean([x['loc_ok'] for x in m])
        print(f'{g}: final {len(rs)} matched {len(m)} det_acc {det:.4f} loc_acc {loc:.4f}')
        for tau in (0.7,0.8,0.9):
            mm=[x for x in m if x['conf']>tau]
            if not mm: continue
            fix=sum((not x['det_ok']) and x['loc_ok'] for x in mm)
            brk=sum(x['det_ok'] and (not x['loc_ok']) for x in mm)
            print(f'  tau {tau} n {len(mm)} loc {np.mean([x["loc_ok"] for x in mm]):.3f} fix {fix} break {brk} '
                  f'fixScore {sum(x["score"] for x in mm if (not x["det_ok"]) and x["loc_ok"]):.1f} '
                  f'brkScore {sum(x["score"] for x in mm if x["det_ok"] and (not x["loc_ok"])):.1f}')
        for ratio in (0.6,0.7,0.8):
            mm=[x for x in m if x['conf']>0.9 and x['ratio']>ratio and x['p_old']<0.3]
            if not mm: continue
            fix=sum((not x['det_ok']) and x['loc_ok'] for x in mm); brk=sum(x['det_ok'] and (not x['loc_ok']) for x in mm)
            print(f'  r>{ratio} pold<.3: n {len(mm)} fix {fix} break {brk} '
                  f'fixScore {sum(x["score"] for x in mm if (not x["det_ok"]) and x["loc_ok"]):.1f} '
                  f'brkScore {sum(x["score"] for x in mm if x["det_ok"] and (not x["loc_ok"])):.1f}')
    Path('/root/autodl-fs/runs_local/dfine_local_iq/postmerge_diag.json').write_text(json.dumps(rows, indent=1)+'\n')

if __name__ == '__main__':
    main()
