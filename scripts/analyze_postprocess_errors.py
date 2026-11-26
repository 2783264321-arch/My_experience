#!/usr/bin/env python3
"""Per-GT error decomposition for fixed candidates + NMS/Soft-NMS/WBF.

For every val GT: does the candidate pool contain a correct-class box at
IoU>=0.5?  After merging, is the GT correct, wrong-class, or missed?  What
are the FP sources (background / duplicate / wrong-class)?  Compare NMS vs
Soft-NMS vs WBF on the same GTs.
"""
from __future__ import annotations
import argparse, json, math, os, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'scripts/detector_upgrade'))
from src.signals.data import N_TIME, SPLIT_MANIFEST, load_manifest
from src.signals.infer import (Prediction, soft_classwise_nms,
                               global_classwise_nms, gt_global)

GROUPS = {"WiFi20": [0,1,2], "WiFi40": [3,4,5], "SRRC": [10,11],
          "Narrow": [6,7,8,9,12,13]}
CAND = Path('/root/autodl-fs/runs_local/dfine_local_iq/val_candidates.npz')
_G = {}

def _group(c):
    for g, cls in GROUPS.items():
        if c in cls: return g
    return 'other'

def _merge(boxes, scores, labels, mode):
    p = Prediction(torch.from_numpy(boxes).float(),
                   torch.from_numpy(scores).float(),
                   torch.from_numpy(labels).long())
    if mode == 'nms':
        f = global_classwise_nms(p, 0.5, 100)
    elif mode == 'soft':
        f = soft_classwise_nms(p, 0.5, 0.5, 1e-3, 100)
    elif mode == 'wbf':
        from eval_sahi_post import wbf
        bb, ss, ll = wbf(boxes, scores, labels, 0.55, 100)
        return bb, ss, ll
    else:
        raise ValueError(mode)
    return f.boxes.numpy(), f.scores.numpy(), f.labels.numpy()

def _sample(i, modes):
    a, b = int(_G['offsets'][i]), int(_G['offsets'][i+1])
    cb = _G['boxes'][a:b]; cs = _G['scores'][a:b]; cl = _G['labels'][a:b].astype(np.int64)
    gb = _G['gts'][i].boxes.numpy(); gl = _G['gts'][i].labels.numpy().astype(np.int64)
    out = {'sample': i, 'n_gt': int(len(gl)), 'modes': {}}
    if len(gb) == 0:
        return out
    iou_cg = box_iou(torch.from_numpy(cb).float(), torch.from_numpy(gb).float()).numpy()
    for mode in modes:
        fb, fs, fl = _merge(cb, cs, cl, mode)
        iou_fg = box_iou(torch.from_numpy(fb).float(), torch.from_numpy(gb).float()).numpy()
        # greedy one-to-one matching by score
        order = np.argsort(-fs)
        matched = -np.ones(len(fs), dtype=np.int64)
        used = np.zeros(len(gb), dtype=bool)
        for d in order:
            cand = np.flatnonzero((iou_fg[d] >= 0.5) & (~used))
            if cand.size:
                gi = int(cand[np.argmax(iou_fg[d, cand])])
                matched[d] = gi; used[gi] = True
        # per-GT rows
        gts_rows = []
        for gi in range(len(gl)):
            fm = np.flatnonzero(iou_fg[:, gi] >= 0.5)
            fc = fm[fl[fm] == gl[gi]] if fm.size else fm
            if matched.tolist().count(gi):
                outcome = 'correct'
            elif fc.size:
                outcome = 'unmatched_correct'
            elif fm.size:
                outcome = 'wrong'
            else:
                outcome = 'missed'
            cm = np.flatnonzero(iou_cg[:, gi] >= 0.5)
            cc = cm[cl[cm] == gl[gi]] if cm.size else cm
            gts_rows.append({
                'gi': gi, 'cls': int(gl[gi]), 'group': _group(int(gl[gi])),
                'outcome': outcome,
                'best_correct_score': float(fs[fc].max()) if fc.size else None,
                'best_any_score': float(fs[fm].max()) if fm.size else None,
                'n_dets_any': int(fm.size), 'n_dets_correct': int(fc.size),
                'n_cands': int(cm.size), 'n_cands_correct': int(cc.size),
                'best_cand_correct_score': float(cs[cc].max()) if cc.size else None,
                'best_cand_any_score': float(cs[cm].max()) if cm.size else None,
                'max_iou_cand': float(iou_cg[cm, gi].max()) if cm.size else None,
            })
        # FP rows: detections not matched one-to-one
        fp = Counter(); fp_score = defaultdict(float)
        for d in range(len(fs)):
            if matched[d] >= 0:
                continue
            if len(gb):
                gi = int(np.argmax(iou_fg[d])); best = float(iou_fg[d, gi])
            else:
                gi, best = -1, 0.0
            if best < 0.5:
                src = 'background'
            elif fl[d] == gl[gi]:
                src = 'duplicate'
            else:
                src = 'wrong_class'
            key = (src, int(fl[d]), _group(int(fl[d])))
            fp[key] += 1; fp_score[key] += float(fs[d])
        out['modes'][mode] = {
            'gts': gts_rows,
            'fp': [{'src': k[0], 'cls': k[1], 'group': k[2],
                    'n': int(v), 'score_sum': float(fp_score[k])}
                   for k, v in fp.items()],
            'n_pred': int(len(fs)),
        }
    return out

def _init():
    torch.set_num_threads(1)
    np.seterr(all='ignore')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--modes', default='nms,soft,wbf')
    args = ap.parse_args()
    modes = args.modes.split(',')
    with np.load(CAND, allow_pickle=True) as zf:
        c = {k: zf[k] for k in ('boxes','scores','labels','sample_offsets')}
    meta = json.loads(CAND.with_suffix('.json').read_text())
    n = int(meta['samples'])
    records = load_manifest(SPLIT_MANIFEST, split='val')[:n]
    gts = [gt_global(r, math.ceil(r.N / (N_TIME - 1))) for r in records]
    _G['boxes'] = c['boxes']; _G['scores'] = c['scores']
    _G['labels'] = c['labels']; _G['offsets'] = c['sample_offsets']; _G['gts'] = gts

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as ex:
        for r in ex.map(_sample, range(n), [modes]*n, chunksize=8):
            rows.append(r)
    payload = {'modes': modes, 'samples': n, 'rows': rows}
    args.out.write_text(json.dumps(payload) + '\n')
    print('wrote', args.out, 'rows', len(rows))

if __name__ == '__main__':
    main()
