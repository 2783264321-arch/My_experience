#!/usr/bin/env python3
"""Reassign/add local-class detections with detector-local arbitration gates."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'scripts/detector_upgrade'))
from src.signals.data import N_TIME, SPLIT_MANIFEST, load_manifest
from src.signals.infer import Prediction, soft_classwise_nms, gt_global
from src.signals.metrics import ImagePredictions
from validate_dual import _eval_set

GROUPS = {"012":[0,1,2], "345":[3,4,5], "1011":[10,11]}
NEED = ('label','gt_label','gt_iou','keep','score','class_scores','candidate_index',
        'sample_index','boxes','bandwidth_mhz','duration_ms','view_type')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', type=Path, default=Path('/root/autodl-fs/runs_local/dfine_local_iq/val_candidates.npz'))
    ap.add_argument('--crops', type=Path, default=Path('/root/autodl-fs/runs_local/dfine_local_iq/val_local_crops.npz'))
    ap.add_argument('--probs', type=Path, default=Path('/root/autodl-fs/runs_local/dfine_local_iq/val_local_probs.npz'))
    ap.add_argument('--mode', choices=('replace','add'), default='replace')
    ap.add_argument('--tau', type=float, default=0.9)
    ap.add_argument('--keep-gate', type=float, default=0.5)
    ap.add_argument('--ratio-min', type=float, default=0.0)
    ap.add_argument('--pold-max', type=float, default=1.0)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--selected-groups', default='012,345,1011')
    ap.add_argument('--skip-baseline', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    with np.load(args.candidates, allow_pickle=True) as zf:
        c = {k: zf[k] for k in ('boxes','scores','labels','sample_offsets')}
    with np.load(args.crops, allow_pickle=True) as zf:
        r = {k: zf[k] for k in NEED}
    with np.load(args.probs, allow_pickle=True) as pf:
        probs_cache = {k: pf[k] for k in pf.files}
    meta = json.loads(args.candidates.with_suffix('.json').read_text())
    n = int(meta['samples'])
    records = load_manifest(SPLIT_MANIFEST, split='val')[:n]
    gts = [gt_global(rr, math.ceil(rr.N / (N_TIME - 1))) for rr in records]
    offsets = c['sample_offsets']
    ci = r['candidate_index'].astype(np.int64)
    orig_labels = c['labels'].astype(np.int64)
    labels = orig_labels.copy()
    boxes = c['boxes'].astype(np.float32)
    scores = c['scores'].astype(np.float32)

    upd = {}
    for g in args.selected_groups.split(','):
        if g not in GROUPS: continue
        idx = probs_cache[f'idx_{g}'].astype(np.int64)
        cls = probs_cache[f'cls_{g}'].astype(np.int64)
        pr = probs_cache[f'probs_{g}']
        pred = pr[:, :-1].argmax(1); conf = pr[:, :-1].max(1); pk = pr[:, -1]
        new = cls[pred]
        old = r['label'][idx].astype(np.int64)
        p_old = r['class_scores'][idx][np.arange(len(idx)), old]
        p_new = r['class_scores'][idx][np.arange(len(idx)), new]
        ratio = p_new / np.maximum(p_old + p_new, 1e-9)
        m = (conf > args.tau) & (pk < args.keep_gate) & (new != old) & \
            (ratio > args.ratio_min) & (p_old < args.pold_max)
        for row, nl, cc in zip(idx[m], new[m], conf[m]):
            j = int(ci[row])
            if j not in upd or cc > upd[j][1]:
                upd[j] = (int(nl), float(cc))
    print('updates', len(upd), 'mode', args.mode, 'tau', args.tau,
          'ratio', args.ratio_min, 'pold', args.pold_max, flush=True)

    def metrics_for(extra=None):
        out = []
        for i in range(n):
            a, b = int(offsets[i]), int(offsets[i+1])
            box = boxes[a:b]; sco = scores[a:b]; lab = labels[a:b]
            if extra and i in extra:
                eb, es, el = extra[i]
                box = np.concatenate([box, eb], 0)
                sco = np.concatenate([sco, es], 0)
                lab = np.concatenate([lab, el], 0)
            p = Prediction(torch.from_numpy(box).float(), torch.from_numpy(sco).float(),
                           torch.from_numpy(lab).long())
            f = soft_classwise_nms(p, 0.5, 0.5, 1e-3, 100)
            out.append(ImagePredictions(f.boxes.numpy(), f.scores.numpy(), f.labels.numpy()))
        m = _eval_set(gts, out)
        m['n_pred'] = int(sum(len(p.boxes) for p in out))
        return m

    res = {'mode': args.mode, 'tau': args.tau, 'ratio_min': args.ratio_min,
           'pold_max': args.pold_max, 'keep_gate': args.keep_gate,
           'selected_groups': args.selected_groups, 'updates': len(upd)}
    if not args.skip_baseline:
        res['baseline'] = metrics_for()
        print('baseline', res['baseline']['map50'], res['baseline']['map50_95'], flush=True)
    if args.mode == 'replace':
        for j, (nl, cc) in upd.items(): labels[j] = nl
        res['variant'] = metrics_for()
    else:
        extra = {}
        for j, (nl, cc) in upd.items():
            si = int(np.searchsorted(offsets, j, side='right') - 1)
            eb, es, el = extra.setdefault(si, ([], [], []))
            eb.append(boxes[j]); es.append(float(scores[j]) * args.alpha); el.append(nl)
        extra = {k: (np.asarray(v[0], np.float32), np.asarray(v[1], np.float32),
                     np.asarray(v[2], np.int64)) for k, v in extra.items()}
        res['variant'] = metrics_for(extra)
        res['extra_dets'] = int(sum(len(v[0]) for v in extra.values()))
    print('variant', res['variant']['map50'], res['variant']['map50_95'], flush=True)
    args.out.write_text(json.dumps(res, indent=2) + '\n')
    print('wrote', args.out)

if __name__ == '__main__':
    main()
