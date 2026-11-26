#!/usr/bin/env python3
"""Track Soft-NMS source candidates, then reassign final merged labels."""
from __future__ import annotations
import argparse, json, math, sys
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
        b = torch.from_numpy(boxes[idx]).float()
        s = torch.from_numpy(scores[idx]).float()
        alive = torch.ones(len(b), dtype=torch.bool)
        while bool(alive.any()):
            masked = s.masked_fill(~alive, float('-inf'))
            j = int(torch.argmax(masked))
            kb.append(b[j].numpy()); ks.append(float(s[j])); kl.append(int(cls)); ki.append(int(idx[j]))
            alive[j] = False
            if not bool(alive.any()): break
            ov = box_iou(b[j].unsqueeze(0), b[alive])[0]
            s[alive] = s[alive] * torch.exp(-(ov ** 2) / max(sigma, 1e-6))
            alive &= s >= score_thr
    if not kb:
        return (np.zeros((0,4),np.float32), np.zeros(0,np.float32),
                np.zeros(0,np.int64), np.zeros(0,np.int64))
    boxes_o = np.stack(kb); scores_o = np.asarray(ks); labels_o = np.asarray(kl); src_o = np.asarray(ki)
    order = np.argsort(-scores_o)[:max_det]
    return boxes_o[order], scores_o[order], labels_o[order], src_o[order]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', type=Path, default=Path('/root/autodl-fs/runs_local/dfine_local_iq/val_candidates.npz'))
    ap.add_argument('--crops', type=Path, default=Path('/root/autodl-fs/runs_local/dfine_local_iq/val_local_crops.npz'))
    ap.add_argument('--probs', type=Path, default=Path('/root/autodl-fs/runs_local/dfine_local_iq/val_local_probs.npz'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    with np.load(args.candidates, allow_pickle=True) as zf:
        c = {k: zf[k] for k in ('boxes','scores','labels','sample_offsets','class_scores')}
    with np.load(args.crops, allow_pickle=True) as zf:
        r = {k: zf[k] for k in NEED}
    with np.load(args.probs, allow_pickle=True) as pf:
        pc = {k: pf[k] for k in pf.files}
    meta = json.loads(args.candidates.with_suffix('.json').read_text())
    n = int(meta['samples'])
    records = load_manifest(SPLIT_MANIFEST, split='val')[:n]
    gts = [gt_global(rr, math.ceil(rr.N / (N_TIME - 1))) for rr in records]
    offsets = c['sample_offsets']
    ci = r['candidate_index'].astype(np.int64)

    # candidate index -> (group, local class index, probs row)
    row_of = {int(j): i for i, j in enumerate(ci)}
    group_of = {}
    for g in GROUPS:
        idx = pc[f'idx_{g}'].astype(np.int64)
        cls = pc[f'cls_{g}'].astype(np.int64)
        pr = pc[f'probs_{g}']
        for k, row in enumerate(idx):
            group_of[int(row)] = (g, cls, pr[k])

    merged = []
    for i in range(n):
        a, b = int(offsets[i]), int(offsets[i+1])
        mb, ms, ml, mi = tracked_soft_nms(c['boxes'][a:b], c['scores'][a:b],
                                          c['labels'][a:b].astype(np.int64))
        merged.append({'sample': i, 'boxes': mb, 'scores': ms, 'labels': ml, 'src': mi + a})
    base = [ImagePredictions(m['boxes'], m['scores'], m['labels']) for m in merged]
    bm = _eval_set(gts, base); bm['n_pred'] = int(sum(len(m['boxes']) for m in merged))
    print('baseline', bm['map50'], bm['map50_95'], flush=True)

    # Precompute the local proposal for each merged detection once.
    props = []
    for m in merged:
        for k in range(len(m['labels'])):
            j = int(m['src'][k]); row = row_of.get(j)
            if row is None or row not in group_of:
                props.append(None); continue
            g, cls, pr = group_of[row]
            pred = int(pr[:-1].argmax()); conf = float(pr[:-1].max()); pk = float(pr[-1])
            new = int(cls[pred]); old = int(m['labels'][k])
            p_old = float(c['class_scores'][j, old]); p_new = float(c['class_scores'][j, new])
            props.append((row, old, new, conf, pk, p_old, p_new,
                          p_new / max(p_old + p_new, 1e-9)))
    print('merged', sum(len(m['labels']) for m in merged),
          'with local', sum(p is not None for p in props), flush=True)

    results = {'baseline': bm, 'variants': []}
    flat = []  # (sample, k, old, new, conf, pk, p_old, p_new, ratio)
    for si, m in enumerate(merged):
        for k in range(len(m['labels'])):
            row = row_of.get(int(m['src'][k]))
            if row is None or row not in group_of:
                flat.append(None); continue
            g, cls, pr = group_of[row]
            pred = int(pr[:-1].argmax()); conf = float(pr[:-1].max()); pk = float(pr[-1])
            old = int(m['labels'][k]); new = int(cls[pred])
            p_old = float(c['class_scores'][int(m['src'][k]), old])
            p_new = float(c['class_scores'][int(m['src'][k]), new])
            flat.append((si, k, old, new, conf, pk, p_old, p_new,
                         p_new / max(p_old + p_new, 1e-9)))
    results['variants'] = []
    offset = 0
    # Evaluate each gate combination by rebuilding labels (the merge is fixed).
    for tau, ratio, pold in ((0.9, 0.7, 0.3), (0.9, 0.8, 0.3),
                             (0.9, 0.6, 0.3), (0.95, 0.7, 0.3)):
        if True:
                changed = 0
                by_sample = {}
                for p in flat:
                    if p is None:
                        continue
                    si, k, old, new, conf, pk, po, pn, rr = p
                    if conf > tau and pk < 0.5 and new != old and rr > ratio and po < pold:
                        label = new; changed += 1
                    else:
                        label = old
                    by_sample.setdefault(si, {})[k] = label
                out = []
                for si, m in enumerate(merged):
                    lab = m['labels'].copy()
                    for k, v in by_sample.get(si, {}).items():
                        lab[k] = v
                    out.append(ImagePredictions(m['boxes'], m['scores'], lab))
                mm = _eval_set(gts, out)
                mm['changed'] = changed
                mm['tau'] = tau; mm['ratio'] = ratio; mm['pold'] = pold
                results['variants'].append(mm)
                print(f'tau {tau} ratio {ratio} pold {pold} changed {changed} '
                      f'{mm["map50"]:.6f} {mm["map50_95"]:.6f}', flush=True)
    args.out.write_text(json.dumps(results, indent=2) + '\n')
    print('wrote', args.out)

if __name__ == '__main__':
    main()
