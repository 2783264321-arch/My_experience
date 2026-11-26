#!/usr/bin/env python3
"""Apply the Wi-Fi expert high-score suppression to a fixed candidate cache.

For val: merge -> suppress -> report baseline vs variant AP.
For test: merge -> suppress -> save filtered predictions for the official
evaluator (test is only run at the final locked evaluation).
"""
from __future__ import annotations
import argparse, json, math, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'scripts/detector_upgrade'))
from src.signals.data import N_TIME, SPLIT_MANIFEST, load_manifest
from src.signals.infer import Prediction, soft_classwise_nms, global_classwise_nms, gt_global
from src.signals.metrics import ImagePredictions
from src.wifi_expert_suppress import suppress, DEFAULT_WEIGHTS
from validate_dual import _eval_set

_G = {}

def _init(): torch.set_num_threads(1)

def _merge(i):
    a, b = int(_G['offsets'][i]), int(_G['offsets'][i+1])
    p = Prediction(torch.from_numpy(_G['boxes'][a:b]).float(),
                   torch.from_numpy(_G['scores'][a:b]).float(),
                   torch.from_numpy(_G['labels'][a:b]).long())
    f = (soft_classwise_nms(p, 0.5, 0.5, 1e-3, 100) if _G['merge'] == 'soft'
         else global_classwise_nms(p, 0.5, 100))
    return ImagePredictions(f.boxes.numpy(), f.scores.numpy(), f.labels.numpy())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', type=Path,
                    default=Path('/root/autodl-fs/runs_local/dfine_local_iq/val_candidates.npz'))
    ap.add_argument('--split', choices=('val', 'test'), default='val')
    ap.add_argument('--merge', choices=('soft', 'nms'), default='soft')
    ap.add_argument('--score-thr', type=float, default=0.3)
    ap.add_argument('--conf-thr', type=float, default=0.9)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument('--expert-arm', choices=('shared','family'), default='shared')
    ap.add_argument('--window-mode', choices=('center','multi'), default='center')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--save-preds', type=Path, default=None)
    ap.add_argument('--skip-suppress', action='store_true')
    args = ap.parse_args()

    with np.load(args.candidates, allow_pickle=True) as zf:
        c = {k: zf[k] for k in ('boxes', 'scores', 'labels', 'sample_offsets')}
    meta = json.loads(args.candidates.with_suffix('.json').read_text())
    n = int(meta['samples'])
    records = load_manifest(SPLIT_MANIFEST, split=args.split)[:n]
    if len(records) != n:
        raise SystemExit(f'{len(records)} records vs {n} candidate samples')
    _G.update(boxes=c['boxes'], scores=c['scores'], labels=c['labels'],
              offsets=c['sample_offsets'], merge=args.merge)
    preds = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as ex:
        for p in ex.map(_merge, range(n), chunksize=8):
            preds.append(p)
    print('merged', sum(len(p.boxes) for p in preds), 'detections', flush=True)

    out = {'split': args.split, 'merge': args.merge,
           'score_thr': args.score_thr, 'conf_thr': args.conf_thr,
           'weights': str(args.weights)}
    gts = [gt_global(r, math.ceil(r.N / (N_TIME - 1))) for r in records]
    base = _eval_set(gts, preds)
    print('baseline', base['map50'], base['map50_95'], flush=True)
    out['baseline'] = base
    if args.skip_suppress:
        filtered, stats = list(preds), {'skipped': True}
    else:
        filtered, stats = suppress(records, preds, score_thr=args.score_thr,
                                   conf_thr=args.conf_thr, workers=args.workers,
                                   weights_dir=args.weights,
                                   expert_arm=args.expert_arm,
                                   window_mode=args.window_mode)
    print('suppress stats', json.dumps(stats), flush=True)
    out['stats'] = stats
    if args.skip_suppress:
        var = dict(base)
    else:
        var = _eval_set(gts, filtered)
    print('variant', var['map50'], var['map50_95'], flush=True)
    out['variant'] = var
    out['delta_map50'] = var['map50'] - base['map50']
    out['delta_map50_95'] = var['map50_95'] - base['map50_95']
    if args.save_preds:
        flat_b, flat_s, flat_l, offs = [], [], [], [0]
        for p in filtered:
            flat_b.append(p.boxes); flat_s.append(p.scores); flat_l.append(p.labels)
            offs.append(offs[-1] + len(p.boxes))
        np.savez(args.save_preds,
                 boxes=np.concatenate(flat_b) if flat_b else np.zeros((0, 4), np.float32),
                 scores=np.concatenate(flat_s) if flat_s else np.zeros(0, np.float32),
                 labels=np.concatenate(flat_l) if flat_l else np.zeros(0, np.int64),
                 sample_offsets=np.asarray(offs, np.int64),
                 sample_ids=np.asarray(meta.get('sample_ids', [])))
        print('saved preds', args.save_preds, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print('wrote', args.out)

if __name__ == '__main__':
    main()
