#!/usr/bin/env python3
"""Sweep expert-confidence thresholds for high-score Wi-Fi FP suppression."""
from __future__ import annotations
import json, math, sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
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

CAND = Path('/root/autodl-fs/runs_local/dfine_local_iq/val_candidates.npz')
PILOT = Path('/root/autodl-fs/runs_local/dfine_local_iq/pilot_wifi_suppress.json')
_G = {}

def _init(): torch.set_num_threads(1)

def _merge_sample(i):
    a, b = int(_G['offsets'][i]), int(_G['offsets'][i+1])
    p = Prediction(torch.from_numpy(_G['boxes'][a:b]).float(),
                   torch.from_numpy(_G['scores'][a:b]).float(),
                   torch.from_numpy(_G['labels'][a:b]).long())
    f = soft_classwise_nms(p, 0.5, 0.5, 1e-3, 100)
    return f.boxes.numpy().astype(np.float32), f.scores.numpy().astype(np.float32), f.labels.numpy().astype(np.int64)

def _eval_thr(thr):
    drop = {(r['si'], r['k']) for r in _G['rows']
            if r['disagree'] and r['expert_conf'] >= thr}
    preds = []
    for si, (mb, ms, ml) in enumerate(_G['merged']):
        keep = [k for k in range(len(ml)) if (si, k) not in drop]
        preds.append(ImagePredictions(mb[keep], ms[keep], ml[keep]))
    m = _eval_set(_G['gts'], preds)
    m['n_pred'] = int(sum(len(p.boxes) for p in preds)); m['dropped'] = len(drop)
    return m

def _init_child(merged, gts, rows):
    torch.set_num_threads(1)
    _G['merged']=merged; _G['gts']=gts; _G['rows']=rows

def main():
    with np.load(CAND, allow_pickle=True) as zf:
        c = {k: zf[k] for k in ('boxes','scores','labels','sample_offsets')}
    meta = json.loads(CAND.with_suffix('.json').read_text()); n = int(meta['samples'])
    records = load_manifest(SPLIT_MANIFEST, split='val')[:n]
    gts = [gt_global(r, math.ceil(r.N/(N_TIME-1))) for r in records]
    _G.update(boxes=c['boxes'], scores=c['scores'], labels=c['labels'], offsets=c['sample_offsets'])
    merged = []
    with ProcessPoolExecutor(max_workers=16, initializer=_init) as ex:
        for r in ex.map(_merge_sample, range(n), chunksize=8): merged.append(r)
    rows = json.loads(PILOT.read_text())['rows']
    thresholds = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    # baseline
    import copy
    g0 = {'merged':merged,'gts':gts,'rows':[]}
    _init_child(merged, gts, [])
    base = _eval_thr(0.0)
    print('baseline', base['map50'], base['map50_95'], 'n_pred', base['n_pred'], flush=True)
    out = {'baseline': base, 'sweep': []}
    _init_child(merged, gts, rows)
    with ProcessPoolExecutor(max_workers=len(thresholds),
                             mp_context=mp.get_context('fork')) as ex:
        for thr, m in zip(thresholds, ex.map(_eval_thr, thresholds)):
            out['sweep'].append(m)
            print(f'thr {thr}: {m["map50"]:.6f} {m["map50_95"]:.6f} dropped {m["dropped"]}', flush=True)
    Path('/root/autodl-fs/runs_local/dfine_local_iq/wifi_expert_sweep.json').write_text(json.dumps(out, indent=2)+'\n')

if __name__ == '__main__':
    main()
