#!/usr/bin/env python3
"""Benchmark the same per-sample Soft-NMS at 1/4/8/16/32 workers."""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'scripts/detector_upgrade'))
from src.signals.data import N_TIME, SPLIT_MANIFEST, load_manifest
from src.signals.infer import Prediction, soft_classwise_nms, gt_global
from src.parallel import parallel_map, cpu_quota

CAND = Path('/root/autodl-fs/runs_local/dfine_local_iq/val_candidates.npz')
_G = {}

def _work(i):
    a, b = int(_G['offsets'][i]), int(_G['offsets'][i+1])
    p = Prediction(torch.from_numpy(_G['boxes'][a:b]).float(),
                   torch.from_numpy(_G['scores'][a:b]).float(),
                   torch.from_numpy(_G['labels'][a:b]).long())
    f = soft_classwise_nms(p, 0.5, 0.5, 1e-3, 100)
    return int(len(f.boxes))

def run(workers):
    t0 = time.perf_counter()
    ns = list(parallel_map(_work, range(_G['n']), workers=workers, chunksize=8))
    return time.perf_counter() - t0, int(sum(ns))

def main():
    with np.load(CAND, allow_pickle=True) as zf:
        c = {k: zf[k] for k in ('boxes','scores','labels','sample_offsets')}
    meta = json.loads(CAND.with_suffix('.json').read_text())
    n = int(meta['samples'])
    _G.update(boxes=c['boxes'], scores=c['scores'], labels=c['labels'],
              offsets=c['sample_offsets'], n=n)
    torch.set_num_threads(1)
    print('quota', cpu_quota(), 'samples', n)
    base = None
    for w in (1, 4, 8, 16, 32):
        dt, npred = run(w)
        if base is None: base = dt
        print(f'workers={w:3d} wall={dt:8.2f}s speedup={base/dt:5.2f}x preds={npred}', flush=True)
    # second 16-worker run after warm-up, to show steady-state
    dt, npred = run(16)
    print(f'workers= 16 warm wall={dt:8.2f}s preds={npred}', flush=True)

if __name__ == '__main__':
    main()
