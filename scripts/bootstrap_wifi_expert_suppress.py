#!/usr/bin/env python3
"""Fast paired image-level bootstrap for the Wi-Fi expert suppression gain.

Precomputes per-image/class/IoU TP flags, then bootstraps images with
multinomial counts (equivalent to resampling images with replacement).
"""
from __future__ import annotations
import argparse, json, math, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing as mp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'scripts/detector_upgrade'))
from src.signals.data import N_TIME, SPLIT_MANIFEST, load_manifest
from src.signals.infer import Prediction, soft_classwise_nms, gt_global
from src.signals.metrics import ImageTargets, ImagePredictions, _match_single

CAND = Path('/root/autodl-fs/runs_local/dfine_local_iq/val_candidates.npz')
PILOT = Path('/root/autodl-fs/runs_local/dfine_local_iq/pilot_wifi_suppress.json')
IOUS = tuple(np.round(np.arange(0.5, 1.0, 0.05), 2))
_G = {}

def _init(): torch.set_num_threads(1)

def _merge_sample(i):
    a, b = int(_G['offsets'][i]), int(_G['offsets'][i+1])
    p = Prediction(torch.from_numpy(_G['boxes'][a:b]).float(),
                   torch.from_numpy(_G['scores'][a:b]).float(),
                   torch.from_numpy(_G['labels'][a:b]).long())
    f = soft_classwise_nms(p, 0.5, 0.5, 1e-3, 100)
    return f.boxes.numpy().astype(np.float32), f.scores.numpy().astype(np.float32), f.labels.numpy().astype(np.int64)

def _ap101(scores, tp, keep, img, gt_counts, cnt):
    w = cnt[img] * keep
    tpw = tp * w
    fpw = (1.0 - tp) * w
    n_gt = float((gt_counts * cnt).sum())
    if n_gt <= 0:
        return None
    if len(scores) == 0:
        return 0.0
    tp_cum = np.cumsum(tpw)
    fp_cum = np.cumsum(fpw)
    rec = tp_cum / n_gt
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    suffix = np.maximum.accumulate(prec[::-1])[::-1]
    ts = np.linspace(0.0, 1.0, 101)
    idx = np.searchsorted(rec, ts, side='left')
    vals = np.where(idx < len(prec), suffix[np.minimum(idx, len(prec)-1)], 0.0)
    return float(vals.mean())

def _map_for(entries, cnt):
    # entries: dict (cls, thr) -> (scores, tp, keep, img, gt_counts)
    out = {}
    for thr in IOUS:
        aps = []
        for cls in range(14):
            e = entries.get((cls, thr))
            if e is None: continue
            ap = _ap101(*e, cnt)
            if ap is not None: aps.append(ap)
        out[thr] = float(np.mean(aps)) if aps else None
    return out

def _run_task(a):
    return _boot_chunk(*a)

def _boot_chunk(seed_b, n_boot, n):
    rng = np.random.default_rng(seed_b)
    res = []
    for _ in range(n_boot):
        cnt = rng.multinomial(n, np.full(n, 1.0/n)).astype(np.float32)
        b = _map_for(_G['base'], cnt)
        v = _map_for(_G['var'], cnt)
        res.append((b[0.5], np.mean([b[t] for t in IOUS if b[t] is not None]),
                    v[0.5], np.mean([v[t] for t in IOUS if v[t] is not None])))
    return res

def _init_child(d):
    torch.set_num_threads(1)
    _G.update(d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=200)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--conf', type=float, default=0.9)
    ap.add_argument('--out', type=Path, default=Path('/root/autodl-fs/runs_local/dfine_local_iq/wifi_expert_bootstrap.json'))
    args = ap.parse_args()
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
    drop = {(r['si'], r['k']) for r in rows if r['disagree'] and r['expert_conf'] >= args.conf}
    print('dropped', len(drop), flush=True)

    # Precompute per-class/thr entries.
    base = {}; var = {}
    gt_counts = {cls: np.zeros(n, np.float32) for cls in range(14)}
    for cls in range(14):
        for i in range(n):
            gt_counts[cls][i] = float((gts[i].labels == cls).sum())
    t0 = __import__('time').perf_counter()
    for i in range(n):
        mb, ms, ml = merged[i]
        gt = ImageTargets(np.asarray(gts[i].boxes), np.asarray(gts[i].labels))
        for cls in range(14):
            pidx = np.flatnonzero(ml == cls)
            if pidx.size == 0: continue
            pred = ImagePredictions(mb[pidx], ms[pidx], ml[pidx])
            for thr in IOUS:
                matches, unmatched = _match_single(gt, pred, thr, class_aware=True, class_id=cls, max_dets=100)
                mpred = {m[0] for m in matches}
                order = np.argsort(-pred.scores, kind='stable')[:100]
                sc = pred.scores[order].astype(np.float32)
                tp = np.isin(order, list(mpred)).astype(np.float32)
                keep = np.array([0.0 if (i, int(pidx[k])) in drop else 1.0 for k in order], np.float32)
                img = np.full(len(sc), i, np.int32)
                for D in (base, var):
                    e = D.setdefault((cls, thr), ([], [], [], []))
                    e[0].append(sc); e[1].append(tp); e[2].append(keep); e[3].append(img)
    print('precompute entries', __import__('time').perf_counter()-t0, flush=True)
    # Concatenate + sort by score desc once.
    for D, use_keep in ((base, True), (var, True)):
        for key in list(D):
            sc, tp, kp, im = D[key]
            sc = np.concatenate(sc); tp = np.concatenate(tp)
            kp = np.concatenate(kp); im = np.concatenate(im)
            order = np.argsort(-sc, kind='stable')
            sc, tp, kp, im = sc[order], tp[order], kp[order], im[order]
            if D is base:
                kp = np.ones_like(kp)
            D[key] = (sc, tp, kp, im, gt_counts[key[0]])
    # full-sample check
    ones = np.ones(n, np.float32)
    b = _map_for(base, ones); v = _map_for(var, ones)
    print('full fast AP base', b[0.5], np.mean([b[t] for t in IOUS if b[t] is not None]))
    print('per-class AP50 base:')
    for cls in range(14):
        e = base.get((cls, 0.5))
        if e is None:
            print(' ', cls, 'NO ENTRIES')
        else:
            print(' ', cls, 'n_entries', len(e[0]), 'n_gt', float(e[4].sum()), 'ap', _ap101(*e, ones))
    print('full fast AP var ', v[0.5], np.mean([v[t] for t in IOUS if v[t] is not None]))
    # bootstrap
    per = max(1, args.boot // args.workers)
    tasks = [(1729 + k, per, n) for k in range(args.workers)]
    _init_child({'base': base, 'var': var})
    allres = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork'),
                             initializer=_init_child, initargs=({'base': base, 'var': var},)) as ex:
        for r in ex.map(_run_task, tasks):
            allres.extend(r)
    arr = np.asarray(allres[:args.boot], np.float64)
    out = {'conf': args.conf, 'dropped': len(drop), 'boot': len(arr),
           'full_base': b, 'full_var': v,
           'boot_mean': arr.mean(0).tolist(),
           'boot_ci50': np.percentile(arr[:,0]-arr[:,2], [2.5, 97.5]).tolist(),
           'boot_ci95': np.percentile(arr[:,1]-arr[:,3], [2.5, 97.5]).tolist(),
           'diff50_mean': float((arr[:,0]-arr[:,2]).mean()),
           'diff95_mean': float((arr[:,1]-arr[:,3]).mean()),
           'p50_le0': float((arr[:,0]-arr[:,2] <= 0).mean()),
           'p95_le0': float((arr[:,1]-arr[:,3] <= 0).mean())}
    print(json.dumps({k:v for k,v in out.items() if k not in ('full_base','full_var')}, indent=1))
    args.out.write_text(json.dumps(out, indent=2)+'\n')

if __name__ == '__main__':
    main()
