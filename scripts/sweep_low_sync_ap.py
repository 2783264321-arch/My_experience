#!/usr/bin/env python3
"""Sweep low-sync confidence thresholds using the cached diagnostic rows."""
from __future__ import annotations
import json, math, pickle, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts')); sys.path.insert(0,str(ROOT/'scripts/detector_upgrade'))
from src.signals.data import N_TIME,SPLIT_MANIFEST,load_manifest
from src.signals.infer import Prediction,soft_classwise_nms,gt_global
from src.signals.metrics import ImagePredictions
from validate_dual import _eval_set

CAND=Path('/root/autodl-fs/runs_local/dfine_local_iq/val_candidates.npz')
PKL=Path('/root/autodl-fs/runs_local/dfine_local_iq/wifi_sync_coverage.pkl')
_G={}
def _init(): torch.set_num_threads(1)
def _merge(i):
    a,b=int(_G['offsets'][i]),int(_G['offsets'][i+1])
    p=Prediction(torch.from_numpy(_G['boxes'][a:b]).float(),torch.from_numpy(_G['scores'][a:b]).float(),torch.from_numpy(_G['labels'][a:b]).long())
    f=soft_classwise_nms(p,0.5,0.5,1e-3,100)
    return f.boxes.numpy(),f.scores.numpy(),f.labels.numpy()
def _eval_variant(name):
    drop=_G['drops'][name]
    preds=[]
    for si,(mb,ms,ml) in enumerate(_G['merged']):
        keep=[k for k in range(len(ml)) if (si,k) not in drop]
        preds.append(ImagePredictions(mb[keep],ms[keep],ml[keep]))
    m=_eval_set(_G['gts'],preds); m['n_pred']=int(sum(len(p.boxes) for p in preds)); m['dropped']=len(drop); m['name']=name
    return m
def main():
    with np.load(CAND,allow_pickle=True) as zf:
        c={k:zf[k] for k in ('boxes','scores','labels','sample_offsets')}
    meta=json.loads(CAND.with_suffix('.json').read_text()); n=int(meta['samples'])
    recs=load_manifest(SPLIT_MANIFEST,split='val')[:n]
    gts=[gt_global(r,math.ceil(r.N/(N_TIME-1))) for r in recs]
    _G.update(boxes=c['boxes'],scores=c['scores'],labels=c['labels'],offsets=c['sample_offsets'])
    merged=[]
    with ProcessPoolExecutor(max_workers=16,initializer=_init) as ex:
        for r in ex.map(_merge,range(n),chunksize=8): merged.append(r)
    with open(PKL,'rb') as f: d=pickle.load(f)
    rows=d['rows']
    drops={}
    strict={(r['si'],r['k']) for r in rows if r.get('status')=='ok' and r.get('disagree') and r.get('conf',0)>=0.9}
    drops['strict']=strict
    for th in (0.90,0.95,0.99):
        s=set(strict)
        s|={(r['si'],r['k']) for r in rows if r.get('status')=='ok_low' and r.get('disagree') and r.get('conf',0)>=th}
        drops[f'low_{th}']=s
    _G.update(merged=merged,gts=gts,drops=drops)
    names=['strict','low_0.9','low_0.95','low_0.99']
    with ProcessPoolExecutor(max_workers=4,mp_context=__import__('multiprocessing').get_context('fork'),initializer=_init) as ex:
        res=list(ex.map(_eval_variant,names))
    base=None
    for m in res:
        if base is None: base=m
        print(f"{m['name']:10s} {m['map50']:.6f} {m['map50_95']:.6f} drops {m['dropped']} "
              f"Δvs_strict {100*(m['map50']-base['map50']):+.3f}/{100*(m['map50_95']-base['map50_95']):+.3f} pp")
    Path('/root/autodl-fs/runs_local/dfine_local_iq/low_sync_ap.json').write_text(json.dumps(res,indent=2)+'\n')
if __name__=='__main__': main()
