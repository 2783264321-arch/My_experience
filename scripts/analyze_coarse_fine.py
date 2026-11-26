#!/usr/bin/env python3
"""Per-GT coarse vs fine vs coarse+fine merge analysis, stratified by size.

First-hand data: raw coarse/fine candidate boxes.  For each GT, find the
highest-score candidate at IoU>=0.5 in each view family, compare their IoU
with the GT and their class, and test a score-weighted coarse+fine box.
IoS (intersection over smaller) distinguishes containment from duplication.
"""
from __future__ import annotations
import argparse, json, math, sys
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
from src.signals.infer import gt_global

CAND = Path('/root/autodl-fs/runs_local/dfine_local_iq/val_candidates.npz')
_G = {}

def _init():
    torch.set_num_threads(1); np.seterr(all='ignore')

def _sample(i):
    a, b = int(_G['offsets'][i]), int(_G['offsets'][i+1])
    cb, cs, cl = _G['boxes'][a:b], _G['scores'][a:b], _G['labels'][a:b].astype(np.int64)
    vt = _G['view_type'][a:b]
    rec = _G['records'][i]
    gb = _G['gts'][i].boxes.numpy(); gl = _G['gts'][i].labels.numpy().astype(np.int64)
    if len(gb) == 0: return []
    iou = box_iou(torch.from_numpy(cb).float(), torch.from_numpy(gb).float()).numpy()
    out = []
    for gi in range(len(gl)):
        m = np.flatnonzero(iou[:, gi] >= 0.5)
        if not m.size: continue
        cm = m[vt[m] == 0]; fm = m[vt[m] == 1]
        gt = int(gl[gi])
        def best(mm):
            if not mm.size: return None
            k = int(mm[np.argmax(cs[mm])])
            return k
        kc, kf = best(cm), best(fm)
        # GT physical size from its normalized box
        bw = float((gb[gi,3]-gb[gi,1]) * rec.B_mhz)
        dur = float((gb[gi,2]-gb[gi,0]) * rec.T_ms)
        bwc = float((cb[kc,3]-cb[kc,1]) * rec.B_mhz) if kc is not None else None
        bwf = float((cb[kf,3]-cb[kf,1]) * rec.B_mhz) if kf is not None else None
        row = {'sample':i,'gi':gi,'gt':gt,'bw_mhz':bw,'dur_ms':dur,
               'n_coarse':int(cm.size),'n_fine':int(fm.size)}
        for name,k in (('c',kc),('f',kf)):
            if k is None:
                row[f'{name}_iou']=None; row[f'{name}_score']=None
                row[f'{name}_ok']=False; row[f'{name}_bw']=None
            else:
                row[f'{name}_iou']=float(iou[k,gi]); row[f'{name}_score']=float(cs[k])
                row[f'{name}_ok']=bool(cl[k]==gt)
                row[f'{name}_bw']=float((cb[k,3]-cb[k,1])*rec.B_mhz)
        # score-weighted merge of the best coarse+fine
        if kc is not None and kf is not None:
            wc, wf = cs[kc], cs[kf]
            bm = (cb[kc]*wc + cb[kf]*wf) / max(wc+wf, 1e-9)
            ioum = float(box_iou(torch.from_numpy(bm[None]).float(),
                                 torch.from_numpy(gb[gi:gi+1]).float())[0,0])
            # IoS between the two source boxes
            inter = max(0.0, min(cb[kc,2],cb[kf,2])-max(cb[kc,0],cb[kf,0])) * \
                    max(0.0, min(cb[kc,3],cb[kf,3])-max(cb[kc,1],cb[kf,1]))
            ac = max(1e-9,(cb[kc,2]-cb[kc,0])*(cb[kc,3]-cb[kc,1]))
            af = max(1e-9,(cb[kf,2]-cb[kf,0])*(cb[kf,3]-cb[kf,1]))
            row['ios_cf']=float(inter/max(1e-9,min(ac,af)))
            row['merge_iou']=ioum
            row['merge_gain']=ioum - max(row['c_iou'],row['f_iou'])
        else:
            row['ios_cf']=None; row['merge_iou']=None; row['merge_gain']=None
        out.append(row)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=16)
    ap.add_argument('--out',type=Path,default=Path('/root/autodl-fs/runs_local/dfine_local_iq/coarse_fine_analysis.json'))
    args=ap.parse_args()
    with np.load(CAND,allow_pickle=True) as zf:
        c={k:zf[k] for k in ('boxes','scores','labels','sample_offsets','view_type')}
    meta=json.loads(CAND.with_suffix('.json').read_text()); n=int(meta['samples'])
    records=load_manifest(SPLIT_MANIFEST,split='val')[:n]
    gts=[gt_global(r,math.ceil(r.N/(N_TIME-1))) for r in records]
    _G.update(boxes=c['boxes'],scores=c['scores'],labels=c['labels'],
              offsets=c['sample_offsets'],view_type=c['view_type'],
              records=records,gts=gts)
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=_init) as ex:
        for r in ex.map(_sample,range(n),chunksize=8): rows.extend(r)
    print('GT rows with any IoU>=0.5 candidate:',len(rows),
          'coarse+fine:',sum(x['c_iou'] is not None and x['f_iou'] is not None for x in rows))
    bins=[('<0.5',0,0.5),('0.5-2',0.5,2),('2-10',2,10),('10-25',10,25),('25-45',25,45),('>45',45,1e9)]
    for name,lo,hi in bins:
        sub=[x for x in rows if lo<=x['bw_mhz']<hi]
        if not sub: continue
        both=[x for x in sub if x['c_iou'] is not None and x['f_iou'] is not None]
        cwin=sum(x['c_iou']>x['f_iou']+1e-9 for x in both)
        fwin=sum(x['f_iou']>x['c_iou']+1e-9 for x in both)
        mg=np.array([x['merge_gain'] for x in both]) if both else np.array([])
        print(f'bw {name:6s} MHz n={len(sub):5d} both={len(both):4d} '
              f'coarse_best {cwin/max(len(both),1)*100:5.1f}% fine_best {fwin/max(len(both),1)*100:5.1f}% '
              f'merge_gain>0 {np.mean(mg>1e-4)*100 if len(mg) else 0:5.1f}% '
              f'mean_gain {np.mean(mg) if len(mg) else 0:+.4f}')
    # duration bins
    dbins=[('<5ms',0,5),('5-20ms',5,20),('20-60ms',20,60),('>60ms',60,1e9)]
    for name,lo,hi in dbins:
        sub=[x for x in rows if lo<=x['dur_ms']<hi]
        if not sub: continue
        both=[x for x in sub if x['c_iou'] is not None and x['f_iou'] is not None]
        cwin=sum(x['c_iou']>x['f_iou']+1e-9 for x in both)
        fwin=sum(x['f_iou']>x['c_iou']+1e-9 for x in both)
        mg=np.array([x['merge_gain'] for x in both]) if both else np.array([])
        print(f'dur {name:7s} n={len(sub):5d} both={len(both):4d} '
              f'coarse_best {cwin/max(len(both),1)*100:5.1f}% fine_best {fwin/max(len(both),1)*100:5.1f}% '
              f'merge_gain>0 {np.mean(mg>1e-4)*100 if len(mg) else 0:5.1f}% mean_gain {np.mean(mg) if len(mg) else 0:+.4f}')
    args.out.write_text(json.dumps(rows)+'\n'); print('wrote',args.out)

if __name__=='__main__': main()
