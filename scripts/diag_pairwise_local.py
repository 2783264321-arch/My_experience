#!/usr/bin/env python3
"""For GTs with a top wrong-class detection, does the local model rank the
correct-class box above the wrong-class box?  Pairwise signal, not argmax."""
from __future__ import annotations
import json, math, sys
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
from src.signals.infer import Prediction, soft_classwise_nms, gt_global

GROUPS = {"012":[0,1,2], "345":[3,4,5], "1011":[10,11]}
_G = {}

def tracked_soft(boxes, scores, labels, iou=0.5, sigma=0.5, score_thr=1e-3, max_det=100):
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
    bo=np.stack(kb); so=np.asarray(ks); lo=np.asarray(kl); io=np.asarray(ki)
    o=np.argsort(-so)[:max_det]
    return bo[o], so[o], lo[o], io[o]

def _init():
    torch.set_num_threads(1); np.seterr(all='ignore')

def _sample(i):
    a,b = int(_G['offsets'][i]), int(_G['offsets'][i+1])
    cb,cs,cl = _G['boxes'][a:b], _G['scores'][a:b], _G['labels'][a:b].astype(np.int64)
    fb,fs,fl,fi = tracked_soft(cb,cs,cl)
    gb = _G['gts'][i].boxes.numpy(); gl = _G['gts'][i].labels.numpy().astype(np.int64)
    if len(gb)==0 or len(fb)==0: return []
    iou = box_iou(torch.from_numpy(fb).float(), torch.from_numpy(gb).float()).numpy()
    out=[]
    for gi in range(len(gl)):
        gt=int(gl[gi]); fm=np.flatnonzero(iou[:,gi]>=0.5)
        if not fm.size: continue
        corr=fm[fl[fm]==gt]; wrong=fm[fl[fm]!=gt]
        if not corr.size or not wrong.size: continue
        cd=int(corr[np.argmax(fs[corr])]); wd=int(wrong[np.argmax(fs[wrong])])
        # local p(gt) and p(wrong label) on each source box
        def lp(det, cls):
            j=int(fi[det]+a); row=_G['row_of'].get(j)
            if row is None or row not in _G['group_of']:
                return None, None
            g, clsarr, pr = _G['group_of'][row]
            p=pr[:-1].astype(float); p=p/max(p.sum(),1e-12)
            pg=float(p[clsarr==gt].sum()) if np.any(clsarr==gt) else 0.0
            pc=float(p[clsarr==cls].sum()) if np.any(clsarr==cls) else 0.0
            return pg,pc
        pg_c, pc_c = lp(cd, gt)
        pg_w, pc_w = lp(wd, int(fl[wd]))
        if pg_c is None or pg_w is None: continue
        out.append({'sample':i,'gt':gt,'score_correct':float(fs[cd]),'score_wrong':float(fs[wd]),
                    'label_wrong':int(fl[wd]),'pg_correct_box':pg_c,'pg_wrong_box':pg_w,
                    'pc_correct_box':pc_c,'pc_wrong_box':pc_w,
                    'pair_pref': pg_c > pg_w, 'flip_wrong': pg_w > pc_w,
                    'keep_correct': pg_c > pc_c,
                    'group': next((k for k,v in GROUPS.items() if gt in v), 'other')})
    return out

def main():
    with np.load('/root/autodl-fs/runs_local/dfine_local_iq/val_candidates.npz', allow_pickle=True) as zf:
        c={k:zf[k] for k in ('boxes','scores','labels','sample_offsets')}
    with np.load('/root/autodl-fs/runs_local/dfine_local_iq/val_local_crops.npz', allow_pickle=True) as zf:
        ci=zf['candidate_index'].astype(np.int64)
    with np.load('/root/autodl-fs/runs_local/dfine_local_iq/val_local_probs.npz', allow_pickle=True) as pf:
        pc={k:pf[k] for k in pf.files}
    meta=json.loads(Path('/root/autodl-fs/runs_local/dfine_local_iq/val_candidates.json').read_text())
    n=int(meta['samples'])
    gts=[gt_global(r, math.ceil(r.N/(N_TIME-1))) for r in load_manifest(SPLIT_MANIFEST, split='val')[:n]]
    _G.update(boxes=c['boxes'],scores=c['scores'],labels=c['labels'],offsets=c['sample_offsets'],gts=gts)
    _G['row_of']={int(j):i for i,j in enumerate(ci)}
    _G['group_of']={}
    for g in GROUPS:
        idx=pc[f'idx_{g}'].astype(np.int64); cls=pc[f'cls_{g}'].astype(np.int64); pr=pc[f'probs_{g}']
        for k,row in enumerate(idx): _G['group_of'][int(row)]=(g,cls,pr[k])
    rows=[]
    with ProcessPoolExecutor(max_workers=16, initializer=_init) as ex:
        for r in ex.map(_sample, range(n), chunksize=8): rows.extend(r)
    print('pairs',len(rows))
    for grp in ('012','345','1011'):
        sub=[x for x in rows if x['group']==grp]
        if not sub: continue
        loss=[x for x in sub if x['score_wrong']>x['score_correct']+1e-9]
        ok=[x for x in sub if x['score_wrong']<=x['score_correct']+1e-9]
        print(f'{grp}: pairs {len(sub)} ranking_loss {len(loss)} ({len(loss)/len(sub)*100:.1f}%)')
        for name,aset in (('loss',loss),('ok',ok)):
            if not aset: continue
            print(f'  {name} pair_pref {np.mean([x["pair_pref"] for x in aset])*100:.1f}% '
                  f'flip_wrong {np.mean([x["flip_wrong"] for x in aset])*100:.1f}% '
                  f'keep_correct {np.mean([x["keep_correct"] for x in aset])*100:.1f}% '
                  f'both {np.mean([x["pair_pref"] and x["flip_wrong"] and x["keep_correct"] for x in aset])*100:.1f}%')
    Path('/root/autodl-fs/runs_local/dfine_local_iq/pairwise_local_diag.json').write_text(json.dumps(rows)+'\n')

if __name__=='__main__': main()
