#!/usr/bin/env python3
"""Class-13 (FM) metadata re-ranker pilot on the fixed val candidate pool.

Train a logistic FM-vs-background classifier on fit candidates, then
re-score only the final class-13 Soft-NMS detections and measure class-13
AP50 / overall AP. No new IQ extraction.
"""
from __future__ import annotations
import json, math, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'scripts/detector_upgrade'))
from src.signals.data import N_TIME, SPLIT_MANIFEST, load_manifest
from src.signals.infer import Prediction, soft_classwise_nms, gt_global
from src.signals.metrics import ImagePredictions, ImageTargets, _match_single, _ap_101

CAND = Path('/root/autodl-fs/runs_local/dfine_local_iq/val_candidates.npz')
_G = {}
FIT = Path('/root/autodl-fs/runs_local/dfine_local_iq/fit_candidates.npz')

def feats(z):
    b=z['boxes']; w=b[:,2]-b[:,0]; h=b[:,3]-b[:,1]
    return np.concatenate([w[:,None],h[:,None],(w*h)[:,None],
                           ((b[:,0]+b[:,2])/2)[:,None],((b[:,1]+b[:,3])/2)[:,None],
                           z['scores'][:,None],z['class_scores'],
                           z['view_type'][:,None].astype(np.float32)],1)

def _merge_worker(i):
    a,b=int(_G['offsets'][i]),int(_G['offsets'][i+1])
    mb,ms,ml,mi=tracked_soft(_G['boxes'][a:b],_G['scores'][a:b],_G['labels'][a:b].astype(np.int64))
    return mb,ms,ml,mi+a

def tracked_soft(boxes,scores,labels,iou=0.5,sigma=0.5,score_thr=1e-3,max_det=100):
    kb,ks,kl,ki=[],[],[],[]
    for cls in sorted(set(labels.tolist())):
        idx=np.flatnonzero(labels==cls); b=torch.from_numpy(boxes[idx]).float(); s=torch.from_numpy(scores[idx]).float()
        alive=torch.ones(len(b),dtype=torch.bool)
        while bool(alive.any()):
            j=int(torch.argmax(s.masked_fill(~alive,float('-inf'))))
            kb.append(b[j].numpy()); ks.append(float(s[j])); kl.append(int(cls)); ki.append(int(idx[j]))
            alive[j]=False
            if not bool(alive.any()): break
            ov=box_iou(b[j].unsqueeze(0),b[alive])[0]; s[alive]*=torch.exp(-(ov**2)/max(sigma,1e-6)); alive&=s>=score_thr
    if not kb: return (np.zeros((0,4),np.float32),np.zeros(0,np.float32),np.zeros(0,np.int64),np.zeros(0,np.int64))
    bo=np.stack(kb); so=np.asarray(ks); lo=np.asarray(kl); io=np.asarray(ki); o=np.argsort(-so)[:max_det]
    return bo[o],so[o],lo[o],io[o]

def main():
    zf=np.load(FIT,allow_pickle=True); fit={k:zf[k] for k in ('boxes','scores','labels','class_scores','gt_label','gt_iou','view_type')}
    zv=np.load(CAND,allow_pickle=True); val={k:zv[k] for k in ('boxes','scores','labels','class_scores','sample_offsets','view_type')}
    # train logistic
    lab=fit['labels'].astype(int); gt=fit['gt_label'].astype(int); iou=fit['gt_iou']
    m=lab==13; pos=m&(gt==13)&(iou>=.5); neg=m&~pos
    rng=np.random.default_rng(0); negidx=np.flatnonzero(neg); order=negidx[np.argsort(-fit['scores'][negidx])]
    negsel=np.concatenate([order[:50000],rng.choice(negidx,50000,replace=False)])
    tr=np.concatenate([np.flatnonzero(pos),negsel]); ytr=np.r_[np.ones(pos.sum()),np.zeros(len(negsel))]
    X=feats(fit); sc=StandardScaler().fit(X[tr])
    clf=LogisticRegression(max_iter=300,class_weight='balanced').fit(sc.transform(X[tr]),ytr)
    print('train',len(tr),'pos',int(ytr.sum()),flush=True)
    # merge val with source indices
    meta=json.loads(CAND.with_suffix('.json').read_text()); n=int(meta['samples'])
    records=load_manifest(SPLIT_MANIFEST,split='val')[:n]
    gts=[gt_global(r,math.ceil(r.N/(N_TIME-1))) for r in records]
    _G.update(boxes=val['boxes'],scores=val['scores'],labels=val['labels'],offsets=val['sample_offsets'])
    def _init(): torch.set_num_threads(1)
    with ProcessPoolExecutor(max_workers=16, initializer=_init) as ex:
        merged=list(ex.map(_merge_worker,range(n),chunksize=8))
    # class-13 final detections and source features
    Xv=feats(val); p_fm=clf.predict_proba(sc.transform(Xv))[:,1]
    dets=[]
    for si,(mb,ms,ml,mi) in enumerate(merged):
        for k in range(len(ml)):
            if int(ml[k])==13: dets.append((si,k,float(ms[k]),int(mi[k]),float(p_fm[int(mi[k])])))
    print('class13 final dets',len(dets),flush=True)
    # class-13 AP50 helper
    def ap13(score_map):
        entries=[]; n_gt=0
        for si,(mb,ms,ml,mi) in enumerate(merged):
            g=gts[si]; gb=np.asarray(g.boxes); gl=np.asarray(g.labels); n_gt+=int((gl==13).sum())
            pi=np.flatnonzero(ml==13)
            if len(pi)==0: continue
            pred=ImagePredictions(mb[pi],ms[pi],ml[pi])
            m,_=_match_single(ImageTargets(gb,gl),pred,0.5,class_aware=True,class_id=13,max_dets=100)
            mp={x[0] for x in m}
            for k in range(len(pi)):
                sc_=score_map.get((si,int(pi[k])),float(ms[pi[k]]))
                entries.append((-sc_, k in mp))
        entries.sort(); tp=np.array([e[1] for e in entries],float); fp=1-tp
        if len(tp)==0 or n_gt==0: return 0.0
        tp_cum=np.cumsum(tp); fp_cum=np.cumsum(fp); prec=tp_cum/np.maximum(tp_cum+fp_cum,1e-12); rec=tp_cum/n_gt
        return _ap_101(prec,rec)
    print('baseline class13 AP50',ap13({}),flush=True)
    for alpha in (0.25,0.5,1.0,2.0,4.0):
        sm={(si,k): s*(p**alpha) for si,k,s,j,p in dets}
        print('alpha',alpha,'AP50',ap13(sm),flush=True)
    # also a simple gated boost (p>=0.9 -> x1.5)
    for th,mult in ((0.9,1.5),(0.95,2.0),(0.8,1.25)):
        sm={(si,k):(s*mult if p>=th else s) for si,k,s,j,p in dets}
        print('gate',th,mult,'AP50',ap13(sm),flush=True)

if __name__=='__main__': main()
