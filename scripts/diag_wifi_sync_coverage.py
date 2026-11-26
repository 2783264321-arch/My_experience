#!/usr/bin/env python3
"""Diagnose Wi-Fi expert sync coverage: per-detection frontend metrics + GT."""
from __future__ import annotations
import json, math, sys, pickle
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'scripts/detector_upgrade'))
from src.signals.data import N_TIME, SPLIT_MANIFEST, load_manifest
from src.signals.infer import Prediction, soft_classwise_nms, gt_global
from src.wifi_expert_suppress import (_frontend_multi, _load_experts, _classify,
                                      box_to_physical, WIFI_CLASSES, DEFAULT_WEIGHTS)

CAND=Path('/root/autodl-fs/runs_local/dfine_local_iq/val_candidates.npz')
_G={}
def _init(): torch.set_num_threads(1)
def _merge(i):
    a,b=int(_G['offsets'][i]),int(_G['offsets'][i+1])
    p=Prediction(torch.from_numpy(_G['boxes'][a:b]).float(),
                 torch.from_numpy(_G['scores'][a:b]).float(),
                 torch.from_numpy(_G['labels'][a:b]).long())
    f=soft_classwise_nms(p,0.5,0.5,1e-3,100)
    return f.boxes.numpy(),f.scores.numpy(),f.labels.numpy()

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
    jobs=[]; refs=[]
    for si,(mb,ms,ml) in enumerate(merged):
        r=recs[si]
        for k in range(len(ml)):
            lab,sc=int(ml[k]),float(ms[k])
            if lab not in WIFI_CLASSES or sc<0.3: continue
            f1,f2,t1,t2=box_to_physical(*mb[k],r.B_mhz,r.f_lo,r.N)
            jobs.append((str(r.bin_path),r.f_lo,r.B_mhz,r.N,(f1+f2)/2,t1,t2,20 if lab<=2 else 40,(0.0,0.5,1.0)))
            refs.append((si,k,lab,sc,mb[k].copy(),(t2-t1)))
    print('jobs',len(jobs),flush=True)
    feats=[]
    with ProcessPoolExecutor(max_workers=16,initializer=_init) as ex:
        for x in ex.map(_frontend_multi,jobs,chunksize=16): feats.append(x)
    ok=[i for i,x in enumerate(feats) if x['status'] in ('ok','ok_low')]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    models=_load_experts(DEFAULT_WEIGHTS,device,'shared')['SHARED']
    probs=_classify(feats,ok,models,device)
    rows=[]
    for i,(si,k,lab,sc,box,dur) in enumerate(refs):
        g=gts[si]; gb=np.asarray(g.boxes); gl=np.asarray(g.labels)
        if len(gb):
            iou=box_iou(torch.from_numpy(box[None]).float(),torch.from_numpy(gb).float())[0].numpy()
            gi=int(np.argmax(iou)); best=float(iou[gi]); matched=best>=0.5; gt=int(gl[gi]) if matched else -1
        else: matched,best,gt=False,0.0,-1
        f=feats[i]
        row={'si':si,'k':k,'label':lab,'score':sc,'dur_ms':dur,'matched':matched,'gt':gt,
             'det_ok':matched and lab==gt,'status':f['status'],
             'metric':f.get('metric'),'ratio':f.get('ratio'),
             'tu_us':f.get('tu_us'),'cp_us':f.get('cp_us'),'pos':f.get('pos'),
             'n_kept':f.get('n_kept')}
        if i in probs:
            p=probs[i]; pm=int(np.argmax(p)); conf=float(p.max()); expert=pm if lab<=2 else 3+pm
            row.update(expert=expert,conf=conf,disagree=expert!=lab,
                       expert_ok=matched and expert==gt)
        rows.append(row)
    with open('/root/autodl-fs/runs_local/dfine_local_iq/wifi_sync_coverage.pkl','wb') as f:
        pickle.dump({'feats':feats,'refs':refs,'probs':probs,'rows':rows}, f)
    Path('/root/autodl-fs/runs_local/dfine_local_iq/wifi_sync_coverage.json').write_text(json.dumps(rows, default=float)+'\n')
    print('saved rows',len(rows))
    # summaries
    for grp in ('all','matched_correct','matched_wrong','background'):
        if grp=='all': sub=rows
        elif grp=='matched_correct': sub=[r for r in rows if r['matched'] and r['det_ok']]
        elif grp=='matched_wrong': sub=[r for r in rows if r['matched'] and not r['det_ok']]
        else: sub=[r for r in rows if not r['matched']]
        okn=sum(r['status'] in ('ok','ok_low') for r in sub)
        low=sum(r['status']=='ok_low' for r in sub)
        print(f'{grp:16s} n={len(sub):5d} sync_ok={okn:4d} ({okn/max(len(sub),1)*100:5.1f}%) '
              f'of_which_low={low:4d} fail={sum(r["status"]=="sync_fail" for r in sub)} '
              f'tu_short={sum(r["status"]=="too_few_symbols" for r in sub)}')
    print('sync_fail metric p10/50/90',[round(float(np.percentile([r['metric'] for r in rows if r['status']=='sync_fail'],q)),3) for q in (10,50,90)])
    print('sync_fail ratio p10/50/90',[round(float(np.percentile([r['ratio'] for r in rows if r['status']=='sync_fail'],q)),1) for q in (10,50,90)])
    print('ok metric p10/50/90',[round(float(np.percentile([r['metric'] for r in rows if r['status']=='ok'],q)),3) for q in (10,50,90)])
    print('ok ratio p10/50/90',[round(float(np.percentile([r['ratio'] for r in rows if r['status']=='ok'],q)),1) for q in (10,50,90)])
    print('ok Tu/CP counts',Counter((r['tu_us'],r['cp_us']) for r in rows if r['status'] in ('ok','ok_low')).most_common(8))
    print('ok pos counts',Counter(r['pos'] for r in rows if r['status'] in ('ok','ok_low')))
    print('low sync pos counts',Counter(r['pos'] for r in rows if r['status']=='ok_low'))
    # expert precision vs metric/ratio bins (matched rows)
    m=[r for r in rows if r['status'] in ('ok','ok_low') and r['matched'] and 'conf' in r]
    for lo,hi in ((0.34,0.5),(0.5,0.8),(0.8,1.5),(1.5,10)):
        sub=[r for r in m if lo<=r['metric']<hi]
        if sub:
            dis=[r for r in sub if r['disagree'] and r['conf']>=0.9]
            for st in ('ok','ok_low'):
                ss=[r for r in sub if r['status']==st]
                if not ss: continue
                dd=[r for r in ss if r['disagree'] and r['conf']>=0.9]
                print(f'metric {lo}-{hi} {st}: n {len(ss)} dis {len(dd)} precision {np.mean([not r["det_ok"] for r in dd]) if dd else float("nan"):.3f} '
                      f'fix {sum((not r["det_ok"]) and r["expert_ok"] for r in dd)} brk {sum(r["det_ok"] for r in dd)}')
    # low-gate precision at conf thresholds
    low=[r for r in rows if r['status']=='ok_low' and r['matched'] and 'conf' in r]
    print('low rows',len(low))
    for th in (0.9,0.95,0.98,0.99,0.995):
        dd=[r for r in low if r['disagree'] and r['conf']>=th]
        if not dd: continue
        print(f'  conf>={th}: dis {len(dd)} precision {np.mean([not r["det_ok"] for r in dd]):.4f} '
              f'fix {sum((not r["det_ok"]) and r["expert_ok"] for r in dd)} brk {sum(r["det_ok"] for r in dd)} '
              f'fixScore {sum(r["score"] for r in dd if (not r["det_ok"]) and r["expert_ok"]):.1f} '
              f'brkScore {sum(r["score"] for r in dd if r["det_ok"]):.1f}')

if __name__=='__main__': main()
