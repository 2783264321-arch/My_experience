#!/usr/bin/env python3
"""Short 5 ms start-region sync on the detections that failed the multi frontend."""
from __future__ import annotations
import json, math, pickle, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from torchvision.ops import box_iou
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts')); sys.path.insert(0,str(ROOT/'scripts/detector_upgrade'))
from src.signals.data import N_TIME,SPLIT_MANIFEST,load_manifest
from src.signals.infer import gt_global
from src.wifi_expert_suppress import (_frontend_short,_load_experts,_classify,
                                      box_to_physical,DEFAULT_WEIGHTS)
PKL=Path('/root/autodl-fs/runs_local/dfine_local_iq/wifi_sync_coverage.pkl')
CAND=Path('/root/autodl-fs/runs_local/dfine_local_iq/val_candidates.npz')
def _init(): torch.set_num_threads(1)
def main():
    with open(PKL,'rb') as f: d=pickle.load(f)
    refs=d['refs']; rows=d['rows']
    with np.load(CAND,allow_pickle=True) as zf:
        meta=json.loads(CAND.with_suffix('.json').read_text()); n=int(meta['samples'])
    recs=load_manifest(SPLIT_MANIFEST,split='val')[:n]
    gts=[gt_global(r,math.ceil(r.N/(N_TIME-1))) for r in recs]
    fail=[i for i,r in enumerate(rows) if r['status'] in ('sync_fail','too_few_symbols')]
    jobs=[]; idxs=[]
    for i in fail:
        si,k,lab,sc,box,dur=refs[i]; r=recs[si]
        f1,f2,t1,t2=box_to_physical(*box,r.B_mhz,r.f_lo,r.N)
        jobs.append((str(r.bin_path),r.f_lo,r.B_mhz,r.N,(f1+f2)/2,t1,t2,20 if lab<=2 else 40,5.0,(-2.0,-1.0,0.0,1.0,2.0)))
        idxs.append(i)
    print('short jobs',len(jobs),flush=True)
    feats=[]
    with ProcessPoolExecutor(max_workers=16,initializer=_init) as ex:
        for x in ex.map(_frontend_short,jobs,chunksize=16): feats.append(x)
    ok=[j for j,x in enumerate(feats) if x['status']=='ok']
    device='cuda' if torch.cuda.is_available() else 'cpu'
    models=_load_experts(DEFAULT_WEIGHTS,device,'shared')['SHARED']
    probs=_classify(feats,ok,models,device)
    out=[]
    for j,i in enumerate(idxs):
        si,k,lab,sc,box,dur=refs[i]; f=feats[j]
        g=gts[si]; gb=np.asarray(g.boxes); gl=np.asarray(g.labels)
        iou=box_iou(torch.from_numpy(box[None]).float(),torch.from_numpy(gb).float())[0].numpy()
        gi=int(np.argmax(iou)); best=float(iou[gi]); matched=best>=0.5; gt=int(gl[gi]) if matched else -1
        row={'si':si,'k':k,'label':lab,'score':sc,'matched':matched,'gt':gt,
             'det_ok':matched and lab==gt,'status':f['status'],
             'metric':f.get('metric'),'ratio':f.get('ratio'),'pos':f.get('pos'),
             'tu_us':f.get('tu_us'),'cp_us':f.get('cp_us'),'n_kept':f.get('n_kept')}
        if j in probs:
            p=probs[j]; pm=int(np.argmax(p)); conf=float(p.max()); expert=pm if lab<=2 else 3+pm
            row.update(expert=expert,conf=conf,disagree=expert!=lab,expert_ok=matched and expert==gt)
        out.append(row)
    print('sync_ok',len(ok),'/',len(jobs),f'{len(ok)/max(len(jobs),1)*100:.1f}%')
    for grp in ('matched_correct','matched_wrong','background'):
        sub=[r for r in out if r['status']=='ok' and ((r['matched'] and r['det_ok']) if grp=='matched_correct' else ((r['matched'] and not r['det_ok']) if grp=='matched_wrong' else not r['matched']))]
        print(' ',grp,len(sub))
    print('pos counts',Counter(r['pos'] for r in out if r['status']=='ok'))
    print('Tu/CP',Counter((r['tu_us'],r['cp_us']) for r in out if r['status']=='ok').most_common(6))
    for th in (0.9,0.95,0.98,0.99,0.995):
        dd=[r for r in out if r.get('status')=='ok' and r.get('disagree') and r.get('conf',0)>=th]
        if not dd: continue
        print(f'conf>={th}: dis {len(dd)} precision {np.mean([not r["det_ok"] for r in dd]):.4f} '
              f'fix {sum((not r["det_ok"]) and r["expert_ok"] for r in dd)} brk {sum(r["det_ok"] for r in dd)} '
              f'fixScore {sum(r["score"] for r in dd if (not r["det_ok"]) and r["expert_ok"]):.1f} '
              f'brkScore {sum(r["score"] for r in dd if r["det_ok"]):.1f}')
    Path('/root/autodl-fs/runs_local/dfine_local_iq/wifi_short_start.json').write_text(json.dumps(out,default=float)+'\n')
    with open('/root/autodl-fs/runs_local/dfine_local_iq/wifi_short_start.pkl','wb') as f:
        pickle.dump({'feats':feats,'refs':[refs[i] for i in idxs],'probs':probs,'rows':out},f)
if __name__=='__main__': main()
