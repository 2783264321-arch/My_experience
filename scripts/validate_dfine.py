#!/usr/bin/env python3
"""Formal D-FINE-L validation: rebuild from checkpoint, native decode, dual protocol."""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts/detector_upgrade'))
from adapters import build_dfine
from src.signals.data import SPLIT_MANIFEST, N_TIME, load_manifest
from src.signals.infer import gt_global
from validate_dual import run_dual

def stratified(records,n):
    by={}
    for r in records: by.setdefault(round(float(r.B_mhz)),[]).append(r)
    out=[]
    for b in sorted(by): out+=by[b][:n]
    return out

def save_npz(path,preds,gts):
    np.savez_compressed(path,
        pred_boxes=np.concatenate([p.boxes.reshape(-1,4) for p in preds]).astype(np.float32),
        pred_scores=np.concatenate([p.scores.reshape(-1) for p in preds]).astype(np.float32),
        pred_labels=np.concatenate([p.labels.reshape(-1) for p in preds]).astype(np.int64),
        pred_sample_index=np.concatenate([np.full(len(p.boxes),i,dtype=np.int64) for i,p in enumerate(preds)]),
        gt_boxes=np.concatenate([g.boxes.reshape(-1,4) for g in gts]).astype(np.float32),
        gt_labels=np.concatenate([g.labels.reshape(-1) for g in gts]).astype(np.int64),
        gt_sample_index=np.concatenate([np.full(len(g.boxes),i,dtype=np.int64) for i,g in enumerate(gts)]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',default='dfine_l_c1_2048x512')
    ap.add_argument('--checkpoint',default='best.pt')
    ap.add_argument('--weights',choices=['ema','model'],default='ema')
    ap.add_argument('--stratified',type=int,default=0)
    ap.add_argument('--split',default='val',choices=['fit','val','test'])
    ap.add_argument('--max-samples',type=int,default=0)
    ap.add_argument('--conf',type=float,default=0.001)
    ap.add_argument('--batch-views',type=int,default=8)
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--output',default='')
    a=ap.parse_args()
    run_dir=ROOT/'runs'/a.run; ck_path=Path(a.checkpoint)
    if not ck_path.is_absolute(): ck_path=run_dir/ck_path
    ck=torch.load(ck_path,map_location='cpu',weights_only=False)
    sd=ck['ema'] if a.weights=='ema' and ck.get('ema') is not None else ck['model']
    model,info=build_dfine(); model.load_state_dict(sd); model=model.cuda().eval()
    assert tuple(model.backbone.stem.stem1.conv.weight.shape)==(32,1,3,3)
    assert int(model.decoder.num_classes)==14
    post=model.postprocessor
    def infer(images):
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            res=post(model(images),torch.tensor([[512,2048]]*images.shape[0],device=images.device))
        out=[]
        for r in res:
            keep=r['scores']>=a.conf          # common candidate threshold
            out.append(torch.cat([r['boxes'][keep],r['scores'][keep,None],
                                  r['labels'][keep,None].float()],1).float())
        return out
    norm_d=json.loads((ROOT/'configs/normalizer.json').read_text()); norm=(norm_d['lo'],norm_d['hi'])
    records=load_manifest(SPLIT_MANIFEST,split=a.split)
    if a.max_samples: records=records[:a.max_samples]
    if a.stratified: records=stratified(records,a.stratified)
    out_dir=Path(a.output) if a.output else run_dir/(f'{a.split}_subset' if a.stratified else f'{a.split}_full')
    out_dir.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter(); res=run_dual(model,records,norm,batch_views=a.batch_views,workers=a.workers,infer_fn=infer)
    elapsed=time.perf_counter()-t0
    gts=[gt_global(r, math.ceil(r.N/(N_TIME-1))) for r in records]
    for name in ('unfiltered','filtered_0_5'): save_npz(out_dir/f'{a.split}_predictions_{name}.npz',res[f'preds_{name}'],gts)
    (out_dir/f'{a.split}_sample_ids.json').write_text(json.dumps([r.sample_id for r in records],indent=2))
    summary={'run':a.run,'checkpoint':str(ck_path),'weights':a.weights,'split':a.split,'n_samples':len(records),
             'n_views':res['n_views'],'n_boundary_dropped':res['n_boundary_dropped'],
             'conf':0.001,'iou':0.5,'max_det':100,'per_view_topk':300,
             'settings':res['settings'],'elapsed_s':elapsed,
             'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'model':info,
             'checkpoint_epoch':ck.get('epoch'),'checkpoint_best_fitness':ck.get('best_fitness')}
    (out_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    f=summary['settings']['filtered_0_5']; u=summary['settings']['unfiltered']
    print(json.dumps({'samples':len(records),'views':res['n_views'],'filtered_map50':f['map50'],
        'filtered_map50_95':f['map50_95'],'unfiltered_map50':u['map50'],
        'unfiltered_map50_95':u['map50_95'],'filtered_recall':f['localization_recall'],
        'elapsed_s':elapsed},ensure_ascii=False))

if __name__=='__main__': main()
