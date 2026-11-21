"""Small val-subset evaluation to sanity-check the full eval path."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts/detector_upgrade'))
from src.signals.data import SPLIT_MANIFEST, build_sample_views, frequency_starts, load_manifest, h_for_n
from src.signals.infer import Prediction, _boundary_mask, _map_view_dets, global_classwise_nms, gt_global
from src.signals.metrics import ImagePredictions, ImageTargets, evaluate
from ultralytics.utils.nms import non_max_suppression

def pick(n_per_b=4):
    val=load_manifest(SPLIT_MANIFEST,split='val'); by={}
    for r in val: by.setdefault(round(float(r.B_mhz)),[]).append(r)
    out=[]
    for b in sorted(by): out+=by[b][:n_per_b]
    return out

def merge(dets,types,starts,F):
    preds=[]
    for det,vt,st in zip(dets,types,starts):
        p=_map_view_dets(det,vt,st,F)
        bm=_boundary_mask(p,vt,st,F)
        if bm.any(): p=Prediction(p.boxes[~bm],p.scores[~bm],p.labels[~bm])
        preds.append(p)
    p=Prediction(torch.cat([x.boxes for x in preds]),torch.cat([x.scores for x in preds]),torch.cat([x.labels for x in preds]))
    return global_classwise_nms(p,0.5,100)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--kind',choices=['yolo26','dfine'],required=True)
    args=ap.parse_args(); import adapters
    norm=json.loads((ROOT/'configs/normalizer.json').read_text()); norm=(norm['lo'],norm['hi'])
    ck=torch.load(ROOT/f'reports/detector_upgrade/short_{args.kind}.pt',map_location='cpu',weights_only=False)
    if args.kind=='yolo26':
        m,_=adapters.build_yolo26(); m.criterion=m.init_criterion(); m.load_state_dict(ck['model']); m=m.cuda().eval(); m.model[-1].end2end=False
        def infer(imgs):
            with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
                return [d.float() for d in non_max_suppression(m(imgs),0.001,0.5,nc=14,max_det=300,max_nms=30000,max_time_img=20)]
    else:
        m,_=adapters.build_dfine(); m.load_state_dict(ck['model']); m=m.cuda().eval(); post=m.postprocessor
        def infer(imgs):
            with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
                res=post(m(imgs),torch.tensor([[512,2048]]*imgs.shape[0],device=imgs.device))
            return [torch.cat([r['boxes'],r['scores'][:,None],r['labels'][:,None].float()],1).float() for r in res]
    gts=[]; preds=[]
    for r in pick():
        F=round(r.B_mhz*700); starts=frequency_starts(F)
        coarse,fines,_=build_sample_views(r,norm,list(range(len(starts))))
        views=[coarse]+fines; imgs=torch.stack([v.tensor for v in views]).cuda()
        dets=infer(imgs)
        merged=merge(dets,[v.view_type for v in views],[v.start for v in views],F)
        preds.append(ImagePredictions(merged.boxes.numpy(),merged.scores.numpy(),merged.labels.numpy()))
        g=gt_global(r,h_for_n(r.N)); gts.append(ImageTargets(g.boxes.numpy(),g.labels.numpy()))
    ev=evaluate(gts,preds,iou_thresholds=(0.5,),max_dets=100)
    print(json.dumps({'kind':args.kind,'samples':len(gts),'gt':sum(len(g.boxes) for g in gts),
        'map50':ev['per_iou']['0.5']['map_class_aware'],'ap_agn':ev['per_iou']['0.5']['ap_class_agnostic'],
        'recall':ev['classification']['localization_recall'],'cls_acc':ev['classification']['classification_accuracy'],
        'n_pred':int(sum(len(p.boxes) for p in preds))}))
if __name__=='__main__': main()
