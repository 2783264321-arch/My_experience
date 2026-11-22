#!/usr/bin/env python3
"""Three-model comparison: shared val protocol, corrected group recall, common-match attribution."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.signals.data import SPLIT_MANIFEST, N_TIME, load_manifest
from src.signals.infer import gt_global
from src.signals.metrics import (ImagePredictions, ImageTargets, _match_single,
                                 evaluate, iou_matrix)

OUT=ROOT/'reports/detector_upgrade'; OUT.mkdir(parents=True,exist_ok=True)
ALL_IOU=tuple(np.round(np.arange(0.5,1.0,0.05),2))
GROUPS={'WiFi_0-5':list(range(6)),'WiFi_20MHz_0-2':[0,1,2],
        'WiFi_40MHz_3-5':[3,4,5],'SRRC_10-11':[10,11],'Narrow_9_12_13':[9,12,13]}
FAM={'WiFi_0-5':range(6),'SRRC_10-11':[10,11],'Narrow_9_12_13':[9,12,13]}

MODELS={
 'YOLO11s': ROOT.parent.parent/'workspace/baseline/runs/yolo11s_c1_directreg_2048x512/stage5_val_filtered_ap95/val_predictions.npz',
 'YOLO26l': Path('/root/autodl-tmp/worktrees/formal-yolo26/runs/yolo26l_c1_2048x512/val_full_best/val_predictions_filtered_0_5.npz'),
 'D-FINE best': ROOT/'runs/dfine_l_c1_2048x512/val_full_best/val_predictions_filtered_0_5.npz',
 'D-FINE last': ROOT/'runs/dfine_l_c1_2048x512/val_full_last/val_predictions_filtered_0_5.npz',
}

def load_preds(path,n):
    z=np.load(path); idx=z['pred_sample_index']
    return [ImagePredictions(z['pred_boxes'][idx==i],z['pred_scores'][idx==i],
                             z['pred_labels'][idx==i]) for i in range(n)]

def overall(gts,preds):
    r05=evaluate(gts,preds,iou_thresholds=(0.5,),max_dets=100)
    r75=evaluate(gts,preds,iou_thresholds=(0.75,),max_dets=100)
    rf=evaluate(gts,preds,iou_thresholds=ALL_IOU,max_dets=100)
    return {'map50':float(r05['per_iou']['0.5']['map_class_aware'] or 0),
            'map50_95':float(np.mean([rf['per_iou'][str(t)]['map_class_aware'] for t in ALL_IOU])),
            'ap_agn50':float(r05['per_iou']['0.5']['ap_class_agnostic'] or 0),
            'ap_agn75':float(r75['per_iou']['0.75']['ap_class_agnostic'] or 0),
            'recall':r05['classification']['localization_recall'],
            'cls_acc':r05['classification']['classification_accuracy'],
            'n_pred':int(sum(len(p.boxes) for p in preds))}

def group_metrics(gts,preds,keep):
    keep=list(keep)
    g=[ImageTargets(gt.boxes[np.isin(gt.labels,keep)],gt.labels[np.isin(gt.labels,keep)]) for gt in gts]
    pa=preds
    pg=[ImagePredictions(pr.boxes[np.isin(pr.labels,keep)],pr.scores[np.isin(pr.labels,keep)],
                         pr.labels[np.isin(pr.labels,keep)]) for pr in preds]
    ra=evaluate(g,pa,iou_thresholds=(0.5,0.75),max_dets=100)
    rs=evaluate(g,pg,iou_thresholds=(0.5,),max_dets=100)
    return {'map50_strict':float(rs['per_iou']['0.5']['map_class_aware'] or 0),
            'ap_agn50':float(ra['per_iou']['0.5']['ap_class_agnostic'] or 0),
            'ap_agn75':float(ra['per_iou']['0.75']['ap_class_agnostic'] or 0),
            'recall_all':ra['classification']['localization_recall'],
            'recall_strict_predclass':rs['classification']['localization_recall'],
            'n_gt':int(ra['per_iou']['0.5']['n_gt'])}

def attribution(gts,preds):
    rows=[]
    for i,(gt,pr) in enumerate(zip(gts,preds)):
        m,_=_match_single(gt,pr,0.5,class_aware=False,max_dets=100)
        matched={gi:pi for pi,gi in m}
        for gi in range(len(gt.boxes)):
            gcls=int(gt.labels[gi])
            if gi in matched:
                pi=matched[gi]
                iou=float(iou_matrix(gt.boxes[gi:gi+1],pr.boxes[pi:pi+1])[0,0])
                rows.append((i,gi,True,gcls,int(pr.labels[pi]),iou))
            else:
                rows.append((i,gi,False,gcls,-1,0.0))
    return rows

def summarize_attr(rows,classes):
    sel=[r for r in rows if r[3] in classes]
    n=len(sel); m=[r for r in sel if r[2]]
    return {'n_gt':n,'matched':len(m),'recall':len(m)/max(n,1),
            'mean_iou':float(np.mean([r[5] for r in m])) if m else 0.0,
            'cls_acc':float(np.mean([r[4]==r[3] for r in m])) if m else 0.0}

def main():
    records=load_manifest(SPLIT_MANIFEST,split='val')
    gts=[gt_global(r,int(np.ceil(r.N/(N_TIME-1)))) for r in records]
    n=len(records)
    out={'protocol':{'split':'val','n_samples':n,'n_gt':int(sum(len(g.boxes) for g in gts)),
                     'conf':0.001,'iou':0.5,'max_det':100,'filter':'fine 0-5 removed',
                     'group_recall':'all predictions class-agnostic, GT restricted to group'},
         'models':{},'attribution':{}}
    attrs={}
    for name,path in MODELS.items():
        if not path.exists():
            print('skip missing',name,path); continue
        preds=load_preds(path,n)
        out['models'][name]=overall(gts,preds)
        out['models'][name]['groups']={g:group_metrics(gts,preds,c) for g,c in GROUPS.items()}
        attrs[name]=attribution(gts,preds)
        out['attribution'][name]={g:summarize_attr(attrs[name],set(c)) for g,c in FAM.items()}
        print(name,round(out['models'][name]['map50'],6),round(out['models'][name]['ap_agn75'],6))
    # common matched GT across the three primary models
    prim=[k for k in ('YOLO11s','YOLO26l','D-FINE best') if k in attrs]
    if len(prim)>=2:
        sets=[{(r[0],r[1]) for r in attrs[k] if r[2]} for k in prim]
        common=set.intersection(*sets)
        out['common_matched']={'models':prim,'n_common':len(common),'per_model':{}}
        for k in prim:
            sel=[r for r in attrs[k] if (r[0],r[1]) in common]
            out['common_matched']['per_model'][k]={
                'mean_iou':float(np.mean([r[5] for r in sel])) if sel else 0.0,
                'cls_acc':float(np.mean([r[4]==r[3] for r in sel])) if sel else 0.0}
            for g,c in FAM.items():
                s=[r for r in sel if r[3] in c]
                out['common_matched']['per_model'][k][g]={
                    'n':len(s),'mean_iou':float(np.mean([r[5] for r in s])) if s else 0.0,
                    'cls_acc':float(np.mean([r[4]==r[3] for r in s])) if s else 0.0}
    (OUT/'three_model_comparison.json').write_text(json.dumps(out,indent=2))
    md=['# 三模型同口径最终对照（val 1,543 / 7,162 GT / 24,975 视图）','',
        '协议：C1 log-power、0.7 ms STFT、单通道 2048×512 粗图＋细片；conf 0.001、'
        'IoU 0.5、全局 top-100；主口径为细片 0-5 过滤。test 封存。','',
        '## 总体','','| 模型 | mAP50 | 50:95 | AP_agn50 | AP_agn75 | 召回 | 分类准确率 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,v in out['models'].items():
        md.append(f"| {name} | {v['map50']:.6f} | {v['map50_95']:.6f} | {v['ap_agn50']:.4f} | {v['ap_agn75']:.4f} | {v['recall']:.4f} | {v['cls_acc']:.4f} |")
    md += ['','## 分组（过滤）','']
    for name,v in out['models'].items():
        md += [f'### {name}','','| 组 | mAP50(strict) | recall(全预测) | recall(旧按类) |','| --- | ---: | ---: | ---: |']
        for g,x in v['groups'].items():
            md.append(f"| {g} | {x['map50_strict']:.4f} | {x['recall_all']:.4f} | {x['recall_strict_predclass']:.4f} |")
        md += ['']
    md += ['## 归因：全部预测框类无关匹配，按 GT 类别分组','','| 模型 | 组 | GT | 匹配 | recall | 平均 IoU | 分类正确率 |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name,groups in out['attribution'].items():
        for g,x in groups.items():
            md.append(f"| {name} | {g} | {x['n_gt']} | {x['matched']} | {x['recall']:.4f} | {x['mean_iou']:.4f} | {x['cls_acc']:.4f} |")
    if 'common_matched' in out:
        cm=out['common_matched']; md += ['','## 共同匹配目标','',
            f"共同匹配 GT：**{cm['n_common']}**（模型：{', '.join(cm['models'])}）",'',
            '| 模型 | 平均 IoU | 分类正确率 |','| --- | ---: | ---: |']
        for name,x in cm['per_model'].items():
            md.append(f"| {name} | {x['mean_iou']:.4f} | {x['cls_acc']:.4f} |")
        md += ['','| 模型 / 组 | n | 平均 IoU | 分类正确率 |','| --- | ---: | ---: | ---: |']
        for name,x in cm['per_model'].items():
            for g in FAM:
                if g in x:
                    md.append(f"| {name} / {g} | {x[g]['n']} | {x[g]['mean_iou']:.4f} | {x[g]['cls_acc']:.4f} |")
    md += ['','## 备注','',
           '- mAP50/50:95 为主指标；分组 recall 为全部预测框的类别无关匹配（修正口径）。',
           '- D-FINE best 为训练中过滤 mAP50 最高轮；last 为第 50 轮，用于展示末段分类退化。',
           '- 归因按 GT 类别分组，先匹配再分组，避免跨组错类被计成漏定位。']
    (OUT/'three_model_comparison.md').write_text('\n'.join(md)+'\n')
    print('wrote', OUT/'three_model_comparison.md')
    print(json.dumps(out['common_matched'],indent=1)[:1200])

if __name__=='__main__': main()
