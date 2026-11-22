#!/usr/bin/env python3
"""After D-FINE training: full-val best.pt and last.pt with the corrected protocol."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports/detector_upgrade'; OUT.mkdir(parents=True,exist_ok=True)
PYTHON='/root/autodl-tmp/workspace/baseline/.venv/bin/python'

def ap75(path):
    import sys; sys.path.insert(0,str(ROOT))
    from src.signals.metrics import ImagePredictions, ImageTargets, evaluate
    z=np.load(path)
    if 'gt_boxes' not in z:
        raise RuntimeError(f'{path} has no GT fields; use corrected validator')
    n=int(z['gt_sample_index'].max())+1 if len(z['gt_sample_index']) else 0
    preds=[ImagePredictions(z['pred_boxes'][z['pred_sample_index']==i],
                            z['pred_scores'][z['pred_sample_index']==i],
                            z['pred_labels'][z['pred_sample_index']==i]) for i in range(n)]
    gts=[ImageTargets(z['gt_boxes'][z['gt_sample_index']==i], z['gt_labels'][z['gt_sample_index']==i]) for i in range(n)]
    r=evaluate(gts,preds,iou_thresholds=(0.75,),max_dets=100)
    return float(r['per_iou']['0.75']['ap_class_agnostic'])

def eval_ckpt(run, ck_name, out_name):
    run_dir=ROOT/'runs'/run; out=run_dir/out_name
    subprocess.run([PYTHON,str(ROOT/'scripts/validate_dfine.py'),'--run',run,
                    '--checkpoint',ck_name,'--output',str(out)],check=True)
    val=json.loads((out/'summary.json').read_text()); f=val['settings']['filtered_0_5']; u=val['settings']['unfiltered']
    ck=__import__('torch').load(run_dir/ck_name,map_location='cpu',weights_only=False)
    return {'checkpoint':ck_name,'epoch':ck.get('epoch'),'global_step':ck.get('global_step'),
            'filtered_map50':f['map50'],'filtered_map50_95':f['map50_95'],
            'unfiltered_map50':u['map50'],'unfiltered_map50_95':u['map50_95'],
            'ap50_agnostic':f['ap50_agnostic'],'ap75_agnostic':ap75(out/'val_predictions_filtered_0_5.npz'),
            'localization_recall':f['localization_recall'],'groups':f['groups'],
            'per_class':f['per_class'],'summary_path':str(out/'summary.json')}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',default='dfine_l_c1_2048x512'); a=ap.parse_args()
    run_dir=ROOT/'runs'/a.run
    if not (run_dir/'summary.json').exists():
        (OUT/'formal_dfine_results.json').write_text(json.dumps({'run':a.run,'status':'failed_or_incomplete'},indent=2)); return
    tr=json.loads((run_dir/'summary.json').read_text())
    ref_y11=0.649725; ref_y26=0.659601
    rows={}
    for ck_name,out_name in [('best.pt','val_full_best'),('last.pt','val_full_last')]:
        if (run_dir/ck_name).exists():
            print(f'evaluating {ck_name} -> {out_name}',flush=True)
            rows[ck_name]=eval_ckpt(a.run,ck_name,out_name)
    best=rows.get('best.pt') or rows.get('last.pt')
    res={'run':a.run,'status':'completed','best_epoch':tr.get('best_epoch'),'total_h':tr.get('total_h'),
         'reference_yolo11':ref_y11,'reference_yolo26l':ref_y26,
         'checkpoints':rows,
         'best_filtered_map50':best['filtered_map50'],
         'best_minus_yolo11_pp':(best['filtered_map50']-ref_y11)*100,
         'best_minus_yolo26_pp':(best['filtered_map50']-ref_y26)*100,
         'val_history':tr.get('val_history'),'config':tr.get('config')}
    (OUT/'formal_dfine_results.json').write_text(json.dumps(res,indent=2))
    md=['# D-FINE-L 正式训练最终结果（修正口径）','',
        f"- total {res['total_h']:.2f} h; best epoch {res['best_epoch']}",
        '- 过滤 mAP@0.5 为最佳权重选择口径；best/last 均完整复评；',
        '- 分组 recall 为**全部预测框**做类别无关匹配后按 GT 组汇总；',
        '- AP@0.75 为不计类别口径；test 封存。','',
        '| 权重 | 轮次 | 过滤 mAP50 | 50:95 | 无过滤 mAP50 | AP_agn50 | AP_agn75 | 召回 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,r in rows.items():
        md.append(f"| {name} | {r['epoch']+1 if r['epoch'] is not None else '-'} | {r['filtered_map50']:.6f} | {r['filtered_map50_95']:.6f} | {r['unfiltered_map50']:.6f} | {r['ap50_agnostic']:.4f} | {r['ap75_agnostic']:.4f} | {r['localization_recall']:.4f} |")
    md += ['',f"对比：YOLO11s 0.649725（{res['best_minus_yolo11_pp']:+.2f} pp）、"
           f"YOLO26l 0.659601（{res['best_minus_yolo26_pp']:+.2f} pp）。",'','## 分组（best）','',
           '| 组 | mAP50 | 50:95 | AP_agn50 | AP_agn75 | recall(全预测) | recall(旧按类) |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for g,v in best['groups'].items():
        md.append(f"| {g} | {v['map50']:.4f} | {v['map50_95']:.4f} | {v['ap50_agnostic']:.4f} | {v['ap75_agnostic']:.4f} | {v['recall']:.4f} | {v['recall_strict_predclass']:.4f} |")
    (OUT/'formal_dfine_results.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({'best_map50':best['filtered_map50'],'delta_y11_pp':res['best_minus_yolo11_pp'],
                      'delta_y26_pp':res['best_minus_yolo26_pp'],'ap75':best['ap75_agnostic']},ensure_ascii=False))

if __name__=='__main__': main()
