#!/usr/bin/env python3
"""After D-FINE training: full-val best.pt, write results and three-model comparison."""
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
    n=int(z['pred_sample_index'].max())+1 if len(z['pred_sample_index']) else 0
    preds=[ImagePredictions(z['pred_boxes'][z['pred_sample_index']==i],
                            z['pred_scores'][z['pred_sample_index']==i],
                            z['pred_labels'][z['pred_sample_index']==i]) for i in range(n)]
    gts=[ImageTargets(z['gt_boxes'][z['gt_sample_index']==i], z['gt_labels'][z['gt_sample_index']==i]) for i in range(n)]
    r=evaluate(gts,preds,iou_thresholds=(0.75,),max_dets=100)
    return float(r['per_iou']['0.75']['ap_class_agnostic'])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',default='dfine_l_c1_2048x512'); a=ap.parse_args()
    run_dir=ROOT/'runs'/a.run
    if not (run_dir/'summary.json').exists():
        (OUT/'formal_dfine_results.json').write_text(json.dumps({'run':a.run,'status':'failed_or_incomplete'},indent=2)); return
    tr=json.loads((run_dir/'summary.json').read_text())
    ck='best.pt' if (run_dir/'best.pt').exists() else 'last.pt'
    vd=run_dir/('val_full_best' if ck=='best.pt' else 'val_full_last')
    subprocess.run([PYTHON,str(ROOT/'scripts/validate_dfine.py'),'--run',a.run,'--checkpoint',ck,'--output',str(vd)],check=True)
    val=json.loads((vd/'summary.json').read_text()); f=val['settings']['filtered_0_5']; u=val['settings']['unfiltered']
    agn75=ap75(vd/'val_predictions_filtered_0_5.npz')
    ref_y11=0.649725; ref_y26=0.659601
    res={'run':a.run,'status':'completed','best_epoch':tr.get('best_epoch'),'total_h':tr.get('total_h'),
         'filtered_map50':f['map50'],'delta_vs_yolo11_pp':(f['map50']-ref_y11)*100,
         'delta_vs_yolo26_pp':(f['map50']-ref_y26)*100,'filtered_map50_95':f['map50_95'],
         'unfiltered_map50':u['map50'],'unfiltered_map50_95':u['map50_95'],
         'ap50_agnostic':f['ap50_agnostic'],'ap75_agnostic':agn75,
         'localization_recall':f['localization_recall'],'groups':f['groups'],'per_class':f['per_class'],
         'val_history':tr.get('val_history'),'config':tr.get('config')}
    (OUT/'formal_dfine_results.json').write_text(json.dumps(res,indent=2))
    md=['# D-FINE-L 正式训练最终结果','',f"- best epoch **{res['best_epoch']}**；总耗时 **{res['total_h']:.2f} h**",
        '', '| 指标 | 值 | 对比 YOLO11s 0.649725 | 对比 YOLO26l 0.659601 |','| --- | ---: | ---: | ---: |',
        f"| 过滤 mAP@0.5 | **{f['map50']:.6f}** | {res['delta_vs_yolo11_pp']:+.2f} pp | {res['delta_vs_yolo26_pp']:+.2f} pp |",
        f"| 过滤 mAP@0.5:0.95 | {f['map50_95']:.6f} | — | — |",
        f"| 无过滤 mAP@0.5 | {u['map50']:.6f} | — | — |",
        f"| 不计类 AP@0.5 | {f['ap50_agnostic']:.6f} | — | — |",
        f"| 不计类 AP@0.75 | {agn75:.6f} | — | — |",
        f"| 定位召回 | {f['localization_recall']:.6f} | — | — |",'','## 分组（过滤）','',
        '| 组 | mAP@0.5 | mAP@0.5:0.95 | recall |','| --- | ---: | ---: | ---: |']
    for g,v in f['groups'].items(): md.append(f"| {g} | {v['map50']:.4f} | {v['map50_95']:.4f} | {v['recall']:.4f} |")
    (OUT/'formal_dfine_results.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({'filtered_map50':f['map50'],'delta_y11_pp':res['delta_vs_yolo11_pp'],
                      'delta_y26_pp':res['delta_vs_yolo26_pp'],'ap75_agnostic':agn75},ensure_ascii=False))

if __name__=='__main__': main()
