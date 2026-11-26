#!/usr/bin/env python3
"""Summarize per-GT post-processing error decomposition."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

P = Path('/root/autodl-fs/runs_local/dfine_local_iq/postprocess_errors.json')
d = json.loads(P.read_text())
rows = d['rows']; modes = d['modes']
GROUPS = ['WiFi20','WiFi40','SRRC','Narrow','other']

# Flatten per mode.
gt = {m: {} for m in modes}   # (sample,gi) -> row
fp = {m: Counter() for m in modes}
fp_score = {m: defaultdict(float) for m in modes}
n_pred = {m: 0 for m in modes}
for r in rows:
    si = r['sample']
    for m in modes:
        md = r['modes'].get(m)
        if md is None: continue
        n_pred[m] += md['n_pred']
        for g in md['gts']:
            gt[m][(si, g['gi'])] = g
        for f in md['fp']:
            key = (f['git'] if 'git' in f else f['group'], f['src'], f['cls'])
            fp[m][key] += f['n']; fp_score[m][key] += f['score_sum']

print('=== GT outcomes per mode ===')
for m in modes:
    c = Counter(g['outcome'] for g in gt[m].values())
    tot = sum(c.values())
    print(f'\n{m}: n_gt={tot} n_pred={n_pred[m]}')
    for k in ('correct','unmatched_correct','wrong','missed'):
        print(f'  {k:18s} {c[k]:5d} {c[k]/max(tot,1)*100:6.2f}%')
    # candidate availability for wrong/missed
    for k in ('wrong','missed','unmatched_correct'):
        sub = [g for g in gt[m].values() if g['outcome'] == k]
        if not sub: continue
        has_corr = sum((g['n_cands_correct'] or 0) > 0 for g in sub)
        has_any = sum((g['n_cands'] or 0) > 0 for g in sub)
        print(f'  {k}: has_correct_candidate {has_corr}/{len(sub)} ({has_corr/len(sub)*100:.1f}%), '
              f'has_any_candidate {has_any}/{len(sub)} ({has_any/len(sub)*100:.1f}%)')
    # per group
    print('  per group:')
    for grp in GROUPS:
        sub = [g for g in gt[m].values() if g['group'] == grp]
        if not sub: continue
        c = Counter(g['outcome'] for g in sub)
        print(f'    {grp:7s} n={len(sub):5d} correct {c["correct"]/len(sub)*100:5.1f}% '
              f'unmatched {c["unmatched_correct"]/len(sub)*100:4.1f}% '
              f'wrong {c["wrong"]/len(sub)*100:4.1f}% missed {c["missed"]/len(sub)*100:4.1f}%')

print('\n=== FP sources per mode (count / score sum) ===')
for m in modes:
    print(f'\n{m}:')
    agg = defaultdict(lambda: [0,0.0])
    for (grp,src,cls),v in fp[m].items():
        agg[(grp,src)][0] += v; agg[(grp,src)][1] += fp_score[m][(grp,src,cls)]
    for (grp,src),(n,s) in sorted(agg.items()):
        print(f'  {grp:7s} {src:12s} n={n:6d} score_sum={s:9.1f}')

print('\n=== Soft vs NMS per-GT transitions ===')
nms = gt['nms']; soft = gt['soft']
keys = set(nms) | set(soft)
trans = Counter(); by_group = defaultdict(Counter)
for k in keys:
    a = nms.get(k, {}).get('outcome'); b = soft.get(k, {}).get('outcome')
    if a is None or b is None: continue
    grp = soft[k]['group']
    if a == 'correct' and b == 'correct': t='both_correct'
    elif a != 'correct' and b == 'correct': t='soft_only'
    elif a == 'correct' and b != 'correct': t='nms_only'
    else: t='both_not_correct'
    trans[t]+=1; by_group[grp][t]+=1
print(dict(trans))
for grp,c in by_group.items():
    print(' ',grp,dict(c))

print('\n=== WBF vs Soft per-GT transitions ===')
wbf = gt['wbf']
trans = Counter(); by_group = defaultdict(Counter)
for k in keys:
    a = soft.get(k, {}).get('outcome'); b = wbf.get(k, {}).get('outcome')
    if a is None or b is None: continue
    grp = soft[k]['group']
    if a == 'correct' and b == 'correct': t='both_correct'
    elif a != 'correct' and b == 'correct': t='wbf_only'
    elif a == 'correct' and b != 'correct': t='soft_only'
    else: t='both_not_correct'
    trans[t]+=1; by_group[grp][t]+=1
print(dict(trans))
for grp,c in by_group.items():
    print(' ',grp,dict(c))

print('\n=== Duplicate structure for correct GTs ===')
for m in modes:
    for grp in GROUPS:
        sub=[g for g in gt[m].values() if g['outcome']=='correct' and g['group']==grp]
        if not sub: continue
        nd=[g['n_dets_any'] for g in sub]; nc=[g['n_dets_correct'] for g in sub]
        print(f'  {m:5s} {grp:7s} n={len(sub):5d} any/det mean {np.mean(nd):4.2f} >1 {np.mean(np.array(nd)>1)*100:4.1f}% '
              f'correct/det mean {np.mean(nc):4.2f} >1 {np.mean(np.array(nc)>1)*100:4.1f}%')
