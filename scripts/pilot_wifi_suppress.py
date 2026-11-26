#!/usr/bin/env python3
"""High-score Wi-Fi FP suppression pilot with the existing OFDM expert."""
from __future__ import annotations
import argparse, json, math, sys, time
from collections import Counter
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
from src.signals.metrics import ImagePredictions
from validate_dual import _eval_set
from baseline3_dsp import read_segment, demod_to_center, band_limit_fft
from baseline3b_ofdm_frontend import (cp_scan_full, strip_symbols,
                                      filter_symbol_rows, legacy_masks)
from baseline3b_histogram import (block_histogram, histogram_for_training,
                                  consecutive_runs)
from baseline3b_train_experts import HistCNN

PASS_METRIC, PASS_RATIO = 0.34, 15.0
SCS_HZ, MAX_SYMBOLS, BLOCK = 312.5e3, 4000, 128
CAND = Path('/root/autodl-fs/runs_local/dfine_local_iq/val_candidates.npz')
WEIGHTS = Path('/root/autodl-tmp/workspace/baseline/runs_local/baseline3b/weights')
_G = {}

def _init():
    torch.set_num_threads(1)

def _merge_sample(i):
    a, b = int(_G['offsets'][i]), int(_G['offsets'][i+1])
    p = Prediction(torch.from_numpy(_G['boxes'][a:b]).float(),
                   torch.from_numpy(_G['scores'][a:b]).float(),
                   torch.from_numpy(_G['labels'][a:b]).long())
    f = soft_classwise_nms(p, 0.5, 0.5, 1e-3, 100)
    return (f.boxes.numpy().astype(np.float32), f.scores.numpy().astype(np.float32),
            f.labels.numpy().astype(np.int64))

def box_to_physical(x1, y1, x2, y2, B_mhz, f_lo, N):
    fs = B_mhz * 1e6; h = math.ceil(N / 511)
    return (f_lo + y1 * B_mhz, f_lo + y2 * B_mhz,
            (x1 * 512 - 0.5) * h / fs * 1000,
            (x2 * 512 - 0.5) * h / fs * 1000)

def frontend(args):
    (bin_path, f_lo, B_mhz, N, f_center, t1_ms, t2_ms, nominal_bw) = args
    try:
        fs = B_mhz * 1e6; f_lo_hz = f_lo * 1e6; B_hz = B_mhz * 1e6
        fc = f_center * 1e6; bw = nominal_bw * 1e6
        f1 = max(f_lo_hz, fc - bw/2); f2 = min(f_lo_hz + B_hz, fc + bw/2)
        if f2 - f1 < bw * 0.5: return {'status': 'invalid_band'}
        ta = max(0.0, t1_ms); tb = min(t2_ms, 1e9); dur = tb - ta
        if dur < 0.5: return {'status': 'invalid_duration'}
        cap = min(dur, 20.0); tc = (ta + tb)/2
        i0 = max(0, int(round((tc - cap/2)/1000*fs)))
        i1 = min(int(N), int(round((tc + cap/2)/1000*fs)))
        if i1 - i0 < 64: return {'status': 'too_short'}
        x = read_segment(bin_path, i0, i1)
        xb = demod_to_center(x, fs, fc, f_lo*1e6)
        xb = band_limit_fft(xb, fs, bw*1.02, edge_frac=0.05)
        k = int(0.5e-3*fs)
        if len(xb) > 2*k: xb = xb[k:-k]
        res = cp_scan_full(xb, fs); b = res['best']
        if b is None: return {'status': 'no_candidate'}
        if b['metric'] < PASS_METRIC or b['ratio'] < PASS_RATIO:
            return {'status': 'sync_fail', 'metric': b['metric'], 'ratio': b['ratio']}
        X = strip_symbols(xb, fs, b['tu_us'], b['cp_us'], b['phi'],
                          cfo_cycles=-b['cfo_cycles'], max_sym=MAX_SYMBOLS)
        occ, pilots, data = legacy_masks(int(round(nominal_bw*1e6/SCS_HZ)))
        X, keep_idx, n_kept, _ = filter_symbol_rows(X, data)
        if n_kept < BLOCK: return {'status': 'too_few_symbols', 'n_kept': n_kept}
        blocks = []
        for a, e in consecutive_runs(keep_idx):
            run = X[a:e]
            for kk in range(len(run)//BLOCK):
                H = block_histogram(run[kk*BLOCK:(kk+1)*BLOCK], data)
                if H is not None:
                    Ht = histogram_for_training(H)
                    if Ht is not None: blocks.append(Ht)
                if len(blocks) >= 4: break
            if len(blocks) >= 4: break
        if not blocks: return {'status': 'no_histogram'}
        return {'status': 'ok', 'blocks': np.stack(blocks).astype(np.float32)}
    except Exception as e:
        return {'status': 'error', 'error': str(e)[:200]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--score-thr', type=float, default=0.3)
    ap.add_argument('--conf-thr', type=float, default=0.9)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--out', type=Path, default=Path('/root/autodl-fs/runs_local/dfine_local_iq/pilot_wifi_suppress.json'))
    args = ap.parse_args()
    with np.load(CAND, allow_pickle=True) as zf:
        c = {k: zf[k] for k in ('boxes','scores','labels','sample_offsets','sample_index')}
    meta = json.loads(CAND.with_suffix('.json').read_text()); n = int(meta['samples'])
    records = load_manifest(SPLIT_MANIFEST, split='val')[:n]
    gts = [gt_global(r, math.ceil(r.N/(N_TIME-1))) for r in records]
    offsets = c['sample_offsets']
    _G.update(boxes=c['boxes'], scores=c['scores'], labels=c['labels'], offsets=offsets)

    t0 = time.perf_counter(); merged = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as ex:
        for mb, ms, ml in ex.map(_merge_sample, range(n), chunksize=8):
            merged.append((mb, ms, ml))
    print(f'merge {time.perf_counter()-t0:.1f}s', flush=True)

    jobs, refs = [], []
    for si, (mb, ms, ml) in enumerate(merged):
        r = records[si]
        for k in range(len(ml)):
            lab, sc = int(ml[k]), float(ms[k])
            if lab not in (0,1,2,3,4,5) or sc < args.score_thr: continue
            f1,f2,t1,t2 = box_to_physical(*mb[k], r.B_mhz, r.f_lo, r.N)
            jobs.append((f'/root/autodl-fs/SpaceNet_Dataset/advanced/{r.sample_id}.bin',
                         r.f_lo, r.B_mhz, r.N, (f1+f2)/2, t1, t2, 20 if lab <= 2 else 40))
            refs.append((si, k, lab, sc, mb[k].copy()))
    print(f'wifi dets score>={args.score_thr}: {len(jobs)}', flush=True)

    t1 = time.perf_counter(); feats = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as ex:
        for x in ex.map(frontend, jobs, chunksize=16): feats.append(x)
    ok = [i for i, x in enumerate(feats) if x['status'] == 'ok']
    print('frontend', dict(Counter(x['status'] for x in feats)),
          f'sync_ok {len(ok)}/{len(feats)} = {len(ok)/max(len(feats),1)*100:.1f}%',
          f'{time.perf_counter()-t1:.1f}s', flush=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = []
    for seed in (1729,1730,1731):
        m = HistCNN(3).to(device)
        m.load_state_dict(torch.load(WEIGHTS/f'SHARED_seed{seed}.pt', map_location=device))
        m.eval(); models.append(m)
    blocks = [feats[i]['blocks'] for i in ok]
    probs = {}
    if blocks:
        maxb = max(x.shape[0] for x in blocks)
        X = np.zeros((len(blocks), maxb, 1, 50, 50), np.float32)
        B = np.zeros((len(blocks), maxb), np.float32)
        for j, x in enumerate(blocks):
            X[j,:x.shape[0]] = x; B[j,:x.shape[0]] = 1
        Xt = torch.from_numpy(X).to(device); Bt = torch.from_numpy(B).to(device)
        acc = 0
        with torch.no_grad():
            for m in models:
                out = F.softmax(m(Xt.reshape(-1,1,50,50)), dim=1).reshape(len(blocks),maxb,3)
                out = (out*Bt[:,:,None]).sum(1)/Bt.sum(1,keepdim=True).clamp_min(1)
                acc = acc + out.cpu().numpy()
        prob = acc/len(models)
        probs = {i: prob[j] for j,i in enumerate(ok)}

    rows = []
    for i in ok:
        si, k, lab, sc, box = refs[i]
        gb = gts[si].boxes.numpy(); gl = gts[si].labels.numpy().astype(np.int64)
        if len(gb):
            iou = box_iou(torch.from_numpy(box[None]).float(), torch.from_numpy(gb).float())[0].numpy()
            gi = int(np.argmax(iou)); best = float(iou[gi]); matched = best >= 0.5
            gt = int(gl[gi]) if matched else -1
        else: matched, best, gt = False, 0.0, -1
        p = probs[i]; pm = int(np.argmax(p)); conf = float(p.max())
        expert = pm if lab <= 2 else 3+pm
        rows.append({'si':si,'k':k,'label':lab,'score':sc,'gt':gt,'matched':matched,
                     'iou':best,'det_ok':matched and lab==gt,'expert':expert,
                     'expert_conf':conf,'expert_ok':matched and expert==gt,
                     'disagree':expert!=lab})
    matched = [x for x in rows if x['matched']]
    dis = [x for x in matched if x['disagree'] and x['expert_conf'] >= args.conf_thr]
    print(f'matched sync-ok {len(matched)}; det_acc {np.mean([x["det_ok"] for x in matched]):.4f} '
          f'expert_acc {np.mean([x["expert_ok"] for x in matched]):.4f}')
    print(f'confident disagreements {len(dis)}; precision(det wrong) '
          f'{np.mean([not x["det_ok"] for x in dis]) if dis else float("nan"):.4f}; '
          f'fixes {sum((not x["det_ok"]) and x["expert_ok"] for x in dis)}; '
          f'breaks {sum(x["det_ok"] for x in dis)}')
    for fam, cls in (('W20',(0,1,2)),('W40',(3,4,5))):
        sub=[x for x in dis if x['label'] in cls]
        if sub:
            print(f'  {fam}: n {len(sub)} precision {np.mean([not x["det_ok"] for x in sub]):.4f} '
                  f'fixes {sum((not x["det_ok"]) and x["expert_ok"] for x in sub)} breaks {sum(x["det_ok"] for x in sub)}')

    drop = {(x['si'],x['k']) for x in rows if x['disagree'] and x['expert_conf'] >= args.conf_thr}
    def metrics(drop_set):
        preds = []
        for si,(mb,ms,ml) in enumerate(merged):
            keep=[k for k in range(len(ml)) if (si,k) not in drop_set]
            preds.append(ImagePredictions(mb[keep],ms[keep],ml[keep]))
        m=_eval_set(gts,preds); m['n_pred']=int(sum(len(p.boxes) for p in preds)); return m
    base=metrics(set()); var=metrics(drop)
    print('baseline',base['map50'],base['map50_95'],'n_pred',base['n_pred'])
    print('suppress',var['map50'],var['map50_95'],'n_pred',var['n_pred'],'dropped',len(drop))
    out={'score_thr':args.score_thr,'conf_thr':args.conf_thr,'n_jobs':len(jobs),
         'sync_ok':len(ok),'sync_rate':len(ok)/max(len(jobs),1),
         'n_matched_sync':len(matched),'n_disagree_conf':len(dis),
         'precision_det_wrong':float(np.mean([not x['det_ok'] for x in dis])) if dis else None,
         'fixes':int(sum((not x['det_ok']) and x['expert_ok'] for x in dis)),
         'breaks':int(sum(x['det_ok'] for x in dis)),
         'baseline':base,'suppress':var,'rows':rows}
    args.out.write_text(json.dumps(out,indent=2)+'\n'); print('wrote',args.out)

if __name__=='__main__': main()
