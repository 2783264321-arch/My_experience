"""Optional post-processing: suppress high-score Wi-Fi detections whose
modulation order the OFDM delta-phase expert confidently rejects.

Pipeline position: after detector inference and Soft-NMS merge, before AP.
Only detections with a Wi-Fi label (0-5) and score >= score_thr are examined.
The expert sees the final box geometry, runs the validated CP/FFT frontend,
builds 50x50 delta-phase histograms, and averages the 3 SHARED seeds.
A detection is dropped when the expert confidence >= conf_thr and the
expert's modulation order differs from the detector's class.
"""
from __future__ import annotations

import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.signals.data import N_TIME
from src.signals.metrics import ImagePredictions

# Validated frontend/histogram/expert code.
from scripts.baseline3_dsp import read_segment, demod_to_center, band_limit_fft
from scripts.baseline3b_ofdm_frontend import (
    cp_scan_full, strip_symbols, filter_symbol_rows, legacy_masks,
)
from scripts.baseline3b_histogram import (
    block_histogram, histogram_for_training, consecutive_runs,
)
from scripts.baseline3b_train_experts import HistCNN

PASS_METRIC = 0.34
PASS_RATIO = 15.0
LOW_METRIC = 0.28
LOW_RATIO = 8.0
SCS_HZ = 312.5e3
MAX_SYMBOLS = 4000
BLOCK = 128
WIFI_CLASSES = (0, 1, 2, 3, 4, 5)
SEEDS = (1729, 1730, 1731)
DEFAULT_WEIGHTS = Path('/root/autodl-fs/runs_local/dfine_wifi_expert/weights')


def box_to_physical(x1, y1, x2, y2, B_mhz, f_lo, N):
    """Normalized global xyxy -> physical f1,f2,t1_ms,t2_ms."""
    fs = B_mhz * 1e6
    h = math.ceil(N / 511)
    return (f_lo + y1 * B_mhz, f_lo + y2 * B_mhz,
            (x1 * 512 - 0.5) * h / fs * 1000,
            (x2 * 512 - 0.5) * h / fs * 1000)


def _frontend(args):
    (bin_path, f_lo, B_mhz, N, f_center, t1_ms, t2_ms, nominal_bw) = args
    try:
        fs = B_mhz * 1e6
        f_lo_hz = f_lo * 1e6
        B_hz = B_mhz * 1e6
        fc = f_center * 1e6
        bw = nominal_bw * 1e6
        f1 = max(f_lo_hz, fc - bw / 2)
        f2 = min(f_lo_hz + B_hz, fc + bw / 2)
        if f2 - f1 < bw * 0.5:
            return {'status': 'invalid_band'}
        ta = max(0.0, t1_ms)
        tb = min(t2_ms, 1e9)
        dur = tb - ta
        if dur < 0.5:
            return {'status': 'invalid_duration'}
        cap = min(dur, 20.0)
        tc = (ta + tb) / 2
        i0 = max(0, int(round((tc - cap / 2) / 1000 * fs)))
        i1 = min(int(N), int(round((tc + cap / 2) / 1000 * fs)))
        if i1 - i0 < 64:
            return {'status': 'too_short'}
        x = read_segment(bin_path, i0, i1)
        xb = demod_to_center(x, fs, fc, f_lo * 1e6)
        xb = band_limit_fft(xb, fs, bw * 1.02, edge_frac=0.05)
        k = int(0.5e-3 * fs)
        if len(xb) > 2 * k:
            xb = xb[k:-k]
        res = cp_scan_full(xb, fs)
        b = res['best']
        if b is None:
            return {'status': 'no_candidate'}
        if b['metric'] < PASS_METRIC or b['ratio'] < PASS_RATIO:
            return {'status': 'sync_fail', 'metric': b['metric'], 'ratio': b['ratio']}
        X = strip_symbols(xb, fs, b['tu_us'], b['cp_us'], b['phi'],
                          cfo_cycles=-b['cfo_cycles'], max_sym=MAX_SYMBOLS)
        _, _, data = legacy_masks(int(round(nominal_bw * 1e6 / SCS_HZ)))
        X, keep_idx, n_kept, _ = filter_symbol_rows(X, data)
        if n_kept < BLOCK:
            return {'status': 'too_few_symbols', 'n_kept': n_kept,
                    'metric': b['metric'], 'ratio': b['ratio'],
                    'tu_us': b['tu_us'], 'cp_us': b['cp_us'], 'pos': pos}
        blocks = []
        for a, e in consecutive_runs(keep_idx):
            run = X[a:e]
            for kk in range(len(run) // BLOCK):
                H = block_histogram(run[kk * BLOCK:(kk + 1) * BLOCK], data)
                if H is not None:
                    Ht = histogram_for_training(H)
                    if Ht is not None:
                        blocks.append(Ht)
                if len(blocks) >= 4:
                    break
            if len(blocks) >= 4:
                break
        if not blocks:
            return {'status': 'no_histogram'}
        return {'status': 'ok', 'blocks': np.stack(blocks).astype(np.float32)}
    except Exception as e:
        return {'status': 'error', 'error': str(e)[:200]}


def _frontend_multi(args):
    """Run CP sync on start/center/end windows; use the best metric window."""
    (bin_path, f_lo, B_mhz, N, f_center, t1_ms, t2_ms, nominal_bw, positions) = args
    try:
        fs = B_mhz * 1e6
        f_lo_hz = f_lo * 1e6
        B_hz = B_mhz * 1e6
        fc = f_center * 1e6
        bw = nominal_bw * 1e6
        f1 = max(f_lo_hz, fc - bw / 2)
        f2 = min(f_lo_hz + B_hz, fc + bw / 2)
        if f2 - f1 < bw * 0.5:
            return {'status': 'invalid_band'}
        ta0 = max(0.0, t1_ms)
        tb0 = min(t2_ms, 1e9)
        dur = tb0 - ta0
        if dur < 0.5:
            return {'status': 'invalid_duration'}
        win = min(20.0, dur)
        best = None
        for p in positions:
            ta = ta0 + float(p) * max(0.0, dur - win)
            tb = ta + win
            i0 = max(0, int(round(ta / 1000 * fs)))
            i1 = min(int(N), int(round(tb / 1000 * fs)))
            if i1 - i0 < 64:
                continue
            x = read_segment(bin_path, i0, i1)
            xb = demod_to_center(x, fs, fc, f_lo * 1e6)
            xb = band_limit_fft(xb, fs, bw * 1.02, edge_frac=0.05)
            k = int(0.5e-3 * fs)
            if len(xb) > 2 * k:
                xb = xb[k:-k]
            res = cp_scan_full(xb, fs)
            b = res['best']
            if b is not None and (best is None or b['metric'] > best[0]['metric']):
                best = (b, xb, float(p))
        if best is None:
            return {'status': 'no_candidate'}
        b, xb, pos = best
        strict = (b['metric'] >= PASS_METRIC and b['ratio'] >= PASS_RATIO)
        low = (b['metric'] >= LOW_METRIC and b['ratio'] >= LOW_RATIO)
        if not low:
            return {'status': 'sync_fail', 'metric': b['metric'], 'ratio': b['ratio'],
                    'tu_us': b['tu_us'], 'cp_us': b['cp_us'], 'pos': pos}
        X = strip_symbols(xb, fs, b['tu_us'], b['cp_us'], b['phi'],
                          cfo_cycles=-b['cfo_cycles'], max_sym=MAX_SYMBOLS)
        _, _, data = legacy_masks(int(round(nominal_bw * 1e6 / SCS_HZ)))
        X, keep_idx, n_kept, _ = filter_symbol_rows(X, data)
        if n_kept < BLOCK:
            return {'status': 'too_few_symbols', 'n_kept': n_kept,
                    'metric': b['metric'], 'ratio': b['ratio'],
                    'tu_us': b['tu_us'], 'cp_us': b['cp_us'], 'pos': pos}
        blocks = []
        for a, e in consecutive_runs(keep_idx):
            run = X[a:e]
            for kk in range(len(run) // BLOCK):
                H = block_histogram(run[kk * BLOCK:(kk + 1) * BLOCK], data)
                if H is not None:
                    Ht = histogram_for_training(H)
                    if Ht is not None:
                        blocks.append(Ht)
                if len(blocks) >= 4:
                    break
            if len(blocks) >= 4:
                break
        if not blocks:
            return {'status': 'no_histogram'}
        return {'status': 'ok' if strict else 'ok_low',
                'blocks': np.stack(blocks).astype(np.float32),
                'metric': b['metric'], 'ratio': b['ratio'],
                'tu_us': b['tu_us'], 'cp_us': b['cp_us'], 'pos': pos,
                'n_kept': n_kept}
    except Exception as e:
        return {'status': 'error', 'error': str(e)[:200]}


def _frontend_short(args):
    """5 ms start-region windows (negative offsets included) + strict gate."""
    (bin_path, f_lo, B_mhz, N, f_center, t1_ms, t2_ms, nominal_bw, win_ms, offsets_ms) = args
    try:
        fs = B_mhz * 1e6
        f_lo_hz = f_lo * 1e6
        B_hz = B_mhz * 1e6
        fc = f_center * 1e6
        bw = nominal_bw * 1e6
        f1 = max(f_lo_hz, fc - bw / 2)
        f2 = min(f_lo_hz + B_hz, fc + bw / 2)
        if f2 - f1 < bw * 0.5:
            return {'status': 'invalid_band'}
        ta0 = max(0.0, t1_ms)
        best = None
        for off in offsets_ms:
            ta = ta0 + float(off)
            tb = ta + float(win_ms)
            i0 = max(0, int(round(ta / 1000 * fs)))
            i1 = min(int(N), int(round(tb / 1000 * fs)))
            if i1 - i0 < 64:
                continue
            x = read_segment(bin_path, i0, i1)
            xb = demod_to_center(x, fs, fc, f_lo * 1e6)
            xb = band_limit_fft(xb, fs, bw * 1.02, edge_frac=0.05)
            k = int(0.5e-3 * fs)
            if len(xb) > 2 * k:
                xb = xb[k:-k]
            res = cp_scan_full(xb, fs)
            b = res['best']
            if b is not None and (best is None or b['metric'] > best[0]['metric']):
                best = (b, xb, float(off))
        if best is None:
            return {'status': 'no_candidate'}
        b, xb, off = best
        if b['metric'] < PASS_METRIC or b['ratio'] < PASS_RATIO:
            return {'status': 'sync_fail', 'metric': b['metric'], 'ratio': b['ratio'],
                    'tu_us': b['tu_us'], 'cp_us': b['cp_us'], 'pos': off}
        X = strip_symbols(xb, fs, b['tu_us'], b['cp_us'], b['phi'],
                          cfo_cycles=-b['cfo_cycles'], max_sym=MAX_SYMBOLS)
        _, _, data = legacy_masks(int(round(nominal_bw * 1e6 / SCS_HZ)))
        X, keep_idx, n_kept, _ = filter_symbol_rows(X, data)
        if n_kept < BLOCK:
            return {'status': 'too_few_symbols', 'n_kept': n_kept,
                    'metric': b['metric'], 'ratio': b['ratio'],
                    'tu_us': b['tu_us'], 'cp_us': b['cp_us'], 'pos': off}
        blocks = []
        for a, e in consecutive_runs(keep_idx):
            run = X[a:e]
            for kk in range(len(run) // BLOCK):
                H = block_histogram(run[kk * BLOCK:(kk + 1) * BLOCK], data)
                if H is not None:
                    Ht = histogram_for_training(H)
                    if Ht is not None:
                        blocks.append(Ht)
                if len(blocks) >= 4:
                    break
            if len(blocks) >= 4:
                break
        if not blocks:
            return {'status': 'no_histogram'}
        return {'status': 'ok', 'blocks': np.stack(blocks).astype(np.float32),
                'metric': b['metric'], 'ratio': b['ratio'],
                'tu_us': b['tu_us'], 'cp_us': b['cp_us'], 'pos': off,
                'n_kept': n_kept}
    except Exception as e:
        return {'status': 'error', 'error': str(e)[:200]}


def _as_numpy(pred):
    return (np.asarray(pred.boxes, dtype=np.float32),
            np.asarray(pred.scores, dtype=np.float32),
            np.asarray(pred.labels, dtype=np.int64))


def _load_experts(weights_dir: Path, device: str, arm: str = 'shared'):
    arms = ('SHARED',) if arm == 'shared' else ('W20', 'W40')
    out = {}
    for a in arms:
        models = []
        for seed in SEEDS:
            path = Path(weights_dir) / f'{a}_seed{seed}.pt'
            if not path.exists():
                raise FileNotFoundError(path)
            m = HistCNN(3).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models.append(m)
        out[a] = models
    return out


def _classify(feats, ok_idx, models, device):
    if not ok_idx:
        return {}
    blocks = [feats[i]['blocks'] for i in ok_idx]
    maxb = max(x.shape[0] for x in blocks)
    X = np.zeros((len(blocks), maxb, 1, 50, 50), np.float32)
    B = np.zeros((len(blocks), maxb), np.float32)
    for j, x in enumerate(blocks):
        X[j, :x.shape[0]] = x
        B[j, :x.shape[0]] = 1
    Xt = torch.from_numpy(X).to(device)
    Bt = torch.from_numpy(B).to(device)
    acc = 0.0
    with torch.no_grad():
        for m in models:
            out = F.softmax(m(Xt.reshape(-1, 1, 50, 50)), dim=1)
            out = out.reshape(len(blocks), maxb, 3)
            out = (out * Bt[:, :, None]).sum(1) / Bt.sum(1, keepdim=True).clamp_min(1)
            acc = acc + out.cpu().numpy()
    prob = acc / len(models)
    return {i: prob[j] for j, i in enumerate(ok_idx)}


def suppress(records: Sequence, preds: Sequence, score_thr: float = 0.3,
             conf_thr: float = 0.9, workers: int = 16,
             weights_dir: Path = DEFAULT_WEIGHTS, device: str | None = None,
             expert_arm: str = 'shared', window_mode: str = 'multi',
             low_sync: bool = False, low_conf: float = 0.99,
             short_start: bool = False, short_conf: float = 0.9,
             short_win_ms: float = 5.0,
             short_offsets: tuple = (-2.0, -1.0, 0.0, 1.0, 2.0)):
    """Return (filtered_preds, stats).

    ``preds[i]`` is the final merged detection set for ``records[i]``.
    Boxes are normalized global xyxy.  No GT is used.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    jobs, refs = [], []
    for si, (rec, pred) in enumerate(zip(records, preds)):
        boxes, scores, labels = _as_numpy(pred)
        for k in range(len(labels)):
            lab, sc = int(labels[k]), float(scores[k])
            if lab not in WIFI_CLASSES or sc < score_thr:
                continue
            f1, f2, t1, t2 = box_to_physical(*boxes[k], rec.B_mhz, rec.f_lo, rec.N)
            job = (str(rec.bin_path), rec.f_lo, rec.B_mhz, rec.N,
                   (f1 + f2) / 2, t1, t2, 20 if lab <= 2 else 40)
            if window_mode == 'multi':
                job = job + ((0.0, 0.5, 1.0),)
            jobs.append(job)
            refs.append((si, k, lab))
    if not jobs:
        return list(preds), {'n_jobs': 0, 'sync_ok': 0, 'n_drops': 0}
    feats = []
    fn = _frontend_multi if window_mode == 'multi' else _frontend
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(fn, jobs, chunksize=16):
            feats.append(r)
    ok_idx = [i for i, r in enumerate(feats) if r['status'] in ('ok', 'ok_low')]
    models = _load_experts(weights_dir, device, expert_arm)
    if expert_arm == 'shared':
        probs = _classify(feats, ok_idx, models['SHARED'], device)
    else:
        probs = {}
        for a, fam in (('W20', 0), ('W40', 1)):
            sub = [i for i in ok_idx if ((refs[i][2] <= 2) == (fam == 0))]
            if sub:
                probs.update(_classify(feats, sub, models[a], device))
    drop = set()
    drops = Counter()
    for i in ok_idx:
        si, k, lab = refs[i]
        p = probs[i]
        pm = int(np.argmax(p))
        conf = float(p.max())
        expert = pm if lab <= 2 else 3 + pm
        status = feats[i]['status']
        thr = conf_thr if status == 'ok' else low_conf
        if low_sync and status == 'ok_low' and conf >= thr and expert != lab:
            drop.add((si, k)); drops[lab] += 1
        elif status == 'ok' and conf >= conf_thr and expert != lab:
            drop.add((si, k)); drops[lab] += 1
    short_stats = {}
    if short_start:
        fail_idx = [i for i, r in enumerate(feats)
                    if r['status'] not in ('ok', 'ok_low')]
        short_jobs = [jobs[i][:8] + (short_win_ms, short_offsets) for i in fail_idx]
        short_feats = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(_frontend_short, short_jobs, chunksize=16):
                short_feats.append(r)
        short_ok = [j for j, r in enumerate(short_feats) if r['status'] == 'ok']
        short_probs = _classify(short_feats, short_ok, models['SHARED'], device)
        short_drops = Counter()
        for j in short_ok:
            i = fail_idx[j]
            si, k, lab = refs[i]
            p = short_probs[j]
            pm = int(np.argmax(p)); conf = float(p.max())
            expert = pm if lab <= 2 else 3 + pm
            if conf >= short_conf and expert != lab:
                drop.add((si, k)); short_drops[lab] += 1; drops[lab] += 1
        short_stats = {
            'n_short_jobs': len(short_jobs), 'short_ok': len(short_ok),
            'short_status_counts': dict(Counter(r['status'] for r in short_feats)),
            'n_short_drops': int(sum(short_drops.values())),
            'short_drops_by_class': dict(sorted(short_drops.items())),
            'short_conf': short_conf, 'short_win_ms': short_win_ms,
        }
    out = []
    for si, pred in enumerate(preds):
        boxes, scores, labels = _as_numpy(pred)
        keep = [k for k in range(len(labels)) if (si, k) not in drop]
        out.append(ImagePredictions(boxes[keep], scores[keep], labels[keep]))
    stats = {
        'n_jobs': len(jobs), 'sync_ok': len(ok_idx),
        'sync_rate': len(ok_idx) / max(len(jobs), 1),
        'status_counts': dict(Counter(r['status'] for r in feats)),
        'n_drops': len(drop), 'drops_by_class': dict(sorted(drops.items())),
        'score_thr': score_thr, 'conf_thr': conf_thr,
        'expert_arm': expert_arm, 'window_mode': window_mode,
        'low_sync': low_sync, 'low_conf': low_conf,
        'short_start': short_start, **short_stats,
    }
    return out, stats
