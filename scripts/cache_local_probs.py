#!/usr/bin/env python3
"""Cache local C-model class probabilities for val_local_crops."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'scripts/detector_upgrade'))
from eval_local_iq_abc import load_model, predict_group

GROUPS = {"012": [0,1,2], "345": [3,4,5], "1011": [10,11]}
CROPS = Path('/root/autodl-fs/runs_local/dfine_local_iq/val_local_crops.npz')
MODELS = Path('/root/autodl-fs/runs_local/dfine_local_iq/models')
OUT = Path('/root/autodl-fs/runs_local/dfine_local_iq/val_local_probs.npz')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with np.load(CROPS, allow_pickle=True) as zf:
    z = {k: zf[k] for k in zf.files}
payload = {}
for g, cls in GROUPS.items():
    cls = np.asarray(cls, dtype=np.int64)
    sel = np.flatnonzero(np.isin(z['label'].astype(np.int64), cls))
    model, ck = load_model(MODELS / g / 'features' / 'best.pt', device)
    meta = np.concatenate([
        z['class_scores'][sel][:, cls].astype(np.float32),
        z['score'][sel][:, None].astype(np.float32),
        np.log1p(z['bandwidth_mhz'][sel])[:, None].astype(np.float32),
        np.log1p(z['duration_ms'][sel])[:, None].astype(np.float32),
        z['view_type'][sel][:, None].astype(np.float32),
    ], axis=1).astype(np.float32)
    probs = predict_group(model, ck, z['local_power'][sel], z['features'][sel],
                          meta, True, device)
    payload[f'idx_{g}'] = sel.astype(np.int64)
    payload[f'cls_{g}'] = cls
    payload[f'probs_{g}'] = probs.astype(np.float32)
    print(g, 'rows', len(sel), 'shape', probs.shape, flush=True)
np.savez_compressed(OUT, **payload)
print('wrote', OUT, flush=True)
