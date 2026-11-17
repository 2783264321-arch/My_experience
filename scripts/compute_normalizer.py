#!/usr/bin/env python3
"""Compute and freeze the fit-only log-power normalizer."""
import json
from pathlib import Path

from src.signals.data import load_manifest, compute_normalizer

out = Path("configs/normalizer.json")
records = load_manifest(split="fit")
stats = compute_normalizer(records, max_samples=128, seed=1729)
out.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in stats.items() if k != "sample_ids"}, ensure_ascii=False, indent=2))
print("wrote", out)
