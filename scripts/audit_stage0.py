#!/usr/bin/env python3
"""Stage-0 audit: paths, split, IQ spec, versions, hashes, resource budget."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.signals.data import DATA_ROOT, SPLIT_MANIFEST, load_manifest, load_iq

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = {
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "ultralytics": __import__("ultralytics").__version__,
            "sahi": __import__("sahi").__version__,
            "scipy": __import__("scipy").__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
            if torch.cuda.is_available() else None,
            "cpus_host": os.cpu_count(),
            "cgroup_cpu_max": open("/sys/fs/cgroup/cpu.max").read().strip()
            if Path("/sys/fs/cgroup/cpu.max").exists() else None,
            "cgroup_memory_max": open("/sys/fs/cgroup/memory.max").read().strip()
            if Path("/sys/fs/cgroup/memory.max").exists() else None,
            "cgroup_memory_gib": round(int(open("/sys/fs/cgroup/memory.max").read()) / 1024 ** 3, 2)
            if Path("/sys/fs/cgroup/memory.max").exists()
            and open("/sys/fs/cgroup/memory.max").read().strip().isdigit() else None,
            "disk_free_gib": round(shutil.disk_usage("/root/autodl-tmp").free / 1024 ** 3, 2),
            "ram_total_gib": round(
                int(open("/proc/meminfo").readline().split()[1]) / 1024 ** 2, 2
            ),
        },
        "paths": {
            "project_root": str(ROOT),
            "data_root": str(DATA_ROOT),
            "split_manifest": str(SPLIT_MANIFEST),
        },
        "hashes": {
            "split_manifest_sha256": sha256(SPLIT_MANIFEST),
            "normalizer_sha256": sha256(ROOT / "configs/normalizer.json"),
            "model_yaml_sha256": sha256(ROOT / "configs/yolo11s_c1_directreg.yaml"),
            "pretrained_sha256": sha256(ROOT / "yolo11s.pt"),
        },
        "split": {},
        "iq_spec": {
            "dtype": "float16 little-endian",
            "layout": "interleaved I,Q",
            "bytes_per_complex": 4,
            "Fs": "B (MHz) per user-confirmed official premise",
            "window_ms": 0.7,
            "n_time": 512,
            "fine_h": 2048,
            "fine_step": 1639,
            "coarse_h": 2048,
        },
        "checks": {},
    }
    all_recs = []
    for split in ["fit", "val", "test"]:
        recs = load_manifest(SPLIT_MANIFEST, split=split)
        all_recs.extend(recs)
        sizes_ok = 0
        json_n_ok = 0
        bounds_ok = 0
        t_ok = 0
        for r in recs:
            if r.bin_path.stat().st_size == 4 * r.N:
                sizes_ok += 1
            js = json.loads(r.json_path.read_text())
            if len(js.get("signals", [])) == len(r.signals):
                json_n_ok += 1
            if all(
                float(s["start_frequency"]) >= r.f_lo
                and float(s["end_frequency"]) <= r.f_hi
                and float(s["start_time"]) >= 0
                and float(s["end_time"]) <= r.T_ms
                and float(s["end_frequency"]) > float(s["start_frequency"])
                and float(s["end_time"]) > float(s["start_time"])
                for s in r.signals
            ):
                bounds_ok += 1
            # N = B * T * 1e6 samples for Fs=B
            if abs(r.N - r.B_mhz * 1e6 * r.T_ms / 1000.0) <= max(2, 1e-5 * r.N):
                t_ok += 1
        out["split"][split] = {
            "n_samples": len(recs),
            "n_instances": sum(len(r.signals) for r in recs),
            "file_size_ok": sizes_ok,
            "json_signal_count_ok": json_n_ok,
            "labels_in_band_and_time": bounds_ok,
            "N_equals_B_T": t_ok,
            "classes": dict(sorted(Counter(
                int(s["class"]) for r in recs for s in r.signals
            ).items())),
        }
    # stratified real-read check (30 samples)
    rng = np.random.default_rng(1729)
    sample_idx = rng.choice(len(all_recs), size=min(30, len(all_recs)), replace=False)
    read_ok = 0
    read_examples = []
    for i in sample_idx:
        r = all_recs[int(i)]
        iq = load_iq(r.bin_path)
        ok = int(iq.numel()) == r.N
        read_ok += int(ok)
        if len(read_examples) < 5:
            read_examples.append({
                "sample_id": r.sample_id, "N_manifest": r.N,
                "N_read": int(iq.numel()), "ok": ok,
            })
    out["checks"]["real_iq_read_30"] = {
        "ok": read_ok, "total": len(sample_idx), "examples": read_examples,
    }
    out["checks"]["unique_sample_ids"] = len({r.sample_id for r in all_recs}) == len(all_recs)
    out["checks"]["split_sums"] = {
        "fit_val": len(load_manifest(SPLIT_MANIFEST, split="fit"))
        + len(load_manifest(SPLIT_MANIFEST, split="val")),
        "test": len(load_manifest(SPLIT_MANIFEST, split="test")),
    }
    out["checks"]["old_split_note"] = (
        "Earlier local audit (6749/751) is obsolete; this run uses the frozen "
        "split_manifest.csv fit 5957 / val 1543 / test 2500."
    )
    out["normalizer"] = json.loads((ROOT / "configs/normalizer.json").read_text())
    out["model"] = {
        "id": "yolo11s_c1_directreg_2048x512",
        "nc": 14, "reg_max": 1, "end2end": False, "input": [1, 1, 2048, 512],
        "stem": "RGB pretrained kernel summed to C1",
    }
    path = ROOT / "reports/stage0_audit.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
