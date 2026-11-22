#!/usr/bin/env python3
"""Run the 40-group SRRC cache trial while monitoring D-FINE step time.

Waits for a stable training segment, records a baseline window, starts the
low-priority single-process trial, and stops it if the rolling median step
time degrades persistently or a validation/save gap appears.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np


def read_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def med(vals):
    return float(np.median(vals)) if len(vals) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trial-out", type=Path, required=True)
    ap.add_argument("--trial-log", type=Path, required=True)
    ap.add_argument("--max-groups", type=int, default=40)
    ap.add_argument("--baseline-steps", type=int, default=200)
    ap.add_argument("--slowdown", type=float, default=1.15)
    ap.add_argument("--sustain", type=int, default=3)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.trial_out.parent.mkdir(parents=True, exist_ok=True)
    if args.trial_out.exists():
        args.trial_out.unlink()

    # Wait for epoch 40 and at least baseline_steps+20 rows in that epoch.
    rows = read_rows(args.log)
    epoch = rows[-1]["epoch"]
    while True:
        rows = read_rows(args.log)
        last = rows[-1]
        epoch = last["epoch"]
        in_epoch = [r for r in rows if r["epoch"] == epoch]
        if int(epoch) >= 40 and len(in_epoch) >= args.baseline_steps + 20:
            break
        time.sleep(10)

    # Baseline: last N rows of the current stable epoch. Require no >5 s gap.
    base = in_epoch[-args.baseline_steps:]
    base_times = [r["time"] for r in base]
    base_steps = np.array([float(r["step_s"]) for r in base])
    baseline = {
        "epoch": int(epoch),
        "step_first": int(base[0]["step"]),
        "step_last": int(base[-1]["step"]),
        "n": len(base),
        "median_s": med(base_steps),
        "p95_s": float(np.percentile(base_steps, 95)),
        "max_s": float(base_steps.max()),
        "first_time": base_times[0],
        "last_time": base_times[-1],
    }

    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    cmd = ["nice", "-n", "19", "env", *[f"{k}={v}" for k, v in env.items()
            if k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                     "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")],
           str(Path("/root/autodl-tmp/workspace/baseline/.venv/bin/python")),
           "-m", "scripts.baseline3srrc_gt_pilot", "prepare"]
    if args.max_groups > 0:
        cmd += ["--max-groups", str(args.max_groups)]
    cmd += ["--out", str(args.trial_out)]
    trial_log = args.trial_log.open("w")
    started = time.time()
    proc = subprocess.Popen(cmd, cwd="/root/autodl-tmp/worktrees/srrc-gt-pilot-prep",
                            env=env, stdout=trial_log, stderr=subprocess.STDOUT)
    during = []
    rolling = []
    slow_checks = 0
    stop_reason = None
    last_step = baseline["step_last"]
    while proc.poll() is None:
        time.sleep(2)
        new_rows = read_rows(args.log)
        for r in new_rows:
            if int(r["step"]) > last_step:
                during.append(r)
                last_step = int(r["step"])
        if during:
            vals = [float(r["step_s"]) for r in during[-50:]]
            rolling.append(med(vals))
            if med(vals) > baseline["median_s"] * args.slowdown:
                slow_checks += 1
            else:
                slow_checks = 0
            if slow_checks >= args.sustain:
                stop_reason = "sustained_slowdown"
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
        # A >10 s gap in the training log means validation/save/epoch change.
        if len(during) >= 2:
            from datetime import datetime
            t0 = datetime.fromisoformat(during[-2]["time"])
            t1 = datetime.fromisoformat(during[-1]["time"])
            if (t1 - t0).total_seconds() > 10:
                stop_reason = "validation_or_save_gap"
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
    proc.wait()
    trial_log.close()
    elapsed = time.time() - started
    if during:
        vals = np.array([float(r["step_s"]) for r in during])
        during_stats = {
            "n": len(during),
            "median_s": med(vals),
            "p95_s": float(np.percentile(vals, 95)),
            "max_s": float(vals.max()),
            "first_step": int(during[0]["step"]),
            "last_step": int(during[-1]["step"]),
            "first_time": during[0]["time"],
            "last_time": during[-1]["time"],
            "delta_median_pp": (med(vals) / baseline["median_s"] - 1.0) * 100.0,
        }
    else:
        during_stats = None
    result = {
        "command": cmd,
        "max_groups": args.max_groups,
        "trial_seconds": elapsed,
        "trial_returncode": proc.returncode,
        "stop_reason": stop_reason,
        "baseline": baseline,
        "during": during_stats,
        "rolling_medians": rolling,
        "trial_out": str(args.trial_out),
        "trial_out_bytes": args.trial_out.stat().st_size if args.trial_out.exists() else None,
        "trial_log": str(args.trial_log),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
