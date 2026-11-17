# SpaceNet baseline - YOLO11s-C1-DirectReg 2048x512

Project root: `/root/autodl-tmp/workspace/baseline`

This is the first reproducible baseline for SpaceNet advanced IQ detection.
Authoritative plan: `docs/Baseline执行计划.md`; execution status:
`docs/status.md`.

```text
IQ (float16 interleaved I/Q, Fs = B)
  -> 0.7 ms periodic Hann STFT, exactly 512 time columns
  -> log power
  -> 2048x512 full-band coarse view + 2048x512 fine frequency slices
     (20% overlap, step 1639, last slice snapped to F-2048)
  -> YOLO11s body / C=1 / nc=14 / reg_max=1 / end2end=False (DirectReg)
  -> all-view inference, coordinate remap, internal-boundary filtering,
     class-wise global NMS, strict class-aware mAP@0.5
```

Data (read-only): `/root/autodl-fs/SpaceNet_Dataset/advanced`.
Frozen split: `fit 5957 / val 1543 / test 2500` from
`/root/autodl-tmp/workspace/spacenet/analysis/outputs/split_manifest.csv`
(SHA256 `224ff3d4...a2a69`). Test is sealed.

Key results and artifacts:

- `reports/stage0_audit.json` - paths, hashes, split, IQ spec, versions.
- `reports/stage1_smoke.md` - real-data shape/coordinate smoke.
- `reports/stage2_resources.md` - FP32/FP16/BF16 acceptance, memory/time.
- `reports/stage3_overfit.md` - 16-source overfit gate, passed at 0.8168.
- `runs/yolo11s_c1_directreg_2048x512/` - 50-epoch run (config, CSV,
  validation history, best/last checkpoints, summary).
- `runs/yolo11s_c1_directreg_2048x512/stage5_val/` - full val predictions,
  metrics, error analysis.
- `runs/yolo11s_c1_directreg_2048x512/stage5_examples/` - qualitative plots.

Commands:

```bash
# 50-epoch training (see scripts/run_training.sh)
./scripts/run_training.sh

# stage-5 full validation
PYTHONPATH=. .venv/bin/python scripts/validate_run.py \
  --run yolo11s_c1_directreg_2048x512 --checkpoint best.pt --workers 12

# error analysis and examples
PYTHONPATH=. .venv/bin/python scripts/analyze_val.py \
  --run yolo11s_c1_directreg_2048x512
PYTHONPATH=. .venv/bin/python scripts/predict_examples.py \
  --run yolo11s_c1_directreg_2048x512 --checkpoint best.pt --limit 10
```

Do not modify original IQ/JSON. Do not create a different split. Do not use
the test split for model selection.
