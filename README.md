# SpaceNet Baseline: YOLO11s-C1-DirectReg 2048x512

Project root: `/root/autodl-tmp/workspace/baseline`

Current execution basis: `docs/Baseline执行计划.md`.

Pipeline:

```text
IQ -> 0.7 ms STFT (512 columns) -> log-power
    -> 2048x512 fine frequency slices + 2048x512 coarse full-band view
    -> YOLO11s C1 / 14 classes / DirectReg (reg_max=1, end2end=False)
    -> SAHI-style slice inference, coordinate remap, class-wise NMS
```

Status: stage 0, environment and implementation setup.

Data (read-only):

```text
/root/autodl-fs/SpaceNet_Dataset/advanced/{train,test}
```

Split:

```text
/root/autodl-tmp/workspace/spacenet/analysis/outputs/split_manifest.csv
fit 5957 / val 1543 / test 2500
```

Do not modify original IQ/JSON or silently deduplicate/ignore labels.
