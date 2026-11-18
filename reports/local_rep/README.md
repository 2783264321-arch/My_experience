# Local representation experiment artifacts

Protocol: `docs/collaboration/README.md` plus the 2026-11-18 request to compare
coarse power, local fine power, and local complex STFT real/imag for 0/1/2 and
3/4/5 modulation separability.

## Data and proposals

- Frozen detector: `runs/yolo11s_c1_directreg_2048x512/best_corrected.pt`.
- Proposal source: `reports/proposal_crops_fit.npz` (fit coarse-view proposals,
  matched to GT at IoU >= 0.5, up to 3 proposals per GT).
- Deduplicated to one highest-score proposal per `(sample_id, GT box)`.
- Kept IoU >= 0.75: **3,754 crops / 2,503 samples**.
- IoU quantiles: 0.05 / 0.50 / 0.95 = 0.908 / 0.984 / 0.996.
- Class counts: 0/1/2/3/4/5 = 705 / 716 / 732 / 530 / 517 / 554.
- WiFi overlap: 1,841 crops; WiFi-clean: 1,913; cross-class WiFi overlap:
  1,604; same-class WiFi overlap: 405; strong WiFi overlap (>=25% of the
  instance area): 1,513. Any-signal overlap: 3,393; any-clean: 361.

## Representations

All representations use the same proposal boxes and 10% physical context.

- `coarse`: existing 0.7 ms coarse power crop, 2048 rows full-band -> 128x256.
- `local_power`: local 0.1 ms (main) complex STFT, local band -> log power,
  resized to 128x256. Window sweep: 0.05, 0.1, 0.2, 0.35, 0.7 ms.
- `local_complex`: real/imag of the **same** local complex STFT coefficients
  and the same 128x256 grid as `local_power`; global phase removed, per-crop
  RMS normalization. This is the requested dual-channel real/imag test.

Normalization: coarse and local power are standardized per crop; complex is
jointly RMS-normalized per crop, then per-crop standardized; all clipped to
[-10, 10].

## Shared holdout and training

- One shared split by IQ sample over all 0-5 crops:
  2,996 train crops / 2,002 samples; 758 holdout crops / 501 samples
  (20% sample holdout, seed 1729).
- Same split for every representation, bandwidth group and seed.
- SmallCNN: 4 conv blocks, 128-d FC; input channels 1 for coarse/power, 2 for
  complex. Same depth/width.
- AdamW, lr 1e-3, weight decay 1e-4, cosine schedule, batch 64, 100 epochs,
  class weights. Same budget for all representations.
- Main comparison: 3 seeds for coarse and local power; 1 seed for complex
  (the complex model overfits the training set to ~99.9% and is at chance on
  the holdout, so extra seeds are not informative).
- Window sweep: local power only, 1 seed, 100 epochs, same split.

## Files

- `proposal_local_win0050.npz` ... `proposal_local_win0700.npz`: aligned
  coarse / local power / local complex arrays plus labels, sample IDs, boxes,
  IoU, and overlap flags. Large, gitignored.
- `representation_comparison_100ep.json`: merged main 100-epoch result.
- `representation_summary_100ep.md`: generated tables.
- `window_*_012.json`, `window_*_345.json`: window sweep results.
- `representation_comparison_win0100.json`: 20-epoch pilot (underfit; kept
  only as a training-budget diagnostic).
- `train_*.log`, `collect_local.log`: run logs.

## Commands

```bash
# collect all windows in one IQ pass
PYTHONPATH=. python scripts/collect_local_proposal_crops.py \
  --win-ms 0.05,0.1,0.2,0.35,0.7 \
  --out 'reports/local_rep/proposal_local_win{win}.npz'

# main 100-epoch comparison
PYTHONPATH=. python scripts/train_local_representation_classifiers.py \
  --data reports/local_rep/proposal_local_win0100.npz \
  --tag win0100 --reps coarse,local_power,local_complex --groups 012,345 \
  --seeds 1729,1730,1731 --epochs 100 --holdout 0.2 --split-seed 1729 \
  --out reports/local_rep/representation_comparison_win0100.json

# window sweep (one seed, local power)
for w in 0050 0200 0350 0700; do
  PYTHONPATH=. python scripts/train_local_representation_classifiers.py \
    --data reports/local_rep/proposal_local_win${w}.npz \
    --tag win${w} --reps local_power --groups 012,345 --seeds 1729 \
    --epochs 100 --holdout 0.2 --split-seed 1729 \
    --out reports/local_rep/window_${w}.json
done
```

## Result

At 100 epochs, local fine power is the only positive representation
(+2.28 pp on 012, +2.72 pp on 345 over coarse; larger gain under WiFi
overlap). Real/imag complex is at chance and overfits. 0.1 ms is the best
balanced window. The next step is a fixed-input stronger classifier/detector
control on coarse vs local fine power, not a real/imag dual-channel branch.
See `reports/modulation_representation_comparison.md` and
`docs/collaboration/opencode-modulation-separability-result.md`.

Test split remains sealed. No original IQ/JSON was modified.
