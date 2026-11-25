"""SpaceNet IQ -> STFT views -> detector targets.

The implementation follows docs/Baseline执行计划.md:
- Fs = B (MHz)
- 0.7 ms periodic Hann window, M=L, no FFT zero-padding
- h=ceil(N/511), N_ext=511h, torch.stft(center=True, pad_mode='constant')
- exactly 512 time columns
- fine views: 2048 frequency bins, 20% overlap, step 1639
- coarse view: frequency power aggregation F -> 2048, same 512 columns
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

DATA_ROOT = Path("/root/autodl-fs/SpaceNet_Dataset/advanced")
SPLIT_MANIFEST = Path("/root/autodl-tmp/workspace/spacenet/analysis/outputs/split_manifest.csv")
WINDOW_MS = 0.7
N_TIME = 512
FINE_H = 2048
FINE_STEP = 1639  # int(2048 * 0.2) = 409 overlap; step = 1639
COARSE_H = 2048

# Scipy FFT backend is numerically equivalent (~4e-7 relative) and ~2x faster
# for the long 0.7 ms windows. Set SPACENET_STFT_BACKEND=torch to fall back.
STFT_BACKEND = os.environ.get("SPACENET_STFT_BACKEND", "scipy").lower()
FFT_THREADS = int(os.environ.get("SPACENET_FFT_THREADS", "1"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class View:
    tensor: torch.Tensor       # [1,H,W] float32
    boxes: torch.Tensor        # [M,4] xyxy normalized
    labels: torch.Tensor       # [M]
    instance_ids: List[int]
    view_type: str             # coarse|fine
    start: int = 0             # fine frequency start
    truncated: Optional[torch.Tensor] = None


@dataclass
class SampleRecord:
    sample_id: str
    split: str
    stem: str
    B_mhz: float
    T_ms: float
    N: int
    f_lo: float
    f_hi: float
    json_path: Path
    bin_path: Path
    signals: list


def _row_to_record(row: dict) -> SampleRecord:
    split = row["split"]
    stem = row["stem"]
    return SampleRecord(
        sample_id=row["sample_id"],
        split=split,
        stem=stem,
        B_mhz=float(row["bandwidth_mhz"]),
        T_ms=float(row["duration_ms"]),
        N=int(row["n_complex_samples"]),
        f_lo=float(row["f_lo_mhz"]),
        f_hi=float(row["f_hi_mhz"]),
        json_path=DATA_ROOT / split / f"{stem}.json",
        bin_path=DATA_ROOT / split / f"{stem}.bin",
        signals=[],
    )


def load_manifest(path: Path = SPLIT_MANIFEST, split: str = "fit") -> List[SampleRecord]:
    import csv

    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split_new"] == split:
                rows.append(_row_to_record(row))
    # load JSON lazily here; 7500 small files is cheap but avoid doing it in workers repeatedly
    for r in rows:
        r.signals = json.loads(r.json_path.read_text(encoding="utf-8"))["signals"]
    return rows


def load_iq(path: Path) -> torch.Tensor:
    raw = np.fromfile(path, dtype="<f2")
    n = len(raw) // 2
    # copy-free-ish conversion from interleaved float16 I/Q
    iq = np.empty(n, dtype=np.complex64)
    iq.real = raw[0::2][:n]
    iq.imag = raw[1::2][:n]
    return torch.from_numpy(iq)


def h_for_n(n: int) -> int:
    return math.ceil(n / (N_TIME - 1))


def stft_power(iq: torch.Tensor, B_mhz: float) -> tuple[torch.Tensor, int, int, int]:
    """Return (linear power [F,512] float32, h, N_ext, F).

    Fs is B MHz. The input is padded on the right to 511*h so that
    center=True produces exactly 512 columns.
    """
    n = iq.numel()
    fs = B_mhz * 1e6
    L = int(round(fs * WINDOW_MS / 1000.0))  # B MHz * 0.7ms = B*700 samples
    if L % 2:
        L += 1
    h = h_for_n(n)
    n_ext = (N_TIME - 1) * h
    if n_ext < n:
        raise ValueError(f"n_ext={n_ext} < n={n}")
    pad = n_ext - n
    x = iq
    if pad:
        x = F.pad(x, (0, pad))
    win = torch.hann_window(L, periodic=True, dtype=torch.float32)
    z = torch.stft(
        x,
        n_fft=L,
        hop_length=h,
        win_length=L,
        window=win,
        center=True,
        pad_mode="constant",
        normalized=False,
        onesided=False,
        return_complex=True,
    )
    # PyTorch returns [F, K]; fftshift frequency rows only.
    z = torch.fft.fftshift(z, dim=0)
    p = z.abs().square() / (fs * win.square().sum())
    return p.to(torch.float32), h, n_ext, L


def frequency_starts(F: int) -> List[int]:
    if F <= FINE_H:
        return [0]
    starts = list(range(0, F - FINE_H + 1, FINE_STEP))
    if starts[-1] != F - FINE_H:
        starts.append(F - FINE_H)
    return starts


def aggregate_frequency(p: torch.Tensor, out_h: int = COARSE_H) -> torch.Tensor:
    """Area-weighted frequency average: each coarse row is the mean of its input bins."""
    f = p.shape[0]
    if f <= out_h:
        # pad the frequency axis if a future sample has fewer than 2048 bins
        return F.interpolate(p[None, None], size=(out_h, p.shape[1]), mode="bilinear", align_corners=False)[0, 0]
    bounds = torch.linspace(0, f, out_h + 1).round().long()
    # guard against duplicate boundaries
    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i - 1]:
            bounds[i] = bounds[i - 1] + 1
    bounds = bounds.clamp(max=f)
    # cumsum difference is equivalent to segment sums and avoids a 2048-iteration loop
    cum = torch.cat([torch.zeros((1, p.shape[1]), dtype=p.dtype, device=p.device), torch.cumsum(p, dim=0)], dim=0)
    sums = cum[bounds[1:]] - cum[bounds[:-1]]
    counts = (bounds[1:] - bounds[:-1]).clamp(min=1).float().view(-1, 1)
    return sums / counts


def _time_x(t_ms: float, B_mhz: float, h: int) -> float:
    # pixel-center convention: t=0 -> 0.5; x is normalized by 512
    fs = B_mhz * 1e6
    return (t_ms / 1000.0) * fs / h + 0.5


def _label_boxes(
    signals: Sequence[dict],
    record: SampleRecord,
    Fbins: int,
    h: int,
    fine_start: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, List[int], torch.Tensor]:
    """Return xyxy normalized boxes, labels, instance ids, truncated flags."""
    boxes = []
    labels = []
    ids = []
    truncated = []
    H = FINE_H if fine_start is not None else COARSE_H
    yscale = H / Fbins
    for sig in signals:
        x1 = _time_x(float(sig["start_time"]), record.B_mhz, h) / N_TIME
        x2 = _time_x(float(sig["end_time"]), record.B_mhz, h) / N_TIME
        y1 = (float(sig["start_frequency"]) - record.f_lo) / record.B_mhz * Fbins
        y2 = (float(sig["end_frequency"]) - record.f_lo) / record.B_mhz * Fbins
        if x2 <= 0 or x1 >= 1:
            continue
        x1 = min(1.0, max(0.0, x1))
        x2 = min(1.0, max(0.0, x2))
        if fine_start is None:
            yy1 = y1 * (COARSE_H / Fbins)
            yy2 = y2 * (COARSE_H / Fbins)
        else:
            yy1 = y1 - fine_start
            yy2 = y2 - fine_start
        # positive frequency intersection with the view
        lo = max(0.0, yy1)
        hi = min(float(H), yy2)
        if hi <= lo or x2 <= x1:
            continue
        boxes.append([x1, lo / H, x2, hi / H])
        labels.append(int(sig["class"]))
        ids.append(int(sig["signal_id"]))
        truncated.append(bool(yy1 < 0 or yy2 > H))
    if not boxes:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
            [],
            torch.zeros((0,), dtype=torch.bool),
        )
    return (
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        ids,
        torch.tensor(truncated, dtype=torch.bool),
    )


def coarse_targets(record: SampleRecord) -> tuple[torch.Tensor, torch.Tensor, List[int], torch.Tensor]:
    """GT for the full-band coarse view, without computing an STFT.

    The native STFT has F = round(B MHz * 0.7 ms * 1000) frequency bins; the
    coarse view aggregates those bins to COARSE_H rows. This helper mirrors
    build_sample_views(..., fine_indices=[]) so cached images can reuse the
    exact same target geometry.
    """
    fbins = int(round(record.B_mhz * WINDOW_MS * 1000.0))
    return _label_boxes(record.signals, record, fbins, h_for_n(record.N), None)


def build_sample_views(
    record: SampleRecord,
    normalizer: tuple[float, float],
    fine_indices: Optional[Iterable[int]] = None,
) -> tuple[View, List[View], dict]:
    """Compute STFT once and return coarse + selected fine views."""
    iq = load_iq(record.bin_path)
    lo, hi = normalizer
    if STFT_BACKEND == "scipy":
        from .stft_fast import (
            aggregate_frequency_fast, normalize_slice, stft_power_fast,
        )
        p, _win, h, L, Fbins = stft_power_fast(iq, record.B_mhz, fft_threads=FFT_THREADS)
        n_ext = (N_TIME - 1) * h

        def norm_power(power_hm: "np.ndarray") -> torch.Tensor:
            return normalize_slice(power_hm, lo, hi).unsqueeze(0)

        coarse_p = aggregate_frequency_fast(p)
    else:
        p, h, n_ext, L = stft_power(iq, record.B_mhz)
        Fbins = p.shape[0]
        denom = max(hi - lo, 1e-6)

        def norm_power(power: torch.Tensor) -> torch.Tensor:
            d = 10.0 * torch.log10(torch.clamp(power, min=torch.finfo(torch.float32).tiny))
            return ((d - lo) / denom).clamp(0.0, 1.0).unsqueeze(0)

        coarse_p = aggregate_frequency(p)

    c_boxes, c_labels, c_ids, c_trunc = _label_boxes(record.signals, record, Fbins, h, None)
    coarse = View(
        tensor=norm_power(coarse_p),
        boxes=c_boxes,
        labels=c_labels,
        instance_ids=c_ids,
        view_type="coarse",
        start=0,
        truncated=c_trunc,
    )

    starts = frequency_starts(Fbins)
    if fine_indices is None:
        chosen = [0]
    else:
        chosen = [int(i) for i in fine_indices]
    fines = []
    for idx in chosen:
        start = starts[idx]
        boxes, labels, ids, trunc = _label_boxes(record.signals, record, Fbins, h, start)
        if STFT_BACKEND == "scipy":
            tensor = norm_power(p[:, start:start + FINE_H])
        else:
            tensor = norm_power(p[start:start + FINE_H])
        fines.append(View(
            tensor=tensor,
            boxes=boxes,
            labels=labels,
            instance_ids=ids,
            view_type="fine",
            start=start,
            truncated=trunc,
        ))
    meta = {
        "sample_id": record.sample_id,
        "B_mhz": record.B_mhz,
        "T_ms": record.T_ms,
        "N": record.N,
        "F": Fbins,
        "h": h,
        "n_ext": n_ext,
        "n_fine": len(starts),
        "fine_starts": starts,
    }
    return coarse, fines, meta


def compute_normalizer(
    records: Sequence[SampleRecord],
    max_samples: int = 128,
    seed: int = 1729,
) -> dict:
    """Fit 0.5/99.5 percentiles of log-power on a B/T-stratified fit subset."""
    rng = random.Random(seed)
    groups: dict[tuple, List[SampleRecord]] = {}
    for r in records:
        groups.setdefault((round(r.B_mhz, 3), round(r.T_ms, 3)), []).append(r)
    chosen = []
    keys = list(groups)
    rng.shuffle(keys)
    while len(chosen) < max_samples:
        progressed = False
        for key in keys:
            if len(chosen) >= max_samples:
                break
            pool = groups[key]
            if pool:
                chosen.append(pool.pop(rng.randrange(len(pool))))
                progressed = True
        if not progressed:
            break
    vals = []
    for r in chosen:
        iq = load_iq(r.bin_path)
        p, _, _, _ = stft_power(iq, r.B_mhz)
        d = 10.0 * torch.log10(torch.clamp(p, min=torch.finfo(torch.float32).tiny))
        coarse = aggregate_frequency(p)
        dcoarse = 10.0 * torch.log10(torch.clamp(coarse, min=torch.finfo(torch.float32).tiny))
        # equal-ish sampling from the two views
        vals.append(d.flatten()[::max(1, d.numel() // 20000)].cpu().numpy())
        vals.append(dcoarse.flatten()[::max(1, dcoarse.numel() // 20000)].cpu().numpy())
    arr = np.concatenate(vals)
    lo, hi = np.percentile(arr, [0.5, 99.5]).tolist()
    return {
        "lo": float(lo),
        "hi": float(hi),
        "n_samples": len(chosen),
        "seed": seed,
        "sample_ids": [r.sample_id for r in chosen],
    }


class SpaceNetViews(torch.utils.data.Dataset):
    """Training dataset: one coarse + one deterministic-rotating fine view per source."""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        normalizer: tuple[float, float],
        seed: int = 1729,
        epoch: int = 0,
    ):
        self.records = list(records)
        self.normalizer = normalizer
        self.seed = seed
        self.epoch = epoch

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.records)

    def _fine_index(self, record: SampleRecord) -> int:
        # number of slices depends only on B; compute from F without STFT
        Fbins = int(round(record.B_mhz * WINDOW_MS * 1000))
        starts = frequency_starts(Fbins)
        rng = random.Random(f"{self.seed}:{record.sample_id}")
        perm = list(range(len(starts)))
        rng.shuffle(perm)
        return perm[self.epoch % len(perm)]

    def __getitem__(self, idx: int):
        record = self.records[idx]
        coarse, fines, meta = build_sample_views(
            record, self.normalizer, [self._fine_index(record)]
        )
        meta["source_index"] = idx
        return {"coarse": coarse, "fine": fines[0], "meta": meta}


class SpaceNetCoarseViews(torch.utils.data.Dataset):
    """Training dataset: exactly one full-band coarse view per source."""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        normalizer: tuple[float, float],
    ):
        self.records = list(records)
        self.normalizer = normalizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        coarse, _, meta = build_sample_views(record, self.normalizer, [])
        meta["source_index"] = idx
        return {
            "tensor": coarse.tensor,
            "boxes": coarse.boxes,
            "labels": coarse.labels,
            "meta": meta,
        }


class CoarseCacheDataset(torch.utils.data.Dataset):
    """Read precomputed BF16-bit coarse views from a .npy memmap.

    The cache stores the exact BFloat16 values the BF16-autocast training path
    would see, packed as int16 bit patterns. __getitem__ expands them to
    float32 without changing the values, so the training step is numerically
    identical to the on-the-fly f32 -> autocast(bf16) path.
    """

    def __init__(
        self,
        records: Sequence[SampleRecord],
        normalizer: tuple[float, float],
        cache_path: str | Path,
        meta_path: str | Path | None = None,
    ):
        self.records = list(records)
        self.normalizer = (float(normalizer[0]), float(normalizer[1]))
        self.cache_path = Path(cache_path)
        meta_path = Path(meta_path) if meta_path else self.cache_path.with_suffix(".json")
        self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))

        if self.meta.get("format") != "coarse-bf16-bits-v1":
            raise ValueError(f"unsupported cache format: {self.meta.get('format')!r}")
        if self.meta.get("dtype") != "bfloat16_bits":
            raise ValueError(f"unsupported cache dtype: {self.meta.get('dtype')!r}")
        ids = [r.sample_id for r in self.records]
        if self.meta.get("sample_ids") != ids:
            raise ValueError("cache sample order does not match the manifest")
        if int(self.meta.get("n_samples", -1)) != len(self.records):
            raise ValueError("cache sample count does not match the manifest")
        if abs(float(self.meta["normalizer_lo"]) - self.normalizer[0]) > 1e-9:
            raise ValueError("cache normalizer lo mismatch")
        if abs(float(self.meta["normalizer_hi"]) - self.normalizer[1]) > 1e-9:
            raise ValueError("cache normalizer hi mismatch")
        if self.meta.get("split_manifest_sha256") != file_sha256(SPLIT_MANIFEST):
            raise ValueError("cache split manifest hash mismatch")

        self.images = np.load(self.cache_path, mmap_mode="r")
        expected = (len(self.records), 1, COARSE_H, N_TIME)
        if tuple(self.images.shape) != expected:
            raise ValueError(f"cache shape {self.images.shape} != {expected}")
        if self.images.dtype != np.int16:
            raise ValueError(f"cache dtype {self.images.dtype} != int16 bit patterns")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        bits = np.asarray(self.images[idx]).copy()
        tensor = torch.from_numpy(bits).view(torch.bfloat16).float()
        boxes, labels, _, _ = coarse_targets(self.records[idx])
        meta = {
            "sample_id": self.records[idx].sample_id,
            "B_mhz": self.records[idx].B_mhz,
            "T_ms": self.records[idx].T_ms,
            "source_index": idx,
        }
        return {
            "tensor": tensor,
            "boxes": boxes,
            "labels": labels,
            "meta": meta,
        }


def collate_coarse(items: list[dict]) -> dict:
    """Stack one coarse view per source into the training/validation batch."""
    images = torch.stack([item["tensor"] for item in items], dim=0)
    boxes = []
    labels = []
    batch_idx = []
    for image_i, item in enumerate(items):
        if len(item["boxes"]):
            boxes.append(item["boxes"])
            labels.append(item["labels"])
            batch_idx.append(torch.full((len(item["boxes"]),), image_i, dtype=torch.long))
    if boxes:
        boxes = torch.cat(boxes, dim=0).float()
        boxes_xywh = torch.cat(
            [(boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]], dim=1
        )
        labels = torch.cat(labels, dim=0).long()
        batch_idx = torch.cat(batch_idx, dim=0).long()
    else:
        boxes_xywh = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.long)
        batch_idx = torch.zeros((0,), dtype=torch.long)
    return {
        "img": images,
        "batch_idx": batch_idx,
        "cls": labels[:, None],
        "bboxes": boxes_xywh,
        "meta": [item["meta"] for item in items],
        "view_types": ["coarse"] * len(items),
        "starts": [0] * len(items),
    }


def collate_views(items: list[dict]) -> dict:
    """Flatten source pairs into a same-shape view batch and YOLO targets."""
    images = []
    boxes = []
    labels = []
    batch_idx = []
    metas = []
    view_types = []
    starts = []
    truncated = []
    for source_i, item in enumerate(items):
        for view in (item["coarse"], item["fine"]):
            image_i = len(images)
            images.append(view.tensor)
            if len(view.boxes):
                boxes.append(view.boxes)
                labels.append(view.labels)
                batch_idx.append(torch.full((len(view.boxes),), image_i, dtype=torch.long))
                if view.truncated is not None:
                    truncated.append(view.truncated.bool())
                else:
                    truncated.append(torch.zeros((len(view.boxes),), dtype=torch.bool))
            view_types.append(view.view_type)
            starts.append(view.start)
            metas.append(item["meta"])
    images = torch.stack(images, dim=0)
    if boxes:
        boxes = torch.cat(boxes, dim=0).float()
        boxes_xywh = torch.cat([(boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]], dim=1)
        labels = torch.cat(labels, dim=0).long()
        batch_idx = torch.cat(batch_idx, dim=0).long()
        truncated = torch.cat(truncated, dim=0).bool()
    else:
        boxes_xywh = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.long)
        batch_idx = torch.zeros((0,), dtype=torch.long)
        truncated = torch.zeros((0,), dtype=torch.bool)
    return {
        "img": images,
        "batch_idx": batch_idx,
        "cls": labels[:, None],
        "bboxes": boxes_xywh,
        "truncated": truncated,
        "meta": metas,
        "view_types": view_types,
        "starts": starts,
    }
