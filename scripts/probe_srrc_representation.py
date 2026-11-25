"""Train-only synthetic SRRC sanity check. No SpaceNet files are read.

Balanced QPSK/16QAM with same SRRC shape, duration and power. Receiver
observes 20 MS/s, R_s=20e6/2.4 and alpha=.2 (10 MHz null-to-null).
Each held-out pair shares nuisance draws across classes. 'sync' features
use the *same received mixture*, with target timing and CFO supplied by
the generator; this is a positive control, not a deployable method.
"""
import argparse
import json
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft, signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FS = 20_000_000
SPS = 12
UP = 5
FS_HI = FS * UP
ALPHA = 0.2
NS = 2 * FS // 1000
START = 240
NFFT = round(FS * 0.0007)
HOP = (FS * 20 // 1000 + 510) // 511
GROUPS = 2048
WINDOW = signal.windows.hann(NFFT, sym=False).astype(np.float32)
BOUNDS = np.round(np.linspace(0, NFFT, GROUPS + 1)).astype(int)
LO, HI = -103.23661804199219, -45.258453369140625
T = np.arange(NS) / FS
EPS = 1e-12


def rrc_taps():
    t = np.arange(-60, 61) / SPS
    a = ALPHA
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (np.sin(np.pi * t * (1 - a)) +
               4 * a * t * np.cos(np.pi * t * (1 + a))) / (
                   np.pi * t * (1 - (4 * a * t) ** 2)
               )
    out[t == 0] = 1 + a * (4 / np.pi - 1)
    edge = np.isclose(np.abs(t), 1 / (4 * a))
    out[edge] = a / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * a)) +
                                  (1 - 2 / np.pi) * np.cos(np.pi / (4 * a)))
    return (out / np.linalg.norm(out)).astype(np.float32)


PULSE = rrc_taps()
GD = len(PULSE) // 2


def packet(rng, label, delay, cfo, phase):
    levels = np.array([-1, 1] if label == 0 else [-3, -1, 1, 3])
    symbols = (rng.choice(levels, NS * UP // SPS + 40) +
               1j * rng.choice(levels, NS * UP // SPS + 40))
    symbols = (symbols / np.sqrt(2 if label == 0 else 10)).astype(np.complex64)
    tx = signal.upfirdn(PULSE, symbols, up=SPS)
    rx = signal.resample_poly(tx[START - delay: START - delay + NS * UP], 1, UP)
    y = rx * np.exp(1j * (2 * np.pi * cfo * T + phase))
    return (y / np.sqrt(np.mean(np.abs(y) ** 2))).astype(np.complex64)


def envelope_features(samples):
    energy = np.abs(samples) ** 2
    energy /= np.mean(energy)
    return np.r_[np.mean(energy**2), np.mean(energy**3),
                 np.quantile(energy, [.05, .10, .25, .5, .75, .9, .95])]


def power_features(x, start):
    # Same window length, capture hop, coarse freq bins and log clipping as
    # the project. Use interior frames (known synthetic time ROI) only.
    frames = sliding_window_view(x, NFFT)[start::HOP]
    z = fft.fftshift(fft.fft(frames * WINDOW, axis=-1), axes=-1)
    p = np.abs(z)**2 / (FS * np.sum(WINDOW**2))
    p = np.add.reduceat(p, BOUNDS[:-1], axis=1) / np.diff(BOUNDS)
    db = np.clip(10 * np.log10(np.maximum(p, 1e-29)), LO, HI)
    s = 10 ** (db / 10)
    s /= s.mean()
    # Coarse power only. 32 frequency-band means; take 8 grouped frequency
    # bins for each of frame-power dispersion, row texture, and lag features.
    mean_psd = np.log(s.mean(0).reshape(32, -1).mean(1) + EPS)
    fsums = s.reshape(s.shape[0], 8, -1).mean(2)
    fsums /= fsums.mean(0)
    g = s.std(0) / np.maximum(s.mean(0), EPS)
    texture = g.reshape(8, -1).mean(1)
    lag = np.abs(np.diff(fsums, axis=0)).mean(0)
    integrated = s.sum(1) / s.sum(1).mean()
    c1stats = np.r_[mean_psd, fsums.std(0), lag, texture,
                    integrated.std(), np.quantile(integrated, [.1, .9])]
    return mean_psd, c1stats


def record(x, target_delay, target_cfo, start):
    quantized = (x.real.astype(np.float16).astype(np.float32)
                 + 1j * x.imag.astype(np.float16).astype(np.float32))
    mean_psd, c1stats = power_features(quantized, start)
    # Use known target CFO/timing solely for the positive control.
    demixed = quantized * np.exp(-2j * np.pi * target_cfo * T)
    upsampled = signal.resample_poly(demixed.astype(np.complex64), UP, 1)
    matched = signal.fftconvolve(upsampled, PULSE)
    first_sym = (target_delay + 2 * GD - START) % SPS
    synced = matched[first_sym + SPS * 16: NS * UP - SPS * 16: SPS]
    return [mean_psd, c1stats, envelope_features(quantized[40:-40]),
            envelope_features(synced)]


def generate(seed, n_pairs):
    conditions = {"no_cochannel": [], "cochannel_0dB": []}
    for ss in np.random.SeedSequence(seed).spawn(n_pairs):
        nr, ir, tr0, tr1 = [np.random.default_rng(k) for k in ss.spawn(4)]
        delay, idelay = nr.integers(0, SPS, size=2)
        cfo, icfo = nr.uniform(-25000, 25000, size=2)
        phase, iphase = nr.uniform(-np.pi, np.pi, size=2)
        gain = 10 ** (nr.uniform(-4, 4) / 20)
        noise = (nr.standard_normal(NS, dtype=np.float32) +
                 1j * nr.standard_normal(NS, dtype=np.float32)) / np.sqrt(2)
        noise *= 10 ** (-12 / 20)
        interferer = packet(ir, int(nr.integers(0, 2)), idelay, icfo, iphase)
        start = int(nr.integers(0, HOP))
        for label, t_rng in enumerate((tr0, tr1)):
            target = packet(t_rng, label, delay, cfo, phase)
            for cond in conditions:
                mixture = gain * (target + noise + (0 if cond == "no_cochannel" else interferer))
                conditions[cond].append(record(mixture, delay, cfo, start))
    return {k: [np.stack([row[i] for row in v]) for i in range(4)]
            for k, v in conditions.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-pairs", type=int, default=96)
    p.add_argument("--val-pairs", type=int, default=96)
    p.add_argument("--train-seed", type=int, default=24101)
    p.add_argument("--val-seed", type=int, default=24102)
    args = p.parse_args()
    t0 = time.perf_counter()
    fit = generate(args.train_seed, args.train_pairs)
    val = generate(args.val_seed, args.val_pairs)
    yt = np.tile([0, 1], args.train_pairs)
    yv = np.tile([0, 1], args.val_pairs)
    out = {"setup": vars(args) | {"fs": FS, "sps_internal": SPS, "alpha": ALPHA,
           "nominal_bw": int(FS_HI / SPS * (1 + ALPHA)), "samples_per_packet": NS,
           "window_samples": NFFT, "hop_samples": HOP, "nominal_snr_db": 12},
           "results": {}}
    boot = np.random.default_rng(24103).integers(0, args.val_pairs,
                    size=(2000, args.val_pairs))
    names = ("mean_coarse_psd", "coarse_power_stats", "received_envelope", "ideal_sync_envelope")
    for cond in fit:
        rows = {}
        for name, x, z in zip(names, fit[cond], val[cond]):
            lr = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000))
            lr.fit(x, yt)
            prob = lr.predict_proba(z)[:, 1]
            pred = (prob >= .5).astype(int)
            pair_acc = ((pred == yv).reshape(-1, 2)).mean(1)
            metrics = {"bal_acc": float(balanced_accuracy_score(yv, pred)),
                       "auc": float(roc_auc_score(yv, prob)),
                       "pair_bootstrap_bal_acc_ci95": np.quantile(pair_acc[boot].mean(1), [.025, .975]).tolist()}
            if "envelope" in name:
                metrics["val_m4_median"] = [float(np.median(z[yv == j, 0])) for j in (0, 1)]
            rows[name] = metrics
        out["results"][cond] = rows
    out["wall_seconds"] = round(time.perf_counter() - t0, 2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
