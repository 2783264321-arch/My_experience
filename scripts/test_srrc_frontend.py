#!/usr/bin/env python3
"""Unit/synthetic tests for the SRRC local-IQ front-end."""
from __future__ import annotations

import math
import unittest

import numpy as np

from scripts.baseline3_dsp import band_limit_fft
from scripts.baseline3_pa_srrc import gen_srrc_clean
from scripts.srrc_frontend import (
    sync_srrc, baseline_features, amplitude_features, phase_diff_features,
    BAND_HZ,
)


def synth(alpha: float, order: int, fs: float = 50e6,
          n_sym: int = 20000, cfo_hz: float = 3e3, seed: int = 0):
    x, meta = gen_srrc_clean(BAND_HZ, alpha, n_sym, order, fs, seed)
    x = x[:int(2.0e-3 * fs)]
    x = band_limit_fft(x, fs, BAND_HZ * 1.02, edge_frac=0.05)
    n = np.arange(len(x), dtype=np.float64)
    x = x * np.exp(2j * np.pi * cfo_hz * n / fs)
    rng = np.random.default_rng(seed + 1)
    p = float(np.mean(np.abs(x) ** 2))
    snr = 20.0
    pn = p / (10 ** (snr / 10.0))
    x = x + math.sqrt(pn / 2) * (rng.standard_normal(len(x)) +
                                 1j * rng.standard_normal(len(x)))
    return x.astype(np.complex64), meta


class FrontendTest(unittest.TestCase):
    def test_synthetic_qpsk_16qam_recovery(self):
        for alpha, order, m4_ref in ((0.2, 4, 1.0), (0.2, 16, 1.32),
                                     (0.35, 4, 1.0), (0.35, 16, 1.32)):
            x, meta = synth(alpha, order, seed=order + int(alpha * 100))
            r = sync_srrc(x, 50e6)
            self.assertEqual(r["status"], "ok", (alpha, order, r))
            self.assertLess(abs(r["R_mhz"] - meta["r_hz"] / 1e6) /
                            (meta["r_hz"] / 1e6), 0.01)
            self.assertLess(abs(r["alpha"] - alpha), 0.08)
            amp = amplitude_features(r["symbols"])
            self.assertLess(abs(amp["m4"] - m4_ref), 0.25,
                            (alpha, order, amp["m4"]))

    def test_phase_diff_only_consecutive(self):
        s = np.exp(1j * np.array([0.0, 0.3, 0.6, 1.0, 1.3, 1.6]))
        idx = np.array([0, 1, 2, 4, 5, 6])
        r = phase_diff_features(s, idx)
        # pairs (0,1),(1,2),(4,5),(5,6) = 4; (2,4) is a gap.
        self.assertEqual(r["n_pairs"], 4)
        self.assertEqual(len(r["features"]), 18)

    def test_noise_rejected(self):
        rng = np.random.default_rng(7)
        x = (rng.standard_normal(100000) +
             1j * rng.standard_normal(100000)).astype(np.complex64)
        x = band_limit_fft(x, 50e6, BAND_HZ * 1.02, edge_frac=0.05)
        r = sync_srrc(x, 50e6)
        self.assertNotEqual(r["status"], "ok", r)

    def test_feature_shapes_and_finite(self):
        rng = np.random.default_rng(8)
        x = (rng.standard_normal(50000) +
             1j * rng.standard_normal(50000)).astype(np.complex64)
        b = baseline_features(x, 50e6)
        self.assertEqual(len(b["features"]), len(b["names"]))
        self.assertTrue(np.all(np.isfinite(b["features"])))
        s = np.exp(1j * rng.uniform(-np.pi, np.pi, 1000))
        a = amplitude_features(s)
        p = phase_diff_features(s)
        self.assertEqual(len(a["features"]), len(a["names"]))
        self.assertEqual(len(p["features"]), len(p["names"]))
        self.assertTrue(np.all(np.isfinite(a["features"])))
        self.assertTrue(np.all(np.isfinite(p["features"])))


if __name__ == "__main__":
    unittest.main()
