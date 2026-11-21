#!/usr/bin/env python3
"""Unit tests for the SRRC ignore-aware AP50 evaluator."""
from __future__ import annotations

import unittest

import numpy as np

from scripts.srrc_ignore_ap import ignore_aware_ap50, drop_gt_ap50
from src.signals.metrics import ImagePredictions, ImageTargets, evaluate


def P(boxes, scores, labels):
    return ImagePredictions(np.asarray(boxes, float),
                            np.asarray(scores, float),
                            np.asarray(labels, int))


def G(boxes, labels):
    return ImageTargets(np.asarray(boxes, float),
                        np.asarray(labels, int))


class IgnoreAPTest(unittest.TestCase):
    def test_all_valid_matches_reference(self):
        gt = G([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8]], [10, 10])
        pred = P([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8],
                  [0.9, 0.9, 0.99, 0.99]],
                 [0.9, 0.8, 0.7], [10, 10, 10])
        ref = evaluate([gt], [pred], iou_thresholds=(0.5,), max_dets=100)
        ref_ap = ref["per_iou"]["0.5"]["per_class"]["10"]["ap"]
        out = ignore_aware_ap50([pred], [gt], [np.zeros((0, 4))], 10)
        self.assertAlmostEqual(out.ap50, ref_ap, places=12)
        self.assertEqual((out.n_tp, out.n_fp, out.n_ignored_pred), (2, 1, 0))

    def test_ignored_hit_not_tp_or_fp(self):
        gt = G([[0.1, 0.1, 0.3, 0.3]], [10])
        ign = np.asarray([[0.6, 0.6, 0.9, 0.9]])
        pred = P([[0.6, 0.6, 0.9, 0.9]], [0.9], [10])
        out = ignore_aware_ap50([pred], [gt], [ign], 10)
        self.assertEqual((out.n_tp, out.n_fp, out.n_ignored_pred), (0, 0, 1))
        self.assertEqual(out.ap50, 0.0)

    def test_valid_hit_is_tp(self):
        gt = G([[0.1, 0.1, 0.3, 0.3]], [10])
        pred = P([[0.1, 0.1, 0.3, 0.3]], [0.9], [10])
        out = ignore_aware_ap50([pred], [gt], [np.zeros((0, 4))], 10)
        self.assertEqual((out.n_tp, out.n_fp, out.n_ignored_pred), (1, 0, 0))
        self.assertEqual(out.ap50, 1.0)

    def test_duplicate_on_valid_is_fp(self):
        gt = G([[0.1, 0.1, 0.3, 0.3]], [10])
        pred = P([[0.1, 0.1, 0.3, 0.3], [0.1, 0.1, 0.3, 0.3]],
                 [0.9, 0.8], [10, 10])
        out = ignore_aware_ap50([pred], [gt], [np.zeros((0, 4))], 10)
        self.assertEqual((out.n_tp, out.n_fp, out.n_ignored_pred), (1, 1, 0))
        self.assertEqual(out.ap50, 1.0)

    def test_valid_priority_over_ignored(self):
        # valid and ignored overlap; prediction is exactly on valid GT
        gt = G([[0.1, 0.1, 0.4, 0.4]], [10])
        ign = np.asarray([[0.1, 0.1, 0.4, 0.4]])
        pred = P([[0.1, 0.1, 0.4, 0.4]], [0.9], [10])
        out = ignore_aware_ap50([pred], [gt], [ign], 10)
        self.assertEqual((out.n_tp, out.n_fp, out.n_ignored_pred), (1, 0, 0))
        self.assertEqual(out.ap50, 1.0)

    def test_duplicate_with_ignored_overlap(self):
        # first prediction matches valid; duplicate hits ignored GT that
        # overlaps the valid box -> duplicate is ignored, not FP.
        gt = G([[0.1, 0.1, 0.4, 0.4]], [10])
        ign = np.asarray([[0.1, 0.1, 0.4, 0.4]])
        pred = P([[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.4, 0.4]],
                 [0.9, 0.8], [10, 10])
        out = ignore_aware_ap50([pred], [gt], [ign], 10)
        self.assertEqual((out.n_tp, out.n_fp, out.n_ignored_pred), (1, 0, 1))
        self.assertEqual(out.ap50, 1.0)

    def test_no_valid_gt(self):
        gt = G([[0.1, 0.1, 0.3, 0.3]], [11])
        pred = P([[0.1, 0.1, 0.3, 0.3]], [0.9], [10])
        out = ignore_aware_ap50([pred], [gt], [np.zeros((0, 4))], 10)
        self.assertIsNone(out.ap50)
        self.assertEqual(out.n_valid_gt, 0)

    def test_does_not_mutate_inputs(self):
        gt = G([[0.1, 0.1, 0.3, 0.3]], [10])
        pred = P([[0.1, 0.1, 0.3, 0.3]], [0.9], [10])
        boxes = pred.boxes.copy(); scores = pred.scores.copy()
        labels = pred.labels.copy()
        gb = gt.boxes.copy(); gl = gt.labels.copy()
        ignore_aware_ap50([pred], [gt], [np.zeros((0, 4))], 10)
        self.assertTrue(np.array_equal(pred.boxes, boxes))
        self.assertTrue(np.array_equal(pred.scores, scores))
        self.assertTrue(np.array_equal(pred.labels, labels))
        self.assertTrue(np.array_equal(gt.boxes, gb))
        self.assertTrue(np.array_equal(gt.labels, gl))

    def test_old_drop_gt_protocol_is_biased(self):
        # one valid GT, one ignored GT; the ignored-box prediction has the
        # higher score, so dropping the ignored GT without ignoring the
        # prediction turns it into a high-score FP.
        gt = G([[0.1, 0.1, 0.3, 0.3], [0.6, 0.6, 0.9, 0.9]], [10, 10])
        gt_valid = G([[0.1, 0.1, 0.3, 0.3]], [10])
        pred = P([[0.1, 0.1, 0.3, 0.3], [0.6, 0.6, 0.9, 0.9]],
                 [0.8, 0.9], [10, 10])
        keep = np.asarray([True, False])
        old = drop_gt_ap50([pred], [gt], 10, [keep])
        new = ignore_aware_ap50([pred], [gt_valid],
                                [np.asarray([[0.6, 0.6, 0.9, 0.9]])], 10)
        self.assertLess(old["ap50"], new.ap50)
        self.assertEqual(new.ap50, 1.0)
        self.assertEqual(new.n_ignored_pred, 1)


if __name__ == "__main__":
    unittest.main()
