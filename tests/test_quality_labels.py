import unittest
import numpy as np
import skfuzzy
from evaluation.quality_labels import QUALITY_BANDS, quality_label, quality_memberships


class DisplayLabelTests(unittest.TestCase):
    def test_original_trapezoids_against_independent_implementation(self):
        x = np.linspace(0, 100, 1001)
        expected = {k: skfuzzy.trapmf(x, p) for k, p in QUALITY_BANDS.items()}
        for i, score in enumerate(x):
            got = quality_memberships(score)
            for k in expected:
                self.assertAlmostEqual(got[k], expected[k][i], places=14)

    def test_overlap_decisions_and_endpoint_shoulders(self):
        for score, label in [(0, "Poor"), (27, "Poor"), (30, "Poor"), (31, "Fair"),
                             (57.5, "Fair"), (60, "Good"), (82.5, "Good"),
                             (85, "Excellent"), (100, "Excellent")]:
            self.assertEqual(quality_label(score), label)
        for score in [-1, 101, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                quality_label(score)
