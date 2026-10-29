import itertools
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
import pandas as pd
from evaluation.frozen_scorer import FrozenScorer
from evaluation.three_features import extract
from evaluation.nr_features import nr_features
from preprocessing.preprocess import preprocess_image, preprocess_dataset
from analysis.monotone_fuzzy import C, memberships

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_features_match_calibration_extractor(self):
        rng = np.random.default_rng(81)
        for im in [np.zeros((17, 25, 3), np.uint8), np.full((17, 25, 3), 255, np.uint8),
                   rng.integers(0, 256, (43, 67, 3), dtype=np.uint8)]:
            expected = nr_features(im.astype(np.float32) / 255)
            np.testing.assert_allclose(extract(im), [expected[k] for k in ("entropy", "contrast", "sharpness")], rtol=0, atol=1e-12)

    def test_saved_calibration_predictions(self):
        df = pd.read_csv(ROOT / "results/nr_features.csv").merge(pd.read_csv(ROOT / "results/nr_fuzzy_scores.csv"), on=["filename", "method"], validate="one_to_one")
        prediction = FrozenScorer().predict(df[["entropy", "contrast", "sharpness"]])
        np.testing.assert_allclose(prediction, df.nr_fuzzy_free, rtol=0, atol=1e-9)

    def test_monotone_rules_partition_and_response(self):
        model = FrozenScorer("nr_mono")
        self.assertGreaterEqual(float((C @ model.theta).min()), -1e-8)
        axes = [np.unique(np.r_[np.linspace(m[0, 0], m[-1, -1], 31), m.ravel(),
                                     m[0, 0] - 1, m[-1, -1] + 1]) for m in model.mfs]
        x = np.array(list(itertools.product(*axes)))
        mu = memberships(x, model.mfs)
        np.testing.assert_allclose(mu.sum(axis=2), 1, rtol=0, atol=1e-12)
        prediction = model.predict(x / model.signs).reshape(tuple(map(len, axes)))
        for axis in range(3):
            self.assertGreaterEqual(float(np.diff(prediction, axis=axis).min()), -1e-8)

    def test_lol_bilinear_preprocessing_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "input"
            im = np.random.default_rng(7).integers(0, 256, (43, 61, 3), dtype=np.uint8)
            for part in ("low", "high"):
                (root / part).mkdir(parents=True)
                cv2.imwrite(str(root / part / "test.png"), im)
            expected = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
            np.testing.assert_array_equal(preprocess_image(root / "low/test.png"), expected)
            output = Path(folder) / "output"
            manifest = preprocess_dataset(root, output)
            self.assertEqual(manifest["counts"], {"low": 1, "high": 1})
            saved = cv2.cvtColor(cv2.imread(str(output / "low/test.png")), cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(saved, (expected * 255).astype(np.uint8))
            with self.assertRaises(FileExistsError):
                preprocess_dataset(root, output)

    def test_demo_uses_normalized_features(self):
        from analysis.realtime_pipeline_demo import extract_features, RealAssessor
        im = np.random.default_rng(41).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        # The recorded demo uses float32 std; the calibration extractor uses float64.
        np.testing.assert_allclose(extract_features(im), extract(im), rtol=1e-6, atol=1e-8)
        score = RealAssessor().infer(*extract_features(im))
        self.assertAlmostEqual(score, FrozenScorer()(extract_features(im)), places=10)


if __name__ == "__main__":
    unittest.main()
