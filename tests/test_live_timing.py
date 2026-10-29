"""Check that timing validation rejects corrupt or partial sessions."""
import copy
import unittest
from realtime.summarize import summarize_rows, statistics


class LiveTimingTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            dict(frame_index='0', warmup='True', capture_start_ns='0',
                 capture_end_ns='1000000', processing_end_ns='3000000',
                 capture_wait_ms='1', processing_ms='2', camera_interval_ms='', gamma='.6'),
            dict(frame_index='1', warmup='False', capture_start_ns='4000000',
                 capture_end_ns='5000000', processing_end_ns='8000000',
                 capture_wait_ms='1', processing_ms='3', camera_interval_ms='4', gamma='.7'),
            dict(frame_index='2', warmup='False', capture_start_ns='9000000',
                 capture_end_ns='10000000', processing_end_ns='16000000',
                 capture_wait_ms='1', processing_ms='6', camera_interval_ms='5', gamma='.8')]

    def test_warmup_predecessor_and_deadline(self):
        rows, seconds = summarize_rows(self.rows, 3, 1)
        result = statistics(rows, seconds)
        self.assertEqual(result['n'], 2)
        self.assertEqual(result['deadline_misses'], 1)
        self.assertEqual(result['processing_ms']['median'], 4.5)
        self.assertAlmostEqual(seconds, .013)
        self.assertAlmostEqual(result['actual_loop_fps'], 2/.013)

    def test_partial_or_corrupt_session_is_rejected(self):
        variants = [self.rows[:-1]]
        for key, value in [('processing_ms', '4'), ('frame_index', '0'),
                           ('warmup', 'True'), ('camera_interval_ms', 'nan')]:
            rows = copy.deepcopy(self.rows); rows[1][key] = value; variants.append(rows)
        for rows in variants:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize_rows(rows, 3, 1)
