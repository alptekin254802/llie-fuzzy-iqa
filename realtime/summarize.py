"""Validate and summarize the archived per-frame camera timings."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import numpy as np


def summarize_rows(rows, total_frames, warmup_frames):
    """Reject incomplete sessions and inconsistent timestamp/measurement pairs."""
    if len(rows) != total_frames or not 0 < warmup_frames < total_frames:
        raise ValueError("Incomplete session or invalid warmup count")
    previous = None
    for i, row in enumerate(rows):
        if int(row['frame_index']) != i or row['warmup'] != str(i < warmup_frames):
            raise ValueError("Frame index or warmup flag mismatch")
        start, capture, end = (int(row[k]) for k in
                              ('capture_start_ns', 'capture_end_ns', 'processing_end_ns'))
        if not start <= capture <= end or (previous and start < previous[1]):
            raise ValueError("Nonmonotonic timestamps")
        checks = {'capture_wait_ms': (capture-start)/1e6,
                  'processing_ms': (end-capture)/1e6}
        if previous:
            checks['camera_interval_ms'] = (capture-previous[0])/1e6
        elif row['camera_interval_ms'] != '':
            raise ValueError("First frame must have no camera interval")
        for key, expected in checks.items():
            value = float(row[key])
            if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"Inconsistent {key} at frame {i}")
        if not math.isfinite(float(row['gamma'])):
            raise ValueError("Nonfinite gamma")
        previous = capture, end
    retained = rows[warmup_frames:]
    seconds = (int(rows[-1]['processing_end_ns']) -
               int(rows[warmup_frames-1]['processing_end_ns']))/1e9
    if seconds <= 0:
        raise ValueError("Nonpositive measured duration")
    return retained, seconds


def statistics(rows, seconds):
    processing = np.array([float(r['processing_ms']) for r in rows])
    intervals = np.array([float(r['camera_interval_ms']) for r in rows])
    n = len(rows)
    misses = int(np.sum(processing > intervals))
    return {'n': n, 'processing_ms': {
        'median': float(np.median(processing)),
        'p95': float(np.quantile(processing, .95)),
        'p99': float(np.quantile(processing, .99)),
        'maximum': float(np.max(processing)), 'mean': float(np.mean(processing))},
        'camera_interval_median_ms': float(np.median(intervals)),
        'camera_interval_mean_ms': float(np.mean(intervals)),
        'deadline_misses': misses, 'deadline_miss_fraction': misses/n,
        'deadline_miss_percent': 100*misses/n, 'measured_loop_seconds': seconds,
        'actual_loop_fps': n/seconds}


def compare(actual, expected, path=''):
    """Compare all recomputed fields with the archived summary."""
    if isinstance(actual, dict):
        for key, value in actual.items():
            compare(value, expected[key], path+'/'+key)
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            raise ValueError('Length mismatch: '+path)
        for i, value in enumerate(actual):
            compare(value, expected[i], path+'/'+str(i))
    elif isinstance(actual, (int, float)):
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Summary mismatch: '+path)
    elif actual != expected:
        raise ValueError('Summary mismatch: '+path)


def summarize_archive(root):
    if (root/'protocol.json').exists() and 'mjpg_1' in (root/'protocol.json').read_text(encoding='utf8'):
        from realtime.verify_mjpg import verify_archive
        return verify_archive(root)
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf8'))
    provenance = json.loads((root/'provenance.json').read_text(encoding='utf8'))
    for name, expected in provenance['exported_file_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Archive hash mismatch: '+name)
    sessions, primary, all_rows = [], [], []
    primary_seconds = all_seconds = 0.0
    for session in protocol['sessions']:
        folder = root/session['id']
        selftest = json.loads((folder/'selftest.json').read_text(encoding='utf8'))
        if (not selftest['passed'] or selftest['frames_completed'] != protocol['selftest']['frames']
                or selftest['original_sha256'] != selftest['lut_sha256']):
            raise ValueError('Failed or incomplete selftest: '+session['id'])
        with (folder/'frames.csv').open(encoding='utf8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        retained, seconds = summarize_rows(rows, session['total_frames'], session['warmup_frames'])
        sessions.append({'session': session['id'], 'exposure': session['exposure'],
                         **statistics(retained, seconds)})
        all_rows.extend(retained); all_seconds += seconds
        if session['exposure'] == -5:
            primary.extend(retained); primary_seconds += seconds
    result = {'sessions': sessions, 'pooled': {
        'primary_minus5': statistics(primary, primary_seconds),
        'all_exposures': statistics(all_rows, all_seconds)}}
    compare(result, json.loads((root/'summary.json').read_text(encoding='utf8')))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parents[1]/'results/live_timing/mjpg_30fps')
    args = parser.parse_args()
    print(json.dumps({'status': 'PASS', **summarize_archive(args.input)}, indent=2))


if __name__ == '__main__':
    main()
