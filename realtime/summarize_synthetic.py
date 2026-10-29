"""Recompute the separate floating-point synthetic timing repeat from CSVs."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from realtime.summarize import compare


def main():
    root = Path(__file__).resolve().parents[1]/'results/synthetic_timing'
    provenance = json.loads((root/'provenance.json').read_text(encoding='utf8'))
    for name, expected in provenance['exported_file_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Archive hash mismatch: '+name)
    summary = json.loads((root/'summary.json').read_text(encoding='utf8'))
    verified = 0
    for filename, section in [('timing_frames.csv', 'variants'), ('profile_frames.csv', 'profile')]:
        with (root/filename).open(encoding='utf8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        for variant in ['original', 'optimized']:
            subset = [r for r in rows if r['variant'] == variant]
            if len(subset) != 400 or len({(r['scene'], r['frame']) for r in subset}) != 400:
                raise ValueError('Incomplete or duplicated frame rows')
            for field, expected in summary[section][variant].items():
                values = np.array([float(r[field]) for r in subset if r[field] != ''])
                if not len(values):
                    if any(v != 0 for v in expected.values()):
                        raise ValueError('Missing nonzero component: '+field)
                    continue
                if len(values) != 400 or not np.isfinite(values).all():
                    raise ValueError('Incomplete or nonfinite component: '+field)
                actual = {'n': 400, 'median_ms': float(np.median(values)),
                          'p95_ms': float(np.quantile(values, .95)),
                          'mean_ms': float(np.mean(values))}
                if 'fps' in expected:
                    actual['fps'] = 1000/actual['median_ms']
                compare({k: actual[k] for k in expected}, expected); verified += 1
    with (root/'equality_selections.csv').open(encoding='utf8', newline='') as stream:
        selections = list(csv.DictReader(stream))
    if (len(selections) != 1600 or
            any(r['original_gamma'] != r['optimized_gamma'] for r in selections)):
        raise ValueError('Selection mismatch')
    print(json.dumps({'status': 'PASS', 'component_summaries_verified': verified,
                      'equal_selections': len(selections), 'median_component_sum_ms': {
                          v: summary['variants'][v]['definition_a_fuzzy']['median_ms']
                          for v in ['original', 'optimized']}}, indent=2))


if __name__ == '__main__':
    main()
