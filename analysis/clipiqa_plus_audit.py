"""CLIP-IQA+ scores on the same LOL enhanced images as the main evaluation."""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, t as student_t
try:
    import torch
    import pyiqa
except ImportError as exc:
    raise SystemExit(f'Missing dependency. Install with: pip install torch torchvision pyiqa\nImport error: {exc}')
from preprocessing.enhancement.methods import apply_clahe, apply_gamma_correction, apply_log_transform, apply_retinex, histogram_equalization
BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
LOW_DIR = BASE / 'preprocessed' / 'low'
CLASSICAL_CSV = RESULTS / 'enhancement_metrics.csv'
SCORES_CSV = RESULTS / 'clipiqa_plus_scores.csv'
METHODS = {'clahe': apply_clahe, 'gamma': apply_gamma_correction, 'log': apply_log_transform, 'retinex': apply_retinex, 'hist_eq': histogram_equalization}

def resolve_image(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if candidate.exists():
        return candidate
    stem = Path(filename).stem
    for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
        alt = directory / f'{stem}{ext}'
        if alt.exists():
            return alt
    raise FileNotFoundError(f'Image not found for {filename!r} in {directory}')

def load_rgb_float(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

def to_tensor(rgb01: np.ndarray, device: str) -> 'torch.Tensor':
    arr = np.ascontiguousarray(np.clip(rgb01, 0.0, 1.0))
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device)

def make_metric(device: str):
    return pyiqa.create_metric('clipiqa+', device=device)

def compute_scores(device: str) -> pd.DataFrame:
    if not CLASSICAL_CSV.exists():
        raise SystemExit(f'Missing {CLASSICAL_CSV}')
    if not LOW_DIR.exists():
        raise SystemExit(f'Missing {LOW_DIR}; cannot recompute CLIP-IQA+ scores.')
    metric = make_metric(device)
    df = pd.read_csv(CLASSICAL_CSV)
    filenames = sorted(df['filename'].unique())
    methods_in_csv = sorted(df['method'].unique())
    unknown = [m for m in methods_in_csv if m not in METHODS]
    if unknown:
        raise SystemExit(f'Unknown methods in {CLASSICAL_CSV.name}: {unknown}')
    print(f'Computing CLIP-IQA+ scores on {device}')
    print(f'Images: {len(filenames)} | methods: {methods_in_csv}')
    rows = []
    t0 = time.perf_counter()
    for i, fname in enumerate(filenames, 1):
        low = load_rgb_float(resolve_image(LOW_DIR, fname))
        for label, fn in METHODS.items():
            enh = np.clip(fn(low), 0.0, 1.0).astype(np.float32)
            x = to_tensor(enh, device)
            with torch.no_grad():
                if device == 'cuda':
                    torch.cuda.synchronize()
                score = float(metric(x).item())
                if device == 'cuda':
                    torch.cuda.synchronize()
            rows.append({'filename': fname, 'method': label, 'clipiqa_plus': score})
        if i % 25 == 0 or i == len(filenames):
            print(f'  {i}/{len(filenames)} images done ({time.perf_counter() - t0:.1f}s)')
    out = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(SCORES_CSV, index=False)
    print(f'Saved: {SCORES_CSV} ({len(out)} rows)')
    return out
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args()
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    compute_scores(device)
if __name__=='__main__':main()
