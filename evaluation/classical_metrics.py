"""Regenerate PSNR, SSIM and entropy for all five LOL enhancement operators."""
from pathlib import Path
import argparse
import pandas as pd,numpy as np
from evaluation.nr_features import METHODS,load_rgb_float
from evaluation.metrics import compute_psnr,compute_ssim,compute_entropy
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preprocessed',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new output file')
    rows=[]
    for path in sorted((args.preprocessed/'low').glob('*.png')):
        low=load_rgb_float(path);high=load_rgb_float(args.preprocessed/'high'/path.name)
        for method,fn in METHODS.items():
            im=np.clip(fn(low),0,1).astype(np.float32)
            rows.append({'filename':path.name,'method':method,'psnr':compute_psnr(im,high),'ssim':compute_ssim(im,high),'entropy':compute_entropy(im)})
    if not rows:raise ValueError('No paired low/high PNG images')
    args.output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(args.output,index=False)
if __name__=='__main__':main()
