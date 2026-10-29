"""Compute pretrained metric scores on human-rated enhanced images."""
import argparse, time, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch, scipy.io as sio
import pyiqa
from analysis.studies import common as c
OUT = c.OUT
NR_METRICS = ['musiq', 'clipiqa', 'clipiqa+', 'niqe', 'brisque']
FR_METRICS = ['lpips', 'lpips-vgg', 'dists']

def items(ds):
    if ds == 'rlie':
        sc = pd.read_csv(c.HUMAN_ROOTS['rlie'] / 'normalized_scores.csv').rename(columns={'Image Name': 'name', 'Score': 'mos'})
        sc['path'] = [str(c.HUMAN_ROOTS['rlie'] / 'enhancement' / n) for n in sc['name']]
        sc['ref'] = [str(c.HUMAN_ROOTS['rlie'] / 'ref' / (n[:3] + '.png')) for n in sc['name']]
    elif ds == 'squarelol':
        r = c.HUMAN_ROOTS['squarelol']
        sc = pd.read_csv(r / 'name_mos.txt', sep='\t', header=None, names=['name', 'mos'])
        sc['path'] = [str(r / 'enhancement' / n) for n in sc['name']]
        sc['ref'] = [str(r / 'all_normal' / ('normal' + n.split('_')[0][3:] + '.png')) for n in sc['name']]
    elif ds == 'lieq':
        r = c.HUMAN_ROOTS['lieq']
        nm = sio.loadmat(r / 'imgNames.mat')['imgNames']
        mos = sio.loadmat(r / 'MOS.mat')['MOSz'].ravel()
        names = [str(nm[i, 2][0]).split('\\')[-1] for i in range(nm.shape[0])]
        sc = pd.DataFrame({'name': names, 'mos': mos})
        sc['path'] = [str(r / 'dataset' / 'Enhancement' / n) for n in names]
        sc['ref'] = [str(r / 'dataset' / 'Reference' / (n[:3] + '.png')) for n in names]
    return sc

def tensor(path, native=False):
    rgb = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if not native:
        rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='squarelol,lieq,rlie')
    ap.add_argument('--native', action='store_true')
    ap.add_argument('--skip-maniqa', action='store_true')
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', dev)
    OUT.mkdir(exist_ok=True)
    timing = []
    for ds in a.datasets.split(','):
        meta = items(ds)
        out = OUT / (f'human_metrics_{ds}_native.csv' if a.native else f'human_metrics_{ds}.csv')
        done = pd.read_csv(out) if out.exists() else pd.DataFrame({'name': meta['name']})
        metrics = NR_METRICS + ([] if a.native or ds == 'rlie' else FR_METRICS) + ([] if a.skip_maniqa else ['maniqa'])
        for m in metrics:
            if m in done.columns and done[m].notna().all():
                print(ds, m, 'done')
                continue
            fn = pyiqa.create_metric(m, device=dev)
            vals = []
            t0 = time.time()
            for _, r in meta.iterrows():
                x = tensor(r['path'], a.native).to(dev)
                with torch.no_grad():
                    v = fn(x, tensor(r['ref']).to(dev)) if m in FR_METRICS else fn(x)
                vals.append(float(v.item()))
            dt = (time.time() - t0) / len(meta) * 1000
            timing.append({'dataset': ds, 'metric': m, 'device': str(dev), 'ms_per_img': dt, 'native': a.native})
            done = done.drop(columns=[m], errors='ignore').merge(pd.DataFrame({'name': meta['name'], m: vals}), on='name')
            done.to_csv(out, index=False)
            print(ds, m, f'{dt:.1f} ms/img', flush=True)
            del fn
            torch.cuda.empty_cache() if dev.type == 'cuda' else None
    if timing:
        pd.DataFrame(timing).to_csv(OUT / 'human_metrics_timing.csv', mode='a', header=not (OUT / 'human_metrics_timing.csv').exists(), index=False)
if __name__ == '__main__':
    main()
