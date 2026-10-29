"""Cache image statistics at native or 256-square resolution."""
import sys, time, numpy as np, pandas as pd, scipy.io as sio
from pathlib import Path
import analysis.studies.common as c

def meta(ds):
    if ds == 'rlie':
        sc = pd.read_csv(c.HUMAN_ROOTS['rlie'] / 'normalized_scores.csv').rename(columns={'Image Name': 'name', 'Score': 'mos'})
        sc['scene'] = sc['name'].str.slice(0, 3)
        sc['path'] = [str(c.HUMAN_ROOTS['rlie'] / 'enhancement' / n) for n in sc['name']]
        sc['method'] = sc['name'].str.split('_').str[1].str.replace('.png', '', regex=False)
    elif ds == 'squarelol':
        root = c.HUMAN_ROOTS['squarelol']
        sc = pd.read_csv(root / 'name_mos.txt', sep='\t', header=None, names=['name', 'mos'])
        sc['scene'] = sc['name'].str.split('_').str[0]
        sc['path'] = [str(root / 'enhancement' / n) for n in sc['name']]
        sc['method'] = sc['name'].str.replace('^low\\d+_', '', regex=True).str.replace('_0\\.png$', '', regex=True)
    elif ds == 'lieq':
        root = c.HUMAN_ROOTS['lieq']
        mos = sio.loadmat(root / 'MOS.mat')['MOSz'].ravel()
        nm = sio.loadmat(root / 'imgNames.mat')['imgNames']
        names = [str(nm[i, 2][0]).split('\\')[-1] for i in range(nm.shape[0])]
        sc = pd.DataFrame({'name': names, 'mos': mos})
        sc['scene'] = sc['name'].str.slice(0, 3)
        sc['path'] = [str(root / 'dataset' / 'Enhancement' / n) for n in names]
        sc['method'] = sc['name'].str.slice(4).str.replace('.png', '', regex=False)
    else:
        raise ValueError(ds)
    return sc.reset_index(drop=True)
if __name__ == '__main__':
    ds, mode = (sys.argv[1], sys.argv[2])
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 150
    m = meta(ds)
    out = c.OUT / f'feat_{ds}_{mode}.csv'
    done = set(pd.read_csv(out)['name']) if out.exists() else set()
    t0 = time.time()
    rows = []
    for _, r in m.iterrows():
        if r['name'] in done:
            continue
        f = c.repo_nr_features(c.load_rgb01(r['path'], mode))
        rows.append({'name': r['name'], 'scene': r['scene'], 'method': r['method'], 'mos': r['mos'], **f})
        if time.time() - t0 > budget:
            break
    if rows:
        pd.DataFrame(rows).to_csv(out, mode='a', header=not out.exists(), index=False)
    n = len(pd.read_csv(out)) if out.exists() else 0
    print(f'{ds}/{mode}: {n}/{len(m)} done')
