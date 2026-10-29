"""Extract BRISQUE features for the human-rated image sets."""
import sys, time, numpy as np, pandas as pd, cv2
from scipy.special import gamma as G
import analysis.studies.common as c
from analysis.studies.features import meta
_g = np.arange(0.2, 10.001, 0.001)
_r_ggd = G(1 / _g) * G(3 / _g) / G(2 / _g) ** 2
_r_aggd = G(2 / _g) ** 2 / (G(1 / _g) * G(3 / _g))

def mscn(img):
    mu = cv2.GaussianBlur(img, (7, 7), 7 / 6, borderType=cv2.BORDER_REPLICATE)
    sd = np.sqrt(np.abs(cv2.GaussianBlur(img * img, (7, 7), 7 / 6, borderType=cv2.BORDER_REPLICATE) - mu * mu))
    return (img - mu) / (sd + 1)

def ggd(x):
    x = x.ravel()
    s2 = np.mean(x ** 2)
    rho = s2 / (np.mean(np.abs(x)) ** 2 + 1e-12)
    return [_g[np.argmin((_r_ggd - rho) ** 2)], s2]

def aggd(x):
    x = x.ravel()
    l = x[x < 0]
    r = x[x >= 0]
    ls = np.sqrt(np.mean(l ** 2)) if l.size else 1e-06
    rs = np.sqrt(np.mean(r ** 2)) if r.size else 1e-06
    gh = ls / rs
    rh = np.mean(np.abs(x)) ** 2 / np.mean(x ** 2)
    Rh = rh * (gh ** 3 + 1) * (gh + 1) / (gh ** 2 + 1) ** 2
    a = _g[np.argmin((_r_aggd - Rh) ** 2)]
    m = (rs - ls) * G(2 / a) / G(1 / a) * np.sqrt(G(1 / a) / G(3 / a))
    return [a, m, ls ** 2, rs ** 2]

def feats(gray):
    out = []
    for s in range(2):
        m = mscn(gray)
        out += ggd(m)
        for sh in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            out += aggd(m * np.roll(m, shift=sh, axis=(0, 1)))
        gray = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
    return out
if __name__ == '__main__':
    ds = sys.argv[1]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 150
    mode = sys.argv[3] if len(sys.argv) > 3 else 'native'
    m = meta(ds)
    out = c.OUT / (f'brisque_{ds}.csv' if mode == 'native' else f'brisque_{ds}_{mode}.csv')
    done = set(pd.read_csv(out)['name']) if out.exists() else set()
    t0 = time.time()
    rows = []
    for _, r in m.iterrows():
        if r['name'] in done:
            continue
        g = cv2.cvtColor(cv2.imread(r['path']), cv2.COLOR_BGR2GRAY).astype(np.float64) if mode == 'native' else cv2.cvtColor((c.load_rgb01(r['path'], mode) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
        rows.append({'name': r['name'], **{f'b{i}': v for i, v in enumerate(feats(g))}})
        if time.time() - t0 > budget:
            break
    if rows:
        pd.DataFrame(rows).to_csv(out, mode='a', header=not out.exists(), index=False)
    print(ds, len(pd.read_csv(out)), '/', len(m), f'{(time.time() - t0) / max(len(rows), 1) * 1000:.0f} ms/img')
