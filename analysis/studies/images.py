"""Resolve image and label records for the configured datasets."""
from analysis.studies.support import *
from evaluation.nr_features import METHODS

def rgb8(rgb):
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)

def entropy_gray(gray):
    h = cv2.calcHist([np.ascontiguousarray(gray)], [0], None, [256], [0, 256]).ravel()
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def basic(rgb):
    g = cv2.cvtColor(rgb8(rgb), cv2.COLOR_RGB2GRAY)
    gf = g.astype(float)
    return {'entropy': entropy_gray(g), 'contrast': float(gf.std() / 255), 'sharpness': float(cv2.Laplacian(gf, cv2.CV_64F).var() / 255 ** 2)}

def regional(rgb):
    glob = basic(rgb)
    g = cv2.cvtColor(rgb8(rgb), cv2.COLOR_RGB2GRAY)
    gf = g.astype(float) / 255
    lap = cv2.Laplacian(g.astype(float), cv2.CV_64F) / 255
    tiles = []
    for yy in np.array_split(np.arange(g.shape[0]), 4):
        for xx in np.array_split(np.arange(g.shape[1]), 4):
            tiles.append(basic(rgb[yy[0]:yy[-1] + 1, xx[0]:xx[-1] + 1]))
    tq = pd.DataFrame(tiles)
    d = {f'V0_{k}': v for k, v in glob.items()}
    for i, t in enumerate(tiles):
        d.update({f'tile{i:02d}_{k}': v for k, v in t.items()})
    masks = {'dark': gf < 1 / 3, 'middle': (gf >= 1 / 3) & (gf <= 2 / 3), 'bright': gf > 2 / 3, 'dark_middle': gf <= 2 / 3}
    for n, mask in masks.items():
        npx = int(mask.sum())
        d[f'{n}_pixels'] = npx
        d[f'{n}_entropy'] = entropy_gray(g[mask].reshape(-1, 1)) if npx else np.nan
        d[f'{n}_contrast'] = float(gf[mask].std()) if npx else np.nan
        d[f'{n}_sharpness'] = float(lap[mask].var()) if npx else glob['sharpness']
    vals = {'V1': (tq.entropy.quantile(0.1), tq.contrast.median(), tq.sharpness.quantile(0.9)), 'V2': (glob['entropy'], glob['contrast'], d['dark_middle_sharpness']), 'V3': (tq.entropy.quantile(0.1), glob['contrast'], d['dark_sharpness'])}
    for v, nums in vals.items():
        d.update(dict(zip([f'{v}_{k}' for k in c.NR], nums)))
    return d

def enhanced(low, method):
    return np.clip(METHODS[method](low), 0, 1).astype(np.float32)

def human_meta(ds):
    x = external(ds)[['name', 'scene', 'method', 'mos']].copy()
    folder = {'squarelol': c.HUMAN_ROOTS['squarelol'] / 'enhancement', 'lieq': c.HUMAN_ROOTS['lieq'] / 'dataset/Enhancement', 'rlie': c.HUMAN_ROOTS['rlie'] / 'enhancement'}[ds]
    x['path'] = [str(folder / n) for n in x.name]
    return x

def choose(df, variant):
    z = df.copy()
    for n in c.NR:
        z[n] = z[f'{variant}_{n}']
    return z

def variant_features(rgb, variant):
    if variant == 'V0':
        return basic(rgb)
    g = cv2.cvtColor(rgb8(rgb), cv2.COLOR_RGB2GRAY)
    gf = g.astype(float)
    if variant == 'V1':
        values = [basic(rgb[y:y + 64, x:x + 64]) for y in range(0, 256, 64) for x in range(0, 256, 64)]
        return dict(zip(c.NR, [np.quantile([t['entropy'] for t in values], 0.1), np.median([t['contrast'] for t in values]), np.quantile([t['sharpness'] for t in values], 0.9)]))
    lap = cv2.Laplacian(gf, cv2.CV_64F) / 255
    mask = gf / 255 <= 2 / 3 if variant == 'V2' else gf / 255 < 1 / 3
    en = entropy_gray(g) if variant == 'V2' else np.quantile([entropy_gray(g[y:y + 64, x:x + 64]) for y in range(0, 256, 64) for x in range(0, 256, 64)], 0.1)
    return {'entropy': en, 'contrast': float(gf.std() / 255), 'sharpness': float(lap[mask].var()) if mask.any() else float(lap.var())}
