"""Extract BRISQUE features from the classical LOL enhancement outputs."""
import sys, time, numpy as np, pandas as pd, cv2
import analysis.studies.common as c
from evaluation.nr_features import METHODS, load_rgb_float, resolve_image
LOW_DIR = c.PREPROCESSED / 'low'
from analysis.studies.brisque_features import feats
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 150
out = c.OUT / 'brisque_lol.csv'
ref = pd.read_csv(c.RES / 'nr_features.csv')
done = set(zip(*[pd.read_csv(out)[k] for k in ['filename', 'method']])) if out.exists() else set()
t0 = time.time()
rows = []
for fname in sorted(ref['filename'].unique()):
    if all(((fname, m) in done for m in METHODS)):
        continue
    low = load_rgb_float(resolve_image(LOW_DIR, fname))
    for label, fn in METHODS.items():
        if (fname, label) in done:
            continue
        enh = np.clip(fn(low), 0.0, 1.0).astype(np.float32)
        f3 = c.repo_nr_features(enh)
        gray = cv2.cvtColor((enh * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
        rows.append({'filename': fname, 'method': label, 'chk_entropy': f3['entropy'], 'chk_sharpness': f3['sharpness'], **{f'b{i}': v for i, v in enumerate(feats(gray))}})
    if time.time() - t0 > budget:
        break
if rows:
    pd.DataFrame(rows).to_csv(out, mode='a', header=not out.exists(), index=False)
d = pd.read_csv(out).merge(ref, on=['filename', 'method'])
print(len(d), '/ 2425; max |entropy diff| =', float((d.chk_entropy - d.entropy).abs().max()), 'max |sharp diff| =', float((d.chk_sharpness - d.sharpness).abs().max()))
