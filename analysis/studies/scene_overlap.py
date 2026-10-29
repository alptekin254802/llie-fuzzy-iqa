"""Identify source-scene overlap between SQUARE-LOL and LOL."""
import glob, numpy as np, pandas as pd, cv2
from pathlib import Path
import analysis.studies.common as c

def thumb(p):
    g = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(cv2.resize(g, (48, 32), interpolation=cv2.INTER_AREA)).astype(np.float64).ravel()
    return (g - g.mean()) / (g.std() + 1e-09)
rated = pd.read_csv(c.HUMAN_ROOTS['squarelol'] / 'name_mos.txt', sep=r'\s+', header=None, names=['name', 'mos'])
scenes = sorted(rated['name'].str.split('_').str[0].unique())
source = {p.stem.replace('normal', 'low', 1): p for p in (c.HUMAN_ROOTS['squarelol'] / 'all_normal').glob('*.png')}
lo = sorted((c.LOL_ROOT / 'high').glob('*.png'))
B = np.stack([thumb(str(p)) for p in lo]); rows = []
for scene in scenes:
    assert scene in source, "Every rated scene requires its normal-light reference."
    a = thumb(str(source[scene])); correlations = B @ a / a.size
    j = int(correlations.argmax()); value = float(correlations[j])
    rows.append(dict(squarelol=scene, best_our485=lo[j].name, corr=value, shared_source=value >= 0.9, status='verified_normal_light_thumbnail'))
df = pd.DataFrame(rows); df.to_csv(c.OUT / 'overlap_squarelol_our485.csv', index=False)
print(df['status'].value_counts().to_dict())
print('Verified shared:', int((df['corr'] >= 0.9).sum()))
print('Verified non-overlap:', int((df['corr'] < 0.9).sum()))
print('Sources outside the rated analysis:', sorted(set(source) - set(scenes)))
