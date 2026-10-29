"""Estimate scene-bootstrap uncertainty for frozen human-dataset transfer."""
import json, numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import analysis.studies.common as c
lol = c.lol_table()
tgt = -lol['lpips'].to_numpy()
signs = c.fit_signs(lol, tgt)
arrs = c.oriented(lol, signs)
mfs = [c.mfset(a) for a in arrs]
fire, fsum = c.firing(arrs, mfs)
thetas = {s: c.calibrate(fire, fsum, tgt, np.arange(len(lol)), seed=s) for s in [42, 1, 2, 3, 4]}
lin = make_pipeline(StandardScaler(), LinearRegression()).fit(np.column_stack(arrs), tgt)
res = {}
rng = np.random.default_rng(11)
for ds in ['squarelol', 'lieq', 'rlie']:
    X = pd.read_csv(c.OUT / f'feat_{ds}_area256.csv')
    mos = X['mos'].to_numpy(float)
    sc = X['scene'].astype(str).to_numpy()
    seeds = {s: round(c.within_group(c.fuzzy_predict(X, signs, mfs, th), mos, sc)['mean_srcc'], 3) for s, th in thetas.items()}
    pf = c.fuzzy_predict(X, signs, mfs, thetas[42])
    pl = lin.predict(np.column_stack(c.oriented(X, signs)))
    per = []
    for g, d in X.assign(pf=pf, pl=pl).groupby('scene'):
        if d['mos'].nunique() > 1:
            per.append((c.srho(d.pf, d.mos), c.srho(d.pl, d.mos)))
    per = np.array(per)
    diffs = per[:, 0] - per[:, 1]
    bs = [np.mean(rng.choice(diffs, len(diffs))) for _ in range(5000)]
    res[ds] = {'fuzzy_within_scene_by_seed': seeds, 'within_scene_fuzzy_minus_linear': [round(float(diffs.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}
(c.OUT / 'transfer_uncertainty.json').write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
