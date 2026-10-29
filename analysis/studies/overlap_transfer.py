"""Evaluate transfer after separating SQUARE-LOL scenes shared with LOL."""
import json, numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import analysis.studies.common as c
X = pd.read_csv(c.OUT / 'feat_squarelol_area256.csv')
ov = pd.read_csv(Path(__file__).resolve().parents[2] / 'results/studies/records/squarelol_lol_scene_matches.csv')
ovset = set(ov.loc[ov['corr'] >= 0.9, 'squarelol'])
newset = set(ov.loc[ov['corr'] < 0.9, 'squarelol'])
assert set(X['scene']) == ovset | newset
signs, mfs, theta = c.lol_frozen_model()
lol = c.lol_table()
tgt = -lol['lpips'].to_numpy()
Xl = np.column_stack(c.oriented(lol, signs))
Xe = np.column_stack(c.oriented(X, signs))
P = {'NR-free fuzzy (LOL-frozen)': c.fuzzy_predict(X, signs, mfs, theta), 'Linear (LOL-trained)': make_pipeline(StandardScaler(), LinearRegression()).fit(Xl, tgt).predict(Xe), 'entropy': X['entropy'].to_numpy(), 'contrast': X['contrast'].to_numpy()}
mos = X['mos'].to_numpy()
sc = X['scene'].to_numpy()
res = {}
for sub, mask in [(f'overlap ({len(ovset)} scenes)', np.isin(sc, list(ovset))), (f'no overlap ({len(newset)} scenes)', np.isin(sc, list(newset))), (f'all ({len(set(sc))})', np.ones(len(X), bool))]:
    res[sub] = {}
    for k, p in P.items():
        w = c.within_group(p[mask], mos[mask], sc[mask])
        res[sub][k] = {'srcc': round(c.srho(p[mask], mos[mask]), 3), 'within_scene_srcc': round(w['mean_srcc'], 3), 'pair_acc': round(w['pair_acc'], 3), 'top1': round(w['top1'], 3), 'scenes': w['groups']}
X['pred'] = P['NR-free fuzzy (LOL-frozen)']
X['r_pred'] = X.groupby('scene')['pred'].rank(ascending=False)
X['r_hum'] = X.groupby('scene')['mos'].rank(ascending=False)
res['mean_rank_by_method (1=best)'] = X.groupby('method')[['r_hum', 'r_pred']].mean().round(2).sort_values('r_hum').to_dict('index')
(c.OUT / 'overlap_transfer.json').write_text(json.dumps(res, indent=1))
for k in list(res)[:3]:
    print(k)
    print(pd.DataFrame(res[k]).T.to_string())
print(pd.DataFrame(res['mean_rank_by_method (1=best)']).T.to_string())
