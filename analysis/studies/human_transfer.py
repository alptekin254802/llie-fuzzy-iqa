"""Evaluate LOL-calibrated models against human scores."""
import sys, json, numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import analysis.studies.common as c
ds, mode = (sys.argv[1], sys.argv[2])
X = pd.read_csv(c.OUT / f'feat_{ds}_{mode}.csv')
mos = X['mos'].to_numpy(float)
sc = X['scene'].astype(str).to_numpy()
signs, mfs, theta = c.lol_frozen_model()
lol = c.lol_table()
tgt = -lol['lpips'].to_numpy()
Xl = np.column_stack(c.oriented(lol, signs))
Xe = np.column_stack(c.oriented(X, signs))
MK = {'Linear': lambda: make_pipeline(StandardScaler(), LinearRegression()), 'SVR': lambda: make_pipeline(StandardScaler(), SVR(C=10.0, gamma='scale')), 'RF': lambda: RandomForestRegressor(n_estimators=300, random_state=c.SEED, n_jobs=2)}
res = {'dataset': ds, 'mode': mode, 'n': int(len(X)), 'scenes': int(len(set(sc)))}
res['feature_quantiles_q05_q50_q95'] = {f: {'LOL': np.quantile(lol[f], [0.05, 0.5, 0.95]).round(4).tolist(), ds: np.quantile(X[f], [0.05, 0.5, 0.95]).round(4).tolist()} for f in c.NR}
zs = {}
preds = {'NR-free fuzzy (LOL-frozen)': c.fuzzy_predict(X, signs, mfs, theta)}
for k, mk in MK.items():
    preds[f'{k} (LOL-trained)'] = mk().fit(Xl, tgt).predict(Xe)
for f in ['entropy', 'contrast', 'brightness']:
    preds[f'{f} (raw)'] = X[f].to_numpy()
preds['sharpness (oriented)'] = -X['sharpness'].to_numpy()
for k, p in preds.items():
    s, kt, pl = c.iqa_scores(p, mos)
    w = c.within_group(np.asarray(p), mos, sc)
    zs[k] = {'srcc': round(s, 3), 'krcc': round(kt, 3), 'plcc': round(pl, 3), 'within_scene_srcc': round(w['mean_srcc'], 3), 'top1': round(w['top1'], 3), 'pair_acc': round(w['pair_acc'], 3)}
res['zero_shot'] = zs
folds = list(c.group_kfold(sc))
rec = {}
arrs = [None]
oof = {k: np.full(len(X), np.nan) for k in ['NR-free fuzzy (MOS-cal)', 'Linear', 'SVR', 'RF']}
for tr, te in folds:
    s_tr = c.fit_signs(X.iloc[tr], mos[tr])
    a_all = c.oriented(X, s_tr)
    mf = [c.mfset(a[tr]) for a in a_all]
    fire, fsum = c.firing(a_all, mf)
    th = c.calibrate(fire, fsum, mos, tr)
    oof['NR-free fuzzy (MOS-cal)'][te] = fire[te] @ th / fsum[te]
    Xa = np.column_stack(a_all)
    for k, mk in MK.items():
        oof[k][te] = mk().fit(Xa[tr], mos[tr]).predict(Xa[te])
for k, p in oof.items():
    fold_s = [c.srho(p[te], mos[te]) for _, te in folds]
    s, kt, pl = c.iqa_scores(p, mos)
    w = c.within_group(p, mos, sc)
    rec[k] = {'srcc_pooled': round(s, 3), 'srcc_foldmean': round(float(np.mean(fold_s)), 3), 'krcc': round(kt, 3), 'plcc': round(pl, 3), 'within_scene_srcc': round(w['mean_srcc'], 3), 'top1': round(w['top1'], 3), 'pair_acc': round(w['pair_acc'], 3)}
res['scene_disjoint_recal'] = rec
(c.OUT / f'human_transfer_{ds}_{mode}.json').write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ['dataset', 'mode', 'n', 'scenes']}))
print(pd.DataFrame(res['zero_shot']).T.to_string())
print(pd.DataFrame(res['scene_disjoint_recal']).T.to_string())
print(json.dumps(res['feature_quantiles_q05_q50_q95']))
