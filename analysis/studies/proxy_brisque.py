"""Fit a BRISQUE-feature regressor to the perceptual proxy target."""
import json, numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import analysis.studies.common as c
BC = [f'b{i}' for i in range(36)]
lol = c.lol_table().merge(pd.read_csv(c.OUT / 'brisque_lol.csv'), on=['filename', 'method'])
tgt = -lol['lpips'].to_numpy()
g = lol['filename'].to_numpy()
mk = lambda: make_pipeline(StandardScaler(), SVR(C=10.0, gamma='scale'))
res = {'LOL_5fold_vs_-LPIPS': {}}
oof = np.full(len(lol), np.nan)
fs = []
for tr, te in c.group_kfold(g):
    oof[te] = mk().fit(lol[BC].to_numpy()[tr], tgt[tr]).predict(lol[BC].to_numpy()[te])
    fs.append(c.srho(oof[te], tgt[te]))
res['LOL_5fold_vs_-LPIPS']['BRISQUE-36+SVR'] = {'foldmean': round(float(np.mean(fs)), 3), 'std': round(float(np.std(fs, ddof=1)), 3), 'within_scene_srcc': round(c.within_group(oof, tgt, g)['mean_srcc'], 3)}
full = mk().fit(lol[BC].to_numpy(), tgt)
signs, mfs, theta = c.lol_frozen_model()
res['zero_shot'] = {}
for ds in ['squarelol', 'lieq', 'rlie']:
    X = pd.read_csv(c.OUT / f'feat_{ds}_area256.csv').merge(pd.read_csv(c.OUT / f'brisque_{ds}_area256.csv'), on='name')
    mos = X['mos'].to_numpy(float)
    sc = X['scene'].astype(str).to_numpy()
    r = {}
    for name, p in [('NR-free fuzzy (LOL-frozen)', c.fuzzy_predict(X, signs, mfs, theta)), ('BRISQUE-36+SVR (LOL -LPIPS trained)', full.predict(X[BC].to_numpy()))]:
        w = c.within_group(p, mos, sc)
        r[name] = {'srcc': round(c.srho(p, mos), 3), 'within_scene_srcc': round(w['mean_srcc'], 3), 'pair_acc': round(w['pair_acc'], 3), 'top1': round(w['top1'], 3)}
    folds = list(c.group_kfold(sc))
    p = np.full(len(X), np.nan)
    for tr, te in folds:
        p[te] = mk().fit(X[BC].to_numpy()[tr], mos[tr]).predict(X[BC].to_numpy()[te])
    w = c.within_group(p, mos, sc)
    r['BRISQUE-36+SVR (MOS-cal, 256)'] = {'srcc': round(c.srho(p, mos), 3), 'within_scene_srcc': round(w['mean_srcc'], 3), 'pair_acc': round(w['pair_acc'], 3), 'top1': round(w['top1'], 3)}
    res['zero_shot'][ds] = r
    print(ds)
    print(pd.DataFrame(r).T.to_string())
print(res['LOL_5fold_vs_-LPIPS'])
(c.OUT / 'proxy_brisque.json').write_text(json.dumps(res, indent=1))
