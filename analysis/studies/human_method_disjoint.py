"""Evaluate human-label calibration with enhancement methods held out."""
import sys, json, numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import analysis.studies.common as c, analysis.studies.human_objectives as h
ds = sys.argv[1]
X = h.load(ds)
y = X['mos'].to_numpy(float)
meth = X['method'].astype(str).to_numpy()
signs, mfs, th0 = c.lol_frozen_model()
D = h.fuzzy_design(X, signs, mfs)
X3 = np.column_stack(c.oriented(X, signs))
B = X[h.BC].to_numpy()
methods = np.array(sorted(set(meth)))
folds = h.scene_folds(X)
res = {}
p = np.full(len(X), np.nan)
for tr, te in folds:
    mm = pd.Series(y[tr]).groupby(meth[tr]).mean()
    p[te] = pd.Series(meth[te]).map(mm).to_numpy()
res['(a) method-mean baseline, scene-disjoint'] = round(h.within_srcc(p, y, h.scene_index(X)), 3)
rng = np.random.default_rng(1)
acc = {m: [] for m in ['fuzzy LOL prior (no labels)', 'fuzzy LS', 'linear-3f', 'BRISQUE-SVR', 'BRISQUE-SVR (not demeaned)']}
for rep in range(10):
    mA = set(rng.choice(methods, len(methods) // 2, replace=False))
    for tr, te in folds:
        trm = np.array([i for i in tr if meth[i] in mA])
        tem = np.array([i for i in te if meth[i] not in mA])
        Xtr = X.iloc[trm].reset_index(drop=True)
        Xte = X.iloc[tem].reset_index(drop=True)
        Str, Ste = (h.scene_index_n(Xtr) if hasattr(h, 'scene_index_n') else None, None)
        gtr = Xtr.groupby('scene').indices
        Str = np.stack([gtr[s] for s in sorted(gtr)])
        gte = Xte.groupby('scene').indices
        Ste = np.stack([gte[s] for s in sorted(gte)])
        ytr, yte = (y[trm], y[tem])
        yc = h.demean(ytr, Str)
        Dc = h.demean(D[trm], Str)
        th = np.linalg.solve(Dc.T @ Dc + 0.001 * np.eye(27), Dc.T @ yc)
        P = {'fuzzy LOL prior (no labels)': D[tem] @ th0, 'fuzzy LS': D[tem] @ th, 'linear-3f': make_pipeline(StandardScaler(), LinearRegression()).fit(h.demean(X3[trm], Str), yc).predict(h.demean(X3[tem], Ste)), 'BRISQUE-SVR': make_pipeline(StandardScaler(), SVR(C=10.0, gamma='scale')).fit(h.demean(B[trm], Str), yc).predict(h.demean(B[tem], Ste)), 'BRISQUE-SVR (not demeaned)': make_pipeline(StandardScaler(), SVR(C=10.0, gamma='scale')).fit(B[trm], ytr).predict(B[tem])}
        for m, pp in P.items():
            acc[m].append(h.within_srcc(pp, yte, Ste))
res['(b) scene and method disjoint, within-scene SRCC over unseen methods [mean, sd]'] = {m: [round(float(np.mean(v)), 3), round(float(np.std(v)), 3)] for m, v in acc.items()}
(c.OUT / f'human_method_disjoint_{ds}.json').write_text(json.dumps(res, indent=1))
print(ds, json.dumps(res, indent=1))
