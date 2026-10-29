"""Fit and evaluate human-label models with scene-disjoint folds."""
import sys, json, numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import analysis.studies.common as c, analysis.studies.human_objectives as h
ds = sys.argv[1]
X = h.load(ds)
y = X['mos'].to_numpy(float)
S_all = h.scene_index(X)
pred = {k: np.full(len(X), np.nan) for k in ['fuzzy DE global (manuscript)', 'fuzzy DE within', 'fuzzy LS within', 'linear-3f within', 'BRISQUE-SVR within', 'BRISQUE-SVR global']}
for tr, te in h.scene_folds(X):
    Xtr = X.iloc[tr]
    s = c.fit_signs(Xtr, y[tr])
    arrs = c.oriented(X, s)
    mfs = [c.mfset(a[tr]) for a in arrs]
    F, fsum = c.firing(arrs, mfs)
    D = F / fsum[:, None]
    Str = h.scene_index(X.iloc[tr].reset_index(drop=True))
    trmap = np.asarray(tr)
    Str = trmap[Str]
    th = c.calibrate(F, fsum, y, tr)
    pred['fuzzy DE global (manuscript)'][te] = D[te] @ th
    Dtr, ytr = (D[tr], y[tr])
    Sloc = h.scene_index(X.iloc[tr].reset_index(drop=True))
    pred['fuzzy DE within'][te] = D[te] @ h.fit_de_within(Dtr, ytr, Sloc)
    pred['fuzzy LS within'][te] = D[te] @ h.fit_ls_within(Dtr, ytr, Sloc)
    X3 = np.column_stack(arrs)
    yc = h.demean(ytr, Sloc)
    Ste = h.scene_index(X.iloc[te].reset_index(drop=True))
    pred['linear-3f within'][te] = make_pipeline(StandardScaler(), LinearRegression()).fit(h.demean(X3[tr], Sloc), yc).predict(h.demean(X3[te], Ste))
    B = X[h.BC].to_numpy()
    pred['BRISQUE-SVR within'][te] = make_pipeline(StandardScaler(), SVR(C=10.0, gamma='scale')).fit(h.demean(B[tr], Sloc), yc).predict(h.demean(B[te], Ste))
    pred['BRISQUE-SVR global'][te] = make_pipeline(StandardScaler(), SVR(C=10.0, gamma='scale')).fit(B[tr], ytr).predict(B[te])
res = {k: {'within_scene_srcc': round(h.within_srcc(p, y, S_all), 3), 'pooled_srcc': round(c.srho(p, y), 3)} for k, p in pred.items()}
(c.OUT / f'human_recalibration_{ds}.json').write_text(json.dumps(res, indent=1))
print(ds)
print(pd.DataFrame(res).T.to_string())

X[['name','scene']].assign(**pred).to_csv(c.OUT/f'human_recalibration_{ds}_predictions.csv',index=False)
