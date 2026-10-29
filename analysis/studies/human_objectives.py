"""Compare pooled and within-scene human-label calibration objectives."""
import numpy as np, pandas as pd
from scipy.optimize import differential_evolution
import analysis.studies.common as c
BC = [f'b{i}' for i in range(36)]

def load(ds):
    X = pd.read_csv(c.OUT / f'feat_{ds}_area256.csv').merge(pd.read_csv(c.OUT / f'brisque_{ds}_area256.csv'), on='name')
    X['scene'] = X['scene'].astype(str)
    return X.sort_values(['scene', 'name']).reset_index(drop=True)

def scene_index(X):
    g = X.groupby('scene').indices
    return np.stack([g[s] for s in sorted(g)])

def ranks(M):
    return np.argsort(np.argsort(M, axis=1), axis=1).astype(float)

def within_srcc(p, y, S):
    from scipy.stats import rankdata
    rp, ry = rankdata(p[S],axis=1,method='average'), rankdata(y[S],axis=1,method='average')
    rp -= rp.mean(1, keepdims=True)
    ry -= ry.mean(1, keepdims=True)
    num = (rp * ry).sum(1)
    den = np.sqrt((rp ** 2).sum(1) * (ry ** 2).sum(1)) + 1e-12
    return float(np.mean(num / den))

def demean(A, S):
    A = np.asarray(A, float).copy()
    for row in S:
        A[row] -= A[row].mean(0)
    return A

def fuzzy_design(X, signs, mfs):
    F, fsum = c.firing(c.oriented(X, signs), mfs)
    return F / fsum[:, None]

def fit_ls_within(D, y, S, ridge=0.001):
    Dc, yc = (demean(D, S), demean(y, S))
    k = D.shape[1]
    return np.linalg.solve(Dc.T @ Dc + ridge * np.eye(k), Dc.T @ yc)

def fit_de_within(D, y, S, seed=c.SEED):
    ry = ranks(y[S])
    ry -= ry.mean(1, keepdims=True)
    ny = np.sqrt((ry ** 2).sum(1))

    def obj(t):
        rp = ranks((D @ t)[S])
        rp -= rp.mean(1, keepdims=True)
        return -np.mean((rp * ry).sum(1) / (np.sqrt((rp ** 2).sum(1)) * ny + 1e-12))
    return differential_evolution(obj, bounds=[(0.0, 100.0)] * D.shape[1], maxiter=c.DE_MAXITER, popsize=c.DE_POPSIZE, seed=seed, tol=0.001, polish=False, workers=1, updating='deferred').x

def scene_folds(X, k=5, seed=c.SEED):
    return list(c.group_kfold(X['scene'].to_numpy(), k, seed))
