"""Capacity ablation. See docs/STUDIES.md for inputs and protocol."""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr
import analysis.studies.common as c
rng = np.random.default_rng(20260929)
lol = c.lol_table()
tgt = -lol['lpips'].to_numpy()
g = lol['filename'].to_numpy()
signs, mfs_all, theta_all = c.lol_frozen_model()

def arr(X):
    return np.column_stack(c.oriented(X, signs))

def mf_degrees(A, mfs):
    return np.column_stack([c.trapmf(A[:, n], *mfs[n][l]) for n in range(3) for l in range(3)])

class Fuzzy:
    name = 'Fuzzy 27 rules, DE (manuscript)'

    def fit(self, A, y):
        self.mfs = [c.mfset(A[:, n]) for n in range(3)]
        f, s = c.firing([A[:, n] for n in range(3)], self.mfs)
        self.th = c.calibrate(f, s, y, np.arange(len(y)))
        return self

    def predict(self, A):
        f, s = c.firing([A[:, n] for n in range(3)], self.mfs)
        return f @ self.th / s

class TSKLS(Fuzzy):
    name = 'Fuzzy 27 rules, LS consequents'

    def fit(self, A, y):
        self.mfs = [c.mfset(A[:, n]) for n in range(3)]
        f, s = c.firing([A[:, n] for n in range(3)], self.mfs)
        self.th = np.linalg.lstsq(f / s[:, None], y, rcond=None)[0]
        return self

class AddMF:
    name = 'Additive memberships (9), linear'

    def fit(self, A, y):
        self.mfs = [c.mfset(A[:, n]) for n in range(3)]
        self.m = LinearRegression().fit(mf_degrees(A, self.mfs), y)
        return self

    def predict(self, A):
        return self.m.predict(mf_degrees(A, self.mfs))

class Lin:
    name = 'Linear (standardised)'

    def fit(self, A, y):
        self.m = make_pipeline(StandardScaler(), LinearRegression()).fit(A, y)
        return self

    def predict(self, A):
        return self.m.predict(A)

class LinRank:
    name = 'Linear on LOL-quantile (rank) features'

    def fit(self, A, y):
        self.ref = [np.sort(A[:, n]) for n in range(3)]
        self.m = LinearRegression().fit(self._t(A), y)
        return self

    def _t(self, A):
        return np.column_stack([np.searchsorted(self.ref[n], A[:, n], side='right') / len(self.ref[n]) for n in range(3)])

    def predict(self, A):
        return self.m.predict(self._t(A))

class LinClip:
    name = 'Linear on clipped features'

    def fit(self, A, y):
        self.lo, self.hi = (A.min(0), A.max(0))
        self.m = make_pipeline(StandardScaler(), LinearRegression()).fit(A, y)
        return self

    def predict(self, A):
        return self.m.predict(np.clip(A, self.lo, self.hi))

class MonoGAM:
    name = 'Monotone additive boosting (stumps)'

    def fit(self, A, y):
        self.m = HistGradientBoostingRegressor(max_depth=1, max_iter=200, learning_rate=0.05, monotonic_cst=[1, 1, 1], random_state=42).fit(A, y)
        return self

    def predict(self, A):
        return self.m.predict(A)

class Entropy:
    name = 'Entropy alone'

    def fit(self, A, y):
        return self

    def predict(self, A):
        return A[:, 0]
MODELS = [Fuzzy, TSKLS, AddMF, Lin, LinRank, LinClip, MonoGAM, Entropy]
Al = arr(lol)
res = {'LOL_cv': {}, 'zero_shot': {}, 'diff_vs_fuzzy_within': {}}
for M in MODELS:
    oof = np.full(len(lol), np.nan)
    fs = []
    for tr, te in c.group_kfold(g):
        oof[te] = M().fit(Al[tr], tgt[tr]).predict(Al[te])
        fs.append(c.srho(oof[te], tgt[te]))
    res['LOL_cv'][M.name] = {'foldmean_srcc': round(float(np.mean(fs)), 3), 'within_scene_srcc': round(c.within_group(oof, tgt, g)['mean_srcc'], 3)}
    print('LOL', M.name, res['LOL_cv'][M.name], flush=True)
fitted = {M.name: M().fit(Al, tgt) for M in MODELS}

def per_scene(p, y, s):
    d = pd.DataFrame({'p': p, 'y': y, 's': s})
    out = {}
    for k, v in d.groupby('s'):
        if len(v) > 1 and v['y'].nunique() > 1:
            r = spearmanr(v['p'], v['y'])[0]
            out[k] = 0.0 if np.isnan(r) else float(r)
    return pd.Series(out)
for ds in ['squarelol', 'lieq', 'rlie']:
    X = pd.read_csv(c.OUT / f'feat_{ds}_area256.csv')
    y = X['mos'].to_numpy(float)
    s = X['scene'].astype(str).to_numpy()
    Ae = arr(X)
    r = {}
    ps = {}
    for name, m in fitted.items():
        p = m.predict(Ae)
        w = c.within_group(p, y, s)
        r[name] = {'srcc': round(c.srho(p, y), 3), 'within_scene_srcc': round(w['mean_srcc'], 3), 'pair_acc': round(w['pair_acc'], 3), 'top1': round(w['top1'], 3)}
        ps[name] = per_scene(p, y, s)
    base = ps[Fuzzy.name]
    scenes = base.index.to_numpy()
    dd = {}
    B = rng.integers(0, len(scenes), size=(2000, len(scenes)))
    for name, v in ps.items():
        if name == Fuzzy.name:
            continue
        diff = (v.reindex(scenes) - base).to_numpy()
        boots = np.nanmean(diff[B], axis=1)
        dd[name] = [round(float(np.nanmean(diff)), 3), round(float(np.quantile(boots, 0.025)), 3), round(float(np.quantile(boots, 0.975)), 3)]
    res['zero_shot'][ds] = r
    res['diff_vs_fuzzy_within'][ds] = dd
    print(ds)
    print(pd.DataFrame(r).T.to_string())
    print('diff vs fuzzy (within, [95% scene bootstrap]):', json.dumps(dd, indent=0), flush=True)
(c.OUT / 'capacity_ablation.json').write_text(json.dumps(res, indent=1))
