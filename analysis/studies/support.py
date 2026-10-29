"""Shared experiment logging, constrained fitting and statistical helpers."""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys, json, hashlib, time, platform, itertools
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np, pandas as pd, cv2, scipy
from scipy.stats import rankdata
from scipy.optimize import minimize, LinearConstraint
from threadpoolctl import threadpool_limits
import analysis.studies.common as c
threadpool_limits(1)
cv2.setNumThreads(1)
ROOT = c.OUT
OUT = c.OUT
LOG = ROOT / 'run.log'
SEED = 20261018
B = 2000

def now():
    return datetime.now(timezone(timedelta(hours=3))).isoformat()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x

def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def begin(task):
    meta = {'task': task, 'started': now(), 'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__, 'opencv': cv2.__version__, 'threads': 1}
    save(OUT / (task + '_run.json'), meta)
    return meta

def finish(meta, result, extra=()):
    result['environment'] = meta | {'finished': now()}
    save(OUT / (meta['task'] + '.json'), result)
    print('Completed', meta['task'], flush=True)

def rho(x, y):
    a = rankdata(x)
    b = rankdata(y)
    a -= a.mean()
    b -= b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den else 0.0

def cv_folds(df):
    for f in sorted(df.fold.unique()):
        yield (np.flatnonzero(df.fold.to_numpy() != f), np.flatnonzero(df.fold.to_numpy() == f))

def within(pred, target, groups):
    return c.within_group(pred, target, groups)

def scene_values(pred, target, groups):
    d = pd.DataFrame({'p': pred, 't': target, 'g': groups})
    return np.array([rho(x.p, x.t) for _, x in d.groupby('g', sort=True) if len(x) > 1 and x.t.nunique() > 1])

def summary(pred, target, groups, folds=None):
    r = {'pooled_srcc': rho(pred, target), 'within': within(pred, target, groups)}
    if folds is not None:
        a = [rho(np.asarray(pred)[te], np.asarray(target)[te]) for _, te in folds]
        r.update(fold_srcc=a, fold_mean=float(np.mean(a)), fold_sd=float(np.std(a)))
    return r

def interval(draws, estimate):
    a = np.asarray(draws)
    lo, hi = np.quantile(a, [0.025, 0.975])
    p = (1 + np.count_nonzero(np.abs(a - estimate) >= abs(estimate))) / (len(a) + 1)
    return {'difference': estimate, 'ci95': [lo, hi], 'p_two_sided_null_centered': p, 'excludes_zero': bool(lo > 0 or hi < 0), 'replicates': len(a), 'seed': SEED}

def paired_within(a, b, target, groups):
    v = scene_values(a, target, groups) - scene_values(b, target, groups)
    rng = np.random.default_rng(SEED)
    draws = np.array([v[rng.integers(len(v), size=len(v))].mean() for _ in range(B)])
    return interval(draws, float(v.mean()))

def paired_bootstrap(a, b, target, groups, fold=None):
    a, b, y = (np.asarray(a), np.asarray(b), np.asarray(target))
    groups = np.asarray(groups)
    u = np.unique(groups)
    ids = [np.flatnonzero(groups == g) for g in u]
    rng = np.random.default_rng(SEED)
    pdiff = []
    fdiff = []
    for _ in range(B):
        ii = np.concatenate([ids[j] for j in rng.integers(len(ids), size=len(ids))])
        pdiff.append(rho(a[ii], y[ii]) - rho(b[ii], y[ii]))
        if fold is not None:
            ff = np.asarray(fold)[ii]
            fdiff.append(np.mean([rho(a[ii[ff == f]], y[ii[ff == f]]) - rho(b[ii[ff == f]], y[ii[ff == f]]) for f in np.unique(fold)]))
    out = {'pooled': interval(pdiff, rho(a, y) - rho(b, y))}
    if fold is not None:
        e = np.mean([rho(a[np.asarray(fold) == f], y[np.asarray(fold) == f]) - rho(b[np.asarray(fold) == f], y[np.asarray(fold) == f]) for f in np.unique(fold)])
        out['fold_mean'] = interval(fdiff, float(e))
    return out
COMBOS = list(itertools.product(range(3), repeat=3))
PAIR = []
for i, t in enumerate(COMBOS):
    for axis in range(3):
        v = list(t)
        v[axis] += 1
        if v[axis] < 3:
            PAIR.append((i, COMBOS.index(tuple(v))))
C = np.zeros((len(PAIR), 27))
for i, (a, b) in enumerate(PAIR):
    C[i, b] = 1
    C[i, a] = -1

def design(train, target, test=None):
    signs = c.fit_signs(train, target)
    arr = c.oriented(train, signs)
    mfs = [c.mfset(x) for x in arr]
    f, s = c.firing(arr, mfs)
    A = f / s[:, None]
    if test is None:
        return (A, signs, mfs)
    f2, s2 = c.firing(c.oriented(test, signs), mfs)
    return (A, f2 / s2[:, None], signs, mfs)

def mono_fit(A, y):
    H = A.T @ A / len(A)
    z = A.T @ y / len(A)
    r = minimize(lambda t: float(t @ H @ t - 2 * t @ z), np.full(27, np.mean(y)), jac=lambda t: 2 * (H @ t - z), method='SLSQP', constraints=[LinearConstraint(C, 0, np.inf)], options={'ftol': 1e-12, 'maxiter': 3000})
    slack = C @ r.x
    if not r.success or slack.min() < -1e-07:
        raise RuntimeError(f'Monotone solver failed: {r.message}, slack={slack.min()}')
    return (r.x, {'success': r.success, 'iterations': r.nit, 'min_constraint_slack': float(slack.min()), 'violations_tolerance_1e7': int(np.sum(slack < -1e-07)), 'mse': float(np.mean((A @ r.x - y) ** 2)), 'rank': int(np.linalg.matrix_rank(A))})

def external(ds):
    return pd.read_csv(OUT / f'feat_{ds}_area256.csv', dtype={'scene': str})
