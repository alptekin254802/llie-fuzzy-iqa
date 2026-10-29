"""Additional product-model fitting and exact-rank helpers."""
from analysis.studies.support import *
import subprocess
sys.dont_write_bytecode = True

def readcsv(p, **kw):
    return pd.read_csv(p, float_precision='round_trip', **kw)
begin2 = begin
finish2 = finish

def prod_design(df, signs, mfs):
    arr = c.oriented(df, signs)
    ms = [np.stack([c.trapmf(arr[n], *mfs[n][j]) for j in range(3)], 1) for n in range(3)]
    f = np.stack([ms[0][:, i] * ms[1][:, j] * ms[2][:, k] for i, j, k in c.COMBOS], 1)
    return f / (f.sum(1, keepdims=True) + 1e-12)

def fit_product(df, y, signs=None):
    signs = c.fit_signs(df, y) if signs is None else signs
    mfs = [c.mfset(v) for v in c.oriented(df, signs)]
    F = prod_design(df, signs, mfs)
    cons = [{'type': 'ineq', 'fun': lambda th, a=a, b=b: th[b] - th[a]} for a, b in PAIR]
    r = minimize(lambda th: np.sum((F @ th - y) ** 2), np.full(27, np.mean(y)), jac=lambda th: 2 * F.T @ (F @ th - y), constraints=cons, method='SLSQP', options={'maxiter': 1000, 'ftol': 1e-14})
    assert np.min(C @ r.x) > -1e-07, r.message
    return {'signs': signs, 'mfs': mfs, 'theta': r.x, 'optimizer': {'success': bool(r.success), 'message': str(r.message), 'min_slack': float(np.min(C @ r.x)), 'sse': float(np.sum((F @ r.x - y) ** 2))}}

def predict(model, df, product=False):
    if product:
        return prod_design(df, model['signs'], model['mfs']) @ np.array(model['theta'])
    return c.fuzzy_predict(df, model['signs'], model['mfs'], np.array(model['theta']))

def within2(p,y,g):
    return c.within_group(p,y,g)

def stats_ms(x):
    return {'n': len(x), 'median_ms': float(np.median(x)), 'p95_ms': float(np.quantile(x, 0.95)), 'mean_ms': float(np.mean(x))}

def diff_ci(v):
    v = np.asarray(v)
    rng = np.random.default_rng(SEED)
    return interval([v[rng.integers(len(v), size=len(v))].mean() for _ in range(B)], float(v.mean()))
