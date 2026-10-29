"""Compare differential-evolution population, iteration and seed settings."""
from analysis.studies.support import *
from scipy.optimize import differential_evolution
from concurrent.futures import ProcessPoolExecutor, as_completed

def vector_objective(f, s, y):
    yr = rankdata(y)
    yr -= yr.mean()
    yn = np.linalg.norm(yr)

    def obj(theta):
        th = np.asarray(theta)
        one = th.ndim == 1
        if one:
            th = th[:, None]
        pred = f @ th / s[:, None]
        r = rankdata(pred, axis=0)
        r -= r.mean(axis=0)
        den = np.sqrt(np.sum(r * r, axis=0)) * yn
        val = -np.divide(yr @ r, den, out=np.zeros(th.shape[1]), where=den > 0)
        return float(val[0]) if one else val
    return obj

def datasets():
    df = pd.read_csv(OUT / 'validation_inputs.csv')
    y = -df.lpips.to_numpy()
    ext = external('squarelol')
    folds = list(cv_folds(df))
    sets = []
    for tr, te in folds:
        A, Ae, sg, mfs = design(df.iloc[tr], y[tr], df.iloc[te])
        f, s = c.firing(c.oriented(df.iloc[tr], sg), mfs)
        fe, se = c.firing(c.oriented(df.iloc[te], sg), mfs)
        sets.append((f, s, y[tr], fe, se, y[te], te))
    A, sg, mfs = design(df, y)
    f, s = c.firing(c.oriented(df, sg), mfs)
    fe, se = c.firing(c.oriented(ext, sg), mfs)
    sets.append((f, s, y, fe, se, ext.mos.to_numpy(), None))
    return (df, ext, sets)

def worker(config, signature):
    pop, it, seed = config
    path = OUT / f'de_sensitivity_p{pop}_g{it}_s{seed}.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old.get('signature') == signature:
            return old
        raise RuntimeError('Checkpoint source/input mismatch')
    df, ext, sets = datasets()
    t = time.perf_counter()
    vals = []
    details = []
    oof = np.zeros(len(df))
    external_score = None
    for k, (f, s, y, fe, se, ye, te) in enumerate(sets):
        st = time.perf_counter()
        obj = vector_objective(f, s, y)
        r = differential_evolution(obj, [(0, 100)] * 27, popsize=pop, maxiter=it, seed=seed, tol=0.001, polish=False, workers=1, updating='deferred', vectorized=True)
        p = fe @ r.x / se
        details.append({'split': k, 'actual_generations': r.nit, 'population_candidates': len(r.population), 'vectorized_calls': r.nfev, 'train_srcc': -r.fun, 'wall_seconds': time.perf_counter() - st, 'theta': r.x})
        if te is not None:
            oof[te] = p
            vals.append(rho(p, ye))
        else:
            external_score = summary(p, ye, ext.scene)
        print(f'DE {config} fit {k + 1}/6', flush=True)
    result = {'popsize_factor': pop, 'maxiter': it, 'seed': seed, 'fold_srcc': vals, 'fold_mean': np.mean(vals), 'fold_sd': np.std(vals), 'pooled_oof': rho(oof, -df.lpips), 'squarelol': external_score, 'wall_seconds': time.perf_counter() - t, 'fits': details, 'signature': signature}
    save(path, result)
    return clean(result)

def main():
    m = begin('de_sensitivity')
    df, ext, sets = datasets()
    f, s, y, *_ = sets[0]
    obj = vector_objective(f, s, y)
    candidates = np.random.default_rng(SEED).uniform(0, 100, (27, 32))
    a = obj(candidates)
    b = np.array([-c.srho(f @ v / s, y) for v in candidates.T])
    err = float(np.max(np.abs(a - b)))
    kw = dict(bounds=[(0, 100)] * 27, popsize=8, maxiter=2, seed=0, tol=0.001, polish=False, workers=1, updating='deferred')
    rv = differential_evolution(obj, vectorized=True, **kw)
    rs = differential_evolution(lambda t: -c.srho(f @ t / s, y), **kw)
    check = {'candidate_objective_max_error': err, 'short_de_objective_difference': abs(rv.fun - rs.fun), 'short_de_theta_max_difference': float(np.max(abs(rv.x - rs.x)))}
    save(OUT / 'de_sensitivity_vectorization_check.json', check)
    if err > 1e-12 or abs(rv.fun - rs.fun) > 1e-12:
        finish(m, {'status': 'VALIDATION_FAILED', 'check': check})
        return
    lsvals = []
    lsoof = np.zeros(len(df))
    lsexternal = None
    for f, s, y, fe, se, ye, te in sets:
        th = np.linalg.lstsq(f / s[:, None], y, rcond=None)[0]
        p = fe @ th / se
        if te is not None:
            lsvals.append(rho(p, ye))
            lsoof[te] = p
        else:
            lsexternal = summary(p, ye, ext.scene)
    result = {'status': 'running', 'validation': check, 'least_squares': {'fold_mean': np.mean(lsvals), 'fold_sd': np.std(lsvals), 'fold_srcc': lsvals, 'pooled_oof': rho(lsoof, -df.lpips), 'squarelol': lsexternal}, 'configurations': []}
    signature = sha(Path(__file__)) + sha(Path(__file__).parent / 'support.py') + sha(OUT / 'validation_inputs.csv') + sha(OUT / 'feat_squarelol_area256.csv')
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(worker, config, signature): config for config in itertools.product((8, 15, 30), (40, 100, 300), (0, 1, 2))}
        for future in as_completed(futures):
            result['configurations'].append(future.result())
            save(OUT / 'de_sensitivity_checkpoint.json', result)
    rows = sorted(result['configurations'], key=lambda r: (r['popsize_factor'], r['maxiter'], r['seed']))
    result['configurations'] = rows
    means = np.array([r['fold_mean'] for r in rows])
    square = np.array([r['squarelol']['within']['mean_srcc'] for r in rows])
    refs = {r['seed']: r['fold_mean'] for r in rows if r['popsize_factor'] == 8 and r['maxiter'] == 40}
    for r in rows:
        r['fold_delta_vs_8_40_same_seed'] = r['fold_mean'] - refs[r['seed']]
    result['summary'] = {'lol_min': means.min(), 'lol_max': means.max(), 'lol_range': np.ptp(means), 'operational_stability_range_le_002': bool(np.ptp(means) <= 0.02), 'squarelol_min': square.min(), 'squarelol_max': square.max(), 'squarelol_range': np.ptp(square), 'total_wall_seconds': time.perf_counter() - started}
    result['status'] = 'complete'
    finish(m, result)
if __name__ == '__main__':
    main()
