"""Fit the model variants used by the validation studies."""
from analysis.studies.support import *
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def fit_model(X, y, kind):
    if kind == 'linear':
        obj = make_pipeline(StandardScaler(), LinearRegression()).fit(X[c.NR], y)
        return {'kind': kind, 'object': obj}
    A, sg, mfs = design(X, y)
    if kind == 'mono':
        th, diag = mono_fit(A, y)
    elif kind == 'de':
        f, s = c.firing(c.oriented(X, sg), mfs)
        th = c.calibrate(f, s, y, np.arange(len(y)))
        diag = {'seed': c.SEED, 'popsize': c.DE_POPSIZE, 'maxiter': c.DE_MAXITER}
    elif kind == 'ls':
        th = np.linalg.lstsq(A, y, rcond=None)[0]
        diag = {}
    else:
        raise ValueError(kind)
    return {'kind': kind, 'signs': sg, 'mfs': mfs, 'theta': th, 'solver': diag}

def predict(model, X):
    if model['kind'] == 'linear':
        return model['object'].predict(X[c.NR])
    return c.fuzzy_predict(X, model['signs'], model['mfs'], model['theta'])

def evaluate(X, humans, kind, tag, methods=None):
    y = -X.lpips.to_numpy()
    folds = list(cv_folds(X))
    oof = np.zeros(len(X))
    models = []
    t = time.perf_counter()
    for j, (tr, te) in enumerate(folds):
        model = fit_model(X.iloc[tr], y[tr], kind)
        oof[te] = predict(model, X.iloc[te])
        models.append(serializable_model(model))
        print(f'{tag}/{kind} CV {j + 1}/5', flush=True)
    result = {'cv': summary(oof, y, X.filename, folds), 'cv_models': models, 'algorithm_disjoint': {}}
    pd.DataFrame({'filename': X.filename, 'method': X.method, 'prediction': oof}).to_csv(OUT / f'study_{tag}_{kind}_oof.csv', index=False)
    for method in sorted(X.method.unique()) if methods is None else methods:
        tr = np.flatnonzero(X.method.to_numpy() != method)
        te = np.flatnonzero(X.method.to_numpy() == method)
        model = fit_model(X.iloc[tr], y[tr], kind)
        p = predict(model, X.iloc[te])
        result['algorithm_disjoint'][method] = {'srcc': rho(p, y[te]), 'n': len(te), 'model': serializable_model(model)}
        pd.DataFrame({'filename': X.filename.iloc[te], 'method': method, 'prediction': p, 'target': y[te]}).to_csv(OUT / f'study_{tag}_{kind}_leave_{method}.csv', index=False)
        print(f'{tag}/{kind} leave {method}', flush=True)
    model = fit_model(X, y, kind)
    result['full_model'] = serializable_model(model)
    result['human'] = {}
    for ds, xx in humans.items():
        p = predict(model, xx)
        result['human'][ds] = summary(p, xx.mos, xx.scene)
        pd.DataFrame({'name': xx.name, 'scene': xx.scene, 'mos': xx.mos, 'prediction': p}).to_csv(OUT / f'study_{tag}_{kind}_{ds}_predictions.csv', index=False)
    result['wall_seconds'] = time.perf_counter() - t
    return result

def serializable_model(model):
    if model['kind'] == 'linear':
        sc, lr = (model['object'].steps[0][1], model['object'].steps[1][1])
        return {'kind': 'linear', 'mean': sc.mean_, 'scale': sc.scale_, 'coefficient': lr.coef_, 'intercept': lr.intercept_}
    return model
