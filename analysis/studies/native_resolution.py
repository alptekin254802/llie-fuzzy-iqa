"""Compare the full enhancement and assessment pipeline at two resolutions."""
from analysis.studies.support_extra import *
from analysis.studies.images import basic, enhanced
from analysis.studies.models import fit_model

def main():
    meta = begin2('native_clean')
    import torch, pyiqa
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    lp = pyiqa.create_metric('lpips', device='cuda')
    hist = c.lol_table()
    names = sorted(hist.filename.unique())
    methods = sorted(hist.method.unique())
    file = OUT / 'native_lpips.csv'
    rows = readcsv(file).to_dict('records') if file.exists() else []
    done = {(r['filename'], r['method']) for r in rows}
    checks = []

    def ten(x):
        return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].to('cuda')
    with torch.inference_mode():
        for i, n in enumerate(names):
            low = c.load_rgb01(c.LOL_ROOT / 'low' / n, 'native')
            high = c.load_rgb01(c.LOL_ROOT / 'high' / n, 'native')
            assert low.shape == high.shape == (400, 600, 3)
            ht = ten(high)
            for m in methods:
                if (n, m) in done:
                    continue
                im = enhanced(low, m)
                torch.cuda.synchronize()
                t = time.perf_counter()
                v = float(lp(ten(im), ht).item())
                torch.cuda.synchronize()
                dt = time.perf_counter() - t
                if dt > 5 and len(rows) > 10:
                    raise RuntimeError('Native LPIPS exceeded5s/image')
                rows.append({'filename': n, 'method': m, 'lpips': v, **basic(im), 'lpips_ms': dt * 1000, 'height': im.shape[0], 'width': im.shape[1], 'low_sha256': sha(c.LOL_ROOT / 'low' / n), 'high_sha256': sha(c.LOL_ROOT / 'high' / n)})
            if i < 10:
                lo = c.load_rgb01(c.PREPROCESSED / 'low' / n, 'native')
                hi = ten(c.load_rgb01(c.PREPROCESSED / 'high' / n, 'native'))
                for m in methods:
                    v = float(lp(ten(enhanced(lo, m)), hi).item())
                    old = float(hist[(hist.filename == n) & (hist.method == m)].lpips.iloc[0])
                    checks.append({'filename': n, 'method': m, 'recomputed': v, 'historical': old, 'absolute_difference': abs(v - old)})
            if (i + 1) % 25 == 0 or i + 1 == len(names):
                pd.DataFrame(rows).to_csv(file, index=False)
                print('native LPIPS', i + 1, '/', len(names), flush=True)
    save(OUT / 'native_protocol_check.json', {'checks': checks, 'max_abs_difference': max((x['absolute_difference'] for x in checks)), 'passed_1e4': max((x['absolute_difference'] for x in checks)) < 0.0001})
    assert max((x['absolute_difference'] for x in checks)) < 0.0001
    native = readcsv(file)
    foldmap = readcsv(OUT / 'validation_inputs.csv').groupby('filename').fold.first()
    res = {'torch': torch.__version__, 'pyiqa': pyiqa.__version__, 'gpu': torch.cuda.get_device_name(0), 'protocol_check_max_abs_lpips': max((x['absolute_difference'] for x in checks)), 'pipelines': {}, 'within_native_minus_256': {}}
    for protocol, X in [('native', native), ('256', hist)]:
        X = X.sort_values(['filename', 'method']).reset_index(drop=True)
        y = -X.lpips.to_numpy()
        folds = list(c.group_kfold(X.filename))
        pipelineres = {}
        assert all((set(X.iloc[te].filename) == set(foldmap[foldmap == j].index) for j, (_, te) in enumerate(folds)))
        for kind in ['de', 'nr_mono']:
            p = np.zeros(len(X))
            models = []
            for j, (tr, te) in enumerate(folds):
                mod = fit_model(X.iloc[tr], y[tr], 'de') if kind == 'de' else fit_product(X.iloc[tr], y[tr])
                p[te] = predict(mod, X.iloc[te], kind == 'nr_mono')
                models.append(mod)
                print(protocol, kind, 'fold', j, flush=True)
            predfile = OUT / f'native_{protocol}_{kind}_oof.csv'
            z = X[['filename', 'method', 'lpips']].copy()
            z['prediction'] = p
            z['fold'] = X.filename.map(foldmap)
            z.to_csv(predfile, index=False)
            tr = np.flatnonzero(X.method.to_numpy() != 'retinex')
            te = np.flatnonzero(X.method.to_numpy() == 'retinex')
            rm = fit_model(X.iloc[tr], y[tr], 'de') if kind == 'de' else fit_product(X.iloc[tr], y[tr])
            rp = predict(rm, X.iloc[te], kind == 'nr_mono')
            rz = X.iloc[te][['filename', 'method', 'lpips']].copy()
            rz['prediction'] = rp
            rz.to_csv(OUT / f'native_{protocol}_{kind}_retinex.csv', index=False)
            full = fit_model(X, y, 'de') if kind == 'de' else fit_product(X, y)
            r = {'pooled_srcc': rho(p, y), 'fold_srcc': [rho(p[te], y[te]) for tr, te in folds], 'within': within2(p, y, X.filename), 'retinex_srcc': rho(rp, y[te]) if len(te) == len(rp) else rho(rp, -rz.lpips), 'cv_models': models, 'retinex_model': rm, 'full_model': full, 'human': {}}
            r['retinex_srcc'] = rho(rp, -rz.lpips.to_numpy())
            r['fold_mean_srcc'] = float(np.mean(r['fold_srcc']))
            for ds in ['squarelol', 'lieq', 'rlie']:
                xx = readcsv(OUT / (f'feat_{ds}_native.csv' if protocol == 'native' else f'feat_{ds}_area256.csv'), dtype={'scene': str})
                hp = predict(full, xx, kind == 'nr_mono')
                r['human'][ds] = {'pooled_srcc': rho(hp, xx.mos), 'within': within2(hp, xx.mos, xx.scene)}
                xx = xx[['name', 'scene', 'mos']].copy()
                xx['prediction'] = hp
                xx.to_csv(OUT / f'native_{protocol}_{kind}_{ds}.csv', index=False)
            pipelineres[kind] = r
            save(OUT / f'native_{protocol}_{kind}_checkpoint.json', r)
        res['pipelines'][protocol] = pipelineres
    for kind in ['de', 'nr_mono']:
        a = readcsv(OUT / f'native_native_{kind}_oof.csv')
        b = readcsv(OUT / f'native_256_{kind}_oof.csv')
        assert (a[['filename', 'method']] == b[['filename', 'method']]).all().all()
        res['within_native_minus_256'][kind] = diff_ci(scene_values(a.prediction, -a.lpips, a.filename) - scene_values(b.prediction, -b.lpips, b.filename))
    finish2(meta, res, [file, OUT / 'native_protocol_check.json'])
if __name__ == '__main__':
    main()
