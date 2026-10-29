"""Compare global image features with regional alternatives."""
from analysis.studies.support import *
from analysis.studies.images import *
from analysis.studies.models import evaluate

def summarize_decisions(result, folder):
    """Rebuild paired decisions from saved predictions, without fitting."""
    folder = Path(folder)
    result['decisions'] = {}
    for v in ('V1', 'V2'):
        result['decisions'][v] = {}
        for kind in ('de',):
            base = result['variants']['V0'][kind]
            cand = result['variants'][v][kind]
            a = pd.read_csv(folder / f'study_regional_{v}_{kind}_squarelol_predictions.csv', dtype={'scene': str}, float_precision='round_trip')
            b = pd.read_csv(folder / f'study_regional_V0_{kind}_squarelol_predictions.csv', dtype={'scene': str}, float_precision='round_trip')
            ci = paired_within(a.prediction, b.prediction, a.mos, a.scene)
            human_contrasts={}
            for ds in ('squarelol','lieq','rlie'):
                aa=pd.read_csv(folder/f'study_regional_{v}_{kind}_{ds}_predictions.csv',dtype={'scene':str},float_precision='round_trip')
                bb=pd.read_csv(folder/f'study_regional_V0_{kind}_{ds}_predictions.csv',dtype={'scene':str},float_precision='round_trip')
                joined=aa.merge(bb[['name','prediction']],on='name',suffixes=('_variant','_base'),validate='one_to_one')
                human_contrasts[ds]=paired_within(joined.prediction_variant,joined.prediction_base,joined.mos,joined.scene)
            ret = cand['algorithm_disjoint']['retinex']['srcc'] - base['algorithm_disjoint']['retinex']['srcc']
            drop = base['cv']['fold_mean'] - cand['cv']['fold_mean']
            result['decisions'][v][kind] = {'retinex_delta': ret, 'squarelol_delta': ci, 'human_differences':human_contrasts, 'lol_fold_mean_drop': drop, 'lol_single_vector_drop': base['cv']['pooled_srcc'] - cand['cv']['pooled_srcc'], 'improves': bool((ret >= 0.1 or ci['ci95'][0] > 0) and drop < 0.02)}
    return result['decisions']


def main():
    m = begin('regional')
    nf = pd.read_csv(c.RES / 'nr_features.csv')
    generated = []
    manifest = {}
    images = []
    for i, name in enumerate(sorted(nf.filename.unique())):
        p = c.PREPROCESSED / 'low' / name
        manifest[str(p)] = sha(p)
        low = c.load_rgb01(p, 'native')
        assert low.shape[:2] == (256, 256)
        for method in sorted(nf.method.unique()):
            im = enhanced(low, method)
            generated.append({'filename': name, 'method': method, **c.repo_nr_features(im)})
        if (i + 1) % 100 == 0:
            print(f'gate {i + 1}/485', flush=True)
    gen = pd.DataFrame(generated)
    joined = nf.merge(gen, on=['filename', 'method'], suffixes=('_old', '_new'), validate='one_to_one')
    error = {k: float(np.max(np.abs(joined[k + '_old'] - joined[k + '_new']))) for k in c.ALL6}
    gate = {'max_abs_error': error, 'passed': bool(all((error[k] <= 1e-06 for k in c.NR))), 'rows': len(joined)}
    save(OUT / 'regional_gate.json', gate)
    if not gate['passed']:
        finish(m, {'status': 'STOPPED_GLOBAL_FEATURE_GATE', 'gate': gate})
        return
    print('GLOBAL GATE PASSED', flush=True)
    allsets = {}
    rows = []
    for i, name in enumerate(sorted(nf.filename.unique())):
        low = c.load_rgb01(c.PREPROCESSED / 'low' / name, 'native')
        for method in sorted(nf.method.unique()):
            im = enhanced(low, method)
            rows.append({'filename': name, 'method': method, **regional(im)})
            if len(images) < 60:
                images.append(im)
        if (i + 1) % 100 == 0:
            print(f'LOL regional {i + 1}/485', flush=True)
    lol = pd.read_csv(OUT / 'validation_inputs.csv')[['filename', 'method', 'lpips', 'fold']].merge(pd.DataFrame(rows), on=['filename', 'method'], validate='one_to_one')
    assert all((set(lol.filename.iloc[a]) == set(lol.filename.iloc[b]) for (_, a), (_, b) in zip(cv_folds(lol), c.group_kfold(lol.filename))))
    allsets['lol'] = lol
    for ds in ('squarelol', 'lieq', 'rlie'):
        rows = []
        for i, r in human_meta(ds).iterrows():
            manifest[r.path] = sha(r.path)
            im = c.load_rgb01(r.path, 'area256')
            rows.append({**r.drop('path').to_dict(), **regional(im)})
            if (i + 1) % 500 == 0:
                print(f'{ds} regional {i + 1}', flush=True)
        allsets[ds] = pd.DataFrame(rows)
    files = []
    for ds, x in allsets.items():
        p = OUT / f'regional_features_{ds}.csv'
        x.to_csv(p, index=False)
        files.append(p)
    save(OUT / 'regional_image_manifest.json', manifest)
    result = {'status': 'complete', 'gate': gate, 'variants': {}, 'fallbacks': {ds: {n: int((x[n + '_pixels'] == 0).sum()) for n in ('dark', 'middle', 'bright', 'dark_middle')} for ds, x in allsets.items()}}
    for v in ('V0', 'V1', 'V2'):
        result['variants'][v] = {}
        for kind in ('de',):
            result['variants'][v][kind] = evaluate(choose(lol, v), {ds: choose(x, v) for ds, x in allsets.items() if ds != 'lol'}, kind, f'regional_{v}')
            save(OUT / 'regional_checkpoint.json', result)
    result['decisions'] = summarize_decisions(result, OUT)
    result['runtime'] = {}
    for v in ('V0', 'V1', 'V2'):
        for im in images[:5]:
            variant_features(im, v)
        times = []
        for _ in range(3):
            for im in images:
                t = time.perf_counter()
                feat = variant_features(im, v)
                times.append((time.perf_counter() - t) * 1000)
                full = regional(im)
                assert max((abs(feat[k] - full[f'{v}_{k}']) for k in c.NR)) < 1e-12
        result['runtime'][v] = {'features_mean_ms': np.mean(times), 'features_median_ms': np.median(times), 'features_p95_ms': np.quantile(times, 0.95), 'n': len(times), 'inference': {}}
        one = pd.DataFrame([variant_features(images[0], v)])
        for kind in ('de',):
            mod = result['variants'][v][kind]['full_model']
            it = []
            for _ in range(200):
                t = time.perf_counter()
                if kind == 'linear':
                    _p = (one[c.NR].to_numpy() - mod['mean']) / mod['scale'] @ mod['coefficient'] + mod['intercept']
                else:
                    _p = c.fuzzy_predict(one, mod['signs'], mod['mfs'], mod['theta'])
                it.append((time.perf_counter() - t) * 1000)
            result['runtime'][v]['inference'][kind] = {'mean_ms': np.mean(it), 'median_ms': np.median(it), 'p95_ms': np.quantile(it, 0.95)}
    finish(m, result, files)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary-from', type=Path)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.summary_from is None:
        main()
    else:
        if args.predictions is None or args.output is None:
            parser.error('--summary-from requires --predictions and --output')
        recorded = json.loads(args.summary_from.read_text(encoding='utf-8'))
        save(args.output, summarize_decisions(recorded, args.predictions))
