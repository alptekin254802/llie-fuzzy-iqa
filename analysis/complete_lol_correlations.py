"""Complete additional-metric LOL correlations from archived predictions only.

Run: python -m analysis.complete_lol_correlations
No inference, fitting, fold creation or score-sign selection occurs here.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'extra': 'results/studies/extra_metrics_lol_scores.csv',
    'richiqa': 'results/studies/richiqa_scores.csv',
    'anchors': 'results/perceptual_metrics.csv',
    'folds': 'results/nr_fuzzy_oof.csv',
}

def read(name):
    return pd.read_csv(ROOT / SOURCES[name], float_precision='round_trip')

def correlations(frame, prediction, target):
    values = [float(spearmanr(prediction[g.index], target[g.index]).statistic)
              for _, g in frame.groupby('fold', sort=True)]
    return {'fold_srcc': values, 'fold_mean_srcc': float(np.mean(values)),
            'fold_std_srcc': float(np.std(values, ddof=0))}

def main():
    base = read('anchors').merge(read('folds'), on=['filename', 'method'], validate='one_to_one')
    base['name'] = base.filename + '__' + base.method
    extra = read('extra')
    assert extra.dataset.eq('lol').all()
    assert (extra.name == extra.filename + '__' + extra.method).all()
    assert extra.scene.eq(extra.filename).all()
    d = base.merge(extra[['name', 'target', 'bpri', 'bmpri', 'stairiqa']], on='name', validate='one_to_one')
    rich = read('richiqa').query("dataset == 'lol'")
    d = d.merge(rich[['name', 'scene', 'score']], on='name', validate='one_to_one').reset_index(drop=True)
    assert len(base) == len(extra) == len(rich) == len(d) == 2425
    assert d.filename.nunique() == 485 and d.groupby('filename').size().eq(5).all()
    assert d.scene.eq(d.filename).all() and d.groupby('filename').fold.nunique().eq(1).all()
    assert d.groupby('fold').size().eq(485).all() and sorted(d.fold.unique()) == list(range(5))
    assert np.allclose(d.target, -d.lpips, atol=1e-14, rtol=0)
    assert np.isfinite(d[['bpri', 'bmpri', 'stairiqa', 'score', 'niqe', 'lpips', 'prediction']]).all().all()
    result = {'protocol': 'Existing five scene-held-out folds; 97 scenes and 485 images per fold; arithmetic mean and population SD (ddof=0). Spearman average ranks; no inference or refitting.',
              'n_images': len(d), 'n_scenes': d.filename.nunique(),
              'fold_sizes': {str(k): len(g) for k,g in d.groupby('fold')},
              'sources_sha256': {v: hashlib.sha256((ROOT/v).read_bytes()).hexdigest() for v in SOURCES.values()},
              'metrics': {}}
    old = json.loads((ROOT/'results/studies/extra_metrics_lol.json').read_text())
    rich_old = json.loads((ROOT/'results/studies/richiqa_results.json').read_text())['datasets']['lol']['metrics']['richiqa']
    for key, col, sign in [('bpri','bpri',-1),('bmpri','bmpri',-1),('stairiqa','stairiqa',1),('richiqa','score',1)]:
        p = sign*d[col].to_numpy()
        values = {a: correlations(d,p,-d[a].to_numpy()) for a in ['lpips','niqe']}
        expected = rich_old if key=='richiqa' else old['metrics'][key]
        assert np.allclose(values['lpips']['fold_srcc'],expected['fold_srcc'],rtol=0,atol=1e-12),key
        result['metrics'][key] = {'raw_score_multiplier':sign, **values}
    control = correlations(d,d.prediction.to_numpy(),-d.niqe.to_numpy())
    expected = json.loads((ROOT/'results/figure_numbers.json').read_text())['nr_fuzzy_free']
    assert round(control['fold_mean_srcc'], 3) == expected['niqe_mean']
    assert round(control['fold_std_srcc'], 3) == expected['niqe_std']
    result['nr_free_niqe_control'] = control
    dest = ROOT/'results/studies/extra_metrics_lol_anchors.json'
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v['niqe'] for k,v in result['metrics'].items()},indent=2))

if __name__ == '__main__':
    main()
