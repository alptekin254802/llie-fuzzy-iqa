"""Compare fixed uncalibrated rule values with LPIPS-fitted rule values.

Run from the repository root:
python -B -m analysis.studies.uncalibrated_rules --output uncalibrated_run

The default run reproduces the two uncalibrated rows and the training-fold
preprocessing check without fitting any consequents. Add --with-calibration
to refit FR-calibrated and NR-free and reproduce the paired scene intervals.
No images or external datasets are needed; inputs are the released LOL CSVs.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from analysis import reference_free_fuzzy as ref

ROOT = Path(__file__).resolve().parents[2]
FEATURES = {'FR-uncalibrated': ['psnr', 'ssim', 'entropy'],
            'NR-uncalibrated': ['entropy', 'contrast', 'sharpness']}


def load_data():
    data = (pd.read_csv(ref.NR_CSV)
            .merge(pd.read_csv(ref.PC_CSV)[['filename', 'method', 'lpips', 'niqe', 'brisque']], on=['filename', 'method'])
            .merge(pd.read_csv(ref.FZ_CSV)[['filename', 'method', 'psnr', 'ssim', 'fuzzy_score']], on=['filename', 'method'])
            .dropna().reset_index(drop=True))
    assert len(data) == 2425 and data.filename.nunique() == 485
    assert data.groupby('filename').size().eq(5).all()
    return data


def preprocessing(data, features, target):
    arrays = ref.orient(data, features, target)
    signs = [spearmanr(data[name], target).statistic < 0 for name in features]
    return arrays, signs, [ref.mfset(x) for x in arrays]


def fixed_firing(data, features, signs, memberships):
    member = [ref._membership_from_mfs(data[name].to_numpy(float) * (-1 if signs[j] else 1), memberships[j])
              for j, name in enumerate(features)]
    fire = np.stack([np.minimum(np.minimum(member[0][:, i], member[1][:, j]), member[2][:, k])
                     for i, j, k in ref.COMBOS], axis=1)
    return fire, fire.sum(axis=1) + 1e-9


def summarize(data, prediction, folds):
    output = {}
    for anchor in ['lpips', 'niqe']:
        target = -data[anchor].to_numpy()
        values = [ref.rho(prediction[test], target[test]) for _, test in folds]
        output[anchor] = dict(mean=float(np.mean(values)), SD=float(np.std(values, ddof=0)), folds=values)
    return output


def run(with_calibration=False):
    data = load_data()
    target = -data.lpips.to_numpy()
    groups = data.filename.to_numpy()
    folds = list(ref.group_kfold(groups, 5, 42))
    fold_ids = np.full(len(data), -1, int)
    for number, (train, test) in enumerate(folds):
        assert set(groups[train]).isdisjoint(set(groups[test]))
        fold_ids[test] = number
    theta = np.array([100 * (i + j + k) / 6 for i, j, k in ref.COMBOS])
    scores = {'Hand-tuned (Mamdani)': data.fuzzy_score.to_numpy()}
    summaries = {'Hand-tuned (Mamdani)': summarize(data, scores['Hand-tuned (Mamdani)'], folds)}
    optional = []
    for model, features in FEATURES.items():
        arrays, _, _ = preprocessing(data, features, target)
        fire, total = ref.build_firing(arrays)
        scores[model] = fire @ theta / total
        summaries[model] = summarize(data, scores[model], folds)
        nested = np.full(len(data), np.nan)
        for train, test in folds:
            _, signs, memberships = preprocessing(data.iloc[train], features, target[train])
            test_fire, test_total = fixed_firing(data.iloc[test], features, signs, memberships)
            nested[test] = test_fire @ theta / test_total
        assert np.isfinite(nested).all()
        optional.append(dict(model=model, **summarize(data, nested, folds)))
        if with_calibration:
            fitted = 'FR-calibrated' if model == 'FR-uncalibrated' else 'NR-free'
            scores[fitted] = np.full(len(data), np.nan)
            for number, (train, test) in enumerate(folds):
                fitted_theta = ref.calibrate(fire, total, target, train, 0.0)
                scores[fitted][test] = fire[test] @ fitted_theta / total[test]
                print(f'{fitted}, fold {number + 1}/5 completed', flush=True)
            summaries[fitted] = summarize(data, scores[fitted], folds)
    order = ['Hand-tuned (Mamdani)', 'FR-uncalibrated', 'FR-calibrated', 'NR-uncalibrated', 'NR-free']
    result = dict(data=dict(images=len(data), scenes=len(np.unique(groups)), methods=5, folds=5),
                  main_preprocessing='pooled, matching main Table 2',
                  rows=[dict(model=name, **summaries[name]) for name in order if name in summaries],
                  optional_training_only=dict(status='COMPUTED', rows=optional, new_consequent_fits=0),
                  consequent_rule='theta_r=100*(l1+l2+l3)/6 in the public COMBOS order',
                  orientation_caveat='Orientation uses negative LPIPS; these models have unfitted rule values but are not fully target-free.')
    if with_calibration:
        from analysis.studies import support
        contrasts = []
        for model, comparator in [('FR-calibrated', 'FR-uncalibrated'), ('NR-free', 'NR-uncalibrated'), ('FR-uncalibrated', 'Hand-tuned (Mamdani)')]:
            paired = support.paired_bootstrap(scores[model], scores[comparator], target, groups, fold=fold_ids)['fold_mean']
            contrasts.append(dict(model=model, comparator=comparator,
                                  contrast=f'{model} minus {comparator}', target='negative LPIPS',
                                  estimand='fold_mean_SRCC', difference=paired['difference'], ci95=paired['ci95'],
                                  B=support.B, seed=support.SEED, interval='95% percentile, unadjusted',
                                  Holm_family_member=False, models_refitted=False))
            print(f'Paired interval: {model} minus {comparator}', flush=True)
        result['bootstrap'] = dict(B=support.B, seed=support.SEED,
                                   unit='whole scene, all5 outputs retained, original fold identity preserved',
                                   estimand='fold_mean_SRCC', target='negative LPIPS',
                                   interval='95% percentile, unadjusted, outside the paper Holm family', contrasts=contrasts)
        means = {name: values['lpips']['mean'] for name, values in summaries.items()}
        result['arithmetic_differences'] = {
            'FR_calibrated_minus_Hand_tuned': means['FR-calibrated'] - means['Hand-tuned (Mamdani)'],
            'FR_uncalibrated_minus_Hand_tuned': means['FR-uncalibrated'] - means['Hand-tuned (Mamdani)'],
            'FR_calibrated_minus_FR_uncalibrated': means['FR-calibrated'] - means['FR-uncalibrated'],
            'NR_free_minus_NR_uncalibrated': means['NR-free'] - means['NR-uncalibrated']}
    result['main_preprocessing'] = 'pooled, matching paper Table2'
    result['orientation_caveat'] = ('Feature orientation signs are derived from LPIPS, so the uncalibrated models '
                                    'are uncalibrated in their consequents but not fully target-free.')
    result['interpretation_limit'] = ('Hand-tuned to uncalibrated combines changes in orientations, memberships, '
                                      'inference and rule consequents; it does not isolate TSK structure alone.')
    result['DE_fits'] = len(FEATURES) * len(folds) if with_calibration else 0
    result['uncalibrated_consequent_fits'] = 0
    result['source'] = dict(scoring='analysis/reference_free_fuzzy.py', bootstrap='analysis/studies/support.py',
                            reproduction='analysis/studies/uncalibrated_rules.py')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('uncalibrated_run'))
    parser.add_argument('--with-calibration', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or output.is_relative_to(ROOT / 'results') or output.is_relative_to(ROOT / 'models'):
        raise ValueError('Choose a separate output directory; archived results and models are read-only.')
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'results.json'
    if destination.exists():
        raise FileExistsError('Use a fresh output directory to preserve previous results.')
    with threadpool_limits(limits=1):
        result = run(args.with_calibration)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for row in result['rows']:
        print(row['model'], 'LPIPS:', row['lpips']['mean'], 'NIQE:', row['niqe']['mean'])


if __name__ == '__main__':
    main()
