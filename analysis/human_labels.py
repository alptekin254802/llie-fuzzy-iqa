"""Join separately obtained human labels by dataset and filename, never by row order."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]

def read_csv(path):
    return pd.read_csv(path, dtype={'scene': str}, float_precision='round_trip')

def load_labels(dataset, root):
    root = Path(root)
    if dataset == 'lieq':
        mos = loadmat(root / 'MOS.mat')['MOSz'].ravel()
        names = loadmat(root / 'imgNames.mat')['imgNames']
        names = [str(names[i, 2][0]).replace('\\', '/').rsplit('/', 1)[-1]
                 for i in range(names.shape[0])]
        labels = pd.DataFrame({'name': names, 'mos': mos})
    elif dataset == 'rlie':
        labels = read_csv(root / 'normalized_scores.csv').rename(
            columns={'Image Name': 'name', 'Score': 'mos'})[['name', 'mos']]
    else:
        raise ValueError('Local label loading supports LIEQ and RLIE.')
    if labels.name.duplicated().any():
        raise ValueError(f'Duplicate image names in {dataset} labels')
    if not np.isfinite(labels.mos.to_numpy(float)).all():
        raise ValueError(f'Non-finite {dataset} labels')
    return labels.set_index('name').mos

def attach_local_mos(data, lieq_root=None, rlie_root=None):
    from analysis.studies.config import HUMAN_ROOTS
    result = data.copy()
    if result.duplicated(['dataset', 'name']).any():
        raise ValueError('Duplicate dataset/name keys in predictions')
    if 'mos' not in result:
        result['mos'] = np.nan
    for dataset, root in [('lieq', lieq_root), ('rlie', rlie_root)]:
        mask = result.dataset.eq(dataset)
        if not mask.any():
            continue
        labels = load_labels(dataset, root or HUMAN_ROOTS[dataset])
        values = result.loc[mask, 'name'].map(labels)
        if values.isna().any():
            missing = result.loc[mask, 'name'][values.isna()].tolist()
            raise ValueError(f'Missing {dataset} labels: {missing[:5]}')
        prior = result.loc[mask, 'mos']
        supplied = prior.notna()
        if supplied.any() and not np.allclose(prior[supplied], values[supplied], rtol=0, atol=1e-12):
            raise ValueError(f'Existing {dataset} labels disagree with the downloaded dataset')
        result.loc[mask, 'mos'] = values.to_numpy()
    if not np.isfinite(result.mos.to_numpy(float)).all():
        raise ValueError('Every image needs a finite human label before analysis')
    return result

def local_output(path):
    path = Path(path).resolve()
    # Full label joins are local artefacts, never released result files.
    if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / 'study_work'):
        raise ValueError('Inside the repository, write local label joins only under ignored study_work/')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT / 'results/lmm_extra/joined.csv')
    p.add_argument('--lieq-root', type=Path)
    p.add_argument('--rlie-root', type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'study_work/lmm_extra/joined.csv')
    a = p.parse_args()
    output = local_output(a.output)
    if output == a.input.resolve():
        p.error('Input and output must be different files')
    data = attach_local_mos(read_csv(a.input), a.lieq_root, a.rlie_root)
    data.to_csv(output, index=False)
    print(f'Joined {len(data)} images by dataset/name; wrote local analysis input to {output}')

if __name__ == '__main__':
    main()
