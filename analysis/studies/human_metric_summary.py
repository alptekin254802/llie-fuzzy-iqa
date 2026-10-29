"""Summarize pretrained metrics against human scores."""
import json, sys, numpy as np, pandas as pd
import analysis.studies.common as c
LOWER_BETTER = {'niqe', 'brisque', 'lpips', 'lpips-vgg', 'dists'}
res = {}
for ds in ['squarelol', 'lieq', 'rlie']:
    for suf in ['', '_native']:
        f = c.OUT / f'human_metrics_{ds}{suf}.csv'
        if not f.exists():
            continue
        X = pd.read_csv(f).merge(pd.read_csv(c.OUT / f'feat_{ds}_area256.csv')[['name', 'scene', 'mos', 'entropy', 'contrast', 'sharpness']], on='name')
        signs, mfs, theta = c.lol_frozen_model()
        X['fuzzy_frozen'] = c.fuzzy_predict(X, signs, mfs, theta)
        mos = X['mos'].to_numpy(float)
        sc = X['scene'].astype(str).to_numpy()
        r = {}
        for m in [k for k in X.columns if k not in ('name', 'scene', 'mos', 'entropy', 'contrast', 'sharpness')]:
            if X[m].isna().any():
                continue
            p = -X[m].to_numpy(float) if m in LOWER_BETTER else X[m].to_numpy(float)
            s, k, pl = c.iqa_scores(p, mos)
            w = c.within_group(p, mos, sc)
            r[m] = {'srcc': round(s, 3), 'plcc': round(pl, 3), 'within_scene_srcc': round(w['mean_srcc'], 3), 'pair_acc': round(w['pair_acc'], 3), 'top1': round(w['top1'], 3)}
        res[ds + suf] = r
        print(ds + suf)
        print(pd.DataFrame(r).T.sort_values('within_scene_srcc', ascending=False).to_string())
(c.OUT / 'human_metrics_analysis.json').write_text(json.dumps(res, indent=1))
