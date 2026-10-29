"""Recompute objective-anchor and demo diagnostics from stored numeric scores.

Run from the repository root: python -B -m analysis.audit_derived
No training, image inference, or image files are required.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from evaluation.frozen_scorer import FrozenScorer
ROOT=Path(__file__).resolve().parents[1]
def main():
    sources=[]
    def csv(name):
        p=ROOT/name;sources.append(p);return pd.read_csv(p)
    keys=['filename','method']
    q=csv('results/validation_predictions.csv')
    for path,cols in [('results/perceptual_metrics.csv',['brisque']),('results/modern_nriqa.csv',['musiq','maniqa','clipiqa']),('results/clipiqa_plus_scores.csv',['clipiqa_plus']),('results/studies/records/lol_svr_predictions.csv',['oof_svr'])]:
        d=csv(path)
        if 'fold' in d:
            check=q[keys+['fold']].merge(d[keys+['fold']],on=keys,validate='one_to_one')
            assert (check.fold_x==check.fold_y).all()
        q=q.merge(d[keys+cols],on=keys,validate='one_to_one')
    assert len(q)==2425 and not q.isna().any().any()
    models={'MUSIQ':'musiq','MANIQA':'maniqa','CLIP-IQA':'clipiqa','CLIP-IQA+':'clipiqa_plus','Linear':'oof_linear','SVR (RBF)':'oof_svr','Random forest':'oof_rf'}
    folds=[];summary={}
    for model,col in models.items():
        summary[model]={}
        for anchor in ['lpips','niqe','brisque']:
            vals=[]
            for fold,g in q.groupby('fold'):
                assert len(g)==485
                v=float(spearmanr(g[col],-g[anchor]).statistic);vals.append(v)
                folds.append(dict(model=model,anchor='negative_'+anchor,fold=int(fold),n=len(g),srcc=v))
            summary[model][anchor]={'mean':float(np.mean(vals)),'std':float(np.std(vals,ddof=0))}
    cross=csv('results/cross_dataset.csv');model=FrozenScorer();pred=model.predict(cross[['entropy','contrast','sharpness']])
    sources.append(ROOT/'models/nr_free.json')
    transfer={a:{'entropy':float(spearmanr(cross.entropy,sgn*cross[a]).statistic),'nr_free':float(spearmanr(pred,sgn*cross[a]).statistic)} for a,sgn in [('musiq',1),('niqe',-1)]}
    exp=csv('results/reviewer_experiments.csv');exp=exp[exp.experiment.eq('leakage')].set_index('model')
    gap=float(exp.loc['FR-calibrated','global_mean']-exp.loc['NR-free','global_mean'])
    v=csv('results/studies/records/demo_frame_changes.csv');assert v.frame.is_unique and v.frame.is_monotonic_increasing
    corr=float(spearmanr(v.input_frame_change.iloc[1:],np.abs(np.diff(v.fixed_score))).statistic)
    result={'protocol':'Five stored scene-held-out LOL folds; Spearman against negative anchor, arithmetic mean and population SD; no refitting.','anchor_correlations':summary,'cross_dataset':transfer,'fr_free_mean_gap':gap,'demo':{'frames':len(v),'paired_changes':len(v)-1,'score_change':'absolute consecutive difference of fixed_score','frame_change':'stored input_frame_change, first row excluded','spearman':corr},'sources_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    out=ROOT/'results/studies/audit_derived.json';out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\r\n')
    pd.DataFrame(folds).to_csv(ROOT/'results/studies/audit_anchor_folds.csv',index=False,lineterminator='\r\n')
    print(json.dumps({'gap':gap,'demo_srcc':corr,'models':list(models)},indent=2))
if __name__=='__main__':main()
