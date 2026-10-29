"""Reproduce targeted and random calibration-scene removal with published NR-free."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from analysis import reference_free_fuzzy as ref
from evaluation.frozen_scorer import FrozenScorer

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'results/studies/targeted_overlap'
FEATURES=['entropy','contrast','sharpness']

def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\r\n')

def summarize(results,random):
    assert len(random)==20 and random.subset_seed.tolist()==list(range(1,21))
    assert random.optimizer_seed.eq(42).all() and random.removed_scenes.eq(231).all()
    targeted=results[results.model.eq('lol254_no_shared')].set_index('square_subset')
    rows=[]
    for subset in ['all_290','nonshared_59']:
        for metric in ['within_scene_srcc','pooled_srcc']:
            values=random[subset+'_'+metric].to_numpy()
            reference=float(targeted.loc[subset,metric])
            rows.append(dict(test_subset=subset,metric=metric,repetitions=len(values),random_mean=float(values.mean()),random_min=float(values.min()),random_max=float(values.max()),overlap_removed_reference=reference,random_lower_count=int(np.sum(values<reference)),random_equal_count=int(np.sum(values==reference)),random_higher_count=int(np.sum(values>reference)),strict_lower_percentile=float(100*np.mean(values<reference))))
    return pd.DataFrame(rows)

def reproduce(out):
    """Keep original join order, float parsing, fitting and frozen inference unchanged."""
    threadpool_limits(1)
    assert (ref.DE_POPSIZE,ref.DE_MAXITER,ref.SEED,ref.DE_WORKERS)==(8,40,42,1)
    assert not out.exists(), 'Choose a new output directory to preserve stored runs.'
    out.mkdir(parents=True)
    inputs=[ROOT/'results'/n for n in ['nr_features.csv','perceptual_metrics.csv','fuzzy_enhancement_results.csv']]
    nf,pc,fz=[pd.read_csv(p) for p in inputs]
    lol=nf.merge(pc[['filename','method','lpips','niqe','brisque']],on=['filename','method']).merge(fz[['filename','method','psnr','ssim','fuzzy_score']],on=['filename','method']).dropna().reset_index(drop=True)
    square=pd.read_csv(DATA/'squarelol_features.csv',dtype={'scene':str})
    matches=pd.read_csv(ROOT/'results/studies/records/squarelol_lol_scene_matches.csv')
    shared=matches[matches.shared_source.eq(True)]
    shared_lol=set(shared.best_our485);shared_square=set(shared.squarelol)
    assert len(lol)==2425 and lol.filename.nunique()==485 and lol.groupby('filename').size().eq(5).all()
    assert len(square)==2900 and square.scene.nunique()==290 and square.groupby('scene').size().eq(10).all()
    assert len(shared_lol)==len(shared_square)==231
    new=~square.scene.isin(shared_square).to_numpy()
    assert square.loc[new,'scene'].nunique()==59 and new.sum()==590
    y=square.mos.to_numpy()
    groups=[np.asarray(ix) for ix in square.groupby('scene',sort=True).indices.values()]

    def fit(train,label):
        target=-train.lpips.to_numpy()
        oriented=ref.orient(train,FEATURES,target)
        mfs=[ref.mfset(x) for x in oriented]
        fire,fsum=ref.build_firing(oriented)
        start=time.perf_counter()
        theta=ref.calibrate(fire,fsum,target,np.arange(len(train)),0.)
        params=dict(signs={f:bool(spearmanr(train[f].to_numpy(),target).statistic<0) for f in FEATURES},mfs=mfs,theta=theta.tolist(),t_norm='minimum',feature_order=FEATURES,training_scenes=int(train.filename.nunique()),training_images=len(train),target='-LPIPS (raw)',epsilon=1e-9,optimizer_seed=42)
        elapsed=time.perf_counter()-start
        dump(out/('model_'+label+'.json'),params)
        return params,elapsed

    def evaluate(params,label):
        prediction=FrozenScorer(params).predict(square[FEATURES].to_numpy())
        pd.DataFrame(dict(name=square.name,scene=square.scene,mos=y,prediction=prediction,nonshared_59=new)).to_csv(out/('predictions_'+label+'.csv'),index=False,lineterminator='\r\n')
        metrics={}
        for subset,mask in [('all_290',np.ones(len(square),dtype=bool)),('nonshared_59',new)]:
            values=[float(spearmanr(prediction[ix],y[ix]).statistic) if np.ptp(prediction[ix])>0 else 0. for ix in groups if mask[ix[0]] and np.ptp(y[ix])>0]
            metrics[subset+'_within_scene_srcc']=float(np.mean(values))
            metrics[subset+'_pooled_srcc']=ref.rho(prediction[mask],y[mask])
        return metrics

    rows=[];baseline=None
    for label,train in [('lol485',lol),('lol254_no_shared',lol.loc[~lol.filename.isin(shared_lol)].reset_index(drop=True))]:
        params,_=fit(train,label);metrics=evaluate(params,label)
        if baseline is None:
            baseline=metrics
            assert f'{metrics["all_290_within_scene_srcc"]:.3f}'=='0.381'
            assert f'{metrics["nonshared_59_within_scene_srcc"]:.3f}'=='0.391'
        for subset,n in [('all_290',290),('nonshared_59',59)]:
            within=metrics[subset+'_within_scene_srcc'];pooled=metrics[subset+'_pooled_srcc']
            rows.append(dict(model=label,lol_training_scenes=params['training_scenes'],lol_training_images=len(train),square_subset=subset,test_scenes=n,test_images=n*10,within_scene_srcc=within,pooled_srcc=pooled,delta_within_vs_485=within-baseline[subset+'_within_scene_srcc'],delta_pooled_vs_485=pooled-baseline[subset+'_pooled_srcc']))
        print(label,metrics,flush=True)
    results=pd.DataFrame(rows);results.to_csv(out/'results.csv',index=False,lineterminator='\r\n')
    random=[];scenes=np.array(sorted(lol.filename.unique()))
    for seed in range(1,21):
        removed=set(np.random.default_rng(seed).choice(scenes,size=231,replace=False).tolist())
        train=lol.loc[~lol.filename.isin(removed)].reset_index(drop=True)
        retained=sorted(set(train.filename))
        assert len(retained)==254 and len(train)==1270
        digest=hashlib.sha256(('\n'.join(retained)+'\n').encode()).hexdigest()
        params,seconds=fit(train,f'random_{seed:02d}');metrics=evaluate(params,f'random_{seed:02d}')
        random.append(dict(subset_seed=seed,optimizer_seed=42,lol_training_scenes=254,lol_training_images=1270,removed_scenes=231,retained_shared_scenes=len(set(retained)&shared_lol),subset_sha256=digest,calibration_seconds=seconds,**metrics))
        pd.DataFrame(random).to_csv(out/'random_subsets.csv',index=False,lineterminator='\r\n')
        print(f'Random subset {seed}/20 complete',flush=True)
    random=pd.DataFrame(random)
    summarize(results,random).to_csv(out/'random_subsets_summary.csv',index=False,lineterminator='\r\n')
    # Runtime is deliberately excluded from deterministic reproduction checks.
    validation={}
    for name,frame in [('results.csv',results),('random_subsets.csv',random)]:
        archived=pd.read_csv(DATA/name)
        for col in archived.columns:
            if col=='calibration_seconds':continue
            if pd.api.types.is_numeric_dtype(archived[col]):
                error=float(np.max(np.abs(archived[col].to_numpy()-frame[col].to_numpy())))
                validation[name+':'+col]=error
                assert error<1e-12,(name,col,error)
            else:assert archived[col].tolist()==frame[col].tolist(),(name,col)
    dump(out/'reproduction_check.json',{'status':'PASS','maximum_absolute_differences':validation})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,help='New directory for full calibration reproduction; omit for CSV-only verification.')
    args=parser.parse_args()
    if args.out:reproduce(args.out)
    else:
        result=summarize(pd.read_csv(DATA/'results.csv'),pd.read_csv(DATA/'random_subsets.csv'))
        archived=pd.read_csv(DATA/'random_subsets_summary.csv')
        pd.testing.assert_frame_equal(result,archived,check_exact=False,rtol=0,atol=1e-12)
        print(result.to_string(index=False))
        print('Stored summary verified.')

if __name__=='__main__':main()
